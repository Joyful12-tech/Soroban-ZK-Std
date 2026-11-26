//! Minimal dudect-style measurement engine (Issue #689).
//!
//! Each comparison runs the same operation on two classes of inputs and asks
//! Welch's t-test whether their execution-time distributions differ. A large
//! `|t|` means the timing (and therefore the control flow or memory access
//! pattern) depends on the input class.
//!
//! Methodology (following Reparaz, Balasch & Verbauwhede, "dudect"):
//! 1. Interleave the two classes in a single measurement loop so that slow
//!    drift (CPU frequency, scheduler pressure) affects both classes equally.
//! 2. Time a single execution per sample, never an average over a loop: an
//!    averaging loop would hide per-input differences.
//! 3. Crop the slowest and fastest 1% of each batch. Interrupts and migration
//!    produce a heavy tail that would otherwise dominate the variance.
//! 4. Accumulate the cropped samples and recompute the test after every batch,
//!    reporting the largest `|t|` seen. A single large value means the timing
//!    distributions separate; a fluctuating value around zero means they do not.

use std::time::{Duration, Instant};

/// Smallest batch worth running the statistics on.
const MIN_BATCH: usize = 64;

/// Deterministic xorshift64* PRNG.
///
/// Keeping our own generator avoids a dev-dependency and, more importantly,
/// makes a failing run reproducible from its seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // A zero state would make xorshift emit only zeros.
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn next_u128(&mut self) -> u128 {
        ((self.next_u64() as u128) << 64) | self.next_u64() as u128
    }

    /// Uniform-ish value in `[0, range)`. The modulo bias is irrelevant here:
    /// inputs are chosen outside the measured region and only need to differ
    /// between classes.
    pub fn below(&mut self, range: u64) -> u64 {
        self.next_u64() % range
    }

    pub fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

/// Name of the clock used for the samples, recorded in the report.
pub const CLOCK: &str = if cfg!(target_arch = "x86_64") {
    "rdtsc"
} else {
    "instant"
};

/// Reads a monotonic timestamp in cycles (x86_64) or nanoseconds (fallback).
///
/// On x86_64 an `lfence` precedes `rdtsc` so the timestamp is not hoisted
/// above the measured operation by the out-of-order core.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn ticks() -> u64 {
    // SAFETY: `rdtsc` and `lfence` are unprivileged on every x86_64 CPU and
    // `_mm_lfence` is available unconditionally (SSE2 is part of the baseline).
    unsafe {
        core::arch::x86_64::_mm_lfence();
        core::arch::x86_64::_rdtsc()
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn ticks() -> u64 {
    use std::sync::OnceLock;

    static START: OnceLock<Instant> = OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

/// One named fixed-vs-random or low-vs-high comparison.
pub struct Comparison {
    pub label: &'static str,
    pub class_a: &'static str,
    pub class_b: &'static str,
}

impl Comparison {
    pub const fn new(label: &'static str, class_a: &'static str, class_b: &'static str) -> Self {
        Self {
            label,
            class_a,
            class_b,
        }
    }
}

/// Aggregated statistics for one comparison.
pub struct ComparisonResult {
    pub label: &'static str,
    pub class_a: &'static str,
    pub class_b: &'static str,
    pub n_a: usize,
    pub n_b: usize,
    pub median_a: u64,
    pub median_b: u64,
    pub final_t: f64,
    pub max_abs_t: f64,
    pub batches: usize,
}

/// Runs one comparison until `budget` elapses and returns its statistics.
///
/// `make_input(class, rng)` produces the operand for class 0 (`class_a`) or 1
/// (`class_b`); `op` returns a value that is consumed by `black_box` so the
/// optimiser cannot delete the work being timed.
pub fn run_comparison<I, M, O>(
    comparison: &Comparison,
    budget: Duration,
    batch_cap: usize,
    rng: &mut Rng,
    mut make_input: M,
    mut op: O,
) -> ComparisonResult
where
    M: FnMut(usize, &mut Rng) -> I,
    O: FnMut(&I) -> u64,
{
    // Warm up: fault in the code pages, prime caches and branch predictors so
    // the first batch is not dominated by one-off costs.
    for _ in 0..64 {
        for class in 0..2 {
            let input = make_input(class, rng);
            std::hint::black_box(op(std::hint::black_box(&input)));
        }
    }

    // Size a batch so that roughly six batches fit into the budget. A fixed
    // batch would make cheap operations run a single huge batch (long stretches
    // of one input class invite drift) and slow ones (G1 scalar multiplication)
    // never complete a second batch, leaving the t-test with too few samples.
    let estimate_start = Instant::now();
    for _ in 0..8 {
        let input = make_input(0, rng);
        std::hint::black_box(op(std::hint::black_box(&input)));
    }
    let per_op = estimate_start.elapsed() / 8;
    let target_batch_time = (budget / 6).max(Duration::from_micros(100));
    let auto = if per_op.is_zero() {
        batch_cap
    } else {
        (target_batch_time.as_secs_f64() / per_op.as_secs_f64()) as usize
    };
    let batch = auto.clamp(MIN_BATCH, batch_cap.max(MIN_BATCH));

    let mut all_a: Vec<u64> = Vec::new();
    let mut all_b: Vec<u64> = Vec::new();
    let mut batch_a: Vec<u64> = Vec::with_capacity(batch);
    let mut batch_b: Vec<u64> = Vec::with_capacity(batch);
    let mut max_abs_t = 0.0_f64;
    let mut final_t = 0.0_f64;
    let mut batches = 0_usize;

    let started = Instant::now();
    while started.elapsed() < budget {
        batch_a.clear();
        batch_b.clear();

        for _ in 0..batch {
            let class = usize::from(rng.next_bool());
            let input = make_input(class, rng);

            let t0 = ticks();
            let out = op(std::hint::black_box(&input));
            let t1 = ticks();
            std::hint::black_box(out);

            let elapsed = t1.wrapping_sub(t0);
            if class == 0 {
                batch_a.push(elapsed);
            } else {
                batch_b.push(elapsed);
            }
        }

        // A batch can come out lopsided; cropping needs a few samples per class.
        if batch_a.len() < 8 || batch_b.len() < 8 {
            continue;
        }

        crop(&mut batch_a);
        crop(&mut batch_b);
        all_a.extend_from_slice(&batch_a);
        all_b.extend_from_slice(&batch_b);

        let t = welch_t(&all_a, &all_b);
        if t.abs() > max_abs_t {
            max_abs_t = t.abs();
        }
        final_t = t;
        batches += 1;

        // Safety valve for very fast operations: ~2M samples is far more than
        // the test needs and keeps memory bounded.
        if all_a.len() + all_b.len() >= 2_000_000 {
            break;
        }
    }

    ComparisonResult {
        label: comparison.label,
        class_a: comparison.class_a,
        class_b: comparison.class_b,
        n_a: all_a.len(),
        n_b: all_b.len(),
        median_a: median(&all_a),
        median_b: median(&all_b),
        final_t,
        max_abs_t,
        batches,
    }
}

/// Drops the fastest and slowest 1% of a batch (interrupt/outlier cropping).
fn crop(samples: &mut Vec<u64>) {
    samples.sort_unstable();
    let cut = (samples.len() / 100).max(1);
    if samples.len() <= 2 * cut {
        return;
    }
    samples.drain(..cut);
    samples.truncate(samples.len() - cut);
}

fn mean_and_variance(samples: &[u64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().map(|&v| v as f64).sum::<f64>() / n;
    let variance = samples
        .iter()
        .map(|&v| {
            let delta = v as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / (n - 1.0);
    (mean, variance)
}

/// Welch's two-sample t-statistic (unequal variances and sample counts).
fn welch_t(a: &[u64], b: &[u64]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let (mean_a, var_a) = mean_and_variance(a);
    let (mean_b, var_b) = mean_and_variance(b);
    let denominator = (var_a / a.len() as f64 + var_b / b.len() as f64).sqrt();
    if denominator <= f64::EPSILON {
        0.0
    } else {
        (mean_a - mean_b) / denominator
    }
}

fn median(samples: &[u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        // Average the two middle values. The slice is sorted, so `hi - lo`
        // cannot underflow and `lo + (hi - lo) / 2` cannot overflow.
        let lo = sorted[mid - 1];
        let hi = sorted[mid];
        lo + (hi - lo) / 2
    } else {
        sorted[mid]
    }
}
