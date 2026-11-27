//! Dudect-style statistical timing-leak harness for BN254 arithmetic
//! (Issue #689).
//!
//! The harness compares execution-time distributions of two input classes
//! (fixed vs. random, and low vs. high Hamming weight) and reports Welch's
//! t-statistic for Fr/Fq multiplication, Fr/Fq inversion and G1 scalar
//! multiplication.
//!
//! Run it with `make ct-test`. It is deliberately not part of
//! `cargo test --release`: timing measurements are noisy on shared runners, so
//! the regular suite only prints a skip notice unless `CT_TEST=1` is set.
//!
//! Environment variables:
//! - `CT_TEST=1`     enable the measurements (otherwise the binary is a no-op)
//! - `CT_BUDGET_MS`  measurement budget per comparison (default 1000 ms)
//! - `CT_BATCH`      samples per batch (default 500)
//! - `CT_THRESHOLD`  `|t|` below which a comparison passes (default 10.0)
//! - `CT_SEED`       PRNG seed, for reproducible runs (default 0x5eed0689)
//! - `CT_STRICT=1`   exit non-zero when a comparison exceeds the threshold
//! - `CT_OUT_DIR`    where the JSON/TSV report is written (default
//!   `target/ct-results`)

mod dudect;

use dudect::{run_comparison, Comparison, ComparisonResult, Rng, CLOCK};
use ethnum::u256;
use soroban_zk_core::{Bn254, G1Affine};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

const DEFAULT_BUDGET_MS: u64 = 1000;
const DEFAULT_BATCH: usize = 500;
const DEFAULT_THRESHOLD: f64 = 10.0;
const DEFAULT_SEED: u64 = 0x5eed_0689;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .unwrap_or(default)
}

fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("1" | "true" | "yes"))
}

/// A fixed, full-width field element well below both BN254 moduli.
fn fixed_element() -> u256 {
    u256::from_words(
        0x1f3a_5c7b_9d2e_4680_1357_9bdf_2468_ace1,
        0x0f1e_2d3c_4b5a_6978_8796_a5b4_c3d2_e1f0,
    )
}

/// A second fixed element, used as the partner operand for multiplication.
fn fixed_partner() -> u256 {
    u256::from_words(
        0x0bad_c0de_baad_f00d_dead_beef_cafe_babe,
        0x0123_4567_89ab_cdef_0123_4567_89ab_cdef,
    )
}

/// Uniform value in `[0, modulus)` from the harness PRNG.
fn random_element(rng: &mut Rng, modulus: u256) -> u256 {
    u256::from_words(rng.next_u128(), rng.next_u128()) % modulus
}

/// A value with a single set bit at position 2..=253. Both BN254 moduli are
/// larger than 2^253, so this is always a valid, non-zero field element.
fn low_hamming_weight(rng: &mut Rng) -> u256 {
    let bit = 2 + rng.below(252) as u32;
    u256::from(1u8) << bit
}

/// A value with exactly two set bits, used for scalar multiplication where
/// `0` and `1` take the documented fast paths.
fn two_bit_scalar(rng: &mut Rng) -> u256 {
    let first = 2 + rng.below(252) as u32;
    let mut second = 2 + rng.below(252) as u32;
    if second == first {
        second = if first == 253 { 252 } else { first + 1 };
    }
    (u256::from(1u8) << first) | (u256::from(1u8) << second)
}

/// The BN254 G1 generator, `(1, 2)`.
fn generator() -> G1Affine {
    G1Affine {
        x: u256::from(1u8),
        y: u256::from(2u8),
    }
}

fn main() {
    if !env_flag("CT_TEST") {
        println!("ct: dudect-style timing harness skipped (run `make ct-test`, or set CT_TEST=1)");
        return;
    }

    let budget_ms = env_u64("CT_BUDGET_MS", DEFAULT_BUDGET_MS);
    let batch = env_u64("CT_BATCH", DEFAULT_BATCH as u64).max(32) as usize;
    let threshold = env_f64("CT_THRESHOLD", DEFAULT_THRESHOLD);
    let strict = env_flag("CT_STRICT");
    let mut rng = Rng::new(env_u64("CT_SEED", DEFAULT_SEED));
    let budget = Duration::from_millis(budget_ms);

    let fr = Bn254::FR_MODULUS;
    let fq = Bn254::FQ_MODULUS;
    let (fixed_a, fixed_b) = (fixed_element(), fixed_partner());
    let point = generator();

    let mut results: Vec<ComparisonResult> = Vec::new();

    // ── Fr multiplication ───────────────────────────────────────────────────
    results.push(run_comparison(
        &Comparison::new("fr_mul: fixed vs random", "fixed", "random"),
        budget,
        batch,
        &mut rng,
        |class, rng| {
            let a = fixed_a;
            let b = if class == 0 {
                fixed_b
            } else {
                random_element(rng, fr)
            };
            (a, b)
        },
        |input: &(u256, u256)| Bn254::mul(input.0, input.1).as_u64(),
    ));
    results.push(run_comparison(
        &Comparison::new("fr_mul: low vs high HW", "low HW", "high HW"),
        budget,
        batch,
        &mut rng,
        |class, rng| {
            let a = fixed_a;
            let b = if class == 0 {
                low_hamming_weight(rng)
            } else {
                random_element(rng, fr)
            };
            (a, b)
        },
        |input: &(u256, u256)| Bn254::mul(input.0, input.1).as_u64(),
    ));

    // ── Fq multiplication ───────────────────────────────────────────────────
    results.push(run_comparison(
        &Comparison::new("fq_mul: fixed vs random", "fixed", "random"),
        budget,
        batch,
        &mut rng,
        |class, rng| {
            let a = fixed_a;
            let b = if class == 0 {
                fixed_b
            } else {
                random_element(rng, fq)
            };
            (a, b)
        },
        |input: &(u256, u256)| Bn254::mul_fq(input.0, input.1).as_u64(),
    ));
    results.push(run_comparison(
        &Comparison::new("fq_mul: low vs high HW", "low HW", "high HW"),
        budget,
        batch,
        &mut rng,
        |class, rng| {
            let a = fixed_a;
            let b = if class == 0 {
                low_hamming_weight(rng)
            } else {
                random_element(rng, fq)
            };
            (a, b)
        },
        |input: &(u256, u256)| Bn254::mul_fq(input.0, input.1).as_u64(),
    ));

    // ── Fr inversion (Fermat, exponent r - 2) ───────────────────────────────
    results.push(run_comparison(
        &Comparison::new("fr_invert: fixed vs random", "fixed", "random"),
        budget,
        batch,
        &mut rng,
        |class, rng| {
            if class == 0 {
                fixed_a
            } else {
                random_element(rng, fr)
            }
        },
        |input: &u256| Bn254::invert(*input).as_u64(),
    ));
    results.push(run_comparison(
        &Comparison::new("fr_invert: low vs high HW", "low HW", "high HW"),
        budget,
        batch,
        &mut rng,
        |class, rng| {
            if class == 0 {
                low_hamming_weight(rng)
            } else {
                random_element(rng, fr)
            }
        },
        |input: &u256| Bn254::invert(*input).as_u64(),
    ));

    // ── Fq inversion (Fermat, exponent q - 2) ───────────────────────────────
    results.push(run_comparison(
        &Comparison::new("fq_invert: fixed vs random", "fixed", "random"),
        budget,
        batch,
        &mut rng,
        |class, rng| {
            if class == 0 {
                fixed_a
            } else {
                random_element(rng, fq)
            }
        },
        |input: &u256| Bn254::invert_fq(*input).as_u64(),
    ));
    results.push(run_comparison(
        &Comparison::new("fq_invert: low vs high HW", "low HW", "high HW"),
        budget,
        batch,
        &mut rng,
        |class, rng| {
            if class == 0 {
                low_hamming_weight(rng)
            } else {
                random_element(rng, fq)
            }
        },
        |input: &u256| Bn254::invert_fq(*input).as_u64(),
    ));

    // ── G1 scalar multiplication ────────────────────────────────────────────
    results.push(run_comparison(
        &Comparison::new("g1_mul: fixed vs random", "fixed", "random"),
        budget,
        batch,
        &mut rng,
        |class, rng| {
            let scalar = if class == 0 {
                fixed_a
            } else {
                random_element(rng, fr)
            };
            (point, scalar)
        },
        |input: &(G1Affine, u256)| input.0.scalar_mul(input.1).x.as_u64(),
    ));
    results.push(run_comparison(
        &Comparison::new("g1_mul: low vs high HW", "low HW", "high HW"),
        budget,
        batch,
        &mut rng,
        |class, rng| {
            let scalar = if class == 0 {
                two_bit_scalar(rng)
            } else {
                random_element(rng, fr)
            };
            (point, scalar)
        },
        |input: &(G1Affine, u256)| input.0.scalar_mul(input.1).x.as_u64(),
    ));

    print_table(&results, threshold, budget_ms, batch);
    write_report(&results, threshold, budget_ms, batch);

    let worst = results.iter().map(|r| r.max_abs_t).fold(0.0_f64, f64::max);
    let leaking: Vec<&ComparisonResult> = results
        .iter()
        .filter(|r| r.max_abs_t >= threshold)
        .collect();

    if leaking.is_empty() {
        println!("\n✅ No comparison exceeded |t| = {threshold:.1} (worst |t| = {worst:.2}).");
    } else {
        println!(
            "\n⚠️  {} of {} comparison(s) exceeded |t| = {threshold:.1} (worst |t| = {worst:.2}):",
            leaking.len(),
            results.len()
        );
        for result in &leaking {
            println!("   - {} ({:.2})", result.label, result.max_abs_t);
        }
        println!("   Histogram-dependent timing is expected for the documented limitations");
        println!("   (variable mul_mod inner loop, G1 add identity branches). See");
        println!("   frontend/content/docs/core-concepts/constant-time-notes.mdx.");
        if strict {
            std::process::exit(1);
        }
    }
}

fn print_table(results: &[ComparisonResult], threshold: f64, budget_ms: u64, batch: usize) {
    println!(
        "dudect-style constant-time measurements: clock={CLOCK}, budget={budget_ms} ms/comparison, batch cap={batch}, threshold |t| < {threshold:.1}"
    );
    println!(
        "{:<32} {:<9} {:<9} {:>9} {:>9} {:>10} {:>8}  verdict",
        "comparison", "class A", "class B", "n A", "n B", "median A", "max|t|"
    );
    for result in results {
        let verdict = if result.max_abs_t >= threshold {
            "LEAK?"
        } else {
            "pass"
        };
        println!(
            "{:<32} {:<9} {:<9} {:>9} {:>9} {:>10} {:>8.2}  {}",
            result.label,
            result.class_a,
            result.class_b,
            result.n_a,
            result.n_b,
            result.median_a,
            result.max_abs_t,
            verdict
        );
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|crates| crates.parent())
        .map_or_else(|| PathBuf::from("."), |root| root.to_path_buf())
}

fn write_report(results: &[ComparisonResult], threshold: f64, budget_ms: u64, batch: usize) {
    // Cargo runs test binaries with the package directory as the working
    // directory, so anchor the default output path at the workspace root: CI
    // uploads `target/ct-results/` from there.
    let out_dir = match std::env::var("CT_OUT_DIR") {
        Ok(value) => PathBuf::from(value),
        Err(_) => workspace_root().join("target/ct-results"),
    };
    if let Err(error) = fs::create_dir_all(&out_dir) {
        eprintln!("ct: could not create {}: {error}", out_dir.display());
        return;
    }

    let mut json = String::new();
    let _ = writeln!(json, "{{");
    let _ = writeln!(json, "  \"tool\": \"dudect-style Welch t-test\",");
    let _ = writeln!(json, "  \"clock\": \"{CLOCK}\",");
    let _ = writeln!(json, "  \"budget_ms\": {budget_ms},");
    let _ = writeln!(json, "  \"batch\": {batch},");
    let _ = writeln!(json, "  \"threshold\": {threshold},");
    let _ = writeln!(json, "  \"cases\": [");
    for (index, result) in results.iter().enumerate() {
        let comma = if index + 1 == results.len() { "" } else { "," };
        let _ = writeln!(
            json,
            "    {{\"name\": \"{}\", \"class_a\": \"{}\", \"class_b\": \"{}\", \"n_a\": {}, \"n_b\": {}, \"median_a_ticks\": {}, \"median_b_ticks\": {}, \"final_t\": {:.4}, \"max_abs_t\": {:.4}, \"batches\": {}, \"verdict\": \"{}\"}}{comma}",
            result.label,
            result.class_a,
            result.class_b,
            result.n_a,
            result.n_b,
            result.median_a,
            result.median_b,
            result.final_t,
            result.max_abs_t,
            result.batches,
            if result.max_abs_t >= threshold {
                "leak-suspected"
            } else {
                "pass"
            }
        );
    }
    let _ = writeln!(json, "  ]");
    let _ = writeln!(json, "}}");

    let mut tsv = String::from(
        "comparison\tclass_a\tclass_b\tn_a\tn_b\tmedian_a\tmedian_b\tfinal_t\tmax_abs_t\tbatches\n",
    );
    for result in results {
        let _ = writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}",
            result.label,
            result.class_a,
            result.class_b,
            result.n_a,
            result.n_b,
            result.median_a,
            result.median_b,
            result.final_t,
            result.max_abs_t,
            result.batches
        );
    }

    for (name, contents) in [
        ("ct-report.json", json.as_str()),
        ("ct-report.tsv", tsv.as_str()),
    ] {
        let path = out_dir.join(name);
        match fs::write(&path, contents) {
            Ok(()) => println!("ct: wrote {}", path.display()),
            Err(error) => eprintln!("ct: could not write {}: {error}", path.display()),
        }
    }
}
