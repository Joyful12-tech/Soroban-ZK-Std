# Soroban-ZK-Std Gas Benchmarks

This suite measures the CPU instruction cost of zero-knowledge cryptographic
primitives within the Soroban VM using the `soroban-sdk` budget API.

Soroban enforces a strict computational budget of **400,000,000 instructions**
per transaction. The results below confirm that a complete Groth16 verification
(including pairing and public-input accumulation) fits well within this limit.

> For the full cost reference including wall-clock Criterion data and
> transaction planning tables, see [GAS.md](./GAS.md).

## Environment

| Property            | Value                            |
|---------------------|----------------------------------|
| Soroban SDK version | 25.3.0                           |
| Stellar Protocol    | 25                               |
| Target              | `wasm32-unknown-unknown`         |
| Measurement API     | `env.cost_estimate().budget().cpu_instruction_cost()` |
| Profile             | `release` (`opt-level = z`, `lto = true`) |

## Results

### Poseidon2 Hashing (host-function — Soroban measured)

`hash_to_field` delegates to `env.crypto_hazmat().poseidon2_permutation()`.

| Operation          | Inputs | CPU Instructions | % of 400 M budget |
|--------------------|-------:|-----------------:|:-----------------:|
| `poseidon2_hash`   | 1      | 1,007,753        | 0.25%             |
| `poseidon2_hash`   | 2      | 2,010,994        | 0.50%             |
| `poseidon2_hash`   | 4      | 3,024,708        | 0.75%             |

### BN254 Field & Curve Arithmetic (WASM guest — wall-clock estimate)

Operations implemented natively in `soroban-zk-core` run as WASM guest code.
The Soroban SDK mock environment does not meter guest WASM instructions,
so these figures use Criterion wall-clock timings × 3,000 instructions/µs.

| Operation          | Est. CPU Instructions | % of 100 M budget |
|--------------------|----------------------:|:-----------------:|
| `Fr::add`          | ~300                  | < 0.001%          |
| `Fr::mul`          | ~669                  | < 0.001%          |
| `Fr::invert`       | ~5,550,000            | 5.55%             |
| `Fq::mul`          | ~4,620                | 0.005%            |
| `Fq::invert`       | ~5,790,000            | 5.79%             |
| `G1::scalar_mul`   | ~69,000,000           | 69.0%             |
| `G1::msm` (n=2)    | ~138,140,000          | 138% †            |
| `G1::msm` (n=4)    | ~276,210,000          | 276% †            |
| `G1::msm` (n=8)    | ~552,350,000          | 552% †            |

† Exceeds the per-operation 100 M limit; split across multiple transactions.

### Groth16 Verification (composite — 1 public input)

Combines an MSM(2) WASM guest cost with a 4-pair native pairing check.

| Component            | CPU Instructions | Source      |
|----------------------|-----------------:|:-----------:|
| MSM(2) accumulator   | ~138,140,000     | wall-clock  |
| 4-pair pairing check | 29,327,515       | Soroban API |
| **Total estimate**   | **~167,467,515** |             |
| % of 400 M budget    | **~41.9 %**      |             |

The pairing-only cost is verified by CI assertion (`cost <= 400_000_000`).

## CI Integration

The instruction-budget suite runs via
`cargo test --release -p soroban-zk-std --bench instruction_cost -- --nocapture`
(or `make bench`):

- All single-primitive asserts enforce `cost ≤ 100,000,000`.
- `groth16_verify` asserts `cost ≤ 400,000,000`.

## CI regression gate (`.github/workflows/bench.yml`)

The Criterion suites (`g1_ops`, `montgomery_ops`, `bulletproofs_vs_groth16`) are
executed on every push to `main` and on every pull request that touches
`crates/**`:

| Event | What happens |
|---|---|
| push to `main` | Runs the suites with `--save-baseline main`, uploads the
`criterion-baseline` artifact and caches `target/criterion`. |
| pull request | Restores the cached baseline, runs the same suites with
`--baseline main`, and fails when a benchmark regressed by more than the
threshold. |
| `workflow_dispatch` | Manual compare run; set `store-baseline` to record a new
reference or `threshold` to override the gate. |

The job summary (and the step log) contains a Markdown table with the baseline
mean, current mean, relative change, 95 % confidence interval and a status per
benchmark.

### Threshold

The gate defaults to **10 %** and is configurable without touching the workflow:

- repository variable `BENCH_THRESHOLD` — a fraction (`0.10`) or a percentage
  (`10%`);
- the `threshold` input of a manual `workflow_dispatch` run (takes precedence);
- `BENCH_MIN_MEAN_NS` (repository variable) — ignore benchmarks whose current
  mean is faster than this many nanoseconds, for suites where sub-microsecond
  timings are pure noise.

### Noise handling

The comparison only fails when the measured change exceeds the threshold **and**
the 95 % confidence interval of the change excludes zero. Benchmarks above the
threshold with an interval that still contains zero are reported as `noisy` in
the summary but do not fail the job. The runner image is pinned to
`ubuntu-24.04`, the Rust toolchain to `1.100.0`, and every suite runs with
`--sample-size 100 --measurement-time 10 --warm-up-time 3`, so a single noisy
sample cannot trip the gate. If a run is flagged as noisy, re-run the job before
suspecting a regression.

### Re-baselining

Baselines are only valid on the same runner image and toolchain. Re-baseline
after an intentional performance change, a rustc/runner upgrade, or a benchmark
rename.

1. Reproduce locally (same toolchain as CI, `rust-toolchain.toml`):

   ```bash
   cargo bench -p soroban-zk-core --all-features -- --save-baseline main
   cargo bench -p soroban-zk-core --all-features -- --baseline main
   ```

2. Re-baseline in CI: run **Benchmark Regression → Run workflow** with
   `store-baseline` enabled on the commit whose numbers should become the
   reference (normally `main` after the intentional change is merged). That
   overwrites the cache key that pull requests restore.
3. Verify the next pull request restores the new baseline — the job log prints
   `Comparing against the stored 'main' baseline.` and the summary is not the
   "no comparison data" notice.

Run `python3 scripts/bench-compare.py --self-test` to exercise the comparison
logic (threshold parsing, boundary values, noise classification, missing
baseline) without running the benchmarks.
