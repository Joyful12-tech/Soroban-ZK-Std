#!/usr/bin/env python3
"""Fail CI when Criterion benchmarks regress against a stored baseline.

The benchmark workflow (.github/workflows/bench.yml) runs the Criterion suites
on `main` with `--save-baseline main` and caches `target/criterion`.  On a pull
request the same suites run with `--baseline main`, which makes Criterion write
`target/criterion/<group>/<bench>/change/estimates.json` holding the relative
change of every benchmark.  This script reads those files and fails when a
benchmark slowed down by more than the configured threshold.

Noise handling: a benchmark only fails the gate when its point estimate is
above the threshold *and* the 95 % confidence interval of the change excludes
zero (Criterion's own significance signal).  Changes above the threshold whose
interval still contains zero are reported as "noisy" but do not fail the job;
the recommended fix is a second run, not a re-baseline.

Usage
-----
  python3 scripts/bench-compare.py --self-test
  python3 scripts/bench-compare.py --criterion-dir target/criterion \\
      --threshold 0.10 --require-baseline

Exit codes: 0 = no regression, 1 = regression (or a required baseline is
missing), 2 = invalid arguments.

Threshold: a fraction ("0.10") or a percentage ("10%").  It can also be supplied
through the BENCH_THRESHOLD environment variable (the workflow exposes it as the
`BENCH_THRESHOLD` repository variable and as a workflow_dispatch input).

The Markdown report is printed to stdout and appended to $GITHUB_STEP_SUMMARY
when that file is set (override with --summary).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

DEFAULT_THRESHOLD = "0.10"

STATUS_UNCHANGED = "unchanged"
STATUS_REGRESSION = "regression"
STATUS_IMPROVEMENT = "improvement"
STATUS_NOISY = "noisy"
STATUS_BELOW_FLOOR = "below-floor"
STATUS_NEW = "new"

STATUS_LABELS = {
    STATUS_UNCHANGED: "➖ unchanged",
    STATUS_REGRESSION: "❌ regression",
    STATUS_IMPROVEMENT: "✅ improvement",
    STATUS_NOISY: "⚠️ noisy (CI includes 0)",
    STATUS_BELOW_FLOOR: "➖ below noise floor",
    STATUS_NEW: "🆕 no baseline",
}


@dataclass(frozen=True)
class Change:
    """One Criterion benchmark compared against the stored baseline."""

    benchmark: str
    change: float
    ci_lower: float
    ci_upper: float
    new_mean_ns: float | None = None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_threshold(raw: object) -> float:
    """Parse a regression threshold given as a fraction or a percentage."""
    text = str(raw).strip()
    if not text:
        raise ValueError("threshold must not be empty")
    if text.endswith("%"):
        value = float(text[:-1]) / 100.0
    else:
        value = float(text)
        if value >= 1.0:
            raise ValueError(
                f"threshold {text!r} looks like a percentage; "
                "write a fraction such as 0.10 or a percentage such as 10%"
            )
    if not 0.0 <= value < 1.0:
        raise ValueError(f"threshold {text!r} must be between 0 and 1 (or 0% and 100%)")
    return value


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _new_mean_ns(bench_dir: Path) -> float | None:
    mean = _read_json(bench_dir / "new" / "estimates.json").get("mean")
    if isinstance(mean, dict) and "point_estimate" in mean:
        return float(mean["point_estimate"])
    return None


def load_changes(criterion_dir: Path) -> list[Change]:
    """Collect every Criterion `change/estimates.json` under `criterion_dir`."""
    changes: list[Change] = []
    criterion_dir = Path(criterion_dir)
    if not criterion_dir.is_dir():
        return changes
    for change_dir in sorted(path for path in criterion_dir.rglob("change") if path.is_dir()):
        data = _read_json(change_dir / "estimates.json")
        mean = data.get("mean")
        if not isinstance(mean, dict) or "point_estimate" not in mean:
            continue
        benchmark = change_dir.parent.relative_to(criterion_dir).as_posix()
        if not benchmark or benchmark == "report" or benchmark.startswith("report/"):
            continue
        interval = mean.get("confidence_interval")
        interval = interval if isinstance(interval, dict) else {}
        point = float(mean["point_estimate"])
        changes.append(
            Change(
                benchmark=benchmark,
                change=point,
                ci_lower=float(interval.get("lower_bound", point)),
                ci_upper=float(interval.get("upper_bound", point)),
                new_mean_ns=_new_mean_ns(change_dir.parent),
            )
        )
    return changes


# ---------------------------------------------------------------------------
# Classification / formatting
# ---------------------------------------------------------------------------

def classify(change: Change, threshold: float, min_mean_ns: float = 0.0) -> str:
    """Classify one benchmark comparison against the gate."""
    if change.new_mean_ns is not None and change.new_mean_ns < min_mean_ns:
        return STATUS_BELOW_FLOOR
    if abs(change.change) <= threshold:
        return STATUS_UNCHANGED
    significant = change.ci_lower > 0.0 if change.change > 0 else change.ci_upper < 0.0
    if not significant:
        return STATUS_NOISY
    return STATUS_REGRESSION if change.change > 0 else STATUS_IMPROVEMENT


def format_duration(nanoseconds: float | None) -> str:
    """Format a wall-clock duration in the largest sensible unit."""
    if nanoseconds is None:
        return "—"
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if nanoseconds >= scale:
            return f"{nanoseconds / scale:.3g} {unit}"
    return f"{nanoseconds:,.0f} ns"


def format_percent(value: float) -> str:
    return f"{value * 100:+.2f}%"


def baseline_mean_ns(change: Change) -> float | None:
    if change.new_mean_ns is None or change.change <= -1.0:
        return None
    return change.new_mean_ns / (1.0 + change.change)


def render_report(
    changes: list[Change],
    threshold: float,
    min_mean_ns: float = 0.0,
    max_rows: int = 50,
) -> list[str]:
    """Build the Markdown summary (job summary + logs)."""
    classified = [(change, classify(change, threshold, min_mean_ns)) for change in changes]
    classified.sort(key=lambda item: item[0].change, reverse=True)
    regressions = [change for change, status in classified if status == STATUS_REGRESSION]

    lines = [
        "## 📊 Criterion benchmark regression gate",
        "",
        f"Regression threshold: **{threshold * 100:.2f}%** "
        f"(a benchmark fails when its mean slows by more than this and the 95% "
        f"confidence interval excludes 0).",
        "",
        f"Compared **{len(classified)}** benchmark(s): "
        f"**{len(regressions)}** regression(s), "
        f"**{sum(1 for _, status in classified if status == STATUS_IMPROVEMENT)}** improvement(s), "
        f"**{sum(1 for _, status in classified if status == STATUS_UNCHANGED)}** unchanged, "
        f"**{sum(1 for _, status in classified if status == STATUS_NOISY)}** noisy.",
        "",
        "| Benchmark | Baseline mean | Current mean | Change | 95% CI | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for change, status in classified[:max_rows]:
        lines.append(
            "| `{bench}` | {base} | {current} | {delta} | {low} … {high} | {status} |".format(
                bench=change.benchmark,
                base=format_duration(baseline_mean_ns(change)),
                current=format_duration(change.new_mean_ns),
                delta=format_percent(change.change),
                low=format_percent(change.ci_lower),
                high=format_percent(change.ci_upper),
                status=STATUS_LABELS.get(status, status),
            )
        )
    if len(classified) > max_rows:
        lines.append("")
        lines.append(f"_{len(classified) - max_rows} further benchmark(s) omitted from the table._")
    if regressions:
        lines.append("")
        lines.append("### ❌ Regressions above threshold")
        lines.append("")
        for change in regressions:
            lines.append(f"- `{change.benchmark}`: {format_percent(change.change)}")
        lines.append("")
        lines.append(
            "Investigate the slow path, or — if the slowdown is intentional — "
            "re-baseline on `main` as documented in BENCHMARK.md."
        )
    lines.append("")
    return lines


def render_missing_baseline() -> list[str]:
    return [
        "## 📊 Criterion benchmark regression gate",
        "",
        "⚠️ No Criterion comparison data was found, so no regression check ran.",
        "",
        "This is expected for the first benchmark run on a new baseline (or after "
        "the benchmark names changed). A baseline is stored by the "
        "`Store main baseline` job; re-run that job (or merge to `main`) and "
        "re-run this workflow.",
        "",
    ]


def emit_markdown(lines: list[str], destination: str | None) -> None:
    text = "\n".join(lines) + "\n"
    print(text)
    if not destination:
        return
    try:
        with open(destination, "a", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:  # pragma: no cover - only hit on misconfigured runners
        print(f"::warning::Could not append to {destination}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail when Criterion benchmarks regress past a threshold.",
    )
    parser.add_argument(
        "--criterion-dir",
        default="target/criterion",
        help="Criterion output directory (default: target/criterion)",
    )
    parser.add_argument(
        "--threshold",
        default=os.environ.get("BENCH_THRESHOLD", DEFAULT_THRESHOLD),
        help="Regression threshold as a fraction (0.10) or percentage (10%%)",
    )
    parser.add_argument(
        "--min-mean-ns",
        type=float,
        default=float(os.environ.get("BENCH_MIN_MEAN_NS") or 0.0),
        help="Ignore benchmarks faster than this mean (ns) to avoid sub-µs noise",
    )
    parser.add_argument(
        "--require-baseline",
        action="store_true",
        help="Fail when no comparison data exists (a baseline was expected)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="Maximum table rows in the summary (default: 50)",
    )
    parser.add_argument(
        "--summary",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="Markdown file to append to (default: $GITHUB_STEP_SUMMARY)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the script's unit tests and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_tests()

    try:
        threshold = parse_threshold(args.threshold)
    except ValueError as exc:
        print(f"::error::Invalid threshold: {exc}", file=sys.stderr)
        return 2
    if args.min_mean_ns < 0:
        print("::error::--min-mean-ns must be >= 0", file=sys.stderr)
        return 2

    changes = load_changes(Path(args.criterion_dir))
    if not changes:
        emit_markdown(render_missing_baseline(), args.summary)
        print(
            f"::warning::No Criterion comparison data under {args.criterion_dir}",
            file=sys.stderr,
        )
        return 1 if args.require_baseline else 0

    lines = render_report(changes, threshold, args.min_mean_ns, max_rows=args.max_rows)
    emit_markdown(lines, args.summary)
    regressions = [
        change
        for change in changes
        if classify(change, threshold, args.min_mean_ns) == STATUS_REGRESSION
    ]
    return 1 if regressions else 0


# ---------------------------------------------------------------------------
# Unit tests (python3 scripts/bench-compare.py --self-test)
# ---------------------------------------------------------------------------

def _write_estimates(path: Path, point: float, lower: float | None = None,
                     upper: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bounds = {
        "confidence_level": 0.95,
        "lower_bound": point if lower is None else lower,
        "upper_bound": point if upper is None else upper,
        "standard_error": 0.0,
    }
    mean = {"confidence_interval": bounds, "point_estimate": point, "standard_error": 0.0}
    path.write_text(
        json.dumps({"mean": mean, "median": mean, "slope": mean, "std_dev": mean}),
        encoding="utf-8",
    )


def _fixture(root: Path, name: str = "g1_ops/g1_add", change: float = 0.25,
             lower: float = 0.12, upper: float = 0.40,
             new_mean_ns: float = 1_000.0) -> Path:
    criterion = root / "criterion"
    bench_dir = criterion / name
    _write_estimates(bench_dir / "new" / "estimates.json", new_mean_ns)
    _write_estimates(bench_dir / "change" / "estimates.json", change, lower, upper)
    return criterion


class ThresholdTests(unittest.TestCase):
    def test_accepts_fraction(self):
        self.assertAlmostEqual(parse_threshold("0.10"), 0.10)
        self.assertAlmostEqual(parse_threshold(0.05), 0.05)

    def test_accepts_percentage(self):
        self.assertAlmostEqual(parse_threshold("12.5%"), 0.125)
        self.assertAlmostEqual(parse_threshold("0%"), 0.0)

    def test_rejects_bare_percentage(self):
        with self.assertRaises(ValueError):
            parse_threshold("10")

    def test_rejects_empty_and_negative(self):
        with self.assertRaises(ValueError):
            parse_threshold("")
        with self.assertRaises(ValueError):
            parse_threshold("-0.1")
        with self.assertRaises(ValueError):
            parse_threshold("100%")


class LoadChangesTests(unittest.TestCase):
    def test_reads_criterion_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            criterion = _fixture(Path(tmp))
            changes = load_changes(criterion)
            self.assertEqual(len(changes), 1)
            change = changes[0]
            self.assertEqual(change.benchmark, "g1_ops/g1_add")
            self.assertAlmostEqual(change.change, 0.25)
            self.assertAlmostEqual(change.ci_lower, 0.12)
            self.assertAlmostEqual(change.new_mean_ns, 1_000.0)

    def test_reads_parameterised_benchmarks(self):
        with tempfile.TemporaryDirectory() as tmp:
            criterion = _fixture(Path(tmp), name="g1_ops/msm/8")
            self.assertEqual(
                [change.benchmark for change in load_changes(criterion)],
                ["g1_ops/msm/8"],
            )

    def test_ignores_missing_and_non_comparison_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            criterion = _fixture(Path(tmp))
            (criterion / "g1_ops" / "g1_add" / "change" / "estimates.json").unlink()
            (criterion / "g1_ops" / "g1_add" / "base").mkdir()
            _write_estimates(criterion / "g1_ops" / "g1_add" / "base" / "estimates.json", 0.9)
            self.assertEqual(load_changes(criterion), [])

    def test_missing_directory_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_changes(Path(tmp) / "absent"), [])


class ClassifyTests(unittest.TestCase):
    THRESHOLD = 0.10

    def _change(self, change: float, lower: float, upper: float,
                new_mean_ns: float = 1_000.0) -> Change:
        return Change("bench", change, lower, upper, new_mean_ns)

    def test_regression_above_threshold(self):
        self.assertEqual(
            classify(self._change(0.101, 0.02, 0.20), self.THRESHOLD),
            STATUS_REGRESSION,
        )

    def test_exact_threshold_is_not_a_regression(self):
        self.assertEqual(
            classify(self._change(0.10, 0.02, 0.20), self.THRESHOLD),
            STATUS_UNCHANGED,
        )

    def test_improvement(self):
        self.assertEqual(
            classify(self._change(-0.50, -0.70, -0.30), self.THRESHOLD),
            STATUS_IMPROVEMENT,
        )

    def test_above_threshold_but_inside_noise_band(self):
        self.assertEqual(
            classify(self._change(0.30, -0.05, 0.60), self.THRESHOLD),
            STATUS_NOISY,
        )

    def test_below_noise_floor(self):
        self.assertEqual(
            classify(self._change(0.50, 0.40, 0.60, new_mean_ns=500.0),
                     self.THRESHOLD, min_mean_ns=1_000.0),
            STATUS_BELOW_FLOOR,
        )


class ReportTests(unittest.TestCase):
    def test_summary_lists_regressions_and_counts(self):
        changes = [
            Change("g1_ops/g1_add", 0.25, 0.12, 0.40, 1_000.0),
            Change("g1_ops/msm/8", -0.30, -0.40, -0.20, 5_000.0),
        ]
        report = "\n".join(render_report(changes, 0.10))
        self.assertIn("g1_ops/g1_add", report)
        self.assertIn("❌ regression", report)
        self.assertIn("✅ improvement", report)
        self.assertIn("1** regression(s)", report)

    def test_duration_formatting(self):
        self.assertEqual(format_duration(None), "—")
        self.assertEqual(format_duration(950.0), "950 ns")
        self.assertEqual(format_duration(1_500.0), "1.5 µs")
        self.assertEqual(format_duration(2_500_000.0), "2.5 ms")


class CliTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(argv)
        return code, out.getvalue()

    def test_fails_on_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            criterion = _fixture(Path(tmp))
            summary = Path(tmp) / "summary.md"
            code, out = self._run([
                "--criterion-dir", str(criterion),
                "--threshold", "0.10",
                "--summary", str(summary),
                "--require-baseline",
            ])
            self.assertEqual(code, 1)
            self.assertIn("regression", out)
            self.assertIn("Benchmark", summary.read_text(encoding="utf-8"))

    def test_passes_when_change_is_within_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            criterion = _fixture(Path(tmp), change=0.05, lower=0.01, upper=0.09)
            code, _ = self._run([
                "--criterion-dir", str(criterion),
                "--threshold", "10%",
                "--summary", str(Path(tmp) / "summary.md"),
            ])
            self.assertEqual(code, 0)

    def test_missing_baseline_is_soft_pass_unless_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--criterion-dir", str(Path(tmp) / "criterion"),
                "--summary", str(Path(tmp) / "summary.md"),
            ]
            self.assertEqual(self._run(argv)[0], 0)
            self.assertEqual(self._run(argv + ["--require-baseline"])[0], 1)

    def test_invalid_threshold_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _ = self._run([
                "--criterion-dir", str(Path(tmp)),
                "--threshold", "10",
                "--summary", str(Path(tmp) / "summary.md"),
            ])
            self.assertEqual(code, 2)


def run_self_tests() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (ThresholdTests, LoadChangesTests, ClassifyTests, ReportTests, CliTests):
        suite.addTests(loader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
