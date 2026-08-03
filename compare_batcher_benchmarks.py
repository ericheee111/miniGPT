"""Strictly compare two hash-bound TokenBatcher benchmark runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minigpt.batcher_benchmark_evidence import (
    InvalidBatcherBenchmarkEvidenceError,
    compare_batcher_benchmarks,
    write_batcher_comparison,
)
from minigpt.benchmark_v2_comparison_policy import (
    InvalidComparisonPolicyError,
    load_comparison_policy,
)


def _parser() -> argparse.ArgumentParser:
    """Build the strict comparison CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--baseline", type=Path, required=True)
    _ = parser.add_argument("--candidate", type=Path, required=True)
    _ = parser.add_argument("--policy", type=Path, required=True)
    _ = parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load, independently recompute, compare, and serialize one run pair."""
    arguments = _parser().parse_args(argv)
    try:
        policy = load_comparison_policy(arguments.policy)
        comparison = compare_batcher_benchmarks(
            arguments.baseline,
            arguments.candidate,
            policy,
        )
        write_batcher_comparison(comparison, arguments.output)
    except (
        InvalidBatcherBenchmarkEvidenceError,
        InvalidComparisonPolicyError,
        OSError,
        ValueError,
    ) as error:
        _ = sys.stderr.write(f"batcher comparison failed: {error}\n")
        return 2
    _ = sys.stdout.write(f"verdict={comparison.verdict}\n")
    _ = sys.stdout.write(f"comparison={arguments.output}\n")
    return 0 if comparison.verdict == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
