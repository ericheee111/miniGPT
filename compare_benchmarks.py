"""Compare two finalized CPU Benchmark v2 run packages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Never, cast

from typing_extensions import override

from minigpt.benchmark_v2_compare import InvalidComparisonInputError, compare_runs, write_comparison
from minigpt.benchmark_v2_comparison_policy import (
    InvalidComparisonPolicyError,
    load_comparison_policy,
)


class _ComparisonArgumentParser(argparse.ArgumentParser):
    """Convert invalid CLI syntax into the documented input-error exit class."""

    @override
    def error(self, message: str) -> Never:
        """Reject invalid syntax without argparse claiming regression exit code 2."""
        raise ValueError(message)


def build_parser() -> argparse.ArgumentParser:
    """Build the typed public CLI parser for a baseline/candidate comparison."""
    parser = _ComparisonArgumentParser(
        description="Compare finalized Benchmark v2 baseline and candidate runs."
    )
    _ = parser.add_argument(
        "--baseline", type=Path, required=True, help="Baseline run_manifest.json path."
    )
    _ = parser.add_argument(
        "--candidate", type=Path, required=True, help="Candidate run_manifest.json path."
    )
    _ = parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="Strict versioned comparison policy YAML path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write deterministic comparison artifacts and return nonzero for invalid source evidence."""
    try:
        arguments = build_parser().parse_args(argv)
        baseline = cast("Path", arguments.baseline)
        candidate = cast("Path", arguments.candidate)
        policy_path = cast("Path", arguments.policy)
        policy = load_comparison_policy(policy_path)
        comparison = compare_runs(baseline, candidate, policy)
        artifacts = write_comparison(comparison)
    except (
        InvalidComparisonInputError,
        InvalidComparisonPolicyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        _ = sys.stderr.write(f"comparison failed: {error}\n")
        return 1
    _ = sys.stdout.write(
        f"comparison_json={artifacts.json_path}\ncomparison_markdown={artifacts.markdown_path}\n"
    )
    if comparison.verdict == "fail":
        return 2
    if comparison.verdict == "not_comparable":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
