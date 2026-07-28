"""Compare two finalized CPU Benchmark v2 run packages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from minigpt.benchmark_v2_compare import InvalidComparisonInputError, compare_runs, write_comparison


def build_parser() -> argparse.ArgumentParser:
    """Build the typed public CLI parser for a baseline/candidate comparison."""
    parser = argparse.ArgumentParser(
        description="Compare finalized Benchmark v2 baseline and candidate runs."
    )
    _ = parser.add_argument(
        "--baseline", type=Path, required=True, help="Baseline run_manifest.json path."
    )
    _ = parser.add_argument(
        "--candidate", type=Path, required=True, help="Candidate run_manifest.json path."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write deterministic comparison artifacts and return nonzero for invalid source evidence."""
    arguments = build_parser().parse_args(argv)
    baseline = cast("Path", arguments.baseline)
    candidate = cast("Path", arguments.candidate)
    try:
        artifacts = write_comparison(compare_runs(baseline, candidate))
    except (InvalidComparisonInputError, OSError, TypeError, ValueError) as error:
        _ = sys.stderr.write(f"comparison failed: {error}\n")
        return 1
    _ = sys.stdout.write(
        f"comparison_json={artifacts.json_path}\ncomparison_markdown={artifacts.markdown_path}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
