"""Run one strict, fresh-process CPU Benchmark v2 configuration."""

from __future__ import annotations

import argparse
import sys
from enum import IntEnum
from pathlib import Path
from typing import cast

from minigpt.benchmark_v2 import run_benchmark_v2
from minigpt.benchmark_v2_config import InvalidBenchmarkV2ConfigError, load_benchmark_v2_config


class BenchmarkV2ExitCode(IntEnum):
    """Return values that let shell callers distinguish Benchmark v2 outcomes."""

    COMPLETE = 0
    RUNTIME_ERROR = 1
    INVALID_CONFIG = 2
    RUN_COLLISION = 3
    PARTIAL = 4
    FAILED = 5


def build_parser() -> argparse.ArgumentParser:
    """Build the typed public CLI parser for one Benchmark v2 execution."""
    parser = argparse.ArgumentParser(
        description="Run one fresh-process CPU Benchmark v2 configuration."
    )
    _ = parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Strict Benchmark v2 YAML configuration path.",
    )
    return parser


def _write_artifact_locations(
    *,
    status: str,
    run_directory: Path,
    run_manifest: Path,
    raw_replicates: Path,
    summary_csv: Path,
    summary_markdown: Path,
) -> None:
    """Print stable key/value evidence locations for a finalized run package."""
    _ = sys.stdout.write(
        "\n".join(
            (
                f"status={status}",
                f"run_directory={run_directory}",
                f"run_manifest={run_manifest}",
                f"raw_replicates={raw_replicates}",
                f"summary_csv={summary_csv}",
                f"summary_markdown={summary_markdown}",
            )
        )
        + "\n"
    )


def _exit_code_for_status(status: str) -> BenchmarkV2ExitCode:
    """Map a finalized run status to its public, typed process exit code."""
    if status == "complete":
        return BenchmarkV2ExitCode.COMPLETE
    if status == "partial":
        return BenchmarkV2ExitCode.PARTIAL
    if status == "failed":
        return BenchmarkV2ExitCode.FAILED
    msg = f"unsupported Benchmark v2 run status {status!r}"
    raise ValueError(msg)


def main(argv: list[str] | None = None) -> int:
    """Run the requested configuration and report durable output paths and outcome."""
    arguments = build_parser().parse_args(argv)
    config_path = cast("Path", arguments.config)
    try:
        config = load_benchmark_v2_config(config_path)
    except InvalidBenchmarkV2ConfigError as error:
        _ = sys.stderr.write(f"invalid benchmark configuration: {error}\n")
        return int(BenchmarkV2ExitCode.INVALID_CONFIG)

    try:
        artifacts = run_benchmark_v2(config)
    except FileExistsError as error:
        _ = sys.stderr.write(f"benchmark run collision: {error}\n")
        return int(BenchmarkV2ExitCode.RUN_COLLISION)
    except (OSError, TypeError, ValueError) as error:
        _ = sys.stderr.write(f"benchmark run failed before finalization: {error}\n")
        return int(BenchmarkV2ExitCode.RUNTIME_ERROR)

    _write_artifact_locations(
        status=artifacts.status,
        run_directory=artifacts.run_directory,
        run_manifest=artifacts.run_manifest_path,
        raw_replicates=artifacts.raw_replicates_path,
        summary_csv=artifacts.summary_csv_path,
        summary_markdown=artifacts.summary_markdown_path,
    )
    try:
        return int(_exit_code_for_status(artifacts.status))
    except ValueError as error:
        _ = sys.stderr.write(f"benchmark run failed after finalization: {error}\n")
        return int(BenchmarkV2ExitCode.RUNTIME_ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
