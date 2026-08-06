"""Run one isolated fresh-process Stage 9 inference benchmark."""

from __future__ import annotations

import argparse
import sys
from enum import IntEnum
from pathlib import Path
from typing import cast

from minigpt.inference_benchmark import run_inference_benchmark
from minigpt.inference_benchmark_config import (
    InvalidInferenceBenchmarkConfigError,
    load_inference_benchmark_config,
)


class InferenceBenchmarkExitCode(IntEnum):
    """Distinguish completed, invalid, partial, and failed benchmark runs."""

    COMPLETE = 0
    RUNTIME_ERROR = 1
    INVALID_CONFIG = 2
    PARTIAL = 4
    FAILED = 5


def build_parser() -> argparse.ArgumentParser:
    """Build the public inference benchmark command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run isolated cached/uncached autoregressive inference benchmarks."
    )
    _ = parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one configuration and print the durable evidence locations."""
    arguments = build_parser().parse_args(argv)
    try:
        config = load_inference_benchmark_config(cast("Path", arguments.config))
    except InvalidInferenceBenchmarkConfigError as error:
        _ = sys.stderr.write(f"invalid inference benchmark configuration: {error}\n")
        return int(InferenceBenchmarkExitCode.INVALID_CONFIG)
    try:
        artifacts = run_inference_benchmark(config)
    except (OSError, TypeError, ValueError) as error:
        _ = sys.stderr.write(f"inference benchmark failed: {error}\n")
        return int(InferenceBenchmarkExitCode.RUNTIME_ERROR)
    _ = sys.stdout.write(
        f"status={artifacts.status}\n"
        f"run_directory={artifacts.run_directory}\n"
        f"run_manifest={artifacts.run_manifest_path}\n"
        f"summary={artifacts.summary_path}\n"
        f"raw_replicates={artifacts.raw_replicates_path}\n"
    )
    if artifacts.status == "complete":
        return int(InferenceBenchmarkExitCode.COMPLETE)
    if artifacts.status == "partial":
        return int(InferenceBenchmarkExitCode.PARTIAL)
    return int(InferenceBenchmarkExitCode.FAILED)


if __name__ == "__main__":
    raise SystemExit(main())
