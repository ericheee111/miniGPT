"""Run descriptive Stage 11A serving profiling outside canonical timing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from minigpt.serving_profile import profile_serving_workload
from minigpt.serving_simulator import SimulatorExecutor


def build_parser() -> argparse.ArgumentParser:
    """Build the descriptive serving profiler CLI parser."""
    parser = argparse.ArgumentParser(description="Profile one Stage 11A serving workload.")
    _ = parser.add_argument("--config", type=Path, required=True)
    _ = parser.add_argument("--scenario", required=True)
    _ = parser.add_argument(
        "--executor",
        choices=[executor.value for executor in SimulatorExecutor],
        required=True,
    )
    _ = parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one descriptive profile and report its artifact paths."""
    arguments = build_parser().parse_args(argv)
    summary, trace = profile_serving_workload(
        cast("Path", arguments.config),
        scenario_name=cast("str", arguments.scenario),
        executor_name=SimulatorExecutor(cast("str", arguments.executor)),
        output_dir=cast("Path", arguments.output),
    )
    print(f"summary={summary}")  # noqa: T201
    print(f"trace={trace}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
