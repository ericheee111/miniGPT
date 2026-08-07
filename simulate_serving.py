"""Run one deterministic Stage 10 offline serving workload."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from minigpt.serving_simulator import (
    load_simulator_config,
    run_executor_equivalence,
    run_simulation,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the offline workload simulator parser."""
    parser = argparse.ArgumentParser(description="Run a deterministic serving workload.")
    _ = parser.add_argument("--config", type=Path, required=True)
    _ = parser.add_argument("--output", type=Path)
    _ = parser.add_argument(
        "--compare-executors",
        action="store_true",
        help="run reference and continuous_decode and verify logical equivalence",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load, run, and report the artifact directory."""
    arguments = build_parser().parse_args(argv)
    config = load_simulator_config(cast("Path", arguments.config))
    output = cast("Path | None", arguments.output)
    if cast("bool", arguments.compare_executors):
        destination = config.output_dir if output is None else output
        comparison = run_executor_equivalence(config, output_dir=destination)
        print(f"output={destination}")  # noqa: T201
        print(f"equivalent={comparison.equivalent}")  # noqa: T201
        return 0
    result = run_simulation(config, output_dir=output)
    print(f"output={result.output_dir}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
