"""Run the configurable CPU training benchmark matrix."""

import argparse
import sys
from pathlib import Path
from typing import cast

from minigpt.benchmark import run_benchmark
from minigpt.benchmark_config import load_benchmark_config


def build_parser() -> argparse.ArgumentParser:
    """Create the CPU benchmark command-line parser."""
    parser = argparse.ArgumentParser(description="Benchmark CPU GPT training configurations.")
    _ = parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the configured matrix and print durable artifact paths."""
    arguments = build_parser().parse_args(argv)
    config_path = cast("Path", arguments.config)
    artifacts = run_benchmark(load_benchmark_config(config_path))
    _ = sys.stdout.write(
        "\n".join(
            (
                f"raw_csv={artifacts.raw_csv}",
                f"summary_csv={artifacts.summary_csv}",
                f"report={artifacts.report_markdown}",
                "",
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
