"""Profile one CPU training configuration and export trace artifacts."""

import argparse
import sys
from pathlib import Path
from typing import cast

from minigpt.benchmark_config import load_benchmark_config
from minigpt.profiling import run_profile


def build_parser() -> argparse.ArgumentParser:
    """Create the CPU profiling command-line parser."""
    parser = argparse.ArgumentParser(description="Profile one CPU GPT training configuration.")
    _ = parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Profile the configured case and print durable artifact paths."""
    arguments = build_parser().parse_args(argv)
    config_path = cast("Path", arguments.config)
    artifacts = run_profile(load_benchmark_config(config_path))
    _ = sys.stdout.write(
        "\n".join(
            (
                f"top_operators_csv={artifacts.top_operators_csv}",
                f"report={artifacts.report_markdown}",
                f"trace={artifacts.trace_json}",
                "",
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
