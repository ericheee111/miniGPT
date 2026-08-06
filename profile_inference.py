"""Capture a separate descriptive Stage 9 inference operator profile."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from minigpt.inference_benchmark_config import load_inference_benchmark_config
from minigpt.inference_profile import run_inference_profile


def build_parser() -> argparse.ArgumentParser:
    """Build the profile command-line parser."""
    parser = argparse.ArgumentParser(description="Profile cached and uncached CPU inference.")
    _ = parser.add_argument("--config", required=True, type=Path)
    _ = parser.add_argument("--output", required=True, type=Path)
    _ = parser.add_argument("--prompt-length", type=int, default=128)
    _ = parser.add_argument("--generated-length", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the explicitly descriptive profile and print its artifact path."""
    arguments = build_parser().parse_args(argv)
    config = load_inference_benchmark_config(cast("Path", arguments.config))
    artifacts = run_inference_profile(
        config,
        cast("Path", arguments.output),
        prompt_length=cast("int", arguments.prompt_length),
        generated_length=cast("int", arguments.generated_length),
    )
    print(artifacts.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
