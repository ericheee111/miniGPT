"""Train miniGPT from a YAML experiment configuration."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from sys import stdout
from typing import TYPE_CHECKING

from minigpt.config import load_experiment_config
from minigpt.trainer import run_training

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the training command-line parser."""
    parser = argparse.ArgumentParser(description="Train the CPU-first character GPT.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load configuration, apply bounded overrides, and run training."""
    arguments = build_parser().parse_args(argv)
    config = load_experiment_config(arguments.config)
    if arguments.max_steps is not None:
        if arguments.max_steps <= 0:
            build_parser().error("--max-steps must be positive")
        if arguments.max_steps <= config.training.warmup_steps:
            build_parser().error("--max-steps must be greater than warmup_steps")
        config = replace(
            config,
            training=replace(config.training, max_steps=arguments.max_steps),
        )
    result = run_training(config, resume_path=arguments.resume)
    _ = stdout.write(
        f"metrics={result.metrics_path}\n"
        f"checkpoint={result.checkpoint_path}\n"
        f"tensorboard={result.tensorboard_dir}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
