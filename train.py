"""Train miniGPT from a YAML experiment configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from sys import stdout
from typing import TYPE_CHECKING, cast

from minigpt.config import load_experiment_config
from minigpt.trainer import run_training

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the training command-line parser."""
    parser = argparse.ArgumentParser(description="Train the CPU-first character GPT.")
    _ = parser.add_argument("--config", type=Path, required=True)
    _ = parser.add_argument("--resume", type=Path)
    _ = parser.add_argument(
        "--run-until-step",
        type=int,
        help="exclusive absolute step boundary for this process",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load configuration and run training to the requested process boundary."""
    arguments = build_parser().parse_args(argv)
    config = load_experiment_config(cast("Path", arguments.config))
    result = run_training(
        config,
        resume_path=cast("Path | None", arguments.resume),
        run_until_step=cast("int | None", arguments.run_until_step),
    )
    output = "\n".join(
        (
            f"metrics={result.metrics_path}",
            f"checkpoint={result.checkpoint_path}",
            f"tensorboard={result.tensorboard_dir}",
        )
    )
    _ = stdout.write(f"{output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
