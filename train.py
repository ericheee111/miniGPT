"""Train miniGPT from a YAML experiment configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from sys import argv as system_argv
from sys import stdout
from typing import TYPE_CHECKING, cast

from minigpt.config import load_experiment_config
from minigpt.run_provenance import (
    RunInvocation,
    begin_run_segment,
    complete_run_segment,
    fail_run_segment,
)
from minigpt.trainer import run_training

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the training command-line parser."""
    parser = argparse.ArgumentParser(description="Train the CPU-first character GPT.")
    _ = parser.add_argument("--config", type=Path, required=True)
    _ = parser.add_argument("--resume", type=Path)
    _ = parser.add_argument(
        "--provenance",
        type=Path,
        help="optional reference-run provenance JSON path",
    )
    _ = parser.add_argument(
        "--run-until-step",
        type=int,
        help="exclusive absolute step boundary for this process",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load configuration and run training to the requested process boundary."""
    invocation_argv = tuple(system_argv[1:] if argv is None else argv)
    arguments = build_parser().parse_args(invocation_argv)
    config_path = cast("Path", arguments.config)
    resume_path = cast("Path | None", arguments.resume)
    provenance_path = cast("Path | None", arguments.provenance)
    run_until_step = cast("int | None", arguments.run_until_step)
    config = load_experiment_config(config_path)
    segment = None
    if provenance_path is not None:
        segment = begin_run_segment(
            provenance_path,
            config_path=config_path,
            config=config,
            invocation=RunInvocation(
                argv=invocation_argv,
                run_until_step=run_until_step,
                resume_path=resume_path,
            ),
        )
    try:
        result = run_training(
            config,
            resume_path=resume_path,
            run_until_step=run_until_step,
        )
    except (OSError, RuntimeError, ValueError):
        if provenance_path is not None and segment is not None:
            fail_run_segment(provenance_path, segment=segment)
        raise
    if provenance_path is not None and segment is not None:
        complete_run_segment(
            provenance_path,
            segment=segment,
            checkpoint_path=result.checkpoint_path,
            final_step=result.final_step,
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
