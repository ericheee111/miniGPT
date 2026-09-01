"""Evaluate a Story Forge checkpoint with bounded deterministic metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from minigpt.story_evaluation import (
    DEFAULT_EVAL_MAX_NEW_TOKENS,
    DEFAULT_EVAL_TEMPERATURE,
    DEFAULT_EVAL_TOP_K,
    DEFAULT_EVAL_VAL_BATCHES,
    evaluate_story_checkpoint,
    write_evaluation_report,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the Story Forge evaluation command-line parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a Story Forge checkpoint with bounded deterministic metrics."
    )
    _ = parser.add_argument("--checkpoint", type=Path, required=True)
    _ = parser.add_argument("--output", type=Path, required=True)
    _ = parser.add_argument("--temperature", type=float, default=DEFAULT_EVAL_TEMPERATURE)
    _ = parser.add_argument("--top-k", type=int, default=DEFAULT_EVAL_TOP_K)
    _ = parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_EVAL_MAX_NEW_TOKENS)
    _ = parser.add_argument("--val-batches", type=int, default=DEFAULT_EVAL_VAL_BATCHES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a bounded Story Forge evaluation and write the report."""
    arguments = build_parser().parse_args(argv)
    checkpoint = cast("Path", arguments.checkpoint)
    if not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)  # noqa: T201
        return 2
    summary = evaluate_story_checkpoint(
        checkpoint,
        max_new_tokens=cast("int", arguments.max_new_tokens),
        temperature=cast("float", arguments.temperature),
        top_k=cast("int | None", arguments.top_k),
        val_batches=cast("int", arguments.val_batches),
    )
    write_evaluation_report(summary, cast("Path", arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
