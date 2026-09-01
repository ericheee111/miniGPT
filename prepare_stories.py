"""Prepare deterministic SimpleStories artifacts for the Story Forge family."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from minigpt.story_data import prepare_simple_stories


def build_parser() -> argparse.ArgumentParser:
    """Create the Story Forge data-preparation command-line parser."""
    parser = argparse.ArgumentParser(
        description="Prepare deterministic SimpleStories Story Forge training data."
    )
    _ = parser.add_argument("--output-dir", type=Path, default=Path("data/story_forge"))
    _ = parser.add_argument(
        "--source-parquet",
        type=Path,
        default=None,
        help="optional local Parquet fixture; otherwise download the pinned official source",
    )
    _ = parser.add_argument("--train-stories", type=int, required=True)
    _ = parser.add_argument("--val-stories", type=int, required=True)
    _ = parser.add_argument("--vocab-size", type=int, required=True)
    _ = parser.add_argument("--min-frequency", type=int, default=2)
    _ = parser.add_argument("--max-token-length", type=int, default=24)
    _ = parser.add_argument("--seed", type=int, required=True)
    _ = parser.add_argument("--batch-size", type=int, default=8192)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare Story Forge data and print the durable artifact paths."""
    arguments = build_parser().parse_args(argv)
    prepared = prepare_simple_stories(
        output_dir=cast("Path", arguments.output_dir),
        source_parquet=cast("Path | None", arguments.source_parquet),
        train_stories=cast("int", arguments.train_stories),
        val_stories=cast("int", arguments.val_stories),
        vocab_size=cast("int", arguments.vocab_size),
        min_frequency=cast("int", arguments.min_frequency),
        max_token_length=cast("int", arguments.max_token_length),
        seed=cast("int", arguments.seed),
        batch_size=cast("int", arguments.batch_size),
    )
    lines = (
        f"output={prepared.output_dir}",
        f"train={prepared.train_path}",
        f"validation={prepared.val_path}",
        f"tokenizer={prepared.tokenizer_path}",
        f"metadata={prepared.metadata_path}",
        "",
    )
    _ = sys.stdout.write("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
