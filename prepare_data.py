"""Prepare Tiny Shakespeare artifacts for MiniTrainGPT."""

import argparse
import sys
from pathlib import Path

from minigpt.data import TINY_SHAKESPEARE_URL, prepare_tiny_shakespeare


class PrepareArguments(argparse.Namespace):
    """Store parsed data-preparation options with concrete types."""

    data_dir: Path
    source_url: str


def build_parser() -> argparse.ArgumentParser:
    """Create the data-preparation command-line parser."""
    parser = argparse.ArgumentParser(description="Prepare Tiny Shakespeare training data.")
    _ = parser.add_argument("--data-dir", type=Path, default=Path("data"))
    _ = parser.add_argument("--source-url", default=TINY_SHAKESPEARE_URL)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare data and print the durable artifact paths."""
    arguments = build_parser().parse_args(argv, namespace=PrepareArguments())
    prepared = prepare_tiny_shakespeare(arguments.data_dir, arguments.source_url)
    lines = (
        f"raw={prepared.raw_path}",
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
