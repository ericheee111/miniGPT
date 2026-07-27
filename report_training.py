"""Generate a validated compact report from reference-training artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from sys import stdout
from typing import TYPE_CHECKING, cast

from minigpt.run_provenance import find_repository_root
from minigpt.training_report import ReportInputs, generate_training_report

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the reference-training report command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate validated CPU reference-training evidence.",
    )
    _ = parser.add_argument("--config", type=Path, required=True)
    _ = parser.add_argument("--metrics", type=Path, required=True)
    _ = parser.add_argument("--samples", type=Path, required=True)
    _ = parser.add_argument("--checkpoint", type=Path, required=True)
    _ = parser.add_argument("--provenance", type=Path, required=True)
    _ = parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate explicit sources and print the generated durable paths."""
    arguments = build_parser().parse_args(argv)
    config_path = cast("Path", arguments.config)
    artifacts = generate_training_report(
        ReportInputs(
            repository_root=find_repository_root(config_path),
            config_path=config_path,
            metrics_path=cast("Path", arguments.metrics),
            samples_path=cast("Path", arguments.samples),
            checkpoint_path=cast("Path", arguments.checkpoint),
            provenance_path=cast("Path", arguments.provenance),
        ),
        cast("Path", arguments.output_dir),
    )
    _ = stdout.write(f"report={artifacts.readme}\nmanifest={artifacts.manifest}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
