"""Generate or verify miniGPT Story Forge v1.1 product evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from minigpt.story_forge_product_evidence import (
    EVIDENCE_DIRECTORY,
    generate_story_forge_product_evidence,
    verify_story_forge_product_evidence,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the Story Forge product-evidence CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit")
    parser.add_argument("--verification-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate an evidence package or verify an existing package."""
    arguments = build_parser().parse_args(argv)
    repository_root = arguments.repository_root.resolve()
    output = (
        repository_root / EVIDENCE_DIRECTORY
        if arguments.output is None
        else arguments.output.resolve()
    )
    if arguments.verify:
        result = verify_story_forge_product_evidence(
            repository_root=repository_root,
            evidence_directory=output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.source_commit is None or arguments.verification_json is None:
        build_parser().error(
            "generation requires --source-commit and --verification-json",
        )
    generated = generate_story_forge_product_evidence(
        repository_root=repository_root,
        source_commit=arguments.source_commit,
        verification_path=arguments.verification_json,
        output_directory=output,
    )
    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
