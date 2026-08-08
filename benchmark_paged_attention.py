"""Run the Stage 13B descriptive paged-attention CPU benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

from minigpt.paged_attention_benchmark import (
    PagedAttentionBenchmarkConfig,
    write_paged_attention_benchmark,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the bounded benchmark command-line parser."""
    parser = argparse.ArgumentParser(description="Benchmark direct paged-attention decode on CPU.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache-access-iterations", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and write stable JSON."""
    arguments = build_parser().parse_args(argv)
    path = write_paged_attention_benchmark(
        cast("Path", arguments.output),
        config=PagedAttentionBenchmarkConfig(
            warmups=cast("int", arguments.warmups),
            repeats=cast("int", arguments.repeats),
            cache_access_iterations=cast("int", arguments.cache_access_iterations),
        ),
    )
    print(path)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
