"""Run the Stage 15 cache-aware batched paged prefill benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

from minigpt.cache_aware_prefill_benchmark import (
    CacheAwarePrefillBenchmarkConfig,
    write_cache_aware_prefill_benchmark,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the strict fresh-process benchmark parser."""
    parser = argparse.ArgumentParser(description="Benchmark cache-aware paged prefill on CPU.")
    _ = parser.add_argument("--output", type=Path, required=True)
    _ = parser.add_argument("--warmups", type=int, default=1)
    _ = parser.add_argument("--repeats", type=int, default=3)
    _ = parser.add_argument("--cv-limit", type=float, default=0.10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and write stable JSON."""
    arguments = build_parser().parse_args(argv)
    path = write_cache_aware_prefill_benchmark(
        cast("Path", arguments.output),
        config=CacheAwarePrefillBenchmarkConfig(
            warmups=cast("int", arguments.warmups),
            repeats=cast("int", arguments.repeats),
            cv_limit=cast("float", arguments.cv_limit),
        ),
    )
    print(path)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
