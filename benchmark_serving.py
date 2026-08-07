"""Run the canonical Stage 11A fresh-process serving benchmark."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import cast

from minigpt.serving_benchmark import run_serving_benchmark


def _source_commit() -> str:
    git_path = shutil.which("git")
    if git_path is None:
        msg = "git executable was not found"
        raise RuntimeError(msg)
    completed = subprocess.run(  # noqa: S603
        [git_path, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    """Build the canonical serving benchmark CLI parser."""
    parser = argparse.ArgumentParser(description="Benchmark Stage 11A serving executors.")
    _ = parser.add_argument("--config", type=Path, required=True)
    _ = parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and report its output and strict verdict."""
    arguments = build_parser().parse_args(argv)
    result = run_serving_benchmark(
        cast("Path", arguments.config),
        source_commit=_source_commit(),
        output_root=cast("Path | None", arguments.output_root),
    )
    print(f"output={result.output_dir}")  # noqa: T201
    print(f"strict_verdict={result.strict_verdict}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
