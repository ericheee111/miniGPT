"""Generate or verify the committed Stage 12 HTTP serving evidence package."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import cast

from minigpt.stage12_evidence import (
    generate_stage12_evidence,
    verify_stage12_evidence,
)


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
    """Build the Stage 12 evidence command-line parser."""
    parser = argparse.ArgumentParser(description="Generate or verify Stage 12 evidence.")
    _ = parser.add_argument("--benchmark-json", type=Path)
    _ = parser.add_argument("--api-examples-json", type=Path)
    _ = parser.add_argument("--lifecycle-json", type=Path)
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/http-serving"),
    )
    _ = parser.add_argument("--source-commit")
    _ = parser.add_argument("--verify", action="store_true")
    return parser


def _required_path(arguments: argparse.Namespace, name: str) -> Path:
    value = getattr(arguments, name)
    if not isinstance(value, Path):
        option = name.replace("_", "-")
        msg = f"--{option} is required unless --verify is used"
        raise TypeError(msg)
    return value


def main(argv: list[str] | None = None) -> int:
    """Generate from fresh inputs, or independently verify an existing package."""
    arguments = build_parser().parse_args(argv)
    output = cast("Path", arguments.output)
    if cast("bool", arguments.verify):
        _ = verify_stage12_evidence(output)
        print(f"verified={output}")  # noqa: T201
        return 0
    result = generate_stage12_evidence(
        benchmark_path=_required_path(arguments, "benchmark_json"),
        api_examples_path=_required_path(arguments, "api_examples_json"),
        lifecycle_path=_required_path(arguments, "lifecycle_json"),
        package_root=output,
        source_commit=cast("str | None", arguments.source_commit) or _source_commit(),
    )
    print(f"output={result}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
