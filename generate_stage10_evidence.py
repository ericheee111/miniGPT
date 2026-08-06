"""Generate or verify the committed Stage 10 serving evidence package."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import cast

from minigpt.stage10_evidence import generate_stage10_evidence, verify_stage10_evidence

DEFAULT_PACKAGE_ROOT = Path("docs/results/serving-control-plane")
DEFAULT_CONFIGS = (
    Path("configs/serving_single_request.yaml"),
    Path("configs/serving_burst_arrivals.yaml"),
    Path("configs/serving_cache_pressure.yaml"),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the evidence generation and verification parser."""
    parser = argparse.ArgumentParser(description="Generate or verify Stage 10 evidence.")
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_PACKAGE_ROOT)
    _ = parser.add_argument("--source-commit")
    _ = parser.add_argument("--verify", action="store_true")
    return parser


def _current_commit() -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        message = "git executable was not found"
        raise FileNotFoundError(message)
    completed = subprocess.run(  # noqa: S603
        [git_executable, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    """Generate a new package or verify an existing package without mutation."""
    arguments = build_parser().parse_args(argv)
    output = cast("Path", arguments.output)
    if cast("bool", arguments.verify):
        _ = verify_stage10_evidence(output)
        print(f"verified={output}")  # noqa: T201
        return 0
    source_commit = cast("str | None", arguments.source_commit)
    generated = generate_stage10_evidence(
        config_paths=DEFAULT_CONFIGS,
        package_root=output,
        source_commit=_current_commit() if source_commit is None else source_commit,
    )
    print(f"generated={generated}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
