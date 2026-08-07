"""Generate the committed Stage 11A decode batching evidence package."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import cast

from minigpt.stage11a_evidence import generate_stage11a_evidence


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
    """Build the Stage 11A evidence generator parser."""
    parser = argparse.ArgumentParser(description="Generate Stage 11A evidence.")
    _ = parser.add_argument("--benchmark-run", type=Path, required=True)
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/decode-continuous-batching"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the package from one verified canonical benchmark run."""
    arguments = build_parser().parse_args(argv)
    result = generate_stage11a_evidence(
        benchmark_run_dir=cast("Path", arguments.benchmark_run),
        simulator_config_paths=(
            Path("configs/serving_stage11a_mixed.yaml"),
            Path("configs/serving_stage11a_overflow.yaml"),
        ),
        package_root=cast("Path", arguments.output),
        source_commit=_source_commit(),
    )
    print(f"output={result}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
