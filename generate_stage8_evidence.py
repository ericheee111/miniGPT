"""Regenerate the committed Stage 8 evidence package."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from minigpt.stage8_evidence import generate_stage8_evidence


def _arguments() -> argparse.Namespace:
    """Parse the deterministic Stage 8 evidence regeneration inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=Path("docs/results/batcher-optimization"),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/benchmark_v2_comparison.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    """Regenerate the committed evidence package from strict raw inputs."""
    arguments = _arguments()
    git = shutil.which("git")
    if git is None:
        msg = "git executable is required"
        raise RuntimeError(msg)
    generated_by_git_sha = subprocess.run(  # noqa: S603 - resolved executable.
        [git, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    generate_stage8_evidence(
        arguments.result_directory.resolve(),
        arguments.policy.resolve(),
        generated_by_git_sha=generated_by_git_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
