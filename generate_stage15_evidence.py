"""Generate the Stage 15 cache-aware batched paged prefill evidence package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from minigpt.stage15_evidence import (
    generate_stage15_batching,
    generate_stage15_correctness,
    generate_stage15_evidence,
    write_stage15_stress,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the Stage 15 package-generation parser."""
    parser = argparse.ArgumentParser(description="Generate hash-bound Stage 15 evidence.")
    _ = parser.add_argument("--benchmark", type=Path, required=True)
    _ = parser.add_argument("--source-commit", required=True)
    _ = parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("docs/results/cache-aware-batched-prefill"),
    )
    _ = parser.add_argument("--work-root", type=Path, default=Path("reports/stage15-evidence"))
    return parser


def _test_evidence(path: Path, tests: Sequence[str]) -> Path:
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and repository-owned tests
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "command": "python -m pytest -q " + " ".join(tests),
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate correctness, batching, stress, lifecycle, and exact package hashes."""
    arguments = build_parser().parse_args(argv)
    work_root = cast("Path", arguments.work_root)
    correctness = generate_stage15_correctness(
        config_root=Path("configs"),
        work_root=work_root / "simulation",
        output_path=work_root / "correctness.json",
    )
    batching = generate_stage15_batching(
        correctness_path=correctness,
        output_path=work_root / "batching.json",
    )
    stress = write_stage15_stress(work_root / "stress.json", operations=1000)
    lifecycle = _test_evidence(
        work_root / "lifecycle_tests.json",
        (
            "tests/test_prefix_serving.py",
            "tests/test_paged_serving.py",
            "tests/test_http_lifecycle.py",
            "tests/test_engine_runner.py",
            "tests/test_cache_aware_prefill_benchmark.py",
        ),
    )
    package = generate_stage15_evidence(
        correctness_path=correctness,
        batching_path=batching,
        benchmark_path=cast("Path", arguments.benchmark),
        stress_path=stress,
        lifecycle_path=lifecycle,
        package_root=cast("Path", arguments.package_root),
        source_commit=cast("str", arguments.source_commit),
    )
    print(package)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
