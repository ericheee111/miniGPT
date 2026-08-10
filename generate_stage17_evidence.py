"""Generate the Stage 17 KV-pressure preemption and recompute evidence package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from minigpt.stage17_evidence import (
    generate_stage17_apc,
    generate_stage17_benchmark,
    generate_stage17_correctness,
    generate_stage17_evidence,
    generate_stage17_scheduling,
    write_stage17_stress,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the Stage 17 evidence-generation parser."""
    parser = argparse.ArgumentParser(description="Generate hash-bound Stage 17 evidence.")
    _ = parser.add_argument("--source-commit", required=True)
    _ = parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("docs/results/kv-pressure-preemption"),
    )
    _ = parser.add_argument("--work-root", type=Path, default=Path("reports/stage17-evidence"))
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
    """Regenerate correctness, scheduling, stress, lifecycle, and exact package hashes."""
    arguments = build_parser().parse_args(argv)
    work_root = cast("Path", arguments.work_root)
    correctness = generate_stage17_correctness(work_root / "correctness.json")
    scheduling = generate_stage17_scheduling(work_root / "scheduling.json")
    apc = generate_stage17_apc(work_root / "apc.json")
    stress = write_stage17_stress(work_root / "stress.json", operations=1000)
    benchmark = generate_stage17_benchmark(
        correctness_path=correctness,
        scheduling_path=scheduling,
        stress_path=stress,
        output_path=work_root / "benchmark.json",
    )
    lifecycle = _test_evidence(
        work_root / "lifecycle_tests.json",
        (
            "tests/test_kv_preemption.py",
            "tests/test_kv_preemption_simulator.py",
            "tests/test_chunked_prefill.py",
            "tests/test_serving_simulator.py",
            "tests/test_prefix_serving.py",
            "tests/test_paged_serving.py",
            "tests/test_http_lifecycle.py",
            "tests/test_engine_runner.py",
        ),
    )
    package = generate_stage17_evidence(
        correctness_path=correctness,
        scheduling_path=scheduling,
        apc_path=apc,
        benchmark_path=benchmark,
        stress_path=stress,
        lifecycle_path=lifecycle,
        package_root=cast("Path", arguments.package_root),
        source_commit=cast("str", arguments.source_commit),
    )
    print(package)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
