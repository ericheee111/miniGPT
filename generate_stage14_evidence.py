"""Generate the Stage 14 Automatic Prefix Caching evidence package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from minigpt.stage14_evidence import generate_stage14_correctness, generate_stage14_evidence

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the Stage 14 package-generation parser."""
    parser = argparse.ArgumentParser(description="Generate hash-bound Stage 14 evidence.")
    _ = parser.add_argument("--benchmark", type=Path, required=True)
    _ = parser.add_argument("--source-commit", required=True)
    _ = parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("docs/results/automatic-prefix-caching"),
    )
    _ = parser.add_argument("--work-root", type=Path, default=Path("reports/stage14-evidence"))
    return parser


def _test_evidence(path: Path, tests: Sequence[str]) -> Path:
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and repository-owned test paths
        command, capture_output=True, text=True, encoding="utf-8", check=False
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
    """Regenerate correctness/test inputs and package exact hashes."""
    arguments = build_parser().parse_args(argv)
    work_root = cast("Path", arguments.work_root)
    correctness = generate_stage14_correctness(
        config_root=Path("configs"),
        work_root=work_root / "simulation",
        output_path=work_root / "correctness.json",
    )
    stress = _test_evidence(work_root / "stress_tests.json", ("tests/test_prefix_cache.py",))
    lifecycle = _test_evidence(
        work_root / "lifecycle_tests.json",
        ("tests/test_prefix_serving.py", "tests/test_http_lifecycle.py"),
    )
    package = generate_stage14_evidence(
        correctness_path=correctness,
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
