"""Generate the Stage 20 unified-CLI/project-doctor evidence package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from minigpt.stage20_evidence import (
    generate_stage20_cli_evidence,
    generate_stage20_doctor_evidence,
    generate_stage20_evidence,
    generate_stage20_packaging_evidence,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the Stage 20 evidence-generation parser."""
    parser = argparse.ArgumentParser(description="Generate hash-bound Stage 20 evidence.")
    _ = parser.add_argument("--source-commit", required=True)
    _ = parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("docs/results/project-doctor"),
    )
    _ = parser.add_argument("--work-root", type=Path, default=Path("reports/stage20-evidence"))
    return parser


def _test_evidence(path: Path, tests: Sequence[str]) -> Path:
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    completed = subprocess.run(  # noqa: S603 - fixed interpreter/repository tests
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
    """Regenerate CLI, doctor, packaging, lifecycle, and exact package hashes."""
    arguments = build_parser().parse_args(argv)
    root = Path.cwd()
    work_root = cast("Path", arguments.work_root)
    cli = generate_stage20_cli_evidence(root, work_root / "cli.json")
    doctor = generate_stage20_doctor_evidence(root, work_root / "doctor.json")
    packaging = generate_stage20_packaging_evidence(root, work_root / "packaging.json")
    lifecycle = _test_evidence(
        work_root / "lifecycle_tests.json",
        (
            "tests/test_cli.py",
            "tests/test_project_doctor.py",
            "tests/test_serving_runtime.py",
            "tests/test_serve_subprocess.py",
            "tests/test_stage19_evidence.py",
            "tests/test_http_server.py",
            "tests/test_engine_runner.py",
        ),
    )
    package = generate_stage20_evidence(
        cli_path=cli,
        doctor_path=doctor,
        packaging_path=packaging,
        lifecycle_path=lifecycle,
        package_root=cast("Path", arguments.package_root),
        source_commit=cast("str", arguments.source_commit),
    )
    print(package)  # noqa: T201 - evidence-generation CLI
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
