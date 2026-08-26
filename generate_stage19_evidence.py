"""Generate the Stage 19 real serving-runtime configuration evidence package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from minigpt.stage19_evidence import (
    generate_stage19_checkpoint_identity,
    generate_stage19_evidence,
    generate_stage19_invalid_combinations,
    generate_stage19_manifest,
    generate_stage19_runtime_wiring,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


_LIFECYCLE_TESTS = (
    "tests/test_serving_runtime.py",
    "tests/test_serve_subprocess.py",
    "tests/test_http_server.py",
    "tests/test_http_lifecycle.py",
    "tests/test_engine_runner.py",
    "tests/test_lazy_kv_reservation.py",
)


def build_parser() -> argparse.ArgumentParser:
    """Create the Stage 19 evidence-generation parser."""
    parser = argparse.ArgumentParser(description="Generate hash-bound Stage 19 evidence.")
    _ = parser.add_argument("--source-commit", required=True)
    _ = parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("docs/results/serving-runtime-configuration"),
    )
    _ = parser.add_argument("--work-root", type=Path, default=Path("reports/stage19-evidence"))
    return parser


def _write_lifecycle(path: Path) -> Path:
    command = [sys.executable, "-m", "pytest", "-q", *_LIFECYCLE_TESTS]
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
                "command": "python -m pytest -q " + " ".join(_LIFECYCLE_TESTS),
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
    """Regenerate runtime, validation, manifest, identity, and lifecycle evidence."""
    arguments = build_parser().parse_args(argv)
    work_root = cast("Path", arguments.work_root)
    wiring = generate_stage19_runtime_wiring(work_root / "runtime_wiring.json")
    invalid = generate_stage19_invalid_combinations(work_root / "invalid_combinations.json")
    manifest = generate_stage19_manifest(work_root / "manifest.json", work_root=work_root)
    identity = generate_stage19_checkpoint_identity(
        work_root / "identity.json",
        work_root / "identity-inputs",
    )
    lifecycle = _write_lifecycle(work_root / "lifecycle_tests.json")
    package = generate_stage19_evidence(
        runtime_wiring_path=wiring,
        invalid_combinations_path=invalid,
        manifest_path=manifest,
        identity_path=identity,
        lifecycle_path=lifecycle,
        package_root=cast("Path", arguments.package_root),
        source_commit=cast("str", arguments.source_commit),
    )
    print(package)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
