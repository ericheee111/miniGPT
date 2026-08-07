"""Generate canonical Stage 13A paged-cache evidence from fresh checks."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from minigpt.stage13a_evidence import generate_stage13a_evidence

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run lifecycle tests and generate the committed hash-bound package."""
    if argv:
        reason = "generate_stage13a_evidence.py does not accept arguments"
        raise ValueError(reason)
    project_root = Path(__file__).resolve().parent
    test_files = [
        "tests/test_paged_kv_cache.py",
        "tests/test_paged_serving.py",
        "tests/test_serving_simulator.py",
        "tests/test_http_lifecycle.py",
        "tests/test_serve_subprocess.py",
    ]
    command = [sys.executable, "-m", "pytest", *test_files, "-q"]
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        _ = sys.stderr.write(completed.stdout)
        _ = sys.stderr.write(completed.stderr)
        return completed.returncode
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lifecycle = {
        "schema_version": 1,
        "exit_code": completed.returncode,
        "test_files": test_files,
        "covered_contracts": [
            "allocator_ownership_and_rollback",
            "dense_paged_equivalence",
            "fifo_capacity_pressure",
            "cancellation_and_failure_isolation",
            "http_disconnect_and_backpressure",
            "graceful_shutdown_zero_leaks",
            "deterministic_allocator_stress",
        ],
    }
    with tempfile.TemporaryDirectory(prefix="minigpt-stage13a-") as temporary:
        temporary_root = Path(temporary)
        lifecycle_path = temporary_root / "lifecycle_tests.json"
        _ = lifecycle_path.write_text(
            json.dumps(lifecycle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        package = generate_stage13a_evidence(
            config_root=project_root / "configs",
            lifecycle_path=lifecycle_path,
            package_root=project_root / "docs" / "results" / "paged-kv-cache-manager",
            work_root=temporary_root / "runs",
            source_commit=source_commit,
        )
    _ = sys.stdout.write(f"{package}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
