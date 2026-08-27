"""Generate the miniGPT v1.0 Stage 21 capstone evidence package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

from minigpt.stage21_evidence import (
    generate_stage21_evidence,
    generate_stage21_registry_evidence,
    generate_stage21_release_evidence,
    generate_stage21_runtime_evidence,
    generate_stage21_version_evidence,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the Stage 21 capstone evidence-generation parser."""
    parser = argparse.ArgumentParser(description="Generate hash-bound miniGPT v1.0 evidence.")
    _ = parser.add_argument("--source-commit", required=True)
    _ = parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("docs/results/v1-release"),
    )
    _ = parser.add_argument("--work-root", type=Path, default=Path("reports/stage21-evidence"))
    _ = parser.add_argument("--exact-resume-path", type=Path, required=True)
    _ = parser.add_argument("--lifecycle-path", type=Path, required=True)
    _ = parser.add_argument("--full-tests-path", type=Path, required=True)
    _ = parser.add_argument("--quality-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate semantic release witnesses and bind pre-collected gate evidence."""
    arguments = build_parser().parse_args(argv)
    root = Path.cwd()
    work_root = cast("Path", arguments.work_root)
    version = generate_stage21_version_evidence(root, work_root / "version.json")
    registry = generate_stage21_registry_evidence(root, work_root / "registry.json")
    release = generate_stage21_release_evidence(root, work_root / "release.json")
    runtime = generate_stage21_runtime_evidence(root, work_root / "runtime.json")
    package = generate_stage21_evidence(
        version_path=version,
        registry_path=registry,
        release_path=release,
        runtime_path=runtime,
        exact_resume_path=cast("Path", arguments.exact_resume_path),
        lifecycle_path=cast("Path", arguments.lifecycle_path),
        full_tests_path=cast("Path", arguments.full_tests_path),
        quality_path=cast("Path", arguments.quality_path),
        package_root=cast("Path", arguments.package_root),
        source_commit=cast("str", arguments.source_commit),
        repository_root=root,
    )
    print(package)  # noqa: T201 - evidence-generation CLI
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
