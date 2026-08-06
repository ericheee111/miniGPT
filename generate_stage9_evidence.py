"""Generate or verify the committed Stage 9 KV-cache evidence package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from minigpt.stage9_evidence import generate_stage9_evidence, verify_stage9_evidence

DEFAULT_PACKAGE_ROOT = Path("docs/results/kv-cache-generation")


def build_parser() -> argparse.ArgumentParser:
    """Build the evidence generation and verification parser."""
    parser = argparse.ArgumentParser(description="Generate or verify Stage 9 evidence.")
    _ = parser.add_argument("--run-manifest", type=Path)
    _ = parser.add_argument("--profile", type=Path)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_PACKAGE_ROOT)
    _ = parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate a new package or verify an existing package without mutation."""
    arguments = build_parser().parse_args(argv)
    output = cast("Path", arguments.output)
    if cast("bool", arguments.verify):
        _ = verify_stage9_evidence(output)
        print(f"verified={output}")
        return 0
    run_manifest = cast("Path | None", arguments.run_manifest)
    profile = cast("Path | None", arguments.profile)
    if run_manifest is None or profile is None:
        msg = "--run-manifest and --profile are required unless --verify is used"
        raise SystemExit(msg)
    generated = generate_stage9_evidence(
        run_manifest_path=run_manifest,
        profile_path=profile,
        package_root=output,
    )
    print(f"generated={generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
