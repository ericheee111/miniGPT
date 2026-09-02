"""Build the deterministic Systems Lab scenario assets from committed evidence."""  # noqa: INP001

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

from minigpt.story_forge_systems import (
    build_systems_lab_assets,
    systems_lab_asset_names,
    verify_systems_lab_assets,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the Systems Lab asset builder parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--repo-root", type=Path, default=Path())
    _ = parser.add_argument("--output", type=Path, default=Path("web/data"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate and verify the four deterministic Systems Lab assets."""
    arguments = build_parser().parse_args(argv)
    values = cast("dict[str, object]", vars(arguments))
    root = cast("Path", values["repo_root"]).resolve()
    output = (cast("Path", values["output"])).resolve()
    build_systems_lab_assets(root, output)
    verified = verify_systems_lab_assets(output)
    for name in systems_lab_asset_names():
        print(f"built {name}")  # noqa: T201
    _ = verified
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
