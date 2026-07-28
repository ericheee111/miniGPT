"""Run a separate CPU operator profile for one configured Benchmark v2 case."""

from __future__ import annotations

import argparse
import sys
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from minigpt.benchmark_v2_config import InvalidBenchmarkV2ConfigError, load_benchmark_v2_config
from minigpt.benchmark_v2_profile import (
    InvalidProfileRunIdError,
    ProfileConfigurationError,
    ProfileRunDirectoryCollisionError,
    ProfileWorkerError,
    run_benchmark_v2_profile,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class ProfileV2ExitCode(IntEnum):
    """Return values that distinguish typed profiler outcomes for shell callers."""

    COMPLETE = 0
    RUNTIME_ERROR = 1
    INVALID_CONFIG = 2
    PROFILE_DISABLED = 3
    RUN_COLLISION = 4
    INVALID_RUN_ID = 5


def _parser() -> argparse.ArgumentParser:
    """Build the typed public command parser."""
    parser = argparse.ArgumentParser(
        description="Run a separate Benchmark v2 CPU operator profile."
    )
    _ = parser.add_argument(
        "--config", required=True, type=Path, help="strict Benchmark v2 YAML path"
    )
    _ = parser.add_argument(
        "--run-id",
        default=None,
        type=str,
        help="optional profile-run identity; an existing directory is rejected",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Load one strict config and print only separate profile artifact locations."""
    parsed = _parser().parse_args(arguments)
    config_path = cast("Path", parsed.config)
    run_id = cast("str | None", parsed.run_id)
    try:
        config = load_benchmark_v2_config(config_path)
    except InvalidBenchmarkV2ConfigError as error:
        _ = sys.stderr.write(f"{error}\n")
        return int(ProfileV2ExitCode.INVALID_CONFIG)
    try:
        artifacts = run_benchmark_v2_profile(config, config_path, run_id=run_id)
    except ProfileConfigurationError as error:
        _ = sys.stderr.write(f"{error}\n")
        return int(ProfileV2ExitCode.PROFILE_DISABLED)
    except InvalidProfileRunIdError as error:
        _ = sys.stderr.write(f"{error}\n")
        return int(ProfileV2ExitCode.INVALID_RUN_ID)
    except ProfileRunDirectoryCollisionError as error:
        _ = sys.stderr.write(f"{error}\n")
        return int(ProfileV2ExitCode.RUN_COLLISION)
    except ProfileWorkerError as error:
        _ = sys.stderr.write(f"{error}\n")
        return int(ProfileV2ExitCode.RUNTIME_ERROR)
    _ = sys.stdout.write(
        "\n".join(
            (
                "status=complete",
                f"profile_directory={artifacts.profile_directory}",
                f"profile_manifest={artifacts.profile_manifest_path}",
                f"top_operators_csv={artifacts.top_operators_csv_path}",
                f"profile_markdown={artifacts.profile_markdown_path}",
                f"trace_json={artifacts.trace_json_path}",
            )
        )
        + "\n"
    )
    return int(ProfileV2ExitCode.COMPLETE)


if __name__ == "__main__":
    raise SystemExit(main())
