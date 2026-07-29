"""Test separate, identity-bound CPU operator profiles for Benchmark v2."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from minigpt.benchmark_v2_config import (
    case_identity,
    load_benchmark_v2_config,
    resolved_config_sha256,
)
from minigpt.benchmark_v2_profile import (
    ProfileRunDirectoryCollisionError,
    run_benchmark_v2_profile,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CLI_PATH = _REPOSITORY_ROOT / "profile_benchmark_v2.py"
_SMOKE_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "benchmark_v2_smoke.yaml"


def test_repository_smoke_config_enables_tiny_separate_profile() -> None:
    """Keep the checked-in smoke profile runnable through its separate CLI."""
    # Given: the repository's authoritative Benchmark v2 smoke configuration.
    config = load_benchmark_v2_config(_SMOKE_CONFIG_PATH)

    # When: its optional profiler settings are read.
    profile = config.profile

    # Then: the tiny profile run is explicitly enabled with a bounded workload.
    assert profile.enabled is True
    assert profile.case_name == "tiny_t1_s32_b2"
    assert profile.warmup_steps == 1
    assert profile.active_steps == 2


def _profile_config_in_temporary_output(tmp_path: Path) -> Path:
    """Copy the profile-enabled smoke config while redirecting its output."""
    copied_config = tmp_path / "benchmark_v2_profile_smoke.yaml"
    _ = shutil.copy2(_SMOKE_CONFIG_PATH, copied_config)
    source = copied_config.read_text(encoding="utf-8")
    source = source.replace(
        "output_root: reports/benchmark-v2", f"output_root: {(tmp_path / 'reports').as_posix()}"
    )
    _ = copied_config.write_text(source, encoding="utf-8")
    return copied_config


def _run_cli(*arguments: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    """Execute the root profiler command from the repository checkout."""
    return subprocess.run(  # noqa: S603 - command is the checkout's fixed interpreter and CLI.
        [sys.executable, str(_CLI_PATH), *arguments],
        capture_output=True,
        check=False,
        cwd=_REPOSITORY_ROOT,
        text=True,
        timeout=timeout,
    )


def test_profile_cli_writes_separate_bound_artifacts(tmp_path: Path) -> None:
    """Write profiled evidence that cannot be mistaken for canonical benchmark evidence."""
    # Given: the tiny v2 smoke methodology with its explicit profile case enabled.
    config_path = _profile_config_in_temporary_output(tmp_path)
    config = load_benchmark_v2_config(config_path)
    selected_case = next(case for case in config.cases if case.name == config.profile.case_name)

    # When: the root command launches its dedicated profiled worker.
    completed = _run_cli("--config", str(config_path))

    # Then: it returns only separate profile artifacts bound to the selected configuration and case.
    assert completed.returncode == 0, completed.stderr
    printed_paths = dict(line.split("=", maxsplit=1) for line in completed.stdout.splitlines())
    assert set(printed_paths) == {
        "status",
        "profile_directory",
        "profile_manifest",
        "top_operators_csv",
        "profile_markdown",
        "trace_json",
    }
    assert printed_paths["status"] == "complete"
    profile_directory = Path(printed_paths["profile_directory"])
    manifest_path = Path(printed_paths["profile_manifest"])
    assert profile_directory.parent.name == "profiles"
    assert {entry.name for entry in profile_directory.iterdir()} == {
        "profile_manifest.json",
        "top_operators.csv",
        "profile_report.md",
        "trace.json",
    }
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    assert manifest["config_sha256"] == resolved_config_sha256(config)
    assert manifest["case_identity"] == case_identity(config, selected_case)
    assert manifest["case_name"] == selected_case.name
    source_run = cast("dict[str, object]", manifest["source_run"])
    assert source_run["kind"] == "profiled_case_only"
    assert source_run["benchmark_run_id"] is None
    assert source_run["config_sha256"] == resolved_config_sha256(config)
    assert source_run["case_identity"] == case_identity(config, selected_case)
    source = cast("dict[str, object]", manifest["source"])
    assert source["config_path"] == config_path.resolve().as_posix()
    assert source["config_sha256"] == resolved_config_sha256(config)
    assert manifest["profiling_timings_are_not_benchmark_timings"] is True
    assert "raw_replicates.jsonl" not in {entry.name for entry in profile_directory.iterdir()}
    assert "summary.csv" not in {entry.name for entry in profile_directory.iterdir()}
    artifacts = cast("list[dict[str, object]]", manifest["artifacts"])
    assert {artifact["path"] for artifact in artifacts} == {
        "top_operators.csv",
        "profile_report.md",
        "trace.json",
    }
    for artifact in artifacts:
        path = profile_directory / cast("str", artifact["path"])
        content = path.read_bytes()
        assert artifact["size_bytes"] == len(content)
        assert artifact["sha256"] == hashlib.sha256(content).hexdigest()
    assert "operator" in (profile_directory / "top_operators.csv").read_text(encoding="utf-8")
    report = (profile_directory / "profile_report.md").read_text(encoding="utf-8")
    assert "Profiler overhead" in report
    assert "not benchmark timings" in report
    assert (profile_directory / "trace.json").stat().st_size > 0


def test_profile_run_rejects_an_existing_identity_directory(tmp_path: Path) -> None:
    """Reject a collision rather than overwrite pre-existing profile evidence."""
    # Given: a valid enabled profile configuration and a pre-existing deterministic run directory.
    config_path = _profile_config_in_temporary_output(tmp_path)
    config = load_benchmark_v2_config(config_path)
    collision_directory = config.output_root / "profiles" / "profile-collision"
    collision_directory.mkdir(parents=True)

    # When: profile evidence is requested with that exact run identity.
    # Then: it refuses to overwrite the existing directory before launching any profile worker.
    with pytest.raises(ProfileRunDirectoryCollisionError, match="already exists"):
        _ = run_benchmark_v2_profile(config, config_path, run_id="profile-collision")


def test_profile_cli_rejects_disabled_profile_configuration(tmp_path: Path) -> None:
    """Expose a typed CLI error when profiling is not enabled by the strict config."""
    # Given: a copy of the canonical smoke config with its profile explicitly disabled.
    config_path = tmp_path / "disabled.yaml"
    _ = shutil.copy2(_SMOKE_CONFIG_PATH, config_path)
    _ = config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("enabled: true", "enabled: false"),
        encoding="utf-8",
    )

    # When: the profile command is invoked.
    completed = _run_cli("--config", str(config_path))

    # Then: it returns the dedicated profile-configuration status without writing evidence.
    assert completed.returncode == 3
    assert "profile.enabled must be true" in completed.stderr
    assert not (tmp_path / "reports").exists()


def test_profile_cli_rejects_existing_profile_run_id(tmp_path: Path) -> None:
    """Map a protected profile-directory collision to the typed public exit code."""
    # Given: an enabled profile config and an already reserved profile evidence identity.
    config_path = _profile_config_in_temporary_output(tmp_path)
    config = load_benchmark_v2_config(config_path)
    run_id = "profile-collision"
    (config.output_root / "profiles" / run_id).mkdir(parents=True)

    # When: the root CLI is asked to use the reserved identity.
    completed = _run_cli("--config", str(config_path), "--run-id", run_id)

    # Then: it refuses to overwrite the evidence with the typed collision status.
    assert completed.returncode == 4
    assert "already exists" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_profile_cli_maps_worker_timeout_to_a_controlled_error(tmp_path: Path) -> None:
    """Convert a profiled-worker timeout into a typed runtime result without a traceback."""
    # Given: a valid tiny profile config with an intentionally impossible worker deadline.
    config_path = _profile_config_in_temporary_output(tmp_path)
    source = config_path.read_text(encoding="utf-8")
    _ = config_path.write_text(
        source.replace("worker_timeout_seconds: 60", "worker_timeout_seconds: 0.001"),
        encoding="utf-8",
    )

    # When: the dedicated profile worker cannot finish before the configured deadline.
    completed = _run_cli("--config", str(config_path))

    # Then: the public command returns its runtime error status instead of leaking an exception.
    assert completed.returncode == 1
    assert "timed out" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_profile_cli_rejects_a_run_id_that_is_not_one_path_component(tmp_path: Path) -> None:
    """Keep optional profile identities contained beneath the configured profile root."""
    # Given: an enabled profile config and a traversal-shaped requested identity.
    config_path = _profile_config_in_temporary_output(tmp_path)
    config = load_benchmark_v2_config(config_path)

    # When: the public CLI receives the unsafe run ID.
    completed = _run_cli("--config", str(config_path), "--run-id", "..\\outside")

    # Then: it rejects the value before it can create evidence outside profiles.
    assert completed.returncode == 5
    assert "single path component" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (config.output_root / "outside").exists()


@pytest.mark.parametrize(
    "run_id",
    [
        "../outside",
        "..\\outside",
        "a/b",
        "a\\b",
        "/absolute",
    ],
)
def test_profile_cli_rejects_path_separator_run_ids_on_every_platform(
    tmp_path: Path, run_id: str
) -> None:
    """Reject forward and backward slashes uniformly on Windows and POSIX runners.

    ``Path`` parsing treats ``\\`` as a separator only on Windows, so the previous
    string-level check let ``"..\\outside"`` through as a single component on Linux
    and the profiler worker ran instead of being rejected. Cover both separator
    forms so the containment rule holds regardless of the host platform.
    """
    # Given: an enabled profile config and a separator-bearing requested identity.
    config_path = _profile_config_in_temporary_output(tmp_path)
    config = load_benchmark_v2_config(config_path)

    # When: the public CLI receives the multi-component run ID.
    completed = _run_cli("--config", str(config_path), "--run-id", run_id)

    # Then: it rejects the value before creating any evidence outside profiles.
    assert completed.returncode == 5
    assert "single path component" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (config.output_root / "outside").exists()
    assert not (config.output_root / "profiles" / run_id).exists()
