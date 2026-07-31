"""Test the public Benchmark v2 command and its canonical configurations."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

from minigpt.benchmark_v2_config import load_benchmark_v2_config

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CLI_PATH = _REPOSITORY_ROOT / "benchmark_v2.py"
_SMOKE_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "benchmark_v2_smoke.yaml"
_REFERENCE_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "benchmark_v2_reference.yaml"
_I7_14700_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "benchmark_v2_i7_14700_stage8.yaml"


def _run_cli(*arguments: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    """Execute the real root CLI from the repository checkout."""
    return subprocess.run(  # noqa: S603 - command is the checkout's fixed interpreter and CLI.
        [sys.executable, str(_CLI_PATH), *arguments],
        capture_output=True,
        check=False,
        cwd=_REPOSITORY_ROOT,
        text=True,
        timeout=timeout,
    )


def _smoke_config_in_temporary_output(tmp_path: Path) -> Path:
    """Copy the canonical smoke config while changing only its report location."""
    copied_config = tmp_path / "benchmark_v2_smoke.yaml"
    _ = shutil.copy2(_SMOKE_CONFIG_PATH, copied_config)
    original_output_root = "output_root: reports/benchmark-v2"
    replacement_output_root = f"output_root: {(tmp_path / 'reports').as_posix()}"
    source = copied_config.read_text(encoding="utf-8")
    if source.count(original_output_root) != 1:
        msg = "canonical smoke config must declare one expected output root"
        raise ValueError(msg)
    _ = copied_config.write_text(
        source.replace(original_output_root, replacement_output_root), encoding="utf-8"
    )
    return copied_config


def test_benchmark_v2_cli_help_describes_required_config() -> None:
    """Keep the public command discoverable through argparse help."""
    # Given: the root command is invoked solely for help.
    # When: argparse renders the command description.
    completed = _run_cli("--help")

    # Then: it succeeds and exposes the required strict configuration input.
    assert completed.returncode == 0
    assert "--config" in completed.stdout
    assert "Benchmark v2" in completed.stdout


def test_benchmark_v2_cli_rejects_invalid_config(tmp_path: Path) -> None:
    """Return the dedicated invalid-config exit code instead of running malformed input."""
    # Given: a YAML document that cannot satisfy the strict v2 schema.
    invalid_config = tmp_path / "invalid.yaml"
    _ = invalid_config.write_text("schema_version: 2\n", encoding="utf-8")

    # When: the public command attempts to load it.
    completed = _run_cli("--config", str(invalid_config))

    # Then: it has the typed configuration failure status and no report location.
    assert completed.returncode == 2
    assert "invalid Benchmark v2 config" in completed.stderr
    assert "run_directory=" not in completed.stdout


def test_benchmark_v2_cli_runs_canonical_smoke_in_fresh_workers(tmp_path: Path) -> None:
    """Produce a complete bounded evidence package without a performance claim."""
    # Given: the committed smoke methodology copied to an isolated report root.
    config_path = _smoke_config_in_temporary_output(tmp_path)

    # When: the real command runs all smoke case replicates.
    completed = _run_cli("--config", str(config_path))

    # Then: every expected artifact is printed and the final manifest is complete.
    assert completed.returncode == 0, completed.stderr
    printed_paths = dict(line.split("=", maxsplit=1) for line in completed.stdout.splitlines())
    assert set(printed_paths) == {
        "status",
        "run_directory",
        "run_manifest",
        "raw_replicates",
        "summary_csv",
        "summary_markdown",
    }
    assert printed_paths["status"] == "complete"
    manifest_path = Path(printed_paths["run_manifest"])
    raw_path = Path(printed_paths["raw_replicates"])
    summary_path = Path(printed_paths["summary_csv"])
    assert manifest_path.is_file()
    assert raw_path.is_file()
    assert summary_path.is_file()
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    raw_records = [
        cast("dict[str, object]", json.loads(line))
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["status"] == "complete"
    assert len(raw_records) == 4
    assert len({record["worker_pid"] for record in raw_records}) == 4
    assert len(summary_path.read_text(encoding="utf-8").splitlines()) == 3
    assert not (Path(printed_paths["run_directory"]) / "profile").exists()
    assert "performance threshold" not in completed.stdout.lower()


def test_reference_and_host_config_separate_portable_and_calibrated_controls() -> None:
    """Keep portable defaults free of one host's affinity and preconditioning."""
    # Given: the portable reference and explicitly named i7-14700 methodology.
    reference = load_benchmark_v2_config(_REFERENCE_CONFIG_PATH)
    host_specific = load_benchmark_v2_config(_I7_14700_CONFIG_PATH)

    # When: its declared one-factor cases are resolved.
    case_names = [case.name for case in reference.cases]

    # Then: both retain the matrix while only the host config pins calibrated controls.
    assert len(case_names) == 10
    assert len(case_names) == len(set(case_names))
    assert reference.replicates == 7
    assert reference.warmup_steps == 10
    assert reference.measurement_steps == 20
    assert reference.cpu_affinity is None
    assert reference.preconditioning.enabled is False
    assert host_specific.warmup_steps == 15
    assert host_specific.measurement_steps == 200
    assert host_specific.cpu_affinity == tuple(range(16))
    assert host_specific.preconditioning.duration_seconds == 120.0
