"""Test guarded baseline and candidate comparison for Benchmark v2 evidence."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import pytest
import yaml

import minigpt.benchmark_v2_compare as compare_module
from minigpt.benchmark_v2_compare import compare_runs, compare_step_times, write_comparison
from minigpt.benchmark_v2_config import (
    case_identity as derive_case_identity,
)
from minigpt.benchmark_v2_config import (
    load_resolved_benchmark_v2_config,
    resolved_config_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from minigpt.benchmark_v2_config import JsonValue


_FIXTURES = Path(__file__).parent / "fixtures" / "benchmark_v2"
_SUMMARY_FIELDS = (
    "case_identity",
    "case_name",
    "replicate_count",
    "success_count",
    "failure_count",
    "median_step_time_ms",
    "min_step_time_ms",
    "max_step_time_ms",
    "population_stddev_step_time_ms",
    "median_absolute_deviation_step_time_ms",
    "coefficient_of_variation_percent",
    "median_tokens_per_second",
    "median_final_rss_mib",
    "max_peak_rss_mib",
    "stability",
)
_CASE_IDENTITY = "a" * 64


def _hash_entry(path: Path) -> dict[str, JsonValue]:
    """Return one strict manifest binding for a small fixture artifact."""
    content = path.read_bytes()
    return {
        "path": path.name,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _write_run_package(  # noqa: PLR0913
    parent: Path,
    name: str,
    *,
    status: str = "complete",
    environment_updates: dict[str, JsonValue] | None = None,
    summary_updates: dict[str, str] | None = None,
    config_sha256: str | None = None,
    threshold_percent: float = 5.0,
    experiment_name: str = "synthetic_benchmark_v2_fixture",
    output_root: str = "reports/benchmark_v2",
    replicates: int = 3,
    minimum_replicates: int = 3,
    max_cv_percent: float = 10.0,
    profile_warmup_steps: int = 1,
    profile_active_steps: int = 1,
    case_identity: str = _CASE_IDENTITY,
) -> Path:
    """Create a hand-checked, hash-bound synthetic package without real timing claims."""
    run_directory = parent / name
    run_directory.mkdir(exist_ok=True)
    case_name = "display-name-can-change" if case_identity == _CASE_IDENTITY else "different-case"
    config: dict[str, JsonValue] = {
        "schema_version": 2,
        "experiment_name": experiment_name,
        "benchmark_seed": 1337,
        "vocab_size": 65,
        "output_root": output_root,
        "worker_timeout_seconds": 60.0,
        "warmup_steps": 0,
        "measurement_steps": 1,
        "replicates": replicates,
        "torch_num_interop_threads": 1,
        "cpu_affinity": None,
        "max_cv_percent": max_cv_percent,
        "minimum_replicates": minimum_replicates,
        "regression_threshold_percent": threshold_percent,
        "relevant_environment_variables": ["OMP_NUM_THREADS"],
        "cases": [
            {
                "name": case_name,
                "model_name": "tiny",
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
                "torch_num_threads": 1,
                "block_size": 32,
                "batch_size": 2,
            }
        ],
        "profile": {
            "enabled": False,
            "case_name": case_name,
            "warmup_steps": profile_warmup_steps,
            "active_steps": profile_active_steps,
        },
    }
    config_content = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
    resolved_config = load_resolved_benchmark_v2_config(
        config_content, run_directory / "resolved_config.yaml"
    )
    computed_config_sha256 = resolved_config_sha256(resolved_config)
    if config_sha256 is not None and config_sha256 != computed_config_sha256:
        msg = "synthetic package config hash must be derived from its resolved config"
        raise ValueError(msg)
    config_sha256 = computed_config_sha256
    case_identity = derive_case_identity(resolved_config, resolved_config.cases[0])
    run_environment: dict[str, JsonValue] = {
        "captured_before_first_worker": True,
        "git": {"commit_sha": "c" * 40, "branch": "main", "dirty": False},
        "platform": "test-platform",
        "machine": "x86_64",
        "cpu_name": "test-cpu",
        "physical_cpu_count": 4,
        "logical_cpu_count": 8,
        "python_version": "3.14.0",
        "torch_version": "2.13.0+cpu",
        "numpy_version": "2.3.0",
        "cuda_available": False,
        "parent_torch_num_threads": 1,
        "parent_torch_num_interop_threads": 1,
        "configured_torch_num_interop_threads": 1,
        "configured_cpu_affinity": None,
        "relevant_environment_variables": {"OMP_NUM_THREADS": "1"},
        "process_priority": "normal",
        "power_scheme": {"value": "balanced", "reason": None},
    }
    environment: dict[str, JsonValue] = {
        "schema_version": 2,
        "run_id": name,
        "run_status": status,
        "config_sha256": config_sha256,
        "run_environment": run_environment,
        "worker_environments": [],
    }
    if environment_updates is not None:
        run_environment.update(environment_updates)
    environment_path = run_directory / "environment.json"
    _ = environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    _ = (run_directory / "resolved_config.yaml").write_text(
        config_content.decode("utf-8"), encoding="utf-8", newline="\n"
    )
    summary: dict[str, str] = {
        "case_identity": case_identity,
        "case_name": case_name,
        "replicate_count": str(replicates),
        "success_count": str(replicates),
        "failure_count": "0",
        "median_step_time_ms": "100.0",
        "min_step_time_ms": "99.0",
        "max_step_time_ms": "101.0",
        "population_stddev_step_time_ms": "0.8",
        "median_absolute_deviation_step_time_ms": "1.0",
        "coefficient_of_variation_percent": "0.8",
        "median_tokens_per_second": "1000.0",
        "median_final_rss_mib": "100.0",
        "max_peak_rss_mib": "120.0",
        "stability": "stable",
    }
    if summary_updates is not None:
        summary.update(summary_updates)
    step_time = float(summary["median_step_time_ms"])
    tokens = float(summary["median_tokens_per_second"])
    summary.update(
        {
            "min_step_time_ms": str(step_time),
            "max_step_time_ms": str(step_time),
            "population_stddev_step_time_ms": "0.0",
            "median_absolute_deviation_step_time_ms": "0.0",
            "coefficient_of_variation_percent": "0.0",
        }
    )
    worker_environment: dict[str, JsonValue] = {
        "platform": "test-platform",
        "python_version": "3.14.0",
        "torch_version": "2.13.0+cpu",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "logical_cpu_count": 8,
        "requested_cpu_affinity": None,
        "effective_cpu_affinity": [0, 1],
        "relevant_environment_variables": {"OMP_NUM_THREADS": "1"},
    }
    environment["worker_environments"] = [
        *[worker_environment for _ in range(replicates)],
    ]
    _ = environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    raw: list[dict[str, JsonValue]] = []
    for index in range(replicates):
        response: dict[str, JsonValue] = {
            "protocol_version": 1,
            "status": "ok",
            "worker_pid": 1000 + index,
            "started_at_utc": "2026-07-28T01:02:03+00:00",
            "ended_at_utc": "2026-07-28T01:02:04+00:00",
            "case_identity": case_identity,
            "case_name": summary["case_name"],
            "replicate_index": index,
            "warmup_steps": 0,
            "measurement_steps": 1,
            "elapsed_seconds": step_time / 1000,
            "step_time_ms": step_time,
            "tokens_per_second": tokens,
            "tokens_per_step": 64,
            "parameter_count": 100,
            "final_rss_mib": 100.0,
            "peak_rss_mib": 120.0,
            "peak_rss_method": "windows_peak_working_set",
            "peak_rss_sampling_interval_ms": None,
            "environment": worker_environment,
        }
        raw.append(
            {
                "status": "ok",
                "case_identity": case_identity,
                "case_name": summary["case_name"],
                "replicate_index": index,
                "worker_pid": 1000 + index,
                "started_at_utc": "2026-07-28T01:02:03+00:00",
                "ended_at_utc": "2026-07-28T01:02:04+00:00",
                "return_code": 0,
                "error_type": None,
                "message": None,
                "stdout": "",
                "stderr": "",
                "worker_response": response,
            }
        )
    _ = (run_directory / "raw_replicates.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in raw), encoding="utf-8", newline="\n"
    )
    summary_path = run_directory / "summary.csv"
    _ = summary_path.write_text(
        ",".join(_SUMMARY_FIELDS)
        + "\n"
        + ",".join(summary[field] for field in _SUMMARY_FIELDS)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _ = (run_directory / "summary.md").write_text("# fixture\n", encoding="utf-8", newline="\n")
    execution_order = [
        {
            "execution_index": index,
            "task_id": f"{case_identity}:{index}",
            "case_name": summary["case_name"],
            "case_identity": case_identity,
            "replicate_index": index,
            "worker_seed": index,
            "worker_pid": 1000 + index,
            "status": "ok",
        }
        for index in range(replicates)
    ]
    _ = (run_directory / "execution_order.json").write_text(
        json.dumps(execution_order, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    artifact_names = (
        "environment.json",
        "resolved_config.yaml",
        "raw_replicates.jsonl",
        "summary.csv",
        "summary.md",
        "execution_order.json",
    )
    manifest: dict[str, JsonValue] = {
        "schema_version": 2,
        "run_id": name,
        "status": status,
        "started_at_utc": "2026-07-28T01:02:03+00:00",
        "ended_at_utc": "2026-07-28T01:02:04+00:00",
        "config_sha256": config_sha256,
        "git": cast("dict[str, JsonValue]", run_environment["git"]),
        "expected_task_count": replicates,
        "completed_task_count": replicates,
        "successful_task_count": replicates,
        "failed_task_count": 0,
        "case_identities": [{"case_name": summary["case_name"], "case_identity": case_identity}],
        "artifacts": [
            _hash_entry(run_directory / artifact_name) for artifact_name in artifact_names
        ],
    }
    manifest_path = run_directory / "run_manifest.json"
    _ = manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest_path


def _rehash_manifest_artifact(manifest_path: Path, artifact_path: Path) -> None:
    """Rebind one deliberately modified synthetic artifact to its fixture manifest."""
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    entries = cast("list[dict[str, object]]", manifest["artifacts"])
    next(entry for entry in entries if entry["path"] == artifact_path.name).update(
        _hash_entry(artifact_path)
    )
    _ = manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def _add_second_case(manifest_path: Path) -> str:
    """Add a hash-bound second three-replicate case to one synthetic fixture package."""
    directory = manifest_path.parent
    second_name = "second-case"
    config_path = directory / "resolved_config.yaml"
    config_document = cast("dict[str, JsonValue]", yaml.safe_load(config_path.read_text()))
    cases = cast("list[dict[str, JsonValue]]", config_document["cases"])
    second_case = dict(cases[0])
    second_case.update({"name": second_name, "torch_num_threads": 2})
    cases.append(second_case)
    config_content = yaml.safe_dump(config_document, sort_keys=False).encode("utf-8")
    resolved_config = load_resolved_benchmark_v2_config(config_content, config_path)
    second_identity = derive_case_identity(resolved_config, resolved_config.cases[1])
    config_sha256 = resolved_config_sha256(resolved_config)
    _ = config_path.write_bytes(config_content)
    raw_path = directory / "raw_replicates.jsonl"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    second_records: list[dict[str, JsonValue]] = []
    for raw_record in raw_records:
        duplicate = cast("dict[str, JsonValue]", json.loads(json.dumps(raw_record)))
        response = cast("dict[str, JsonValue]", duplicate["worker_response"])
        worker_pid = cast("int", duplicate["worker_pid"]) + 100
        duplicate.update(
            {
                "case_identity": second_identity,
                "case_name": second_name,
                "worker_pid": worker_pid,
            }
        )
        response.update(
            {
                "case_identity": second_identity,
                "case_name": second_name,
                "worker_pid": worker_pid,
            }
        )
        worker_environment = cast("dict[str, JsonValue]", response["environment"])
        worker_environment["torch_num_threads"] = 2
        second_records.append(duplicate)
    raw_records.extend(second_records)
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records),
        encoding="utf-8",
        newline="\n",
    )
    summary_path = directory / "summary.csv"
    header, first_row = summary_path.read_text(encoding="utf-8").splitlines()
    summary = dict(zip(header.split(","), first_row.split(","), strict=True))
    summary.update({"case_identity": second_identity, "case_name": second_name})
    _ = summary_path.write_text(
        header
        + "\n"
        + first_row
        + "\n"
        + ",".join(summary[field] for field in _SUMMARY_FIELDS)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    environment_path = directory / "environment.json"
    environment = cast(
        "dict[str, JsonValue]", json.loads(environment_path.read_text(encoding="utf-8"))
    )
    environment["config_sha256"] = config_sha256
    worker_environments = cast("list[JsonValue]", environment["worker_environments"])
    worker_environments.extend(cast("list[JsonValue]", json.loads(json.dumps(worker_environments))))
    for worker in worker_environments[3:]:
        cast("dict[str, JsonValue]", worker)["torch_num_threads"] = 2
    _ = environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    order_path = directory / "execution_order.json"
    order = cast("list[dict[str, JsonValue]]", json.loads(order_path.read_text(encoding="utf-8")))
    order.extend(
        {
            "execution_index": 3 + index,
            "task_id": f"{second_identity}:{index}",
            "case_name": second_name,
            "case_identity": second_identity,
            "replicate_index": index,
            "worker_seed": 3 + index,
            "worker_pid": 1100 + index,
            "status": "ok",
        }
        for index in range(3)
    )
    _ = order_path.write_text(
        json.dumps(order, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    manifest = cast("dict[str, JsonValue]", json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest.update(
        {
            "config_sha256": config_sha256,
            "expected_task_count": 6,
            "completed_task_count": 6,
            "successful_task_count": 6,
        }
    )
    case_identities = cast("list[JsonValue]", manifest["case_identities"])
    case_identities.append({"case_name": second_name, "case_identity": second_identity})
    entries = cast("list[dict[str, JsonValue]]", manifest["artifacts"])
    for artifact_path in (config_path, raw_path, summary_path, environment_path, order_path):
        next(entry for entry in entries if entry["path"] == artifact_path.name).update(
            _hash_entry(artifact_path)
        )
    _ = manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return second_identity


def _write_two_case_incomplete_candidate(
    parent: Path, *, status: Literal["partial", "failed"]
) -> tuple[Path, Path, str]:
    """Create strict two-case evidence interrupted after one success or before any raw result."""
    baseline = _write_run_package(parent, "baseline")
    _ = _add_second_case(baseline)
    candidate = _write_run_package(parent, "candidate")
    _ = _add_second_case(candidate)
    directory = candidate.parent
    raw_path = directory / "raw_replicates.jsonl"
    summary_path = directory / "summary.csv"
    environment_path = directory / "environment.json"
    order_path = directory / "execution_order.json"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    order = cast("list[dict[str, JsonValue]]", json.loads(order_path.read_text(encoding="utf-8")))
    if status == "partial":
        retained_records = raw_records[:1]
        retained_identity = cast("str", retained_records[0]["case_identity"])
        summary_header, first_summary, _second_summary = summary_path.read_text(
            encoding="utf-8"
        ).splitlines()
        summary = dict(zip(summary_header.split(","), first_summary.split(","), strict=True))
        summary.update(
            {
                "replicate_count": "1",
                "success_count": "1",
                "failure_count": "0",
                "stability": "insufficient_samples",
            }
        )
        summary_content = (
            summary_header + "\n" + ",".join(summary[field] for field in _SUMMARY_FIELDS) + "\n"
        )
        order[0]["status"] = "ok"
        order[0]["worker_pid"] = retained_records[0]["worker_pid"]
    else:
        retained_records = []
        retained_identity = ""
        summary_header = summary_path.read_text(encoding="utf-8").splitlines()[0]
        summary_content = summary_header + "\n"
    for entry in order[1 if status == "partial" else 0 :]:
        entry["status"] = "pending"
        entry["worker_pid"] = None
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in retained_records),
        encoding="utf-8",
        newline="\n",
    )
    _ = summary_path.write_text(summary_content, encoding="utf-8", newline="\n")
    environment = cast(
        "dict[str, JsonValue]", json.loads(environment_path.read_text(encoding="utf-8"))
    )
    environment["run_status"] = status
    environment["worker_environments"] = (
        cast("list[JsonValue]", environment["worker_environments"])[:1]
        if status == "partial"
        else []
    )
    _ = environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    _ = order_path.write_text(
        json.dumps(order, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    manifest = cast("dict[str, JsonValue]", json.loads(candidate.read_text(encoding="utf-8")))
    manifest.update(
        {
            "status": status,
            "completed_task_count": len(retained_records),
            "successful_task_count": len(retained_records),
            "failed_task_count": 0,
        }
    )
    entries = cast("list[dict[str, JsonValue]]", manifest["artifacts"])
    for path in (raw_path, summary_path, environment_path, order_path):
        next(entry for entry in entries if entry["path"] == path.name).update(_hash_entry(path))
    _ = candidate.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return baseline, candidate, retained_identity


def test_compare_runs_accepts_two_case_partial_evidence_after_first_success(
    tmp_path: Path,
) -> None:
    """A report with pending cases after its first success remains valid partial evidence."""
    # Given: a strict two-case package interrupted after one successful worker result.
    baseline, candidate, observed_identity = _write_two_case_incomplete_candidate(
        tmp_path, status="partial"
    )

    # When: comparison reconciles summary rows only with observed raw case identities.
    comparison = compare_runs(baseline, candidate)

    # Then: it retains the aligned observed delta but explicitly refuses a performance verdict.
    assert comparison.verdict == "not_comparable"
    assert "candidate run status is partial" in comparison.reasons
    assert tuple(case.case_identity for case in comparison.case_comparisons) == (observed_identity,)
    assert comparison.case_comparisons[0].step_time_change_percent == 0.0


def test_compare_runs_accepts_two_case_failed_evidence_before_any_raw_record(
    tmp_path: Path,
) -> None:
    """A failed package with only pending tasks remains valid evidence rather than corruption."""
    # Given: a strict two-case package finalized before any worker could append raw evidence.
    baseline, candidate, _ = _write_two_case_incomplete_candidate(tmp_path, status="failed")

    # When: comparison loads the empty raw and summary artifacts with their pending execution order.
    comparison = compare_runs(baseline, candidate)

    # Then: the failed status is explicit, without fabricated case statistics.
    assert comparison.verdict == "not_comparable"
    assert "candidate run status is failed" in comparison.reasons
    assert not comparison.case_comparisons


def test_compare_runs_rejects_a_partial_observed_case_without_its_summary(
    tmp_path: Path,
) -> None:
    """An incomplete run may omit pending cases but cannot omit an observed case summary."""
    # Given: an otherwise valid partial package has one observed raw case and an empty summary.
    baseline, candidate, _ = _write_two_case_incomplete_candidate(tmp_path, status="partial")
    summary_path = candidate.parent / "summary.csv"
    header = summary_path.read_text(encoding="utf-8").splitlines()[0]
    _ = summary_path.write_text(header + "\n", encoding="utf-8", newline="\n")
    _rehash_manifest_artifact(candidate, summary_path)

    # When/Then: comparison rejects the forged omission rather than dropping observed raw evidence.
    with pytest.raises(
        ValueError, match=r"raw replicate case identities do not match summary\.csv"
    ):
        _ = compare_runs(baseline, candidate)


def _write_partial_failure_package(parent: Path, *, worker_declared: bool) -> tuple[Path, Path]:
    """Create one complete baseline and one valid partial candidate with a chosen failure origin."""
    baseline = _write_run_package(parent, "baseline")
    candidate = _write_run_package(parent, "candidate")
    raw_path = candidate.parent / "raw_replicates.jsonl"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    failed = raw_records[-1]
    failed.update(
        {
            "status": "error",
            "return_code": 1,
            "error_type": "RuntimeError",
            "message": "synthetic worker failure",
            "worker_response": None,
        }
    )
    if worker_declared:
        failed["worker_response"] = {
            "protocol_version": 1,
            "status": "error",
            "worker_pid": failed["worker_pid"],
            "started_at_utc": failed["started_at_utc"],
            "ended_at_utc": failed["ended_at_utc"],
            "case_identity": failed["case_identity"],
            "case_name": failed["case_name"],
            "replicate_index": failed["replicate_index"],
            "error_type": "RuntimeError",
            "message": "synthetic worker failure",
        }
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records),
        encoding="utf-8",
        newline="\n",
    )
    order_path = candidate.parent / "execution_order.json"
    order = cast("list[dict[str, JsonValue]]", json.loads(order_path.read_text(encoding="utf-8")))
    order[-1]["status"] = "error"
    _ = order_path.write_text(
        json.dumps(order, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    summary_path = candidate.parent / "summary.csv"
    summary_header, summary_record = summary_path.read_text(encoding="utf-8").splitlines()
    summary = dict(zip(summary_header.split(","), summary_record.split(","), strict=True))
    summary.update(
        {"success_count": "2", "failure_count": "1", "stability": "insufficient_samples"}
    )
    _ = summary_path.write_text(
        summary_header + "\n" + ",".join(summary[field] for field in _SUMMARY_FIELDS) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    environment_path = candidate.parent / "environment.json"
    environment = cast(
        "dict[str, JsonValue]", json.loads(environment_path.read_text(encoding="utf-8"))
    )
    environment["run_status"] = "partial"
    workers = cast("list[JsonValue]", environment["worker_environments"])
    environment["worker_environments"] = workers[:-1]
    _ = environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    manifest = cast("dict[str, JsonValue]", json.loads(candidate.read_text(encoding="utf-8")))
    manifest.update({"status": "partial", "successful_task_count": 2, "failed_task_count": 1})
    entries = cast("list[dict[str, JsonValue]]", manifest["artifacts"])
    for path in (raw_path, summary_path, environment_path, order_path):
        next(entry for entry in entries if entry["path"] == path.name).update(_hash_entry(path))
    _ = candidate.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return baseline, candidate


def test_compare_runs_aligns_identity_not_display_name_and_writes_deterministic_outputs(
    tmp_path: Path,
) -> None:
    """Compare aligned stable fixtures with one matching resolved case identity."""
    # Given: two complete, hash-bound runs with one matching resolved workload identity.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(
        tmp_path,
        "candidate",
        summary_updates={
            "median_step_time_ms": "105.0",
            "median_tokens_per_second": "952.3809523809524",
        },
    )

    # When: the candidate is compared to the baseline and materialized beside its manifest.
    comparison = compare_runs(baseline, candidate)
    artifacts = write_comparison(comparison)

    # Then: equality passes and outputs are immutable-source-safe siblings of the candidate run.
    assert comparison.verdict == "pass"
    assert len(comparison.case_comparisons[0].case_identity) == 64
    assert comparison.case_comparisons[0].regressed is False
    assert artifacts.json_path == candidate.parent.parent / "candidate-comparison-baseline.json"
    assert artifacts.markdown_path == candidate.parent.parent / "candidate-comparison-baseline.md"
    assert '"verdict": "pass"' in artifacts.json_path.read_text(encoding="utf-8")
    assert "| display-name-can-change |" in artifacts.markdown_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "candidate_value"),
    [
        ("cpu_name", "different-cpu"),
        ("python_version", "3.13.0"),
        ("torch_version", "2.14.0+cpu"),
        ("numpy_version", "2.4.0"),
        ("power_scheme", {"value": "high performance", "reason": None}),
        ("relevant_environment_variables", {"OMP_NUM_THREADS": "2"}),
    ],
)
def test_compare_runs_reports_performance_environment_mismatches_but_keeps_deltas(
    tmp_path: Path, field: str, candidate_value: JsonValue
) -> None:
    """Refuse a verdict after any performance-relevant environment mismatch."""
    # Given: one otherwise aligned candidate whose supplied compatibility field differs.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(
        tmp_path,
        "candidate",
        environment_updates={field: candidate_value},
        summary_updates={"median_step_time_ms": "110.0", "median_tokens_per_second": "909.0"},
    )

    # When: comparison checks the complete environment evidence.
    comparison = compare_runs(baseline, candidate)

    # Then: a named mismatch blocks pass/fail while retaining the observed descriptive time delta.
    assert comparison.verdict == "not_comparable"
    assert comparison.environment_mismatches[0].field == field
    assert comparison.case_comparisons[0].step_time_change_percent == pytest.approx(10.0)
    assert comparison.case_comparisons[0].regressed is None


def test_compare_runs_ignores_git_provenance_differences(tmp_path: Path) -> None:
    """Allow code-provenance differences when all performance compatibility fields match."""
    # Given: aligned runs that intentionally have different commit SHA, branch, and dirty evidence.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    environment_path = candidate.parent / "environment.json"
    environment = cast(
        "dict[str, object]", json.loads(environment_path.read_text(encoding="utf-8"))
    )
    run_environment = cast("dict[str, object]", environment["run_environment"])
    run_environment["git"] = {"commit_sha": "d" * 40, "branch": "feature", "dirty": True}
    _ = environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    manifest_path = candidate
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest["git"] = run_environment["git"]
    entries = cast("list[dict[str, object]]", manifest["artifacts"])
    environment_entry = next(entry for entry in entries if entry["path"] == "environment.json")
    environment_entry.update(_hash_entry(environment_path))
    _ = manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    # When: comparison processes the validated provenance variation.
    comparison = compare_runs(baseline, candidate)

    # Then: Git identity differences remain visible only as provenance, not a performance blocker.
    assert comparison.verdict == "pass"
    assert not comparison.environment_mismatches


@pytest.mark.parametrize(
    ("status", "summary_updates", "case_identity", "reason"),
    [
        ("partial", None, _CASE_IDENTITY, "candidate run status is partial"),
        ("complete", None, "d" * 64, "case identity sets do not align"),
    ],
)
def test_compare_runs_refuses_regression_verdict_for_ineligible_evidence(
    tmp_path: Path,
    status: str,
    summary_updates: dict[str, str] | None,
    case_identity: str,
    reason: str,
) -> None:
    """Keep aligned descriptive information but never score incomplete or unsuitable evidence."""
    # Given: a valid baseline and a candidate with one comparison-eligibility failure.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(
        tmp_path,
        "candidate",
        status=status,
        summary_updates=summary_updates,
        case_identity=case_identity,
    )

    # When: guarded comparison evaluates the synthetic evidence.
    comparison = compare_runs(baseline, candidate)

    # Then: it emits an explicit refusal reason rather than a pass/fail regression claim.
    assert comparison.verdict == "not_comparable"
    assert reason in comparison.reasons
    if case_identity == _CASE_IDENTITY:
        assert comparison.case_comparisons[0].step_time_change_percent == 0.0


def test_compare_runs_allows_full_config_differences_excluded_from_case_identity(
    tmp_path: Path,
) -> None:
    """Use each complete run's own stable-sample rules despite allowed config hash differences."""
    # Given: aligned workloads with distinct output/report/stability and threshold settings.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(
        tmp_path,
        "candidate",
        experiment_name="candidate_report_only_settings",
        output_root="reports/candidate-only",
        threshold_percent=6.0,
        replicates=5,
        minimum_replicates=4,
        max_cv_percent=2.0,
        profile_warmup_steps=2,
        profile_active_steps=3,
        summary_updates={
            "median_step_time_ms": "105.5",
            "median_tokens_per_second": "947.8672985781991",
        },
    )

    # When: comparison evaluates matching case identities and compatible actual environments.
    comparison = compare_runs(baseline, candidate)

    # Then: full config hashes may differ, and the candidate threshold controls the verdict.
    baseline_manifest = cast(
        "dict[str, JsonValue]", json.loads(baseline.read_text(encoding="utf-8"))
    )
    candidate_manifest = cast(
        "dict[str, JsonValue]", json.loads(candidate.read_text(encoding="utf-8"))
    )
    assert baseline_manifest["config_sha256"] != candidate_manifest["config_sha256"]
    assert comparison.verdict == "pass"
    assert "config_sha256 differs" not in comparison.reasons
    assert comparison.regression_threshold_percent == 6.0
    assert comparison.case_comparisons[0].step_time_change_percent == pytest.approx(5.5)
    assert comparison.case_comparisons[0].regressed is False


def test_compare_step_times_uses_a_strict_regression_threshold() -> None:
    """Flag only changes strictly above the configured relative-step-time boundary."""
    # Given: hand-calculated equality and just-over-threshold candidate measurements.
    # When: the pure comparison helper applies the same threshold to both inputs.
    equality = compare_step_times(100.0, 105.0, threshold_percent=5.0)
    exceeded = compare_step_times(100.0, 105.01, threshold_percent=5.0)

    # Then: equality is not a regression, while a larger relative increase is.
    assert equality.regressed is False
    assert exceeded.regressed is True


def test_compare_runs_rejects_a_complete_manifest_without_raw_replicate_evidence(
    tmp_path: Path,
) -> None:
    """Reject a self-declared complete/stable summary when its bound raw artifact is empty."""
    # Given: both complete packages are valid, then one rehashes a forged empty raw artifact.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    raw_path = candidate.parent / "raw_replicates.jsonl"
    _ = raw_path.write_text("", encoding="utf-8", newline="\n")
    manifest = cast("dict[str, object]", json.loads(candidate.read_text(encoding="utf-8")))
    entries = cast("list[dict[str, object]]", manifest["artifacts"])
    next(entry for entry in entries if entry["path"] == "raw_replicates.jsonl").update(
        _hash_entry(raw_path)
    )
    _ = candidate.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    # When/Then: comparison refuses forged completeness before rendering a pass verdict.
    with pytest.raises(ValueError, match=r"raw replicate.*manifest|raw replicate.*summary"):
        _ = compare_runs(baseline, candidate)


def test_compare_runs_reconciles_partial_raw_failures_before_refusing_a_verdict(
    tmp_path: Path,
) -> None:
    """Retain failed raw records when recomputing a partial run's case summary."""
    # Given: a valid candidate is made partial by changing one bound raw result into a failure.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    raw_path = candidate.parent / "raw_replicates.jsonl"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    raw_records[-1].update(
        {
            "status": "error",
            "return_code": 1,
            "error_type": "RuntimeError",
            "message": "synthetic worker failure",
            "worker_response": None,
        }
    )
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records),
        encoding="utf-8",
        newline="\n",
    )
    order_path = candidate.parent / "execution_order.json"
    order = cast("list[dict[str, JsonValue]]", json.loads(order_path.read_text(encoding="utf-8")))
    order[-1]["status"] = "error"
    _ = order_path.write_text(
        json.dumps(order, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    summary_path = candidate.parent / "summary.csv"
    summary_header, summary_record = summary_path.read_text(encoding="utf-8").splitlines()
    summary = dict(zip(summary_header.split(","), summary_record.split(","), strict=True))
    summary.update(
        {"success_count": "2", "failure_count": "1", "stability": "insufficient_samples"}
    )
    _ = summary_path.write_text(
        summary_header + "\n" + ",".join(summary[field] for field in _SUMMARY_FIELDS) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    environment_path = candidate.parent / "environment.json"
    environment = cast(
        "dict[str, JsonValue]", json.loads(environment_path.read_text(encoding="utf-8"))
    )
    environment["run_status"] = "partial"
    workers = cast("list[JsonValue]", environment["worker_environments"])
    environment["worker_environments"] = workers[:-1]
    _ = environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    manifest = cast("dict[str, JsonValue]", json.loads(candidate.read_text(encoding="utf-8")))
    manifest.update({"status": "partial", "successful_task_count": 2, "failed_task_count": 1})
    entries = cast("list[dict[str, JsonValue]]", manifest["artifacts"])
    for path in (raw_path, summary_path, environment_path, order_path):
        next(entry for entry in entries if entry["path"] == path.name).update(_hash_entry(path))
    _ = candidate.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    # When: comparison recomputes statistics from every raw record, including the failure.
    comparison = compare_runs(baseline, candidate)

    # Then: partial evidence is described but never receives a regression pass/fail verdict.
    assert comparison.verdict == "not_comparable"
    assert "candidate run status is partial" in comparison.reasons
    assert "candidate case has insufficient successful replicates" in comparison.reasons


def test_compare_step_times_flags_the_next_representable_value_above_threshold() -> None:
    """Do not suppress a genuine floating-point increase above an exact percentage boundary."""
    # Given: a candidate step time one representable value above exact five-percent equality.
    candidate = math.nextafter(105.0, float("inf"))

    # When: the guard evaluates its mathematical strictly-greater threshold.
    result = compare_step_times(100.0, candidate, threshold_percent=5.0)

    # Then: any representable increase above equality is a regression.
    assert result.regressed is True


def test_write_comparison_rolls_back_the_first_output_if_the_second_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave no one-sided artifact when the Markdown half of the pair cannot be published."""
    # Given: a valid comparison and a writer that fails only for its Markdown artifact.
    comparison = compare_runs(
        _write_run_package(tmp_path, "baseline"), _write_run_package(tmp_path, "candidate")
    )
    writer_name = "_atomic_write_new"
    original = cast("Callable[[Path, bytes], None]", getattr(compare_module, writer_name))

    def fail_markdown(path: Path, content: bytes) -> None:
        if path.suffix == ".md":
            message = "markdown write failed"
            raise OSError(message)
        original(path, content)

    monkeypatch.setattr(compare_module, writer_name, fail_markdown)

    # When/Then: publishing reports the failure and removes the already-written JSON sibling.
    with pytest.raises(OSError, match="markdown write failed"):
        _ = write_comparison(comparison)
    assert not (tmp_path / "candidate-comparison-baseline.json").exists()


def test_write_comparison_rejects_existing_sibling_artifacts(tmp_path: Path) -> None:
    """Refuse deterministic output-name collisions instead of overwriting comparison evidence."""
    # Given: an aligned comparison has already written its two immutable-run sibling artifacts.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    comparison = compare_runs(baseline, candidate)
    _ = write_comparison(comparison)

    # When/Then: the same deterministic comparison is requested again.
    with pytest.raises(FileExistsError, match="already exists"):
        _ = write_comparison(comparison)


def test_compare_cli_writes_sibling_artifacts_and_returns_nonzero_for_invalid_input(
    tmp_path: Path,
) -> None:
    """Run the typed public CLI against strict fixtures and a malformed manifest path."""
    # Given: a valid synthetic baseline/candidate pair and the root comparison entrypoint.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    script = Path(__file__).parents[1] / "compare_benchmarks.py"
    command = [
        sys.executable,
        str(script),
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
    ]

    # When: the public CLI receives valid evidence and then a missing candidate path.
    successful = subprocess.run(command, capture_output=True, check=False, text=True)  # noqa: S603
    invalid = subprocess.run(  # noqa: S603
        [*command[:-1], str(tmp_path / "missing" / "run_manifest.json")],
        capture_output=True,
        check=False,
        text=True,
    )

    # Then: valid comparison emits sibling paths, while invalid data is a nonzero process result.
    assert successful.returncode == 0
    assert "comparison_json=" in successful.stdout
    assert (tmp_path / "candidate-comparison-baseline.json").is_file()
    assert invalid.returncode != 0
    assert "comparison failed:" in invalid.stderr


def test_fixed_fixture_packages_are_hash_valid_and_have_no_real_timing_claims() -> None:
    """Compare compact committed fixtures that only exercise the artifact contract."""
    # Given: the fixed baseline/candidate packages committed for CLI and loader coverage.
    baseline = _FIXTURES / "baseline" / "run_manifest.json"
    candidate = _FIXTURES / "candidate" / "run_manifest.json"

    # When: their strict package contents are compared.
    comparison = compare_runs(baseline, candidate)

    # Then: their synthetic-only report keeps an explicitly neutral comparison result.
    assert comparison.verdict == "pass"
    assert comparison.case_comparisons[0].step_time_change_percent == 0.0


def test_compare_runs_rejects_duplicate_keys_inside_a_nested_raw_worker_environment(
    tmp_path: Path,
) -> None:
    """Reject JSON whose parser would otherwise silently keep the last nested object key."""
    # Given: a valid raw artifact is rehashed after duplicating one nested worker-control key.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    raw_path = candidate.parent / "raw_replicates.jsonl"
    _ = raw_path.write_text(
        raw_path.read_text(encoding="utf-8").replace(
            '"platform": "test-platform"',
            '"platform": "forged", "platform": "test-platform"',
            1,
        ),
        encoding="utf-8",
        newline="\n",
    )
    _rehash_manifest_artifact(candidate, raw_path)

    # When/Then: strict decoding refuses the duplicate instead of accepting the final value.
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _ = compare_runs(baseline, candidate)


def test_compare_runs_rejects_nonfinite_environment_json_constants(tmp_path: Path) -> None:
    """Reject JavaScript-style non-finite constants before compatibility checks can inspect them."""
    # Given: a bound environment replaces a finite CPU count with a permissive JSON NaN token.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    environment_path = candidate.parent / "environment.json"
    _ = environment_path.write_text(
        environment_path.read_text(encoding="utf-8").replace(
            '"physical_cpu_count": 4', '"physical_cpu_count": NaN'
        ),
        encoding="utf-8",
        newline="\n",
    )
    _rehash_manifest_artifact(candidate, environment_path)

    # When/Then: strict JSON constants reject non-finite evidence before delta calculation.
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        _ = compare_runs(baseline, candidate)


def test_compare_runs_rejects_an_empty_bound_execution_order(tmp_path: Path) -> None:
    """Reject a manifest-bound order that cannot account for any finalized raw task."""
    # Given: a complete package rebinds an empty execution-order artifact.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    order_path = candidate.parent / "execution_order.json"
    _ = order_path.write_text("[]\n", encoding="utf-8", newline="\n")
    _rehash_manifest_artifact(candidate, order_path)

    # When/Then: comparison must reconcile the order before accepting the otherwise valid summaries.
    with pytest.raises(ValueError, match="execution_order"):
        _ = compare_runs(baseline, candidate)


def test_compare_runs_rejects_malformed_worker_peak_method_and_environment(tmp_path: Path) -> None:
    """Reject invalid worker methodology evidence before comparing its otherwise valid medians."""
    # Given: a candidate rebinds unsupported peak-memory method and malformed actual affinity.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    raw_path = candidate.parent / "raw_replicates.jsonl"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    response = cast("dict[str, JsonValue]", raw_records[0]["worker_response"])
    environment = cast("dict[str, JsonValue]", response["environment"])
    response["peak_rss_method"] = "unsupported"
    environment["effective_cpu_affinity"] = []
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records),
        encoding="utf-8",
        newline="\n",
    )
    _rehash_manifest_artifact(candidate, raw_path)

    # When/Then: the full worker schema refuses either malformed methodology field.
    with pytest.raises(ValueError, match=r"peak RSS evidence|effective_cpu_affinity"):
        _ = compare_runs(baseline, candidate)


def test_compare_runs_rejects_malformed_worker_environment_affinity(tmp_path: Path) -> None:
    """Reject an invalid actual worker affinity with otherwise valid response evidence."""
    # Given: a candidate keeps valid peak evidence but rebinds an empty effective affinity.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    raw_path = candidate.parent / "raw_replicates.jsonl"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    response = cast("dict[str, JsonValue]", raw_records[0]["worker_response"])
    environment = cast("dict[str, JsonValue]", response["environment"])
    environment["effective_cpu_affinity"] = []
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records),
        encoding="utf-8",
        newline="\n",
    )
    _rehash_manifest_artifact(candidate, raw_path)

    # When/Then: the worker's exact actual-affinity contract is enforced.
    with pytest.raises(ValueError, match="effective_cpu_affinity"):
        _ = compare_runs(baseline, candidate)


def test_compare_runs_rejects_execution_order_status_or_pid_that_disagrees_with_raw(
    tmp_path: Path,
) -> None:
    """Reject a bound execution order that rewrites a completed task's process evidence."""
    # Given: a complete candidate changes its first execution entry without changing raw evidence.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    order_path = candidate.parent / "execution_order.json"
    order = cast("list[dict[str, JsonValue]]", json.loads(order_path.read_text(encoding="utf-8")))
    order[0].update({"status": "error", "worker_pid": 9999})
    _ = order_path.write_text(
        json.dumps(order, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    _rehash_manifest_artifact(candidate, order_path)

    # When/Then: the ordered task must remain identical to its raw task status and PID.
    with pytest.raises(ValueError, match=r"execution_order\.json disagrees"):
        _ = compare_runs(baseline, candidate)


def test_compare_runs_rejects_case_scoped_worker_control_swaps(tmp_path: Path) -> None:
    """Reject case-level control swaps even when global control counts remain equal."""
    # Given: two cases exchange their actual candidate worker thread counts.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    second_identity = _add_second_case(baseline)
    _ = _add_second_case(candidate)
    raw_path = candidate.parent / "raw_replicates.jsonl"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    for raw_record in raw_records:
        response = cast("dict[str, JsonValue]", raw_record["worker_response"])
        environment = cast("dict[str, JsonValue]", response["environment"])
        environment["torch_num_threads"] = (
            1 if raw_record["case_identity"] == second_identity else 2
        )
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records),
        encoding="utf-8",
        newline="\n",
    )
    environment_path = candidate.parent / "environment.json"
    environment_document = cast(
        "dict[str, JsonValue]", json.loads(environment_path.read_text(encoding="utf-8"))
    )
    workers = cast("list[dict[str, JsonValue]]", environment_document["worker_environments"])
    for index, worker in enumerate(workers):
        worker["torch_num_threads"] = 2 if index < 3 else 1
    _ = environment_path.write_text(
        json.dumps(environment_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash_manifest_artifact(candidate, raw_path)
    _rehash_manifest_artifact(candidate, environment_path)

    # When: comparison aligns controls by durable case/replicate identity, not a global Counter.
    comparison = compare_runs(baseline, candidate)

    # Then: equal global control counts cannot hide the per-case methodology swap.
    assert second_identity in {case.case_identity for case in comparison.case_comparisons}
    assert comparison.verdict == "not_comparable"
    assert "applied worker controls differ" in comparison.reasons


def test_compare_runs_rejects_multiple_actual_control_signatures_within_one_case(
    tmp_path: Path,
) -> None:
    """Reject a case whose successful workers did not apply one normalized methodology signature."""
    # Given: one replicate in a complete candidate records a different actual affinity.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    raw_path = candidate.parent / "raw_replicates.jsonl"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    response = cast("dict[str, JsonValue]", raw_records[0]["worker_response"])
    cast("dict[str, JsonValue]", response["environment"])["effective_cpu_affinity"] = [0]
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records),
        encoding="utf-8",
        newline="\n",
    )
    environment_path = candidate.parent / "environment.json"
    environment = cast(
        "dict[str, JsonValue]", json.loads(environment_path.read_text(encoding="utf-8"))
    )
    workers = cast("list[dict[str, JsonValue]]", environment["worker_environments"])
    workers[0]["effective_cpu_affinity"] = [0]
    _ = environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    _rehash_manifest_artifact(candidate, raw_path)
    _rehash_manifest_artifact(candidate, environment_path)

    # When: comparison reduces controls per aligned case instead of relying on replicate positions.
    comparison = compare_runs(baseline, candidate)

    # Then: it fails closed because the one candidate case used multiple control signatures.
    assert comparison.verdict == "not_comparable"
    assert "candidate case has inconsistent applied worker controls" in comparison.reasons


@pytest.mark.parametrize(
    "forgery",
    [
        "null_pid",
        "zero_pid",
        "null_timestamps",
        "invalid_timestamps",
        "reversed_timestamps",
        "null_return_code",
        "zero_return_code",
        "boolean_return_code",
        "nested_pid_mismatch",
    ],
)
def test_compare_runs_rejects_incomplete_outer_evidence_for_worker_declared_failures(
    tmp_path: Path, forgery: str
) -> None:
    """Require complete outer lifecycle evidence whenever a real worker emitted failure JSON."""
    # Given: a valid worker-declared partial failure is forged at one outer/nested contract field.
    baseline, candidate = _write_partial_failure_package(tmp_path, worker_declared=True)
    raw_path = candidate.parent / "raw_replicates.jsonl"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    failed = raw_records[-1]
    nested = cast("dict[str, JsonValue]", failed["worker_response"])
    if forgery == "null_pid":
        failed["worker_pid"] = None
        nested["worker_pid"] = None
    elif forgery == "zero_pid":
        failed["worker_pid"] = 0
        nested["worker_pid"] = 0
    elif forgery == "null_timestamps":
        failed["started_at_utc"] = None
        failed["ended_at_utc"] = None
        nested["started_at_utc"] = None
        nested["ended_at_utc"] = None
    elif forgery == "invalid_timestamps":
        failed["started_at_utc"] = "not-a-timestamp"
        nested["started_at_utc"] = "not-a-timestamp"
    elif forgery == "reversed_timestamps":
        failed["started_at_utc"] = "2026-07-28T01:02:05+00:00"
        failed["ended_at_utc"] = "2026-07-28T01:02:04+00:00"
        nested["started_at_utc"] = failed["started_at_utc"]
        nested["ended_at_utc"] = failed["ended_at_utc"]
    elif forgery == "null_return_code":
        failed["return_code"] = None
    elif forgery == "zero_return_code":
        failed["return_code"] = 0
    elif forgery == "boolean_return_code":
        failed["return_code"] = True
    else:
        nested["worker_pid"] = 9999
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records),
        encoding="utf-8",
        newline="\n",
    )
    _rehash_manifest_artifact(candidate, raw_path)

    # When/Then: a declared worker failure cannot use parent-only nullable lifecycle evidence.
    with pytest.raises(
        ValueError,
        match=(
            r"worker-declared failure|raw worker_pid|raw return_code|raw started_at_utc|"
            r"raw worker lifecycle|failed raw record|failed raw worker response"
        ),
    ):
        _ = compare_runs(baseline, candidate)


@pytest.mark.parametrize("worker_declared", [True, False])
def test_compare_runs_accepts_valid_worker_declared_and_parent_only_failures(
    tmp_path: Path, *, worker_declared: bool
) -> None:
    """Accept valid worker-declared and parent-only failures as partial, non-comparable evidence."""
    # Given: equivalent partial packages differing only in whether the worker supplied failure JSON.
    baseline, candidate = _write_partial_failure_package(tmp_path, worker_declared=worker_declared)

    # When: the strict parser validates the appropriate failure-origin contract.
    comparison = compare_runs(baseline, candidate)

    # Then: both retain evidence yet refuse a performance pass/fail verdict for partial data.
    assert comparison.verdict == "not_comparable"
    assert "candidate run status is partial" in comparison.reasons


def test_compare_runs_accepts_exact_parent_invalid_response_with_zero_return_code(
    tmp_path: Path,
) -> None:
    """Accept production-shaped invalid child output after the child exits successfully."""
    # Given: malformed zero-exit stdout leaves only parent-owned failure evidence in the raw record.
    baseline, candidate = _write_partial_failure_package(tmp_path, worker_declared=False)
    raw_path = candidate.parent / "raw_replicates.jsonl"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    raw_records[-1].update(
        {
            "worker_pid": None,
            "started_at_utc": None,
            "ended_at_utc": None,
            "return_code": 0,
            "error_type": "InvalidWorkerResponse",
            "message": "Expecting value: line 1 column 1 (char 0)",
            "worker_response": None,
        }
    )
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records),
        encoding="utf-8",
        newline="\n",
    )
    _rehash_manifest_artifact(candidate, raw_path)
    order_path = candidate.parent / "execution_order.json"
    order = cast("list[dict[str, JsonValue]]", json.loads(order_path.read_text(encoding="utf-8")))
    order[-1].update({"status": "error", "worker_pid": None})
    _ = order_path.write_text(
        json.dumps(order, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    _rehash_manifest_artifact(candidate, order_path)

    # When: strict comparison loads parent-only failure evidence.
    comparison = compare_runs(baseline, candidate)

    # Then: the partial run remains valid evidence but cannot receive a performance verdict.
    assert comparison.verdict == "not_comparable"
    assert "candidate run status is partial" in comparison.reasons


def test_compare_runs_rejects_zero_return_code_when_a_worker_declared_failure_exists(
    tmp_path: Path,
) -> None:
    """Keep zero exit codes invalid when a nested worker failure claims the process failed."""
    # Given: a valid worker-declared failure forges only its parent process return code.
    baseline, candidate = _write_partial_failure_package(tmp_path, worker_declared=True)
    raw_path = candidate.parent / "raw_replicates.jsonl"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    raw_records[-1]["return_code"] = 0
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records),
        encoding="utf-8",
        newline="\n",
    )
    _rehash_manifest_artifact(candidate, raw_path)

    # When/Then: only the parent-classified null-response path may retain a zero return code.
    with pytest.raises(ValueError, match=r"worker-declared failure.*return_code"):
        _ = compare_runs(baseline, candidate)


def test_compare_runs_accepts_empty_worker_declared_failure_messages(tmp_path: Path) -> None:
    """Allow production-valid empty strings while retaining outer/nested message equality."""
    # Given: a worker-declared failure records an empty message in both exact protocol locations.
    baseline, candidate = _write_partial_failure_package(tmp_path, worker_declared=True)
    raw_path = candidate.parent / "raw_replicates.jsonl"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    failed = raw_records[-1]
    failed["message"] = ""
    cast("dict[str, JsonValue]", failed["worker_response"])["message"] = ""
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records),
        encoding="utf-8",
        newline="\n",
    )
    _rehash_manifest_artifact(candidate, raw_path)

    # When/Then: empty but equal protocol messages remain valid partial evidence.
    assert compare_runs(baseline, candidate).verdict == "not_comparable"


@pytest.mark.parametrize("nested", [False, True])
def test_compare_runs_rejects_nonstring_worker_declared_failure_messages(
    tmp_path: Path, *, nested: bool
) -> None:
    """Require message fields to be strings while permitting only the empty-string value."""
    # Given: a worker-declared failure forges either its outer or nested message type.
    baseline, candidate = _write_partial_failure_package(tmp_path, worker_declared=True)
    raw_path = candidate.parent / "raw_replicates.jsonl"
    raw_records = cast(
        "list[dict[str, JsonValue]]",
        [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()],
    )
    failed = raw_records[-1]
    target = cast("dict[str, JsonValue]", failed["worker_response"]) if nested else failed
    target["message"] = 42
    _ = raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in raw_records),
        encoding="utf-8",
        newline="\n",
    )
    _rehash_manifest_artifact(candidate, raw_path)

    # When/Then: malformed message types cannot be treated as failure diagnostics.
    with pytest.raises(ValueError, match="message"):
        _ = compare_runs(baseline, candidate)
