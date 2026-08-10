from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from minigpt.stage17_evidence import (
    Stage17EvidenceVerificationError,
    generate_stage17_apc,
    generate_stage17_benchmark,
    generate_stage17_correctness,
    generate_stage17_evidence,
    generate_stage17_scheduling,
    run_stage17_stress,
    verify_stage17_evidence,
    write_stage17_stress,
)


def _write(path: Path, document: dict[str, object]) -> Path:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_stage17_stress_is_deterministic_and_releases_resources() -> None:
    first = run_stage17_stress(operations=100)
    second = run_stage17_stress(operations=100)

    assert first == second
    assert cast("int", first["operations"]) >= 100
    assert cast("int", first["preemptions"]) > 0
    assert cast("int", first["resumes"]) > 0
    assert cast("int", first["recompute_tokens"]) > 0
    assert first["all_requests_terminal"] is True
    assert first["all_resources_released"] is True
    assert first["terminal_prefill_logits_released"] is True


def test_stage17_package_binds_pressure_contracts_and_exact_hashes(tmp_path: Path) -> None:
    correctness = generate_stage17_correctness(tmp_path / "correctness.json")
    scheduling = generate_stage17_scheduling(tmp_path / "scheduling.json")
    apc = generate_stage17_apc(tmp_path / "apc.json")
    stress = write_stage17_stress(tmp_path / "stress.json", operations=100)
    benchmark = generate_stage17_benchmark(
        correctness_path=correctness,
        scheduling_path=scheduling,
        stress_path=stress,
        output_path=tmp_path / "benchmark.json",
    )
    lifecycle = _write(tmp_path / "lifecycle.json", {"schema_version": 1, "exit_code": 0})

    package = generate_stage17_evidence(
        correctness_path=correctness,
        scheduling_path=scheduling,
        apc_path=apc,
        benchmark_path=benchmark,
        stress_path=stress,
        lifecycle_path=lifecycle,
        package_root=tmp_path / "package",
        source_commit="stage17-test-source",
    )
    manifest = verify_stage17_evidence(package)
    summary = cast(
        "dict[str, object]",
        json.loads((package / "summary.json").read_text(encoding="utf-8")),
    )

    assert manifest["stage"] == "17"
    assert summary["kv_pressure_preemption"] is True
    assert summary["recompute_resume"] is True
    assert summary["recompute_does_not_sample"] is True
    assert summary["per_request_rng_equivalence"] is True
    assert summary["overflow_sliding_window_equivalence"] is True
    assert summary["per_tick_budget_respected"] is True
    assert summary["apc_shared_refs_released"] is True
    assert summary["resume_uses_private_recompute"] is True
    assert summary["no_starvation_finite_workload"] is True
    assert summary["benchmark_strict_verdict"] == "descriptive_only"
    assert summary["wall_clock_performance_improvement"] is False
    assert summary["dynamic_kv_reservation"] is False
    assert summary["cpu_swap"] is False
    readme = (package / "README.md").read_text(encoding="utf-8")
    assert readme.endswith("\n")
    assert not readme.endswith("\n\n")

    correctness_copy = package / "evidence" / "correctness.json"
    _ = correctness_copy.write_text(
        correctness_copy.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(Stage17EvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage17_evidence(package)
