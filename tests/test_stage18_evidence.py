from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from minigpt.stage18_evidence import (
    Stage18EvidenceVerificationError,
    generate_stage18_apc,
    generate_stage18_benchmark,
    generate_stage18_correctness,
    generate_stage18_evidence,
    generate_stage18_scheduling,
    run_stage18_stress,
    verify_stage18_evidence,
    write_stage18_stress,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, document: dict[str, object]) -> Path:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_stage18_stress_is_deterministic_and_releases_resources() -> None:
    first = run_stage18_stress(operations=100)
    second = run_stage18_stress(operations=100)

    assert first == second
    assert cast("int", first["operations"]) >= 100
    assert cast("int", first["reservation_growths"]) > 0
    assert cast("int", first["reservation_growth_blocked"]) > 0
    assert cast("int", first["growth_pressure_preemptions"]) > 0
    assert cast("int", first["preemptions"]) > 0
    assert cast("int", first["resumes"]) > 0
    assert first["all_requests_terminal"] is True
    assert first["all_resources_released"] is True
    assert first["terminal_prefill_logits_released"] is True


def test_stage18_package_binds_growth_contracts_and_exact_hashes(tmp_path: Path) -> None:
    correctness = generate_stage18_correctness(tmp_path / "correctness.json")
    scheduling = generate_stage18_scheduling(tmp_path / "scheduling.json")
    apc = generate_stage18_apc(tmp_path / "apc.json")
    stress = write_stage18_stress(tmp_path / "stress.json", operations=100)
    benchmark = generate_stage18_benchmark(
        correctness_path=correctness,
        scheduling_path=scheduling,
        stress_path=stress,
        output_path=tmp_path / "benchmark.json",
    )
    lifecycle = _write(tmp_path / "lifecycle.json", {"schema_version": 1, "exit_code": 0})

    package = generate_stage18_evidence(
        correctness_path=correctness,
        scheduling_path=scheduling,
        apc_path=apc,
        benchmark_path=benchmark,
        stress_path=stress,
        lifecycle_path=lifecycle,
        package_root=tmp_path / "package",
        source_commit="stage18-test-source",
    )
    manifest = verify_stage18_evidence(package)
    summary = cast(
        "dict[str, object]",
        json.loads((package / "summary.json").read_text(encoding="utf-8")),
    )

    assert manifest["stage"] == "18"
    assert summary["lazy_kv_reservation"] is True
    assert summary["controlled_overcommit"] is True
    assert summary["growth_before_model_work"] is True
    assert summary["growth_work_free"] is True
    assert summary["growth_pressure_preemption"] is True
    assert summary["immediate_growth_retry"] is True
    assert summary["per_request_rng_equivalence"] is True
    assert summary["overflow_sliding_window_equivalence"] is True
    assert summary["intrinsic_impossible_heads_rejected"] is True
    assert summary["recompute_resume_equivalence"] is True
    assert summary["per_tick_budget_respected"] is True
    assert summary["apc_shared_refs_released"] is True
    assert summary["resume_uses_private_recompute"] is True
    assert summary["no_starvation_finite_workload"] is True
    assert summary["benchmark_strict_verdict"] == "descriptive_only"
    assert summary["wall_clock_performance_improvement"] is False
    assert summary["cpu_swap"] is False
    assert summary["partial_block_copy_on_write"] is False
    assert summary["new_http_api"] is False
    readme = (package / "README.md").read_text(encoding="utf-8")
    assert readme.endswith("\n")
    assert not readme.endswith("\n\n")

    correctness_copy = package / "evidence" / "correctness.json"
    _ = correctness_copy.write_text(
        correctness_copy.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(Stage18EvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage18_evidence(package)
