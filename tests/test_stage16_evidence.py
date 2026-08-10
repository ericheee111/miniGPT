from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from minigpt.stage16_evidence import (
    Stage16EvidenceVerificationError,
    generate_stage16_benchmark,
    generate_stage16_correctness,
    generate_stage16_evidence,
    generate_stage16_scheduling,
    run_stage16_stress,
    verify_stage16_evidence,
    write_stage16_stress,
)


def _write(path: Path, document: dict[str, object]) -> Path:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_stage16_stress_is_deterministic_and_releases_every_block() -> None:
    # Given/When: the same mixed chunk/APC/cancellation workload runs twice.
    first = run_stage16_stress(operations=100)
    second = run_stage16_stress(operations=100)

    # Then: its identity is deterministic and explicit cleanup empties the pool.
    assert first == second
    assert cast("int", first["operations"]) >= 100
    assert cast("int", first["chunk_count"]) > 0
    assert first["all_resources_released"] is True
    assert first["terminal_prefill_logits_released"] is True
    assert first["active_refs_after_cleanup"] == 0
    assert first["private_blocks_after_cleanup"] == 0
    assert first["reservations_after_cleanup"] == 0
    assert first["allocated_blocks_after_cleanup"] == 0


def test_stage16_package_binds_scheduler_claims_and_exact_hashes(tmp_path: Path) -> None:
    # Given: fresh correctness, scheduling, structural benchmark, stress, and lifecycle inputs.
    correctness = generate_stage16_correctness(tmp_path / "correctness.json")
    scheduling = generate_stage16_scheduling(tmp_path / "scheduling.json")
    benchmark = generate_stage16_benchmark(
        correctness_path=correctness,
        scheduling_path=scheduling,
        output_path=tmp_path / "benchmark.json",
    )
    stress = write_stage16_stress(tmp_path / "stress.json", operations=100)
    lifecycle = _write(tmp_path / "lifecycle.json", {"schema_version": 1, "exit_code": 0})

    # When: the evidence package is generated and independently verified.
    package = generate_stage16_evidence(
        correctness_path=correctness,
        scheduling_path=scheduling,
        benchmark_path=benchmark,
        stress_path=stress,
        lifecycle_path=lifecycle,
        package_root=tmp_path / "package",
        source_commit="stage16-test-source",
    )
    manifest = verify_stage16_evidence(package)
    summary = cast(
        "dict[str, object]",
        json.loads((package / "summary.json").read_text(encoding="utf-8")),
    )

    # Then: scheduling semantics, APC preservation, and the no-speedup boundary are explicit.
    assert manifest["stage"] == "16"
    assert summary["chunked_prefill"] is True
    assert summary["token_budget_scheduler"] is True
    assert summary["decode_prefill_interleaving"] is True
    assert summary["overflow_budget_accounting"] is True
    assert summary["terminal_prefill_logits_released"] is True
    assert summary["apc_batched_prefill_opt_in"] is True
    assert summary["apc_batched_prefill_default"] is False
    assert summary["intermediate_chunks_block_aligned"] is True
    assert summary["partial_final_chunk_supported"] is True
    assert summary["intermediate_chunks_sample"] is False
    assert summary["per_request_rng_equivalence"] is True
    assert summary["apc_prefix_reuse_preserved"] is True
    assert summary["historical_kv_materialized"] is False
    assert summary["benchmark_strict_verdict"] == "descriptive_only"
    assert summary["wall_clock_performance_improvement"] is False
    assert summary["partial_block_copy_on_write"] is False
    assert summary["preemption"] is False

    # When/Then: any post-manifest mutation is rejected by the exact hash verifier.
    with (package / "evidence" / "scheduling.json").open("a", encoding="utf-8") as stream:
        _ = stream.write(" ")
    with pytest.raises(Stage16EvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage16_evidence(package)
