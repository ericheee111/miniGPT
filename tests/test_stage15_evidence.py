from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from minigpt.stage15_evidence import (
    Stage15EvidenceVerificationError,
    generate_stage15_batching,
    generate_stage15_correctness,
    generate_stage15_evidence,
    run_stage15_stress,
    verify_stage15_evidence,
    write_stage15_stress,
)


def _write(path: Path, document: dict[str, object]) -> Path:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _benchmark(path: Path) -> Path:
    sequential = {
        "cache_aware_prefill_model_calls": 5,
        "cache_aware_prefill_batches": 0,
        "batched_suffix_requests": 0,
        "max_suffix_prefill_batch_size": 1,
        "suffix_useful_tokens": 12,
        "suffix_padded_tokens": 12,
        "avoided_prefill_tokens": 32,
    }
    batched = {
        "cache_aware_prefill_model_calls": 2,
        "cache_aware_prefill_batches": 2,
        "batched_suffix_requests": 5,
        "average_suffix_prefill_batch_size": 2.5,
        "max_suffix_prefill_batch_size": 4,
        "suffix_useful_tokens": 12,
        "suffix_padded_tokens": 12,
        "avoided_prefill_tokens": 32,
    }
    return _write(
        path,
        {
            "schema_version": 1,
            "correctness_equivalent": True,
            "model_call_reduction": True,
            "strict_verdict": "fail",
            "wall_clock_performance_improvement": False,
            "workloads": {
                "repeated_prefix_short_suffix": {
                    "strict_verdict": "fail",
                    "strategies": {
                        "apc_sequential": sequential,
                        "apc_batched": batched,
                    },
                }
            },
        },
    )


def test_stage15_stress_is_deterministic_and_releases_every_block() -> None:
    # Given/When: the same mixed hit/miss/promotion/cancellation stress runs twice.
    first = run_stage15_stress(operations=100)
    second = run_stage15_stress(operations=100)

    # Then: its logical identity is deterministic and both runs end fully empty.
    assert first == second
    assert cast("int", first["operations"]) >= 100
    assert cast("int", first["prefix_hits"]) > 0
    assert first["all_resources_released"] is True
    assert first["active_refs_after_cleanup"] == 0
    assert first["private_blocks_after_cleanup"] == 0
    assert first["reservations_after_cleanup"] == 0


def test_stage15_package_binds_batching_claim_policy_and_exact_hashes(tmp_path: Path) -> None:
    # Given: fresh sequential/batched correctness, structural batching, stress, and lifecycle data.
    project_root = Path(__file__).resolve().parents[1]
    correctness = generate_stage15_correctness(
        config_root=project_root / "configs",
        work_root=tmp_path / "simulations",
        output_path=tmp_path / "correctness.json",
    )
    batching = generate_stage15_batching(
        correctness_path=correctness,
        output_path=tmp_path / "batching.json",
    )
    stress = write_stage15_stress(tmp_path / "stress.json", operations=100)
    lifecycle = _write(tmp_path / "lifecycle.json", {"schema_version": 1, "exit_code": 0})
    benchmark = _benchmark(tmp_path / "benchmark.json")

    # When: the evidence package is generated and independently verified.
    package = generate_stage15_evidence(
        correctness_path=correctness,
        batching_path=batching,
        benchmark_path=benchmark,
        stress_path=stress,
        lifecycle_path=lifecycle,
        package_root=tmp_path / "package",
        source_commit="stage15-test-source",
    )
    manifest = verify_stage15_evidence(package)
    summary = cast(
        "dict[str, object]",
        json.loads((package / "summary.json").read_text(encoding="utf-8")),
    )

    # Then: fake batching, scope limits, strict fail, and no wall-clock claim are explicit.
    assert manifest["stage"] == "15"
    assert summary["batched_prefill_opt_in"] is True
    assert summary["batched_prefill_default"] is False
    assert summary["fake_batching_guard_passed"] is True
    assert summary["sequential_model_calls"] == 5
    assert summary["batched_model_calls"] == 2
    assert summary["model_call_reduction"] == 3
    assert summary["benchmark_strict_verdict"] == "fail"
    assert summary["wall_clock_performance_improvement"] is False
    assert summary["historical_kv_materialized"] is False
    assert summary["chunked_prefill"] is False
    assert summary["partial_block_copy_on_write"] is False
    assert summary["preemption"] is False

    # When/Then: any post-manifest mutation is rejected by the exact hash verifier.
    with (package / "evidence" / "batching.json").open("a", encoding="utf-8") as stream:
        _ = stream.write(" ")
    with pytest.raises(Stage15EvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage15_evidence(package)
