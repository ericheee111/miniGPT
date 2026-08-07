from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from minigpt.stage13a_evidence import (
    SCENARIO_CONFIGS,
    STRESS_STEPS,
    Stage13AEvidenceVerificationError,
    generate_stage13a_evidence,
    run_allocator_stress,
    verify_stage13a_evidence,
)


def _lifecycle(path: Path) -> Path:
    _ = path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "exit_code": 0,
                "covered_contracts": ["test fixture"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_allocator_stress_is_deterministic_and_releases_every_block() -> None:
    # Given/When: the invariant workload is repeated from the same fixed seed.
    first = run_allocator_stress()
    second = run_allocator_stress()

    # Then: the trace is stable and final capacity is fully free.
    assert first == second
    assert first["steps"] == STRESS_STEPS
    assert first["trace_sha256"]
    final_metrics = first["final_metrics"]
    assert isinstance(final_metrics, dict)
    assert final_metrics["free_blocks"] == final_metrics["total_blocks"]
    assert final_metrics["allocated_blocks"] == 0
    assert final_metrics["reserved_blocks"] == 0
    reuse_count = final_metrics["block_reuse_count"]
    assert isinstance(reuse_count, int)
    assert reuse_count > 0


def test_evidence_generation_verification_and_tamper_detection(tmp_path: Path) -> None:
    # Given: fresh simulator inputs and a successful lifecycle command record.
    project_root = Path(__file__).resolve().parents[1]
    package_root = tmp_path / "package"

    # When: all seven scenarios and allocator stress are packaged.
    package = generate_stage13a_evidence(
        config_root=project_root / "configs",
        lifecycle_path=_lifecycle(tmp_path / "lifecycle.json"),
        package_root=package_root,
        work_root=tmp_path / "runs",
        source_commit="stage13a-test-commit",
    )
    manifest = verify_stage13a_evidence(package)
    summary = cast(
        "dict[str, object]",
        json.loads((package / "summary.json").read_text(encoding="utf-8")),
    )
    scenarios = cast(
        "dict[str, object]",
        json.loads((package / "evidence" / "scenarios.json").read_text(encoding="utf-8")),
    )

    # Then: membership, equivalence, zero leaks, and the performance boundary are explicit.
    assert manifest["stage"] == "13A"
    assert summary["scenario_count"] == len(SCENARIO_CONFIGS)
    assert summary["dense_paged_equivalence_scenarios"] == 5
    assert summary["all_resources_released"] is True
    assert summary["dense_materialization_remains"] is True
    assert summary["paged_attention"] is False
    scenario_rows = scenarios["scenarios"]
    assert isinstance(scenario_rows, list)
    assert len(cast("list[object]", scenario_rows)) == len(SCENARIO_CONFIGS)
    readme = (package / "README.md").read_text(encoding="utf-8")
    assert "not PagedAttention" in readme
    assert "no speedup is claimed" in readme

    # When/Then: any later mutation invalidates the outer artifact hash.
    _ = (package / "evidence" / "lifecycle_tests.json").write_text(
        "mutated\n",
        encoding="utf-8",
    )
    with pytest.raises(Stage13AEvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage13a_evidence(package)
