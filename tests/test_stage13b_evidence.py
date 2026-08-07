from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from minigpt.paged_attention_benchmark import (
    PagedAttentionBenchmarkConfig,
    write_paged_attention_benchmark,
)
from minigpt.stage13b_evidence import (
    CORRECTNESS_CONFIGS,
    Stage13BEvidenceVerificationError,
    generate_stage13b_correctness,
    generate_stage13b_evidence,
    verify_stage13b_evidence,
)


def _lifecycle(path: Path) -> Path:
    _ = path.write_text(
        json.dumps({"schema_version": 1, "exit_code": 0}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_stage13b_evidence_binds_correctness_benchmark_and_claims(tmp_path: Path) -> None:
    # Given: fresh three-strategy simulations and a bounded real CPU benchmark.
    project_root = Path(__file__).resolve().parents[1]
    correctness = generate_stage13b_correctness(
        config_root=project_root / "configs",
        work_root=tmp_path / "runs",
        output_path=tmp_path / "correctness.json",
    )
    benchmark = write_paged_attention_benchmark(
        tmp_path / "benchmark.json",
        config=PagedAttentionBenchmarkConfig(
            warmups=0,
            repeats=2,
            cache_access_iterations=5,
        ),
    )

    # When: Stage 13B packages the inputs and verifies exact hashes.
    package = generate_stage13b_evidence(
        correctness_path=correctness,
        benchmark_path=benchmark,
        lifecycle_path=_lifecycle(tmp_path / "lifecycle.json"),
        package_root=tmp_path / "package",
        source_commit="stage13b-test-commit",
    )
    manifest = verify_stage13b_evidence(package)
    summary = cast(
        "dict[str, object]",
        json.loads((package / "summary.json").read_text(encoding="utf-8")),
    )
    correctness_document = cast(
        "dict[str, object]",
        json.loads((package / "evidence" / "correctness.json").read_text(encoding="utf-8")),
    )

    # Then: equivalence, no-materialization scope, dense overflow, and caveat are explicit.
    assert manifest["stage"] == "13B"
    assert summary["all_correctness_checks_passed"] is True
    assert summary["all_resources_released"] is True
    assert summary["normal_decode_dense_materialization"] is False
    assert summary["overflow_reprefill_remains_dense"] is True
    assert summary["speedup_claim"] is False
    scenarios = correctness_document["scenarios"]
    assert isinstance(scenarios, list)
    assert len(cast("list[object]", scenarios)) == len(CORRECTNESS_CONFIGS)
    readme = (package / "README.md").read_text(encoding="utf-8")
    assert "no speedup is claimed" in readme

    # When/Then: a later mutation fails the outer hash contract.
    _ = (package / "evidence" / "benchmark.json").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(Stage13BEvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage13b_evidence(package)
