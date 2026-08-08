from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from minigpt.stage14_evidence import (
    Stage14EvidenceVerificationError,
    generate_stage14_correctness,
    generate_stage14_evidence,
    verify_stage14_evidence,
)


def _write(path: Path, document: object) -> Path:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_stage14_correctness_binds_dense_paged_direct_and_apc(tmp_path: Path) -> None:
    # Given/When: the canonical APC simulator workload runs all four storage/execution paths.
    project_root = Path(__file__).resolve().parents[1]
    output = generate_stage14_correctness(
        config_root=project_root / "configs",
        work_root=tmp_path / "runs",
        output_path=tmp_path / "correctness.json",
    )
    document = cast(
        "dict[str, object]",
        json.loads(output.read_text(encoding="utf-8")),
    )

    # Then: output, RNG, lifecycle, and logical events share one bound identity.
    assert document["equivalent"] is True
    hashes = cast("dict[str, str]", document["correctness_sha256"])
    assert set(hashes) == {"dense", "materialized", "direct", "apc"}
    assert len(set(hashes.values())) == 1
    assert document["avoided_prefill_tokens"] == 12
    assert document["active_resources_released"] is True


def test_stage14_package_binds_claim_policy_and_exact_hashes(tmp_path: Path) -> None:
    # Given: passing correctness/stress/lifecycle inputs and a strict benchmark failure.
    correctness = _write(
        tmp_path / "correctness.json",
        {"equivalent": True, "active_resources_released": True},
    )
    command_result = {"exit_code": 0}
    stress = _write(tmp_path / "stress.json", command_result)
    lifecycle = _write(tmp_path / "lifecycle.json", command_result)
    benchmark = _write(
        tmp_path / "benchmark.json",
        {
            "correctness_equivalent": True,
            "strict_verdict": "fail",
            "wall_clock_performance_improvement": False,
            "workloads": {
                "exact": {
                    "strict_verdict": "fail",
                    "strategies": {
                        "paged_direct_apc": {
                            "prefix_hit_requests": 1,
                            "prefix_hit_tokens": 4,
                            "avoided_prefill_tokens": 4,
                            "evictions": 0,
                            "prefix_hit_request_ratio": 0.5,
                            "prefix_hit_token_ratio": 0.5,
                        }
                    },
                }
            },
        },
    )

    # When: the package is generated and verified.
    package = generate_stage14_evidence(
        correctness_path=correctness,
        benchmark_path=benchmark,
        stress_path=stress,
        lifecycle_path=lifecycle,
        package_root=tmp_path / "package",
        source_commit="stage14-test-commit",
    )
    manifest = verify_stage14_evidence(package)
    summary = cast(
        "dict[str, object]",
        json.loads((package / "summary.json").read_text(encoding="utf-8")),
    )

    # Then: no wall-clock claim escapes a non-pass verdict and Stage 14 scope is explicit.
    assert manifest["stage"] == "14"
    assert summary["full_block_sharing_only"] is True
    assert summary["partial_tail_private"] is True
    assert summary["partial_block_copy_on_write"] is False
    assert summary["avoided_prefill_tokens"] == 4
    assert summary["benchmark_strict_verdict"] == "fail"
    assert summary["wall_clock_performance_improvement"] is False
    readme = (package / "README.md").read_text(encoding="utf-8")
    assert "Python/PyTorch reference implementation" in readme
    assert "no partial-block sharing or copy-on-write" in readme

    # When/Then: later artifact mutation fails the outer hash contract.
    _ = (package / "evidence" / "benchmark.json").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(Stage14EvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage14_evidence(package)
