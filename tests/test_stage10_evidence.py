from __future__ import annotations

from pathlib import Path

import pytest

from minigpt.stage10_evidence import (
    EvidenceVerificationError,
    generate_stage10_evidence,
    verify_stage10_evidence,
)

CONFIGS = (
    Path("configs/serving_single_request.yaml"),
    Path("configs/serving_burst_arrivals.yaml"),
    Path("configs/serving_cache_pressure.yaml"),
)


def test_stage10_evidence_generation_is_complete_and_hash_verified(tmp_path: Path) -> None:
    # Given: the three committed deterministic workload configurations.
    package_root = tmp_path / "serving-control-plane"

    # When: the evidence package is generated and independently verified.
    generated = generate_stage10_evidence(
        config_paths=CONFIGS,
        package_root=package_root,
        source_commit="0123456789abcdef",
    )
    manifest = verify_stage10_evidence(generated)

    # Then: all required reports, timelines, summaries, configs, and top-level docs are bound.
    assert manifest["stage"] == "10"
    assert (generated / "README.md").is_file()
    assert (generated / "summary.json").is_file()
    for scenario in ("single-request", "burst-arrivals", "cache-pressure-cancellation"):
        scenario_root = generated / "evidence" / scenario
        assert {path.name for path in scenario_root.iterdir()} == {
            "events.jsonl",
            "requests.csv",
            "summary.json",
            "timeline.md",
            "workload.yaml",
        }


def test_stage10_evidence_verifier_rejects_mutation(tmp_path: Path) -> None:
    # Given: a generated package with one modified event stream.
    package_root = generate_stage10_evidence(
        config_paths=CONFIGS,
        package_root=tmp_path / "serving-control-plane",
        source_commit="0123456789abcdef",
    )
    events_path = package_root / "evidence" / "single-request" / "events.jsonl"
    _ = events_path.write_text("mutated\n", encoding="utf-8", newline="\n")

    # When/Then: verification detects the byte/hash mismatch.
    with pytest.raises(EvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage10_evidence(package_root)
