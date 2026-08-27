from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from minigpt.stage19_evidence import (
    Stage19EvidenceVerificationError,
    generate_stage19_checkpoint_identity,
    generate_stage19_evidence,
    generate_stage19_invalid_combinations,
    generate_stage19_manifest,
    generate_stage19_runtime_wiring,
    verify_stage19_evidence,
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


def test_stage19_runtime_wiring_proves_legacy_and_lazy_defaults(tmp_path: Path) -> None:
    # Given: the Stage 19 runtime wiring generator.
    output = generate_stage19_runtime_wiring(tmp_path / "runtime_wiring.json")

    # When: the wiring document is inspected.
    document = cast(
        "dict[str, object]",
        json.loads(output.read_text(encoding="utf-8")),
    )

    # Then: legacy defaults, lazy wiring, and idle cleanup are all witnessed.
    assert document["legacy_defaults_preserved"] is True
    scheduler = cast("dict[str, object]", document["legacy_scheduler_defaults"])
    assert scheduler["max_scheduled_tokens"] is None
    assert scheduler["prefill_chunk_tokens"] is None
    assert scheduler["kv_preemption"] is False
    assert scheduler["lazy_kv_reservation"] is False
    lazy = cast("dict[str, object]", document["lazy_scheduler_resolved"])
    assert lazy["lazy_kv_reservation"] is True
    assert lazy["kv_preemption"] is True
    assert cast("float", lazy["kv_overcommit_ratio"]) == pytest.approx(2.0)
    assert document["executor"] == "PagedAttentionExecutor"
    assert document["apc_prefill_strategy"] == "sequential"
    assert document["batched_apc_wired"] is True
    assert document["idle_resources_released"] is True


def test_stage19_invalid_combinations_are_all_rejected(tmp_path: Path) -> None:
    # Given: the Stage 19 invalid-combination generator.
    output = generate_stage19_invalid_combinations(tmp_path / "invalid_combinations.json")

    # When: the rejection document is inspected.
    document = cast(
        "dict[str, object]",
        json.loads(output.read_text(encoding="utf-8")),
    )

    # Then: every documented combination is rejected with a reason.
    rejected = cast("dict[str, object]", document["rejected_combinations"])
    assert document["all_rejected"] is True
    assert len(rejected) >= 8
    assert all(isinstance(reason, str) and reason for reason in rejected.values())


def test_stage19_manifest_evidence_is_deterministic_and_atomic(tmp_path: Path) -> None:
    # Given: the Stage 19 manifest generator bound to one work root.
    output = generate_stage19_manifest(tmp_path / "manifest.json", work_root=tmp_path)

    # When: the manifest evidence document is inspected.
    document = cast(
        "dict[str, object]",
        json.loads(output.read_text(encoding="utf-8")),
    )

    # Then: deterministic bytes, portability, and atomic replacement are witnessed.
    assert document["deterministic_bytes"] is True
    assert document["lf_only"] is True
    assert document["no_absolute_paths"] is True
    assert document["no_timestamps"] is True
    assert document["atomic_replacement"] is True
    assert document["no_temporary_leftovers"] is True
    assert document["claim_policy_bounded"] is True
    digest = cast("str", document["manifest_bytes_sha256"])
    assert len(digest) == 64


def test_stage19_checkpoint_identity_binds_real_files(tmp_path: Path) -> None:
    # Given: the Stage 19 identity generator and a scratch directory.
    output = generate_stage19_checkpoint_identity(tmp_path / "identity.json", tmp_path)

    # When: the identity document is inspected.
    document = cast(
        "dict[str, object]",
        json.loads(output.read_text(encoding="utf-8")),
    )

    # Then: real checkpoint/tokenizer digests are recorded and bound.
    assert document["identity_bound"] is True
    assert len(cast("str", document["checkpoint_sha256"])) == 64
    assert len(cast("str", document["tokenizer_sha256"])) == 64


def test_stage19_package_binds_contracts_and_exact_hashes(tmp_path: Path) -> None:
    # Given: generated Stage 19 inputs and a passing lifecycle witness.
    runtime_wiring = generate_stage19_runtime_wiring(tmp_path / "runtime_wiring.json")
    invalid_combinations = generate_stage19_invalid_combinations(
        tmp_path / "invalid_combinations.json"
    )
    manifest = generate_stage19_manifest(tmp_path / "manifest.json", work_root=tmp_path)
    identity = generate_stage19_checkpoint_identity(tmp_path / "identity.json", tmp_path)
    lifecycle = _write(tmp_path / "lifecycle.json", {"schema_version": 1, "exit_code": 0})

    # When: the hash-bound package is generated and verified.
    package = generate_stage19_evidence(
        runtime_wiring_path=runtime_wiring,
        invalid_combinations_path=invalid_combinations,
        manifest_path=manifest,
        identity_path=identity,
        lifecycle_path=lifecycle,
        package_root=tmp_path / "package",
        source_commit="stage19-test-source",
    )
    evidence_manifest = verify_stage19_evidence(package)
    summary = cast(
        "dict[str, object]",
        json.loads((package / "summary.json").read_text(encoding="utf-8")),
    )

    # Then: contracts hold, claims stay bounded, and hashes bind every artifact.
    assert evidence_manifest["stage"] == "19"
    assert summary["legacy_defaults_preserved"] is True
    assert summary["stage15_apc_prefill_strategy_flag"] is True
    assert summary["stage16_token_budget_flags"] is True
    assert summary["stage17_preemption_flag"] is True
    assert summary["stage18_lazy_reservation_flag"] is True
    assert summary["typed_policy_validation"] is True
    assert summary["runtime_manifest"] is True
    assert summary["deterministic_manifest_bytes"] is True
    assert summary["atomic_manifest_replacement"] is True
    assert summary["checkpoint_tokenizer_identity_bound"] is True
    assert summary["http_schema_unchanged"] is True
    assert summary["lifecycle_passed"] is True
    assert summary["benchmark_strict_verdict"] == "descriptive_only"
    assert summary["wall_clock_performance_improvement"] is False
    assert summary["public_production_security_readiness"] is False
    readme = (package / "README.md").read_text(encoding="utf-8")
    assert readme.endswith("\n")
    assert not readme.endswith("\n\n")

    # When: one bound artifact byte is modified.
    wiring_copy = package / "evidence" / "runtime_wiring.json"
    _ = wiring_copy.write_text(
        wiring_copy.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    # Then: verification rejects the corrupted package.
    with pytest.raises(Stage19EvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage19_evidence(package)


def test_stage19_package_requires_passing_lifecycle(tmp_path: Path) -> None:
    # Given: generated inputs but a failing lifecycle witness.
    runtime_wiring = generate_stage19_runtime_wiring(tmp_path / "runtime_wiring.json")
    invalid_combinations = generate_stage19_invalid_combinations(
        tmp_path / "invalid_combinations.json"
    )
    manifest = generate_stage19_manifest(tmp_path / "manifest.json", work_root=tmp_path)
    identity = generate_stage19_checkpoint_identity(tmp_path / "identity.json", tmp_path)
    lifecycle = _write(tmp_path / "lifecycle.json", {"schema_version": 1, "exit_code": 1})

    # When: the package is generated.
    # Then: the failing lifecycle witness is rejected.
    with pytest.raises(Stage19EvidenceVerificationError, match="lifecycle"):
        _ = generate_stage19_evidence(
            runtime_wiring_path=runtime_wiring,
            invalid_combinations_path=invalid_combinations,
            manifest_path=manifest,
            identity_path=identity,
            lifecycle_path=lifecycle,
            package_root=tmp_path / "package",
            source_commit="stage19-test-source",
        )


def test_stage19_verification_error_allows_traceback_assignment() -> None:
    error = Stage19EvidenceVerificationError("injected")

    error.__traceback__ = None

    assert error.__traceback__ is None
