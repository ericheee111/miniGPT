import json
from pathlib import Path
from typing import cast

from minigpt.stage9_evidence import verify_stage9_evidence


def test_committed_stage9_evidence_is_complete_and_hash_bound() -> None:
    # Given: the committed Stage 9 report package generated from raw evidence.
    package_root = Path("docs/results/kv-cache-generation")

    # When: all artifact and nested run-manifest hashes are recomputed.
    manifest = verify_stage9_evidence(package_root)
    summary = cast(
        "dict[str, object]",
        json.loads((package_root / "summary.json").read_text(encoding="utf-8")),
    )
    readme = (package_root / "README.md").read_text(encoding="utf-8")

    # Then: correctness, performance, overflow, memory, and profiler evidence stay distinct.
    assert manifest["stage"] == "stage9-kv-cache-generation"
    assert summary["stage"] == "stage9-kv-cache-generation"
    assert cast("dict[str, object]", summary["correctness"])["prefill_exact_equal"] is True
    performance = cast("dict[str, object]", summary["performance"])
    assert performance["case_count"] == 12
    assert performance["strict_verdict"] in {"pass", "not_comparable"}
    for heading in (
        "Correctness and generation semantics",
        "Canonical performance",
        "Cache memory and overflow",
        "TTFT, TPOT, throughput, and profiler",
    ):
        assert heading in readme
