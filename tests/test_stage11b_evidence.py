from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from minigpt.stage11b_evidence import (
    Stage11BEvidenceVerificationError,
    generate_stage11b_evidence,
    verify_stage11b_evidence,
)


def _write_json(path: Path, document: dict[str, object]) -> None:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _benchmark_fixture(root: Path) -> Path:
    root.mkdir()
    summary: dict[str, object] = {
        "schema_version": 1,
        "strict_verdict": "pass",
        "scenarios": [
            {
                "scenario": "burst-equal-length",
                "strict_verdict": "pass",
                "performance_conclusion": "improved",
                "speedup_continuous_decode_over_continuous": 1.2,
                "continuous_decode": {
                    "median_median_ttft_seconds": 0.008,
                    "median_request_throughput_per_second": 80.0,
                },
                "continuous": {
                    "median_average_prefill_batch_size": 4.0,
                    "median_prompt_padding_waste_ratio": 0.0,
                    "median_median_ttft_seconds": 0.01,
                    "median_request_throughput_per_second": 100.0,
                },
            }
        ],
    }
    _write_json(root / "summary.json", summary)
    _ = (root / "raw_replicates.jsonl").write_text(
        '{"status":"ok"}\n',
        encoding="utf-8",
        newline="\n",
    )
    artifacts: list[dict[str, object]] = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.iterdir())
    ]
    _write_json(
        root / "artifact_manifest.json",
        {"schema_version": 1, "source_commit": "fixture", "artifacts": artifacts},
    )
    return root


def test_stage11b_evidence_binds_benchmark_and_three_executor_simulator(
    tmp_path: Path,
) -> None:
    # Given: a verified benchmark fixture and one Stage 11B mixed simulator workload.
    benchmark = _benchmark_fixture(tmp_path / "benchmark")
    package = tmp_path / "batched-prefill"

    # When: the Stage 11B evidence package is generated and independently verified.
    generated = generate_stage11b_evidence(
        benchmark_run_dir=benchmark,
        simulator_config_paths=(Path("configs/serving_stage11b_mixed.yaml"),),
        package_root=package,
        source_commit="0123456789abcdef",
    )
    manifest = verify_stage11b_evidence(generated)

    # Then: benchmark data, three executors, prefill events, and limitations are hash-bound.
    assert manifest["stage"] == "11B"
    assert (generated / "evidence" / "benchmark" / "raw_replicates.jsonl").is_file()
    simulator = generated / "evidence" / "simulator" / "stage11b-mixed"
    for executor in ("reference", "continuous_decode", "continuous"):
        assert (simulator / executor / "events.jsonl").is_file()
        assert (simulator / executor / "prefill_events.jsonl").is_file()
    raw_equivalence = cast(
        "object",
        json.loads((simulator / "equivalence.json").read_text(encoding="utf-8")),
    )
    assert isinstance(raw_equivalence, dict)
    equivalence = cast("dict[str, object]", raw_equivalence)
    assert equivalence["equivalent"] is True
    assert "cache_accounting" in cast("list[str]", equivalence["checked_contracts"])
    readme = (generated / "README.md").read_text(encoding="utf-8")
    assert "decode" in readme.lower()
    assert "prompt padding is more expensive" in readme.lower()
    assert "Throughput" in readme
    assert "not paged attention" in readme

    # When/Then: any later mutation invalidates the outer package manifest.
    _ = (simulator / "equivalence.json").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(Stage11BEvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage11b_evidence(generated)
