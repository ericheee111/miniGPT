from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from minigpt.stage11a_evidence import (
    Stage11AEvidenceVerificationError,
    generate_stage11a_evidence,
    verify_stage11a_evidence,
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
                "scenario": "burst-2",
                "strict_verdict": "pass",
                "speedup_reference_over_continuous": 1.25,
                "continuous_decode": {
                    "median_average_decode_batch_size": 2.0,
                    "median_padding_waste_ratio": 0.0,
                    "median_token_throughput_per_second": 100.0,
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


def test_stage11a_evidence_binds_benchmark_and_simulator_equivalence(tmp_path: Path) -> None:
    # Given: a verified benchmark fixture and one committed mixed simulator workload.
    benchmark = _benchmark_fixture(tmp_path / "benchmark")
    package = tmp_path / "decode-continuous-batching"

    # When: the Stage 11A evidence package is generated and independently verified.
    generated = generate_stage11a_evidence(
        benchmark_run_dir=benchmark,
        simulator_config_paths=(Path("configs/serving_stage11a_mixed.yaml"),),
        package_root=package,
        source_commit="0123456789abcdef",
    )
    manifest = verify_stage11a_evidence(generated)

    # Then: docs, raw benchmark data, dual simulator outputs, and required limitations are bound.
    assert manifest["stage"] == "11A"
    assert (generated / "evidence" / "benchmark" / "raw_replicates.jsonl").is_file()
    simulator = generated / "evidence" / "simulator" / "stage11a-mixed"
    assert (simulator / "reference" / "events.jsonl").is_file()
    assert (simulator / "continuous_decode" / "events.jsonl").is_file()
    raw_equivalence = cast(
        "object",
        json.loads((simulator / "equivalence.json").read_text(encoding="utf-8")),
    )
    assert isinstance(raw_equivalence, dict)
    equivalence = cast("dict[str, object]", raw_equivalence)
    assert equivalence["equivalent"] is True
    readme = (generated / "README.md").read_text(encoding="utf-8")
    assert "Prefill remains" in readme
    assert "not paged attention" in readme
    assert "padding" in readme.lower()

    # When/Then: any later mutation invalidates the package hash contract.
    _ = (simulator / "equivalence.json").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(Stage11AEvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage11a_evidence(generated)
