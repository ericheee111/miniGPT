from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from minigpt.stage12_evidence import (
    Stage12EvidenceVerificationError,
    generate_stage12_evidence,
    verify_stage12_evidence,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, document: dict[str, object]) -> Path:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _benchmark_fixture(path: Path) -> Path:
    latency = {"p50": 0.01, "p95": 0.02, "p99": 0.03}
    return _write_json(
        path,
        {
            "duration_seconds": 1.0,
            "cases": [
                {
                    "concurrency": 2,
                    "prompt_kind": "mixed",
                    "stream": True,
                    "requests_per_second": 20.0,
                    "generated_tokens_per_second": 160.0,
                    "ttft_seconds": latency,
                    "tpot_seconds": latency,
                    "e2e_seconds": latency,
                    "http_error_count": 0,
                    "cancellation_count": 0,
                }
            ],
            "measurements": [{"status_code": 200}],
            "engine": {
                "peak_active_requests": 2,
                "average_prefill_batch_size": 1.5,
                "average_decode_batch_size": 2.0,
            },
        },
    )


def _api_fixture(path: Path) -> Path:
    return _write_json(
        path,
        {
            "non_stream": {
                "curl": "curl -X POST http://127.0.0.1:8000/v1/completions",
                "response": {"choices": [{"text": " JULIET"}]},
            },
            "streaming": {
                "curl": "curl -N -X POST http://127.0.0.1:8000/v1/completions",
                "sse_lines": ['data: {"choices":[{"text":" "}]}', "data: [DONE]"],
            },
            "concurrent": {"status_codes": [200, 200], "texts": ["A", "B"]},
        },
    )


def test_stage12_evidence_binds_http_benchmark_and_lifecycle(tmp_path: Path) -> None:
    # Given: fresh HTTP measurements, API examples, and passing lifecycle tests.
    benchmark = _benchmark_fixture(tmp_path / "benchmark.json")
    api_examples = _api_fixture(tmp_path / "api.json")
    lifecycle = _write_json(
        tmp_path / "lifecycle.json",
        {"command": "pytest lifecycle", "exit_code": 0, "output": "7 passed"},
    )
    package = tmp_path / "http-serving"

    # When: the package is generated and independently verified.
    generated = generate_stage12_evidence(
        benchmark_path=benchmark,
        api_examples_path=api_examples,
        lifecycle_path=lifecycle,
        package_root=package,
        source_commit="0123456789abcdef",
    )
    manifest = verify_stage12_evidence(generated)

    # Then: all raw evidence and the documented API/system boundary are hash-bound.
    assert manifest["stage"] == 12
    summary = cast(
        "dict[str, object]",
        json.loads((generated / "summary.json").read_text(encoding="utf-8")),
    )
    assert summary["benchmark_scope"] == "http_end_to_end"
    assert summary["benchmark_cases"] == 1
    assert summary["http_error_count"] == 0
    readme = (generated / "README.md").read_text(encoding="utf-8")
    assert "EngineRunner" in readme
    assert "data: [DONE]" in readme
    assert "Stage 11 executor benchmark" in readme
    assert "backpressure" in readme

    # When/Then: mutation after generation invalidates the artifact manifest.
    target = generated / "evidence" / "lifecycle.json"
    _ = target.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(Stage12EvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage12_evidence(generated)
