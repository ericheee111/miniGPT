from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from minigpt.batcher_benchmark import (
    BatcherBenchmarkCase,
    InvalidBatcherBenchmarkConfigError,
    batcher_case_identity,
    load_batcher_benchmark_config,
    run_batcher_benchmark,
)

if TYPE_CHECKING:
    from minigpt.benchmark_v2_config import JsonValue


def _write_config(path: Path, output_root: Path, *, unexpected: str = "") -> None:
    """Write one small strict config that is safe to execute in tests."""
    _ = path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "experiment_name: batcher-smoke",
                f'output_root: "{output_root.as_posix()}"',
                "worker_timeout_seconds: 30",
                "warmup_batches: 1",
                "measurement_batches: 3",
                "replicates: 2",
                "minimum_replicates: 2",
                "max_cv_percent: 100.0",
                "corpus_tokens: 512",
                "seed: 123",
                "cpu_affinity: null",
                "cases:",
                "  - name: smoke-b4-t16",
                "    batch_size: 4",
                "    block_size: 16",
                unexpected,
            ),
        ),
        encoding="utf-8",
    )


def test_load_batcher_benchmark_config_rejects_unknown_keys(tmp_path: Path) -> None:
    # Given: an otherwise valid config with an unrecognized case field.
    config_path = tmp_path / "batcher.yaml"
    _write_config(config_path, tmp_path / "reports", unexpected="    surprise: true")

    # When: the strict loader parses the config.
    with pytest.raises(
        InvalidBatcherBenchmarkConfigError,
        match="unexpected key 'surprise'",
    ):
        _ = load_batcher_benchmark_config(config_path)


def test_batcher_case_identity_ignores_name_but_tracks_workload() -> None:
    # Given: aliases with identical resolved work and one changed batch size.
    original = BatcherBenchmarkCase(name="original", batch_size=16, block_size=128)
    renamed = BatcherBenchmarkCase(name="renamed", batch_size=16, block_size=128)
    changed = BatcherBenchmarkCase(name="original", batch_size=32, block_size=128)

    # When: workload identities are calculated.
    original_identity = batcher_case_identity(original, corpus_tokens=4096, seed=7)

    # Then: display-only names do not matter, while a workload control does.
    assert original_identity == batcher_case_identity(renamed, corpus_tokens=4096, seed=7)
    assert original_identity != batcher_case_identity(changed, corpus_tokens=4096, seed=7)


def test_run_batcher_benchmark_writes_valid_fresh_process_evidence(tmp_path: Path) -> None:
    # Given: a minimal benchmark config with two requested fresh-process replicates.
    config_path = tmp_path / "batcher.yaml"
    output_root = tmp_path / "reports"
    _write_config(config_path, output_root)
    config = load_batcher_benchmark_config(config_path)

    # When: the isolated batch-only benchmark runs.
    artifacts = run_batcher_benchmark(config)

    # Then: raw evidence and its manifest are complete, linked, and schema-versioned.
    assert artifacts.status == "complete"
    raw_lines = artifacts.raw_replicates_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 2
    raw_documents = [cast("dict[str, JsonValue]", json.loads(line)) for line in raw_lines]
    assert all(document["status"] == "ok" for document in raw_documents)
    replicate_indexes = {cast("int", document["replicate_index"]) for document in raw_documents}
    assert replicate_indexes == {0, 1}
    manifest = cast(
        "dict[str, JsonValue]",
        json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8")),
    )
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "complete"
    assert manifest["raw_replicate_count"] == 2
    assert artifacts.summary_csv_path.is_file()
    assert artifacts.summary_markdown_path.is_file()


def test_committed_stage8_evidence_manifest_verifies_all_artifacts() -> None:
    # Given: the compact Stage 8 package and its reviewable artifact manifest.
    result_directory = Path(__file__).parents[1] / "docs" / "results" / "batcher-optimization"
    manifest_value = cast(
        "object",
        json.loads(
            (result_directory / "artifact_manifest.json").read_text(encoding="utf-8"),
        ),
    )
    assert isinstance(manifest_value, dict)
    manifest = cast("dict[str, object]", manifest_value)
    artifacts_value = manifest["artifacts"]
    assert isinstance(artifacts_value, list)
    artifacts = cast("list[object]", artifacts_value)

    # When: every declared path, byte count, and SHA-256 is recomputed.
    verified_paths: set[str] = set()
    for entry_value in artifacts:
        assert isinstance(entry_value, dict)
        entry = cast("dict[str, object]", entry_value)
        relative_path = entry["path"]
        expected_sha256 = entry["sha256"]
        expected_size = entry["size_bytes"]
        assert isinstance(relative_path, str)
        assert isinstance(expected_sha256, str)
        assert isinstance(expected_size, int)
        artifact_path = result_directory / relative_path
        verified_paths.add(relative_path)
        assert artifact_path.stat().st_size == expected_size
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == expected_sha256

    # Then: the report, summary, and every copied evidence file are bound by the manifest.
    expected_paths = {
        path.relative_to(result_directory).as_posix()
        for path in result_directory.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    assert verified_paths == expected_paths
