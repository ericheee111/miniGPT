from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
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
from minigpt.batcher_benchmark_evidence import (
    InvalidBatcherBenchmarkEvidenceError,
    compare_batcher_benchmarks,
    load_batcher_benchmark_run,
    write_batcher_comparison,
)
from minigpt.benchmark_v2_comparison_policy import load_comparison_policy

if TYPE_CHECKING:
    from minigpt.benchmark_v2_config import JsonValue


def _write_config(path: Path, output_root: Path, *, unexpected: str = "") -> None:
    """Write one small strict config that is safe to execute in tests."""
    _ = path.write_text(
        "\n".join(
            (
                "schema_version: 2",
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
                "relevant_environment_variables: []",
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
    assert manifest["schema_version"] == 2
    assert manifest["status"] == "complete"
    assert manifest["raw_replicate_count"] == 2
    assert re.fullmatch(
        r"[0-9a-f]{40}",
        cast("str", manifest["evidence_generator_git_commit_sha"]),
    )
    artifact_entries = cast("list[dict[str, JsonValue]]", manifest["artifacts"])
    assert {cast("str", entry["path"]) for entry in artifact_entries} == {
        "environment.json",
        "execution_order.json",
        "raw_replicates.jsonl",
        "resolved_config.yaml",
        "summary.csv",
        "summary.md",
    }
    for entry in artifact_entries:
        artifact_path = artifacts.run_directory / cast("str", entry["path"])
        content = artifact_path.read_bytes()
        assert entry["size_bytes"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
    loaded = load_batcher_benchmark_run(artifacts.run_manifest_path)
    assert loaded.summaries[0].replicate_count == 2
    assert len({record.worker_pid for record in loaded.records}) == 2
    assert artifacts.summary_csv_path.is_file()
    assert artifacts.summary_markdown_path.is_file()


def test_batcher_loader_rejects_changed_bound_raw_evidence(tmp_path: Path) -> None:
    # Given: one complete hash-bound batch-only run.
    config_path = tmp_path / "batcher.yaml"
    _write_config(config_path, tmp_path / "reports")
    artifacts = run_batcher_benchmark(load_batcher_benchmark_config(config_path))

    # When: a raw replicate is modified after the manifest was finalized.
    with artifacts.raw_replicates_path.open("a", encoding="utf-8") as stream:
        _ = stream.write("{}\n")

    # Then: the strict loader rejects the package before trusting its summary.
    with pytest.raises(
        InvalidBatcherBenchmarkEvidenceError,
        match="hash or size mismatch",
    ):
        _ = load_batcher_benchmark_run(artifacts.run_manifest_path)


def test_batcher_comparison_uses_one_policy_and_writes_machine_json(tmp_path: Path) -> None:
    # Given: two same-code complete runs and one explicit shared comparison policy.
    config_path = tmp_path / "batcher.yaml"
    _write_config(config_path, tmp_path / "reports")
    config = load_batcher_benchmark_config(config_path)
    baseline = run_batcher_benchmark(config)
    candidate = run_batcher_benchmark(config)
    policy_path = tmp_path / "policy.yaml"
    _ = policy_path.write_text(
        """\
schema_version: 1
minimum_successful_replicates: 2
max_cv_percent: 100.0
require_equal_replicate_count: true
regression_threshold_percent: 100.0
""",
        encoding="utf-8",
    )
    policy = load_comparison_policy(policy_path)

    # When: the strict comparison is computed and serialized.
    comparison = compare_batcher_benchmarks(
        baseline.run_manifest_path,
        candidate.run_manifest_path,
        policy,
    )
    output_path = tmp_path / "comparison.json"
    write_batcher_comparison(comparison, output_path)
    document = cast(
        "dict[str, JsonValue]",
        json.loads(output_path.read_text(encoding="utf-8")),
    )

    # Then: both inputs are evaluated under the exact same policy identity.
    assert document["policy_sha256"] == policy.sha256
    assert document["verdict"] == "pass"
    case = cast("list[dict[str, JsonValue]]", document["cases"])[0]
    assert case["baseline_stability"] == "stable"
    assert case["candidate_stability"] == "stable"
    assert isinstance(case["batch_time_change_percent"], float)


def test_committed_stage8_candidate_sha_is_consistent_and_exists() -> None:
    # Given: every committed Stage 8 surface that declares the optimized candidate.
    result_directory = Path(__file__).parents[1] / "docs" / "results" / "batcher-optimization"
    report = (result_directory / "README.md").read_text(encoding="utf-8")
    summary = cast(
        "dict[str, JsonValue]",
        json.loads((result_directory / "summary.json").read_text(encoding="utf-8")),
    )
    batcher_manifest = cast(
        "dict[str, JsonValue]",
        json.loads(
            (result_directory / "evidence" / "batcher-candidate-manifest.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    reference_manifest = cast(
        "dict[str, JsonValue]",
        json.loads(
            (result_directory / "evidence" / "reference-candidate-manifest.json").read_text(
                encoding="utf-8",
            ),
        ),
    )

    # When: their candidate Git identities are extracted.
    report_match = re.search(r"The candidate used\s+`([0-9a-f]+)`", report)
    assert report_match is not None
    reference_git = cast("dict[str, JsonValue]", reference_manifest["git"])
    candidate_shas = {
        report_match.group(1),
        cast("str", summary["candidate_git_commit_sha"]),
        cast("str", batcher_manifest["git_commit_sha"]),
        cast("str", reference_git["commit_sha"]),
    }

    # Then: one canonical lowercase commit exists in the local Git object database.
    assert len(candidate_shas) == 1
    candidate_sha = candidate_shas.pop()
    assert re.fullmatch(r"[0-9a-f]{40}", candidate_sha)
    git = shutil.which("git")
    assert git is not None
    completed = subprocess.run(  # noqa: S603 - resolved Git executable and validated SHA.
        [git, "cat-file", "-e", f"{candidate_sha}^{{commit}}"],
        cwd=Path(__file__).parents[1],
        check=False,
    )
    assert completed.returncode == 0


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
