import json
from dataclasses import replace
from pathlib import Path

import pytest

from minigpt.inference_benchmark import (
    InferenceReplicate,
    compare_inference_modes,
    run_inference_benchmark,
    summarize_inference_replicates,
    verify_inference_run_manifest,
)
from minigpt.inference_benchmark_config import (
    InvalidInferenceBenchmarkConfigError,
    expand_inference_cases,
    load_inference_benchmark_config,
)
from minigpt.inference_benchmark_worker import InferenceWorkerRequest, run_worker_request


def write_config(path: Path, *, output_root: Path, canonical: bool = False) -> Path:
    prompt_lengths = "[16, 32, 64, 128]" if canonical else "[4]"
    generated_lengths = "[8, 32, 64]" if canonical else "[3]"
    _ = path.write_text(
        f"""schema_version: 1
experiment_name: stage9-test
benchmark_seed: 20260806
vocab_size: 65
output_root: {output_root.as_posix()}
worker_timeout_seconds: 30.0
warmup_iterations: 0
measurement_iterations: 1
replicates: 2
minimum_replicates: 2
max_cv_percent: 5.0
torch_num_threads: 1
torch_num_interop_threads: 1
cpu_affinity: null
relevant_environment_variables: [OMP_NUM_THREADS, MKL_NUM_THREADS]
model:
  block_size: 256
  n_layer: 2
  n_head: 2
  n_embd: 16
  dropout: 0.0
  bias: false
batch_size: 1
prompt_lengths: {prompt_lengths}
generated_lengths: {generated_lengths}
""",
        encoding="utf-8",
        newline="\n",
    )
    return path


def make_replicate(
    *,
    mode: str,
    replicate_index: int,
    end_to_end_time_ms: float,
    environment_signature: str = "same-host",
) -> InferenceReplicate:
    return InferenceReplicate(
        status="ok",
        case_name="p16-g8",
        mode=mode,
        replicate_index=replicate_index,
        worker_pid=100 + replicate_index,
        prefill_time_ms=1.0,
        time_to_first_token_ms=1.0,
        median_decode_time_ms=0.5,
        generated_tokens_per_second=8_000 / end_to_end_time_ms,
        end_to_end_time_ms=end_to_end_time_ms,
        peak_rss_mib=200.0,
        kv_cache_bytes=4_096 if mode == "cached" else 0,
        environment_signature=environment_signature,
        worker_response={"status": "ok"},
        error_type=None,
        message=None,
        stdout="{}\n",
        stderr="",
    )


def test_config_expands_required_canonical_matrix(tmp_path: Path) -> None:
    # Given: the Stage 9 canonical prompt and generated-length matrix.
    config = load_inference_benchmark_config(
        write_config(tmp_path / "canonical.yaml", output_root=tmp_path / "runs", canonical=True)
    )

    # When: logical inference cases are expanded.
    cases = expand_inference_cases(config)

    # Then: all 4x3 non-overflow batch-one cases are explicit and deterministic.
    assert len(cases) == 12
    assert {(case.prompt_length, case.generated_length) for case in cases} == {
        (prompt, generated) for prompt in (16, 32, 64, 128) for generated in (8, 32, 64)
    }
    assert all(case.batch_size == 1 for case in cases)
    assert all(case.prompt_length + case.generated_length <= 256 for case in cases)


def test_config_rejects_canonical_overflow_case(tmp_path: Path) -> None:
    # Given: one requested measurement would cross the learned-position window.
    path = write_config(tmp_path / "overflow.yaml", output_root=tmp_path / "runs")
    content = path.read_text(encoding="utf-8").replace(
        "generated_lengths: [3]", "generated_lengths: [253]"
    )
    _ = path.write_text(content, encoding="utf-8", newline="\n")

    # When/Then: canonical configuration cannot silently benchmark overflow fallback.
    with pytest.raises(InvalidInferenceBenchmarkConfigError, match="exceeds block_size"):
        _ = load_inference_benchmark_config(path)


@pytest.mark.parametrize("mode", ["cached", "uncached"])
def test_worker_reports_complete_deterministic_inference_metrics(tmp_path: Path, mode: str) -> None:
    # Given: one small deterministic inference request in a fresh-worker shape.
    config = load_inference_benchmark_config(
        write_config(tmp_path / "worker.yaml", output_root=tmp_path / "runs")
    )
    case = expand_inference_cases(config)[0]
    request = InferenceWorkerRequest.from_config(config, case, mode=mode, replicate_index=0)

    # When: the unprofiled worker executes forced-token generation.
    result = run_worker_request(request)

    # Then: all required latency, throughput, memory, and identity fields are present.
    assert result.status == "ok"
    assert result.mode == mode
    assert result.prefill_time_ms > 0.0
    assert result.time_to_first_token_ms == result.prefill_time_ms
    assert result.median_decode_time_ms > 0.0
    assert result.generated_tokens_per_second > 0.0
    assert result.end_to_end_time_ms > 0.0
    assert result.peak_rss_mib > 0.0
    expected_cache_bytes = 2 * 2 * 1 * (4 + 3 - 1) * 16 * 4
    assert result.kv_cache_bytes == (expected_cache_bytes if mode == "cached" else 0)


def test_strict_comparison_requires_stable_compatible_equal_replicates() -> None:
    # Given: two complete stable mode samples from the same worker environment.
    uncached = tuple(
        make_replicate(mode="uncached", replicate_index=index, end_to_end_time_ms=value)
        for index, value in enumerate((10.0, 10.1, 9.9))
    )
    cached = tuple(
        make_replicate(mode="cached", replicate_index=index, end_to_end_time_ms=value)
        for index, value in enumerate((6.0, 6.1, 5.9))
    )
    uncached_summary = summarize_inference_replicates(
        uncached, minimum_replicates=3, max_cv_percent=5.0
    )
    cached_summary = summarize_inference_replicates(
        cached, minimum_replicates=3, max_cv_percent=5.0
    )

    # When: strict comparison validates stability and compatibility.
    comparison = compare_inference_modes(uncached_summary, cached_summary)

    # Then: only comparable evidence receives pass and may support a speedup statement.
    assert comparison.verdict == "pass"
    assert comparison.end_to_end_change_percent == pytest.approx(-40.0)
    assert comparison.reasons == ()

    incompatible_cached = replace(cached[-1], environment_signature="different-host")
    incompatible_summary = summarize_inference_replicates(
        (*cached[:-1], incompatible_cached), minimum_replicates=3, max_cv_percent=5.0
    )
    incompatible = compare_inference_modes(uncached_summary, incompatible_summary)
    assert incompatible.verdict == "not_comparable"
    assert "environment" in " ".join(incompatible.reasons)


def test_run_manifest_binds_every_artifact_and_detects_tampering(tmp_path: Path) -> None:
    # Given: a two-mode, two-replicate smoke run with real fresh subprocesses.
    config = load_inference_benchmark_config(
        write_config(tmp_path / "smoke.yaml", output_root=tmp_path / "runs")
    )
    artifacts = run_inference_benchmark(config)

    # When: the manifest is independently verified.
    verified = verify_inference_run_manifest(artifacts.run_manifest_path)

    # Then: raw evidence, environment, config, order, and summary are all hash-bound.
    assert verified["status"] == "complete"
    bound_artifacts = verified["artifacts"]
    assert isinstance(bound_artifacts, list)
    assert len(bound_artifacts) == 6
    raw_path = artifacts.run_directory / "raw_replicates.jsonl"
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all(json.loads(line)["worker_pid"] for line in lines)

    _ = raw_path.write_text(raw_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        _ = verify_inference_run_manifest(artifacts.run_manifest_path)
