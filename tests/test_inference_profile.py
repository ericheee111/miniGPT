import json
from pathlib import Path
from typing import cast

from minigpt.inference_benchmark_config import load_inference_benchmark_config
from minigpt.inference_profile import run_inference_profile


def write_profile_config(path: Path, output_root: Path) -> Path:
    _ = path.write_text(
        f"""schema_version: 1
experiment_name: stage9-profile-test
benchmark_seed: 20260806
vocab_size: 65
output_root: {output_root.as_posix()}
worker_timeout_seconds: 30.0
warmup_iterations: 0
measurement_iterations: 1
replicates: 1
minimum_replicates: 1
max_cv_percent: 20.0
torch_num_threads: 1
torch_num_interop_threads: 1
cpu_affinity: null
relevant_environment_variables: []
model:
  block_size: 32
  n_layer: 1
  n_head: 2
  n_embd: 16
  dropout: 0.0
  bias: false
batch_size: 1
prompt_lengths: [4]
generated_lengths: [3]
""",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_inference_profile_is_separate_descriptive_evidence(tmp_path: Path) -> None:
    # Given: a tiny deterministic profile case independent of benchmark timers.
    config = load_inference_benchmark_config(
        write_profile_config(tmp_path / "profile.yaml", tmp_path / "runs")
    )
    output_path = tmp_path / "profile.json"

    # When: cached and uncached operator profiles are captured.
    artifacts = run_inference_profile(
        config,
        output_path,
        prompt_length=4,
        generated_length=3,
        operator_limit=5,
    )
    document = cast(
        "dict[str, object]", json.loads(artifacts.output_path.read_text(encoding="utf-8"))
    )

    # Then: the artifact explicitly cannot be consumed as canonical timing evidence.
    assert document["descriptive_only"] is True
    assert document["canonical_timing_source"] is False
    modes = cast("dict[str, object]", document["modes"])
    assert set(modes) == {"cached", "uncached"}
    assert all(cast("dict[str, object]", mode)["top_operators"] for mode in modes.values())
