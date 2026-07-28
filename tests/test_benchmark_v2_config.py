"""Test strict parsing and stable identities for Benchmark v2."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from minigpt.benchmark_v2_config import (
    InvalidBenchmarkV2ConfigError,
    case_identity,
    load_benchmark_v2_config,
    resolved_config_document,
    resolved_config_sha256,
)

if TYPE_CHECKING:
    from pathlib import Path

    from minigpt.benchmark_v2_types import BenchmarkV2Case


def write_v2_config(tmp_path: Path, replacement: str = "") -> Path:
    """Write a valid v2 config, optionally replacing one literal fragment."""
    document = """\
schema_version: 2
experiment_name: cpu_benchmark_v2
benchmark_seed: 1337
vocab_size: 65
output_root: reports/benchmark_v2
worker_timeout_seconds: 60.0
warmup_steps: 2
measurement_steps: 5
replicates: 3
torch_num_interop_threads: 1
cpu_affinity: [0, 1]
max_cv_percent: 5.0
minimum_replicates: 2
regression_threshold_percent: 3.0
relevant_environment_variables: [OMP_NUM_THREADS, MKL_NUM_THREADS]
models:
  tiny:
    n_layer: 2
    n_head: 2
    n_embd: 64
cases:
  - name: tiny_t1_s32_b2
    model_name: tiny
    torch_num_threads: 1
    block_size: 32
    batch_size: 2
  - name: tiny_t2_s32_b2
    model_name: tiny
    torch_num_threads: 2
    block_size: 32
    batch_size: 2
profile:
  enabled: true
  case_name: tiny_t1_s32_b2
  warmup_steps: 2
  active_steps: 3
"""
    path = tmp_path / "benchmark_v2.yaml"
    _ = path.write_text(document.replace("REPLACEMENT", replacement), encoding="utf-8")
    return path


def test_v2_config_resolves_explicit_cases_and_stable_hash(tmp_path: Path) -> None:
    """Parse explicit cases in declaration order with stable resolved identities."""
    # Given: a schema-v2 YAML with one model and two explicit cases.
    path = write_v2_config(tmp_path)

    # When: it is parsed twice.
    first = load_benchmark_v2_config(path)
    second = load_benchmark_v2_config(path)

    # Then: order and identities are stable without creating a Cartesian product.
    assert [case.name for case in first.cases] == ["tiny_t1_s32_b2", "tiny_t2_s32_b2"]
    assert resolved_config_sha256(first) == resolved_config_sha256(second)
    assert case_identity(first, first.cases[0]) != case_identity(first, first.cases[1])
    assert resolved_config_document(first)["output_root"] == "reports/benchmark_v2"


def test_v2_config_accepts_zero_top_level_warmup(tmp_path: Path) -> None:
    """Allow measurements with no warmup while still rejecting negative values."""
    # Given: a schema-v2 config with zero top-level warmup steps.
    path = write_v2_config(tmp_path)
    document = path.read_text(encoding="utf-8").replace(
        "warmup_steps: 2\nmeasurement_steps: 5",
        "warmup_steps: 0\nmeasurement_steps: 5",
    )
    _ = path.write_text(document, encoding="utf-8")

    # When: the strict config parser loads it.
    config = load_benchmark_v2_config(path)

    # Then: zero is preserved as a valid no-warmup methodology.
    assert config.warmup_steps == 0


@pytest.mark.parametrize(
    ("original", "updated"),
    [
        ("output_root: reports/benchmark_v2", "output_root: reports/another_location"),
        ("worker_timeout_seconds: 60.0", "worker_timeout_seconds: 120.0"),
        ("replicates: 3", "replicates: 5"),
        ("max_cv_percent: 5.0", "max_cv_percent: 7.5"),
        ("minimum_replicates: 2", "minimum_replicates: 3"),
        ("regression_threshold_percent: 3.0", "regression_threshold_percent: 4.0"),
        (
            "relevant_environment_variables: [OMP_NUM_THREADS, MKL_NUM_THREADS]",
            "relevant_environment_variables: [OPENBLAS_NUM_THREADS]",
        ),
        ("  enabled: true", "  enabled: false"),
        ("  case_name: tiny_t1_s32_b2", "  case_name: tiny_t2_s32_b2"),
        ("  warmup_steps: 2\n  active_steps: 3", "  warmup_steps: 4\n  active_steps: 3"),
        ("  active_steps: 3", "  active_steps: 4"),
    ],
)
def test_case_identity_excludes_execution_and_reporting_settings(
    tmp_path: Path, original: str, updated: str
) -> None:
    """Keep one workload identity stable across each non-workload setting."""
    # Given: the same explicit workload with one set of output and report settings.
    path = write_v2_config(tmp_path)
    first = load_benchmark_v2_config(path)
    first_identity = case_identity(first, first.cases[0])

    # When: one execution, reporting, environment, or profile field changes.
    _ = path.write_text(
        path.read_text(encoding="utf-8").replace(original, updated), encoding="utf-8"
    )
    second = load_benchmark_v2_config(path)

    # Then: provenance changes, but the workload's case identity does not.
    assert resolved_config_sha256(first) != resolved_config_sha256(second)
    assert first_identity == case_identity(second, second.cases[0])


@pytest.mark.parametrize(
    ("original", "updated"),
    [
        (
            "warmup_steps: 2\nmeasurement_steps: 5",
            "warmup_steps: 3\nmeasurement_steps: 5",
        ),
        ("measurement_steps: 5", "measurement_steps: 6"),
        ("torch_num_interop_threads: 1", "torch_num_interop_threads: 2"),
        ("cpu_affinity: [0, 1]", "cpu_affinity: [1, 2]"),
        ("benchmark_seed: 1337", "benchmark_seed: 1338"),
        ("vocab_size: 65", "vocab_size: 66"),
    ],
)
def test_case_identity_changes_for_workload_and_methodology_settings(
    tmp_path: Path, original: str, updated: str
) -> None:
    """Distinguish workloads and benchmark methodologies that affect a case."""
    # Given: one explicit workload and methodology.
    path = write_v2_config(tmp_path)
    first = load_benchmark_v2_config(path)

    # When: one workload or methodology field changes independently.
    _ = path.write_text(
        path.read_text(encoding="utf-8").replace(original, updated), encoding="utf-8"
    )
    second = load_benchmark_v2_config(path)

    # Then: the case identity changes.
    assert case_identity(first, first.cases[0]) != case_identity(second, second.cases[0])


def changed_case(case: BenchmarkV2Case, field: str) -> BenchmarkV2Case:
    """Return one case with exactly one field changed for identity coverage."""
    changes = {
        "name": replace(case, name="different_name"),
        "model_name": replace(case, model_name="different_model"),
        "n_layer": replace(case, n_layer=case.n_layer + 1),
        "n_head": replace(case, n_head=case.n_head + 1),
        "n_embd": replace(case, n_embd=case.n_embd + 1),
        "torch_num_threads": replace(case, torch_num_threads=case.torch_num_threads + 1),
        "block_size": replace(case, block_size=case.block_size + 1),
        "batch_size": replace(case, batch_size=case.batch_size + 1),
    }
    try:
        return changes[field]
    except KeyError as error:
        msg = f"unsupported case field {field!r}"
        raise ValueError(msg) from error


@pytest.mark.parametrize(
    "field",
    [
        "name",
        "model_name",
        "n_layer",
        "n_head",
        "n_embd",
        "torch_num_threads",
        "block_size",
        "batch_size",
    ],
)
def test_case_identity_changes_for_every_case_field(tmp_path: Path, field: str) -> None:
    """Include every explicit case field in the workload identity."""
    # Given: one parsed explicit case.
    config = load_benchmark_v2_config(write_v2_config(tmp_path))
    case = config.cases[0]

    # When: exactly one case field changes.
    changed = changed_case(case, field)

    # Then: the workload identity changes.
    assert case_identity(config, case) != case_identity(config, changed)


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        ("schema_version: 2 => schema_version: 1", "schema_version"),
        ("benchmark_seed: 1337 => benchmark_seed: 1337\nunexpected: true", "unexpected key"),
        ("name: tiny_t2_s32_b2 => name: tiny_t1_s32_b2", "duplicate case name"),
        ("model_name: tiny => model_name: unknown", "unknown model"),
        ("warmup_steps: 2 => warmup_steps: -1", "warmup_steps"),
        ("measurement_steps: 5 => measurement_steps: 0", "measurement_steps"),
        ("replicates: 3 => replicates: 0", "replicates"),
        ("worker_timeout_seconds: 60.0 => worker_timeout_seconds: 0", "worker_timeout_seconds"),
        ("n_head: 2 => n_head: 3", "divisible"),
        ("cpu_affinity: [0, 1] => cpu_affinity: []", "cpu_affinity"),
        ("cpu_affinity: [0, 1] => cpu_affinity: [-1]", "cpu_affinity"),
        ("cpu_affinity: [0, 1] => cpu_affinity: [0, 0]", "cpu_affinity"),
        ("minimum_replicates: 2 => minimum_replicates: 4", "minimum_replicates"),
    ],
)
def test_v2_config_rejects_invalid_schema_values(
    tmp_path: Path, replacement: str, reason: str
) -> None:
    """Reject every invalid v2 schema value before a benchmark can start."""
    # Given: a valid config with one invalid schema value.
    path = write_v2_config(tmp_path)
    original, updated = replacement.split(" => ")
    _ = path.write_text(
        path.read_text(encoding="utf-8").replace(original, updated), encoding="utf-8"
    )

    # When/Then: loading rejects the invalid value with a useful reason.
    with pytest.raises(InvalidBenchmarkV2ConfigError, match=reason):
        _ = load_benchmark_v2_config(path)
