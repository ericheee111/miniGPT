from __future__ import annotations

from typing import cast

import pytest

from minigpt.cache_aware_prefill_benchmark import (
    CacheAwarePrefillBenchmarkConfig,
    run_cache_aware_prefill_worker,
)


def test_cache_aware_benchmark_requires_three_fresh_samples() -> None:
    # Given/When/Then: strict median/MAD/CV evidence rejects undersampling.
    with pytest.raises(ValueError, match="at least three"):
        _ = CacheAwarePrefillBenchmarkConfig(repeats=2)


def test_repeated_prefix_worker_preserves_logic_and_reduces_model_calls() -> None:
    # Given/When: Stage 14 sequential and Stage 15 batched APC run one isolated workload.
    sequential = run_cache_aware_prefill_worker(
        "repeated_prefix_short_suffix",
        "apc_sequential",
    )
    batched = run_cache_aware_prefill_worker(
        "repeated_prefix_short_suffix",
        "apc_batched",
    )

    # Then: tokens/RNG/events match while four suffix rows collapse into one model call.
    for key in ("generated_tokens", "generator_state_hashes", "terminal_states", "logical_events"):
        assert batched[key] == sequential[key]
    assert sequential["cache_aware_prefill_model_calls"] == 5
    assert batched["cache_aware_prefill_model_calls"] == 2
    assert batched["suffix_prefill_batch_sizes"] == [1, 4]
    assert batched["max_suffix_prefill_batch_size"] == 4
    assert sequential["avoided_prefill_tokens"] == batched["avoided_prefill_tokens"] == 32
    assert sequential["all_resources_released"] is True
    assert batched["all_resources_released"] is True


def test_padding_pressure_worker_obeys_suffix_padding_policy() -> None:
    # Given/When: variable suffixes run under a strict twenty-five-percent padding limit.
    batched = run_cache_aware_prefill_worker("padding_pressure", "apc_batched")

    # Then: grouping is based on suffix compute and never exceeds the configured waste ratio.
    assert isinstance(batched["suffix_padding_waste_ratio"], float)
    assert batched["suffix_padding_waste_ratio"] <= 0.25
    assert cast("int", batched["suffix_useful_tokens"]) < cast(
        "int", batched["suffix_padded_tokens"]
    )
    assert batched["all_resources_released"] is True
