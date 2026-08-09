from __future__ import annotations

import pytest

from minigpt.prefix_cache_benchmark import (
    PrefixCacheBenchmarkConfig,
    run_prefix_cache_worker,
)


def test_benchmark_config_requires_three_fresh_samples() -> None:
    # Given/When/Then: strict median/MAD/CV evidence rejects undersampling.
    with pytest.raises(ValueError, match="at least three"):
        _ = PrefixCacheBenchmarkConfig(repeats=2)


def test_exact_repeated_worker_preserves_results_and_skips_second_prefill() -> None:
    # Given/When: identical direct/APC workers run the exact repeated prompt workload.
    direct = run_prefix_cache_worker("exact_repeated_prompt", "paged_direct")
    apc = run_prefix_cache_worker("exact_repeated_prompt", "paged_direct_apc")

    # Then: logical output/RNG/events match and APC avoids the repeated full prompt.
    for key in ("generated_tokens", "generator_state_hashes", "terminal_states", "logical_events"):
        assert apc[key] == direct[key]
    assert direct["avoided_prefill_tokens"] == 0
    assert apc["avoided_prefill_tokens"] == 4
    assert apc["reused_blocks"] == 2
    assert direct["all_resources_released"] is True
    assert apc["all_resources_released"] is True


def test_eviction_pressure_worker_evicts_only_idle_prefix_blocks() -> None:
    # Given/When: unique prompts exceed the tiny APC resident pool sequentially.
    result = run_prefix_cache_worker("eviction_pressure", "paged_direct_apc")

    # Then: deterministic zero-ref eviction occurs and final cleanup remains leak-free.
    assert isinstance(result["evictions"], int)
    assert result["evictions"] > 0
    assert result["all_resources_released"] is True
