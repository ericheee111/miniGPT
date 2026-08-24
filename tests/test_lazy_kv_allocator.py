from __future__ import annotations

import random

import pytest
import torch

from minigpt.layers import KVCache, LayerKVCache
from minigpt.paged_kv_cache import (
    PagedKVCacheCapacityError,
    PagedKVCacheConfig,
    PagedKVCachePool,
    PrefixCacheNamespace,
)


def _pool(*, blocks: int = 8, prefix_cache: bool = False) -> PagedKVCachePool:
    namespace = (
        PrefixCacheNamespace(
            model_checkpoint_identity="stage18-allocator",
            model_config_identity="stage18-config",
            dtype="torch.float32",
            device="cpu",
            block_tokens=2,
            cache_schema_version=1,
            position_embedding_semantics="learned_absolute_v1",
        )
        if prefix_cache
        else None
    )
    return PagedKVCachePool(
        PagedKVCacheConfig(block_tokens=2, num_blocks=blocks),
        n_layer=2,
        n_head=2,
        head_size=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
        prefix_cache_namespace=namespace,
    )


def _cache(length: int, *, offset: float = 0.0) -> KVCache:
    layers: list[LayerKVCache] = []
    for layer_index in range(2):
        values = torch.arange(2 * length * 3, dtype=torch.float32).reshape(1, 2, length, 3)
        key = values + offset + layer_index * 100.0
        layers.append(LayerKVCache(key=key, value=key + 0.5))
    return tuple(layers)


def _logits(length: int) -> torch.Tensor:
    return torch.arange(length * 11, dtype=torch.float32).reshape(length, 11)


def test_legacy_reservation_keeps_current_equal_to_lifetime_maximum() -> None:
    pool = _pool(blocks=4)

    pool.reserve("legacy", 3)

    table = pool.request_cache("legacy")
    assert table.reserved_blocks == table.max_blocks == 3
    assert pool.current_total_reserved_blocks("legacy") == 3
    assert pool.metrics().reservation_growth_count == 0
    pool.verify_invariants()


def test_lazy_reservation_grows_transactionally_without_allocating_cache() -> None:
    pool = _pool(blocks=4)
    pool.reserve("lazy", 1, max_blocks=4)
    before = pool.metrics()

    assert pool.can_grow_reservation("lazy", 3) is True
    grown = pool.grow_reservation("lazy", 3)

    table = pool.request_cache("lazy")
    after = pool.metrics()
    assert grown == 2
    assert table.reserved_blocks == 3
    assert table.max_blocks == 4
    assert table.block_ids == ()
    assert after.allocated_blocks == before.allocated_blocks == 0
    assert after.reservation_growth_count == 1
    assert after.reservation_growth_blocks == 2
    assert pool.grow_reservation("lazy", 2) == 0
    assert pool.metrics().reservation_growth_count == 1
    pool.verify_invariants()


def test_growth_capacity_failure_is_non_mutating() -> None:
    pool = _pool(blocks=4)
    pool.reserve("first", 2, max_blocks=4)
    pool.reserve("second", 2, max_blocks=4)
    before_first = pool.request_cache("first")
    before_metrics = pool.metrics()

    assert pool.can_grow_reservation("first", 3) is False
    with pytest.raises(PagedKVCacheCapacityError, match="growth requires"):
        _ = pool.grow_reservation("first", 3)

    assert pool.request_cache("first") == before_first
    assert pool.metrics() == before_metrics
    pool.verify_invariants()


def test_growth_cannot_exceed_lifetime_maximum() -> None:
    pool = _pool(blocks=4)
    pool.reserve("request", 1, max_blocks=2)

    assert pool.can_grow_reservation("request", 3) is False
    with pytest.raises(PagedKVCacheCapacityError, match="exceeds max 2"):
        _ = pool.grow_reservation("request", 3)

    assert pool.current_total_reserved_blocks("request") == 1
    pool.verify_invariants()


def test_apc_promotion_preserves_current_total_and_allows_later_growth() -> None:
    pool = _pool(blocks=5, prefix_cache=True)
    prompt = (1, 2, 3, 4)
    lookup = pool.reserve_with_prefix(
        "request",
        reserved_blocks=2,
        max_blocks=4,
        prompt_tokens=prompt,
    )
    pool.write_prefill_suffix("request", _cache(4))

    _ = pool.promote_prompt_blocks(
        "request",
        prompt_tokens=prompt,
        prefix_hit_tokens=0,
        suffix_logits=_logits(4),
    )

    promoted = pool.request_cache("request")
    assert lookup.prefix_hit_blocks == 0
    assert promoted.shared_blocks == 2
    assert promoted.reserved_blocks == 0
    assert promoted.max_blocks == 4
    assert pool.current_total_reserved_blocks("request") == 2
    assert pool.grow_reservation("request", 3) == 1
    grown = pool.request_cache("request")
    assert grown.reserved_blocks == 1
    assert pool.current_total_reserved_blocks("request") == 3
    pool.verify_invariants()


def test_apc_overflow_rebuild_requires_current_protection_not_lifetime_maximum() -> None:
    pool = _pool(blocks=5, prefix_cache=True)
    prompt = (1, 2, 3, 4)
    _ = pool.reserve_with_prefix(
        "request",
        reserved_blocks=2,
        max_blocks=3,
        prompt_tokens=prompt,
    )
    pool.write_prefill_suffix("request", _cache(4))
    _ = pool.promote_prompt_blocks(
        "request",
        prompt_tokens=prompt,
        prefix_hit_tokens=0,
        suffix_logits=_logits(4),
    )
    before = pool.request_cache("request")

    with pytest.raises(PagedKVCacheCapacityError, match="current protection is 2"):
        pool.rebuild("request", _cache(6, offset=100.0))

    assert pool.request_cache("request") == before
    assert pool.grow_reservation("request", 3) == 1
    pool.rebuild("request", _cache(6, offset=100.0))
    rebuilt = pool.request_cache("request")
    assert rebuilt.cache_length == 6
    assert rebuilt.shared_blocks == 0
    assert rebuilt.reserved_blocks == 3
    assert rebuilt.max_blocks == 3
    pool.verify_invariants()


def test_deterministic_lazy_growth_stress_preserves_allocator_invariants() -> None:
    pool = _pool(blocks=24)
    generator = random.Random(20260824)  # noqa: S311 - deterministic allocator stress
    active: dict[str, int] = {}
    next_request = 0

    for mutation in range(2_000):
        action = generator.randrange(4)
        if (action == 0 or not active) and len(active) < 8:
            current = generator.randint(1, 2)
            lifetime = generator.randint(current, 4)
            if pool.can_reserve(current):
                request_id = f"lazy-{next_request}"
                next_request += 1
                pool.reserve(request_id, current, max_blocks=lifetime)
                initial_length = generator.randint(1, current * pool.config.block_tokens)
                pool.write_prefill(request_id, _cache(initial_length, offset=float(mutation)))
                active[request_id] = lifetime
        elif active:
            request_id = generator.choice(tuple(active))
            table = pool.request_cache(request_id)
            current = pool.current_total_reserved_blocks(request_id)
            if action == 1 and current < active[request_id]:
                target = generator.randint(current + 1, active[request_id])
                if pool.can_grow_reservation(request_id, target):
                    _ = pool.grow_reservation(request_id, target)
            elif action == 2 and table.cache_length < current * pool.config.block_tokens:
                pool.append(request_id, _cache(table.cache_length + 1, offset=float(mutation)))
            else:
                pool.release(request_id)
                del active[request_id]
        pool.verify_invariants()

    for request_id in tuple(active):
        pool.release(request_id)
    metrics = pool.metrics()
    assert metrics.reservation_growth_count > 0
    assert metrics.reservation_growth_blocks > 0
    assert metrics.reserved_blocks == 0
    assert metrics.allocated_blocks == 0
    assert metrics.free_blocks == metrics.total_blocks
    pool.verify_invariants()
