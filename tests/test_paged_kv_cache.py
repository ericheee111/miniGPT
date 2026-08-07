from __future__ import annotations

import random
from typing import TYPE_CHECKING, cast

import pytest
import torch

from minigpt.layers import KVCache, LayerKVCache
from minigpt.paged_kv_cache import (
    PagedKVCacheCapacityError,
    PagedKVCacheConfig,
    PagedKVCacheOwnershipError,
    PagedKVCachePool,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _pool(*, block_tokens: int = 2, num_blocks: int = 8) -> PagedKVCachePool:
    return PagedKVCachePool(
        PagedKVCacheConfig(block_tokens=block_tokens, num_blocks=num_blocks),
        n_layer=2,
        n_head=2,
        head_size=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _cache(length: int, *, offset: float = 0.0) -> KVCache:
    layers: list[LayerKVCache] = []
    for layer_index in range(2):
        positions = torch.arange(length * 3, dtype=torch.float32).reshape(1, 1, length, 3)
        heads = torch.arange(2, dtype=torch.float32).reshape(1, 2, 1, 1) * 100.0
        values = positions + heads
        key = values + offset + layer_index * 1000.0
        value = key + 500.0
        layers.append(LayerKVCache(key=key, value=value))
    return tuple(layers)


def _assert_cache_equal(left: KVCache, right: KVCache) -> None:
    assert len(left) == len(right)
    for left_layer, right_layer in zip(left, right, strict=True):
        assert torch.equal(left_layer.key, right_layer.key)
        assert torch.equal(left_layer.value, right_layer.value)


def test_pool_layout_prefill_materialize_and_tail_fragmentation() -> None:
    # Given: a two-token block pool and one reservation spanning three blocks.
    pool = _pool(block_tokens=2, num_blocks=5)
    source = _cache(5)
    pool.reserve("request", 3)

    # When: compact per-layer cache is scattered into the request block table.
    pool.write_prefill("request", source)

    # Then: physical layout, logical mapping, materialization, and tail waste are exact.
    assert tuple(pool.key_blocks.shape) == (2, 5, 2, 2, 3)
    assert tuple(pool.value_blocks.shape) == (2, 5, 2, 2, 3)
    table = pool.request_cache("request")
    assert table.block_ids == (0, 1, 2)
    assert table.cache_length == 5
    assert table.reserved_blocks == 3
    _assert_cache_equal(pool.materialize("request"), source)
    metrics = pool.metrics()
    assert metrics.used_token_slots == 5
    assert metrics.allocated_token_slots == 6
    assert metrics.internal_fragmentation_tokens == 1
    assert metrics.internal_fragmentation_ratio == pytest.approx(1 / 6)
    pool.verify_invariants()


def test_append_crosses_block_boundary_and_reuses_non_contiguous_blocks() -> None:
    # Given: interleaved request allocations that make a later table non-contiguous.
    pool = _pool(block_tokens=2, num_blocks=6)
    pool.reserve("first", 2)
    pool.reserve("second", 2)
    pool.write_prefill("first", _cache(2, offset=10.0))
    pool.write_prefill("second", _cache(2, offset=20.0))
    pool.release("first")

    # When: second crosses a block boundary after the lower physical ID is recycled.
    expected = _cache(3, offset=20.0)
    pool.append("second", expected)

    # Then: its physical blocks need not be contiguous and dense semantics remain exact.
    assert pool.request_cache("second").block_ids == (1, 0)
    _assert_cache_equal(pool.materialize("second"), expected)
    assert pool.metrics().block_reuse_count == 1
    pool.release("second")
    metrics = pool.metrics()
    assert metrics.free_blocks == metrics.total_blocks
    assert metrics.reserved_blocks == 0
    assert metrics.allocated_blocks == 0


def test_reservation_prevents_overcommit_and_double_free() -> None:
    # Given: one request has reserved all but one physical block.
    pool = _pool(num_blocks=3)
    pool.reserve("large", 2)
    pool.reserve("small", 1)

    # When/Then: another reservation cannot overcommit the pool.
    with pytest.raises(PagedKVCacheCapacityError, match="0 available"):
        pool.reserve("blocked", 1)
    pool.release("large")
    with pytest.raises(PagedKVCacheOwnershipError, match="no reservation"):
        pool.release("large")
    pool.release("small")
    pool.verify_invariants()


def test_prefill_and_append_failures_roll_back_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a reserved request and injected failures after allocation begins.
    pool = _pool(block_tokens=2, num_blocks=4)
    pool.reserve("request", 2)
    original_write_cache = cast(
        "Callable[[PagedKVCachePool, KVCache, list[int], int], None]",
        vars(PagedKVCachePool)["_write_cache"],
    )

    def fail_write_cache(
        self: PagedKVCachePool,
        cache: KVCache,
        block_ids: list[int],
        cache_length: int,
    ) -> None:
        original_write_cache(self, cache, block_ids, cache_length)
        message = "injected prefill write failure"
        raise RuntimeError(message)

    monkeypatch.setattr(PagedKVCachePool, "_write_cache", fail_write_cache)
    with pytest.raises(RuntimeError, match="prefill write"):
        pool.write_prefill("request", _cache(2))
    assert pool.request_cache("request").block_ids == ()
    assert pool.metrics().allocated_blocks == 0
    pool.verify_invariants()

    monkeypatch.setattr(PagedKVCachePool, "_write_cache", original_write_cache)
    pool.write_prefill("request", _cache(2))

    def fail_write_token(
        self: PagedKVCachePool,
        cache: KVCache,
        *,
        block_id: int,
        offset: int,
        token_index: int,
    ) -> None:
        del self, cache, block_id, offset, token_index
        message = "injected append write failure"
        raise RuntimeError(message)

    monkeypatch.setattr(PagedKVCachePool, "_write_token", fail_write_token)
    with pytest.raises(RuntimeError, match="append write"):
        pool.append("request", _cache(3))
    assert pool.request_cache("request").block_ids == (0,)
    assert pool.request_cache("request").cache_length == 2
    _assert_cache_equal(pool.materialize("request"), _cache(2))
    pool.verify_invariants()


def test_overflow_rebuild_failure_restores_old_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an existing multi-block request and a replacement overflow window.
    pool = _pool(block_tokens=2, num_blocks=5)
    original = _cache(3, offset=10.0)
    pool.reserve("request", 2)
    pool.write_prefill("request", original)
    original_write_cache = cast(
        "Callable[[PagedKVCachePool, KVCache, list[int], int], None]",
        vars(PagedKVCachePool)["_write_cache"],
    )

    def corrupt_then_fail(
        self: PagedKVCachePool,
        cache: KVCache,
        block_ids: list[int],
        cache_length: int,
    ) -> None:
        original_write_cache(self, cache, block_ids, cache_length)
        message = "injected overflow rebuild failure"
        raise RuntimeError(message)

    # When: the rebuild writes replacement data and then fails before commit.
    monkeypatch.setattr(PagedKVCachePool, "_write_cache", corrupt_then_fail)
    with pytest.raises(RuntimeError, match="overflow rebuild"):
        pool.rebuild("request", _cache(4, offset=99.0))

    # Then: block table, logical length, tensors, and ownership return to the old state.
    assert pool.request_cache("request").block_ids == (0, 1)
    assert pool.request_cache("request").cache_length == 3
    _assert_cache_equal(pool.materialize("request"), original)
    pool.verify_invariants()


def test_deterministic_allocator_stress_preserves_every_invariant() -> None:
    # Given: a deterministic randomized allocation/append/rebuild/free workload.
    pool = _pool(block_tokens=4, num_blocks=32)
    random_source = random.Random(20260807)  # noqa: S311
    active: dict[str, int] = {}
    next_request = 0

    # When: thousands of ownership transitions are checked after every operation.
    for step in range(3000):
        action = random_source.randrange(4)
        if (action == 0 or not active) and pool.metrics().reserved_blocks < pool.config.num_blocks:
            available = pool.config.num_blocks - pool.metrics().reserved_blocks
            reserved = random_source.randint(1, min(4, available))
            request_id = f"request-{next_request}"
            next_request += 1
            pool.reserve(request_id, reserved)
            initial_length = random_source.randint(1, reserved * pool.config.block_tokens)
            pool.write_prefill(request_id, _cache(initial_length, offset=float(step)))
            active[request_id] = reserved
        elif active:
            request_id = random_source.choice(tuple(active))
            table = pool.request_cache(request_id)
            capacity = active[request_id] * pool.config.block_tokens
            if action == 1 and table.cache_length < capacity:
                pool.append(request_id, _cache(table.cache_length + 1, offset=float(step)))
            elif action == 2:
                rebuilt_length = random_source.randint(1, capacity)
                pool.rebuild(request_id, _cache(rebuilt_length, offset=float(step)))
            else:
                pool.release(request_id)
                del active[request_id]
        pool.verify_invariants()

    # Then: final cleanup returns the entire pool while preserving reuse evidence.
    for request_id in tuple(active):
        pool.release(request_id)
        pool.verify_invariants()
    metrics = pool.metrics()
    assert metrics.free_blocks == metrics.total_blocks
    assert metrics.allocated_blocks == 0
    assert metrics.reserved_blocks == 0
    assert metrics.allocation_count == metrics.free_count
    assert metrics.block_reuse_count > 0


@pytest.mark.parametrize("invalid", [-1, True])
def test_required_blocks_rejects_invalid_lengths(invalid: int) -> None:
    pool = _pool()
    with pytest.raises(ValueError, match="cache_tokens"):
        _ = pool.required_blocks(invalid)
