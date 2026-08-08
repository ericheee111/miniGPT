from __future__ import annotations

import random
from typing import TYPE_CHECKING, cast

import pytest
import torch

from minigpt.layers import KVCache, LayerKVCache
from minigpt.paged_kv_cache import (
    PagedKVCacheCapacityError,
    PagedKVCacheConfig,
    PagedKVCachePool,
    PhysicalBlockState,
    PrefixCacheNamespace,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _namespace(identity: str = "checkpoint-a") -> PrefixCacheNamespace:
    return PrefixCacheNamespace(
        model_checkpoint_identity=identity,
        model_config_identity="gpt-config-a",
        dtype="torch.float32",
        device="cpu",
        block_tokens=2,
        cache_schema_version=1,
        position_embedding_semantics="learned_absolute_v1",
    )


def _pool(*, blocks: int = 8, identity: str = "checkpoint-a") -> PagedKVCachePool:
    return PagedKVCachePool(
        PagedKVCacheConfig(block_tokens=2, num_blocks=blocks),
        n_layer=2,
        n_head=2,
        head_size=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
        prefix_cache_namespace=_namespace(identity),
    )


def _cache(length: int, *, offset: float = 0.0) -> KVCache:
    layers: list[LayerKVCache] = []
    for layer in range(2):
        values = torch.arange(2 * length * 3, dtype=torch.float32).reshape(1, 2, length, 3)
        key = values + offset + layer * 100.0
        layers.append(LayerKVCache(key=key, value=key + 0.5))
    return tuple(layers)


def _logits(length: int) -> torch.Tensor:
    return torch.arange(length * 11, dtype=torch.float32).reshape(length, 11)


def _populate(pool: PagedKVCachePool, request_id: str, tokens: tuple[int, ...]) -> None:
    lookup = pool.reserve_with_prefix(
        request_id,
        reserved_blocks=pool.required_blocks(len(tokens)),
        prompt_tokens=tokens,
    )
    suffix = len(tokens) - lookup.prefix_hit_tokens
    pool.write_prefill_suffix(request_id, _cache(suffix) if suffix else ())
    _ = pool.promote_prompt_blocks(
        request_id,
        prompt_tokens=tokens,
        prefix_hit_tokens=lookup.prefix_hit_tokens,
        suffix_logits=_logits(suffix),
    )


def test_hash_chain_binds_namespace_history_and_exact_token_metadata() -> None:
    # Given: two namespaces and token blocks with the same final block.
    first = _pool(identity="checkpoint-a")
    second = _pool(identity="checkpoint-b")

    # When: each pool derives its full-block prefix hash chain.
    chain_a = first.prefix_hash_chain((1, 2, 7, 8))
    chain_b = second.prefix_hash_chain((1, 2, 7, 8))
    different_parent = first.prefix_hash_chain((3, 4, 7, 8))

    # Then: namespace and all historical tokens participate in block identity.
    assert chain_a != chain_b
    assert chain_a[1] != different_parent[1]
    assert first.prefix_block_tokens(chain_a[0]) is None


def test_exact_and_partial_hits_share_only_complete_immutable_blocks() -> None:
    # Given: one released four-token prompt cached as two full immutable blocks.
    pool = _pool()
    _populate(pool, "first", (1, 2, 3, 4))
    first_ids = pool.request_cache("first").block_ids
    pool.release("first")

    # When: exact and partial-tail prompts are admitted.
    exact = pool.reserve_with_prefix("exact", reserved_blocks=2, prompt_tokens=(1, 2, 3, 4))
    partial = pool.reserve_with_prefix("partial", reserved_blocks=3, prompt_tokens=(1, 2, 3, 4, 5))

    # Then: both reuse the same two physical IDs and the tail remains private after prefill.
    assert exact.prefix_hit_blocks == 2
    assert exact.prefix_hit_tokens == 4
    assert partial.prefix_hit_tokens == 4
    assert pool.request_cache("exact").block_ids == first_ids
    assert pool.request_cache("partial").block_ids == first_ids
    pool.write_prefill_suffix("partial", _cache(1))
    partial_ids = pool.request_cache("partial").block_ids
    assert partial_ids[:2] == first_ids
    assert pool.block_state(partial_ids[-1]) is PhysicalBlockState.PRIVATE
    assert pool.metrics().active_shared_references == 4
    pool.verify_invariants()


def test_duplicate_promotion_uses_one_canonical_block_and_releases_private_copy() -> None:
    # Given: two requests that both miss before either prompt is promoted.
    pool = _pool(blocks=4)
    tokens = (1, 2)
    first = pool.reserve_with_prefix("first", reserved_blocks=1, prompt_tokens=tokens)
    second = pool.reserve_with_prefix("second", reserved_blocks=1, prompt_tokens=tokens)
    assert first.prefix_hit_blocks == second.prefix_hit_blocks == 0
    pool.write_prefill_suffix("first", _cache(2))
    pool.write_prefill_suffix("second", _cache(2, offset=1000.0))
    duplicate_id = pool.request_cache("second").block_ids[0]

    # When: both requests promote the same token prefix in owner-thread order.
    _ = pool.promote_prompt_blocks(
        "first", prompt_tokens=tokens, prefix_hit_tokens=0, suffix_logits=_logits(2)
    )
    promotion = pool.promote_prompt_blocks(
        "second", prompt_tokens=tokens, prefix_hit_tokens=0, suffix_logits=_logits(2)
    )

    # Then: the second table attaches the first canonical block and frees its duplicate.
    canonical = pool.request_cache("first").block_ids[0]
    assert pool.request_cache("second").block_ids == (canonical,)
    assert duplicate_id != canonical
    assert promotion.duplicate_private_blocks_released == 1
    assert pool.block_state(duplicate_id) is PhysicalBlockState.FREE
    assert pool.metrics().prefix_cache_blocks == 1
    pool.verify_invariants()


def test_duplicate_promotion_rolls_back_partial_canonical_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: two private copies admitted before the first two-block prompt is promoted.
    pool = _pool(blocks=6)
    tokens = (1, 2, 3, 4)
    for request_id in ("canonical", "duplicate"):
        _ = pool.reserve_with_prefix(request_id, reserved_blocks=2, prompt_tokens=tokens)
        pool.write_prefill_suffix(request_id, _cache(4))
    _ = pool.promote_prompt_blocks(
        "canonical",
        prompt_tokens=tokens,
        prefix_hit_tokens=0,
        suffix_logits=_logits(4),
    )
    duplicate_before = pool.request_cache("duplicate")
    metrics_before = pool.metrics()
    validator = cast(
        "Callable[[int, str, tuple[int, ...], str], None]",
        pool._validate_canonical,  # pyright: ignore[reportPrivateUsage]
    )
    calls = 0

    def fail_second_validation(
        block_id: int,
        prefix_hash: str,
        token_block: tuple[int, ...],
        fingerprint: str,
    ) -> None:
        nonlocal calls
        validator(block_id, prefix_hash, token_block, fingerprint)
        calls += 1
        if calls == 2:
            reason = "injected second canonical validation failure"
            raise RuntimeError(reason)

    monkeypatch.setattr(pool, "_validate_canonical", fail_second_validation)

    # When: promotion fails after its first duplicate block was tentatively attached.
    with pytest.raises(RuntimeError, match="second canonical"):
        _ = pool.promote_prompt_blocks(
            "duplicate",
            prompt_tokens=tokens,
            prefix_hit_tokens=0,
            suffix_logits=_logits(4),
        )

    # Then: table, refcounts, ownership, free list, and reservation are fully restored.
    assert pool.request_cache("duplicate") == duplicate_before
    assert pool.metrics() == metrics_before
    pool.verify_invariants()


def test_lru_evicts_zero_ref_blocks_but_never_active_shared_blocks() -> None:
    # Given: a two-block pool with one cached zero-ref prefix.
    pool = _pool(blocks=2)
    _populate(pool, "cached", (1, 2))
    cached_id = pool.request_cache("cached").block_ids[0]
    pool.release("cached")
    assert pool.metrics().evictable_blocks == 1

    # When: a private two-block request needs the whole physical pool.
    pool.reserve("private", 2)
    pool.write_prefill("private", _cache(4))

    # Then: the zero-ref block is evicted deterministically before allocation.
    assert pool.block_state(cached_id) is PhysicalBlockState.PRIVATE
    assert pool.metrics().prefix_cache_evictions == 1
    pool.release("private")

    # Given/When: the prefix is active, another two-block reservation cannot overcommit it.
    _populate(pool, "active", (1, 2))
    with pytest.raises(PagedKVCacheCapacityError):
        pool.reserve("blocked", 2)
    assert pool.block_state(pool.request_cache("active").block_ids[0]) is PhysicalBlockState.SHARED
    pool.verify_invariants()


def test_release_and_shutdown_clear_refs_private_ownership_and_reservations() -> None:
    # Given: concurrent exact hits plus a request-private tail.
    pool = _pool()
    _populate(pool, "first", (1, 2, 3))
    pool.release("first")
    _ = pool.reserve_with_prefix("one", reserved_blocks=2, prompt_tokens=(1, 2, 9))
    _ = pool.reserve_with_prefix("two", reserved_blocks=1, prompt_tokens=(1, 2))
    pool.write_prefill_suffix("one", _cache(1))

    # When: catastrophic/graceful owner cleanup releases everything.
    pool.release_all()

    # Then: this implementation also clears zero-ref prefix residency.
    metrics = pool.metrics()
    assert metrics.free_blocks == metrics.total_blocks
    assert metrics.allocated_blocks == 0
    assert metrics.reserved_blocks == 0
    assert metrics.prefix_cache_blocks == 0
    assert metrics.active_shared_references == 0
    pool.verify_invariants()


def test_deterministic_random_prefix_cache_stress_preserves_every_invariant() -> None:
    # Given: a tiny pool and a deterministic high-reuse/low-reuse prompt mixture.
    generator = random.Random(20260809)  # noqa: S311 - deterministic non-security stress input
    pool = _pool(blocks=12)
    active: dict[str, tuple[int, ...]] = {}
    common = (
        (1, 2),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (7, 8, 9, 10),
    )

    # When: thousands of lookups, shares, allocations, promotions, decodes, and releases run.
    for mutation in range(5_000):
        admit = not active or (len(active) < 4 and generator.random() < 0.62)
        if admit:
            request_id = f"stress-{mutation}"
            if generator.random() < 0.72:
                tokens = common[generator.randrange(len(common))]
            else:
                length = generator.randint(1, 6)
                tokens = tuple(generator.randrange(23) for _ in range(length))
            max_blocks = pool.required_blocks(len(tokens) + 1)
            if pool.can_reserve_with_prefix(
                reserved_blocks=max_blocks,
                prompt_tokens=tokens,
            ):
                lookup = pool.reserve_with_prefix(
                    request_id,
                    reserved_blocks=max_blocks,
                    prompt_tokens=tokens,
                )
                suffix = len(tokens) - lookup.prefix_hit_tokens
                pool.write_prefill_suffix(request_id, _cache(suffix) if suffix else ())
                _ = pool.promote_prompt_blocks(
                    request_id,
                    prompt_tokens=tokens,
                    prefix_hit_tokens=lookup.prefix_hit_tokens,
                    suffix_logits=_logits(suffix),
                )
                if generator.random() < 0.35:
                    pool.append_delta(request_id, _cache(1))
                active[request_id] = tokens
        else:
            request_id = generator.choice(tuple(active))
            pool.release(request_id)
            del active[request_id]

        # Then: every individual mutation leaves unique legal ownership/refcount state.
        pool.verify_invariants()

    pool.release_all()
    metrics = pool.metrics()
    assert metrics.prefix_lookup_requests > 1_000
    assert metrics.prefix_hit_requests > 0
    assert metrics.block_reuse_count > 0
    assert metrics.prefix_cache_evictions > 0
    assert metrics.active_shared_references == 0
    assert metrics.reserved_blocks == 0
    assert metrics.free_blocks == metrics.total_blocks
    pool.verify_invariants()
