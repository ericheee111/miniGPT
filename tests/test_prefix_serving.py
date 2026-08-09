from __future__ import annotations

from typing import TYPE_CHECKING, Never, final

import torch

from minigpt.engine_runner import EngineRunner, RunnerConfig, RunnerEventType
from minigpt.model import GPT
from minigpt.paged_kv_cache import (
    KVCacheBackend,
    PagedKVCacheConfig,
    PagedKVCachePool,
    PrefixCacheNamespace,
)
from minigpt.serving import (
    APCPrefillStrategy,
    EngineConfig,
    EngineEventType,
    GenerationRequest,
    PagedAttentionExecutor,
    PrefillBatchConfig,
    RequestStatus,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest

    from minigpt.layers import KVCache, PagedKVCacheView
    from minigpt.paged_kv_cache import PrefixCachePromotion


@final
class StepClock:
    def __init__(self, step: float) -> None:
        self._step = step
        self._current = 0.0

    def __call__(self) -> float:
        current = self._current
        self._current += self._step
        return current


def _model() -> GPT:
    original = torch.get_rng_state()
    try:
        _ = torch.default_generator.manual_seed(20260809)
        value = GPT(
            GPTConfig(
                vocab_size=23,
                block_size=8,
                n_layer=2,
                n_head=2,
                n_embd=8,
                dropout=0.0,
                bias=False,
            )
        ).eval()
    finally:
        torch.set_rng_state(original)
    return value


def _namespace(model: GPT) -> PrefixCacheNamespace:
    return PrefixCacheNamespace(
        model_checkpoint_identity="test-checkpoint-sha256",
        model_config_identity="test-gpt-config-sha256",
        dtype=str(model.token_embedding.weight.dtype),
        device=str(model.token_embedding.weight.device),
        block_tokens=2,
        cache_schema_version=1,
        position_embedding_semantics="learned_absolute_v1",
    )


def _engine(
    model: GPT,
    *,
    prefix_cache: bool,
    blocks: int = 16,
    prefix_prefill_strategy: APCPrefillStrategy = APCPrefillStrategy.BATCHED,
    prefill_config: PrefillBatchConfig | None = None,
) -> ServingEngine:
    paged = PagedKVCacheConfig(block_tokens=2, num_blocks=blocks)
    pool = PagedKVCachePool.from_model(
        paged,
        model,
        prefix_cache_namespace=_namespace(model) if prefix_cache else None,
    )
    executor = PagedAttentionExecutor(
        model,
        pool,
        prefill_config=prefill_config,
        prefix_prefill_strategy=prefix_prefill_strategy,
        clock=StepClock(0.001),
        telemetry_clock=StepClock(0.0001),
    )
    return ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(max_active_requests=4, max_cached_tokens=32),
            block_size=model.config.block_size,
            kv_cache_backend=KVCacheBackend.PAGED,
            paged_kv_cache=paged,
        ),
        executor=executor,
        paged_cache_pool=pool,
        clock=lambda: 0.0,
    )


def _run(engine: ServingEngine, *, start: int = 0) -> None:
    for tick in range(start, start + 100):
        if engine.is_idle:
            return
        engine.tick(now=tick * 0.01)
    assert engine.is_idle


def _request(
    request_id: str,
    prompt: tuple[int, ...],
    *,
    seed: int = 17,
    max_new_tokens: int = 4,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_tokens=prompt,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )


def test_exact_repeated_prompt_skips_all_prefill_and_preserves_tokens_and_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: identical direct-paged engines with APC disabled/enabled.
    model = _model()
    reference = _engine(model, prefix_cache=False)
    cached = _engine(model, prefix_cache=True)
    prompt = (1, 2, 3, 4)
    reference.submit(_request("reference", prompt))
    _run(reference)
    cached.submit(_request("prime", prompt, seed=3, max_new_tokens=2))
    _run(cached)

    def forbid_full_prefix_recompute(_token_ids: torch.Tensor) -> Never:
        reason = "exact prefix hit reran dense full prefill"
        raise AssertionError(reason)

    monkeypatch.setattr(model, "prefill_with_all_logits", forbid_full_prefix_recompute)
    monkeypatch.setattr(model, "prefill_paged_batch", forbid_full_prefix_recompute)

    # When: the same prompt and seed run after its full blocks are resident.
    cached.submit(_request("hit", prompt))
    _run(cached, start=100)

    # Then: no prompt token is recomputed and sampling is bit-for-bit equivalent.
    expected = reference.request_state("reference")
    actual = cached.request_state("hit")
    assert actual.prefix_hit_blocks == 2
    assert actual.prefix_hit_tokens == len(prompt)
    assert actual.prefill_tokens_computed == 0
    assert actual.generated_tokens == expected.generated_tokens
    assert torch.equal(actual.generator.get_state(), expected.generator.get_state())
    metrics = cached.metrics()
    assert metrics.avoided_prefill_tokens >= len(prompt)
    assert metrics.prefix_hit_requests == 1
    assert any(event.event_type is EngineEventType.PREFIX_HIT for event in cached.events)


def test_four_same_length_apc_suffixes_use_one_batched_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: four admitted requests reuse the same canonical four-token prefix.
    model = _model()
    engine = _engine(model, prefix_cache=True)
    engine.submit(_request("prime", (1, 2, 3, 4), max_new_tokens=1))
    _run(engine)
    model_calls = 0
    original = model.prefill_paged_batch

    def count_model_call(
        token_ids: torch.Tensor,
        new_token_lengths: torch.Tensor,
        cache_views: Sequence[PagedKVCacheView | None],
        past_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, KVCache]:
        nonlocal model_calls
        model_calls += 1
        return original(token_ids, new_token_lengths, cache_views, past_lengths)

    monkeypatch.setattr(model, "prefill_paged_batch", count_model_call)

    def reject_materialize(_request_id: str) -> Never:
        reason = "batched APC suffix prefill materialized historical K/V"
        raise AssertionError(reason)

    pool = engine.paged_cache_pool
    assert pool is not None
    monkeypatch.setattr(pool, "materialize", reject_materialize)
    for index, suffix in enumerate((5, 6, 7, 8)):
        engine.submit(
            _request(
                f"batch-{index}",
                (1, 2, 3, 4, suffix),
                seed=index + 1,
                max_new_tokens=1,
            )
        )

    # When: one tick admits, then the next prefill step evaluates all four suffix rows.
    engine.tick(now=1.0)
    engine.tick(now=1.01)

    # Then: the Stage 15 path performs one real batched model call, not four wrapped calls.
    assert model_calls == 1
    observation = engine.prefill_observations[-1]
    assert observation.batch_size == 4
    assert observation.useful_prompt_tokens == 4
    assert observation.padded_prompt_tokens == 4
    assert observation.model_calls == 1
    assert observation.execution_mode.value == "batched_apc_suffix"
    pool.verify_invariants()


def test_batched_apc_matches_sequential_for_mixed_hits_lengths_and_rng() -> None:
    # Given: equivalent sequential and batched APC engines with one resident prefix.
    model = _model()
    sequential = _engine(
        model,
        prefix_cache=True,
        prefix_prefill_strategy=APCPrefillStrategy.SEQUENTIAL,
        prefill_config=PrefillBatchConfig(max_padding_ratio=1.0),
    )
    batched = _engine(
        model,
        prefix_cache=True,
        prefix_prefill_strategy=APCPrefillStrategy.BATCHED,
        prefill_config=PrefillBatchConfig(max_padding_ratio=1.0),
    )
    for engine in (sequential, batched):
        engine.submit(_request("prime", (1, 2, 3, 4), seed=5, max_new_tokens=1))
        _run(engine)
    requests = (
        _request("exact", (1, 2, 3, 4), seed=11),
        _request("short", (1, 2, 3, 4, 5), seed=13),
        _request("long", (1, 2, 3, 4, 6, 7), seed=17),
        _request("miss", (9, 10, 11), seed=19),
    )

    # When: both strategies run the same mixed FIFO admission set to completion.
    for request in requests:
        sequential.submit(request)
        batched.submit(request)
    _run(sequential, start=100)
    _run(batched, start=100)

    # Then: batch composition cannot couple per-request tokens, RNG, or terminal state.
    for request in requests:
        expected = sequential.request_state(request.request_id)
        actual = batched.request_state(request.request_id)
        assert actual.status is expected.status
        assert actual.generated_tokens == expected.generated_tokens
        assert torch.equal(actual.generator.get_state(), expected.generator.get_state())
        assert actual.prefix_hit_tokens == expected.prefix_hit_tokens
    observations = batched.prefill_observations
    exact = next(item for item in observations if item.request_ids == ("exact",))
    suffix_batch = next(item for item in observations if "short" in item.request_ids)
    assert exact.model_calls == 0
    assert exact.execution_mode.value == "exact_cache_hit"
    assert suffix_batch.request_ids == ("short", "long", "miss")
    assert suffix_batch.model_calls == 1
    assert suffix_batch.useful_prompt_tokens == 6
    assert suffix_batch.padded_prompt_tokens == 9
    for engine in (sequential, batched):
        assert engine.paged_cache_pool is not None
        engine.paged_cache_pool.verify_invariants()


def test_batched_apc_model_failure_fails_whole_batch_without_pool_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: two prefix-hit suffix rows and a failure at the batched model boundary.
    model = _model()
    engine = _engine(model, prefix_cache=True)
    engine.submit(_request("prime", (1, 2, 3, 4), max_new_tokens=1))
    _run(engine)
    pool = engine.paged_cache_pool
    assert pool is not None

    def fail_batch(
        _token_ids: torch.Tensor,
        _new_token_lengths: torch.Tensor,
        _cache_views: Sequence[PagedKVCacheView | None],
        _past_lengths: torch.Tensor,
    ) -> Never:
        reason = "injected cache-aware batch failure"
        raise RuntimeError(reason)

    monkeypatch.setattr(model, "prefill_paged_batch", fail_batch)
    for request_id, suffix in (("first", 5), ("second", 6)):
        engine.submit(_request(request_id, (1, 2, 3, 4, suffix), max_new_tokens=1))

    # When: admission completes and the shared model call fails before any suffix scatter.
    engine.tick(now=1.0)
    engine.tick(now=1.01)

    # Then: both rows fail and request refs/reservations/private ownership are released.
    for request_id in ("first", "second"):
        state = engine.request_state(request_id)
        assert state.status is RequestStatus.FAILED
        assert state.failure_reason == "RuntimeError: injected cache-aware batch failure"
    observation = engine.prefill_observations[-1]
    assert observation.batch_failed
    assert observation.model_calls == 1
    metrics = pool.metrics()
    assert metrics.active_shared_references == 0
    assert metrics.private_blocks == 0
    assert metrics.reserved_blocks == 0
    assert metrics.allocated_blocks == metrics.prefix_cache_blocks
    pool.verify_invariants()


def test_concurrent_batched_misses_canonicalize_duplicate_prompt_blocks() -> None:
    # Given: two concurrent misses with the same complete four-token prompt.
    engine = _engine(_model(), prefix_cache=True)
    for request_id, seed in (("first", 23), ("second", 29)):
        engine.submit(_request(request_id, (1, 2, 3, 4), seed=seed, max_new_tokens=1))

    # When: both rows prefill together and owner-thread promotion runs in FIFO order.
    engine.tick(now=0.0)
    engine.tick(now=0.01)

    # Then: one canonical chain survives; the duplicate private chain is released.
    assert all(
        engine.request_state(request_id).status is RequestStatus.FINISHED
        for request_id in ("first", "second")
    )
    observation = engine.prefill_observations[-1]
    assert observation.request_ids == ("first", "second")
    assert observation.model_calls == 1
    pool = engine.paged_cache_pool
    assert pool is not None
    metrics = pool.metrics()
    assert metrics.prefix_cache_blocks == 2
    assert metrics.allocated_blocks == 2
    assert metrics.active_shared_references == 0
    assert metrics.private_blocks == 0
    assert metrics.reserved_blocks == 0
    pool.verify_invariants()


def test_batched_apc_isolates_one_row_sampling_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: two prefix-hit rows and a sampler that fails only its first row.
    model = _model()
    engine = _engine(model, prefix_cache=True)
    engine.submit(_request("prime", (1, 2, 3, 4), max_new_tokens=1))
    _run(engine)
    original = torch.multinomial
    calls = 0

    def fail_first_sample(
        probabilities: torch.Tensor,
        num_samples: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 1:
            reason = "injected first-row sample failure"
            raise RuntimeError(reason)
        return original(probabilities, num_samples, generator=generator)

    monkeypatch.setattr(torch, "multinomial", fail_first_sample)
    engine.submit(_request("bad", (1, 2, 3, 4, 5), max_new_tokens=1))
    engine.submit(_request("good", (1, 2, 3, 4, 6), max_new_tokens=1))

    # When: both suffixes share one successful model call and scatter samples per row.
    engine.tick(now=1.0)
    engine.tick(now=1.01)

    # Then: only the injected row fails; its peer and all pool ownership remain valid.
    assert engine.request_state("bad").status is RequestStatus.FAILED
    assert engine.request_state("good").status is RequestStatus.FINISHED
    assert engine.prefill_observations[-1].model_calls == 1
    pool = engine.paged_cache_pool
    assert pool is not None
    metrics = pool.metrics()
    assert metrics.active_shared_references == 0
    assert metrics.private_blocks == 0
    assert metrics.reserved_blocks == 0
    pool.verify_invariants()


def test_batched_apc_promotion_failure_isolates_request_and_rolls_back_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: two concurrent batched misses and an injected first-request promotion failure.
    engine = _engine(_model(), prefix_cache=True)
    pool = engine.paged_cache_pool
    assert pool is not None
    original = pool.promote_prompt_blocks

    def fail_first_promotion(
        request_id: str,
        *,
        prompt_tokens: tuple[int, ...],
        prefix_hit_tokens: int,
        suffix_logits: torch.Tensor,
    ) -> PrefixCachePromotion:
        if request_id == "bad":
            reason = "injected Stage 15 promotion failure"
            raise RuntimeError(reason)
        return original(
            request_id,
            prompt_tokens=prompt_tokens,
            prefix_hit_tokens=prefix_hit_tokens,
            suffix_logits=suffix_logits,
        )

    monkeypatch.setattr(pool, "promote_prompt_blocks", fail_first_promotion)
    engine.submit(_request("bad", (1, 2, 3, 4), max_new_tokens=1))
    engine.submit(_request("good", (5, 6, 7, 8), max_new_tokens=1))

    # When: prefill succeeds for both rows but owner-thread promotion fails for only the first.
    engine.tick(now=0.0)
    engine.tick(now=0.01)

    # Then: the failed request is released and the peer's canonical blocks stay valid.
    assert engine.request_state("bad").status is RequestStatus.FAILED
    assert engine.request_state("good").status is RequestStatus.FINISHED
    metrics = pool.metrics()
    assert metrics.active_shared_references == 0
    assert metrics.private_blocks == 0
    assert metrics.reserved_blocks == 0
    assert metrics.allocated_blocks == metrics.prefix_cache_blocks == 2
    pool.verify_invariants()


def test_common_prefix_concurrent_requests_share_ids_and_release_refs_independently() -> None:
    # Given: a cached four-token prefix.
    engine = _engine(_model(), prefix_cache=True)
    engine.submit(_request("prime", (1, 2, 3, 4), max_new_tokens=1))
    _run(engine)

    # When: three active requests attach the same prefix with different private tails.
    requests = (
        _request("x", (1, 2, 3, 4, 5), seed=1),
        _request("y", (1, 2, 3, 4, 6), seed=2),
        _request("z", (1, 2, 3, 4, 7), seed=3),
    )
    for request in requests:
        engine.submit(request)
    engine.tick(now=1.0)
    pool = engine.paged_cache_pool
    assert pool is not None
    shared_ids = tuple(pool.request_cache(request.request_id).block_ids for request in requests)
    assert shared_ids[0] == shared_ids[1] == shared_ids[2]
    assert pool.metrics().active_shared_references == 6

    # When/Then: cancelling one decrefs without affecting the two peers.
    engine.cancel("x", at=1.01)
    engine.tick(now=1.01)
    assert engine.request_state("x").status is RequestStatus.CANCELLED
    assert pool.metrics().active_shared_references == 4
    _run(engine, start=102)
    assert engine.request_state("y").status is RequestStatus.FINISHED
    assert engine.request_state("z").status is RequestStatus.FINISHED
    assert pool.metrics().active_shared_references == 0
    pool.verify_invariants()


def test_partial_hit_and_learned_position_overflow_match_uncached_direct_backend() -> None:
    # Given: APC and uncached direct engines with identical model weights.
    model = _model()
    reference = _engine(model, prefix_cache=False)
    cached = _engine(model, prefix_cache=True)
    cached.submit(_request("prime", (1, 2, 3, 4), seed=5, max_new_tokens=1))
    _run(cached)
    request = _request("work", (1, 2, 3, 4, 5), seed=29, max_new_tokens=6)
    reference.submit(request)
    cached.submit(request)

    # When: a private partial tail is prefetched and decode crosses learned-position capacity.
    _run(reference)
    _run(cached, start=100)

    # Then: only full blocks hit and dense overflow rebuild preserves logical results.
    expected = reference.request_state("work")
    actual = cached.request_state("work")
    assert actual.prefix_hit_tokens == 4
    assert actual.prefix_miss_tokens == 1
    assert actual.prefill_tokens_computed == 1
    assert actual.generated_tokens == expected.generated_tokens
    assert torch.equal(actual.generator.get_state(), expected.generator.get_state())
    assert any(event.used_fallback for event in cached.events if event.request_id == "work")
    assert cached.paged_cache_pool is not None
    cached.paged_cache_pool.verify_invariants()


def test_prefix_hit_stream_backpressure_and_shutdown_leave_no_refs_or_reservations() -> None:
    # Given: a canonical prompt block and a runner with an intentionally tiny unconsumed stream.
    engine = _engine(_model(), prefix_cache=True)
    pool = engine.paged_cache_pool
    assert pool is not None
    runner = EngineRunner(
        engine=engine,
        config=RunnerConfig(command_queue_size=8, stream_buffer_size=1),
    )
    runner.start()
    try:
        prime = runner.submit(
            _request("prime", (1, 2, 3, 4), seed=5, max_new_tokens=1),
            stream=False,
        )
        _ = prime.future.result(timeout=5.0)

        # When: an exact-hit request emits without any consumer draining its token queue.
        handle = runner.submit(
            _request("backpressured", (1, 2, 3, 4), seed=7, max_new_tokens=8),
            stream=True,
        )
        result = handle.future.result(timeout=5.0)
        metrics = runner.metrics()

        # Then: backpressure cancels the request and releases its shared/private capacity.
        assert result.status is RequestStatus.CANCELLED
        assert any(event.event_type is RunnerEventType.BACKPRESSURE for event in runner.events)
        assert metrics.active_shared_references == 0
        assert metrics.allocated_blocks == metrics.prefix_cache_blocks
        assert metrics.reserved_blocks == 0
    finally:
        runner.shutdown()

    # Then: shutdown also clears the optional zero-ref resident prefix cache.
    final = pool.metrics()
    assert final.active_shared_references == 0
    assert final.allocated_blocks == 0
    assert final.reserved_blocks == 0
    assert final.free_blocks == final.total_blocks
    pool.verify_invariants()
