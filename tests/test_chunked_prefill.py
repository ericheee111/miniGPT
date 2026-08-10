from __future__ import annotations

from typing import Never, final

import pytest
import torch

from minigpt.model import GPT
from minigpt.paged_kv_cache import (
    KVCacheBackend,
    PagedKVCacheConfig,
    PagedKVCachePool,
    PhysicalBlockState,
    PrefixCacheNamespace,
)
from minigpt.serving import (
    EngineConfig,
    EngineEventType,
    GenerationRequest,
    InvalidServingConfigError,
    PagedAttentionExecutor,
    PrefillBatchConfig,
    RequestStatus,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.settings import GPTConfig


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
        _ = torch.default_generator.manual_seed(20260810)
        model = GPT(
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
    return model


def _namespace(model: GPT) -> PrefixCacheNamespace:
    return PrefixCacheNamespace(
        model_checkpoint_identity="stage16-test-checkpoint",
        model_config_identity="stage16-test-config",
        dtype=str(model.token_embedding.weight.dtype),
        device=str(model.token_embedding.weight.device),
        block_tokens=2,
        cache_schema_version=1,
        position_embedding_semantics="learned_absolute_v1",
    )


def _engine(
    model: GPT,
    *,
    chunked: bool,
    prefix_cache: bool = False,
    prefill_batch_limit: int = 16,
) -> ServingEngine:
    paged = PagedKVCacheConfig(block_tokens=2, num_blocks=16)
    pool = PagedKVCachePool.from_model(
        paged,
        model,
        prefix_cache_namespace=_namespace(model) if prefix_cache else None,
    )
    executor = PagedAttentionExecutor(
        model,
        pool,
        prefill_config=PrefillBatchConfig(
            max_batch_size=4,
            max_batch_tokens=prefill_batch_limit,
            max_padding_ratio=1.0,
        ),
        clock=StepClock(0.001),
        telemetry_clock=StepClock(0.0001),
    )
    scheduler = (
        SchedulerConfig(
            max_active_requests=4,
            max_cached_tokens=32,
            max_scheduled_tokens=8,
            prefill_chunk_tokens=2,
        )
        if chunked
        else SchedulerConfig(max_active_requests=4, max_cached_tokens=32)
    )
    return ServingEngine(
        config=EngineConfig(
            scheduler=scheduler,
            block_size=model.config.block_size,
            kv_cache_backend=KVCacheBackend.PAGED,
            paged_kv_cache=paged,
        ),
        executor=executor,
        paged_cache_pool=pool,
        clock=lambda: 0.0,
    )


def _request(
    request_id: str,
    prompt: tuple[int, ...],
    *,
    seed: int = 17,
    max_new_tokens: int = 3,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_tokens=prompt,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )


def _run(engine: ServingEngine, *, start: int = 0) -> None:
    for tick in range(start, start + 100):
        if engine.is_idle:
            return
        engine.tick(now=tick * 0.01)
    assert engine.is_idle


def test_chunked_prefill_matches_unchunked_tokens_and_rng() -> None:
    # Given: identical paged engines with and without the Stage 16 scheduler.
    model = _model()
    reference = _engine(model, chunked=False)
    chunked = _engine(model, chunked=True)
    prompt = (1, 2, 3, 4, 5, 6)
    reference.submit(_request("reference", prompt))
    chunked.submit(_request("chunked", prompt))

    # When: both requests run to completion with the same request-local seed.
    _run(reference)
    _run(chunked)

    # Then: chunk boundaries do not change sampling or prompt work accounting.
    expected = reference.request_state("reference")
    actual = chunked.request_state("chunked")
    assert actual.generated_tokens == expected.generated_tokens
    assert torch.equal(actual.generator.get_state(), expected.generator.get_state())
    assert actual.prefill_tokens_computed == len(prompt)
    starts = [
        event
        for event in chunked.events
        if event.event_type is EngineEventType.PREFILL_CHUNK_STARTED
    ]
    assert len(starts) == 3
    finishes = [
        event
        for event in chunked.events
        if event.event_type is EngineEventType.PREFILL_CHUNK_FINISHED
    ]
    assert len(finishes) == 3


def test_chunked_prefill_reuses_apc_prefix_and_matches_reference_rng() -> None:
    # Given: APC has a resident four-token prefix and the next request adds a suffix.
    model = _model()
    reference = _engine(model, chunked=True)
    cached = _engine(model, chunked=True, prefix_cache=True)
    cached.submit(_request("prime", (1, 2, 3, 4), seed=3, max_new_tokens=1))
    _run(cached)
    prompt = (1, 2, 3, 4, 5, 6)
    reference.submit(_request("reference", prompt))
    cached.submit(_request("hit", prompt))

    # When: the suffix is evaluated through Stage 16 after the APC admission hit.
    _run(reference)
    _run(cached, start=100)

    # Then: only the two-token suffix is computed and sampling remains equivalent.
    expected = reference.request_state("reference")
    actual = cached.request_state("hit")
    assert actual.prefix_hit_tokens == 4
    assert actual.prefill_tokens_computed == 2
    assert actual.generated_tokens == expected.generated_tokens
    assert torch.equal(actual.generator.get_state(), expected.generator.get_state())
    metrics = cached.metrics()
    assert metrics.chunked_prefill_useful_tokens >= 6
    assert metrics.prefix_hit_tokens >= 4
    pool = cached.paged_cache_pool
    assert pool is not None
    pool.verify_invariants()


def test_partial_final_chunk_keeps_tail_private_until_request_finishes() -> None:
    # Given: a five-token prompt requires two full chunks plus a one-token final chunk.
    engine = _engine(_model(), chunked=True, prefix_cache=True)
    engine.submit(_request("partial", (1, 2, 3, 4, 5), max_new_tokens=2))
    engine.tick(now=0.0)
    engine.tick(now=0.01)
    engine.tick(now=0.02)

    # When: the final one-token prompt chunk is committed and first-token sampling completes.
    engine.tick(now=0.03)

    # Then: complete prompt blocks are shared, while the incomplete tail remains request-private.
    state = engine.request_state("partial")
    assert state.status is RequestStatus.DECODING
    assert state.cached_tokens == 5
    pool = engine.paged_cache_pool
    assert pool is not None
    table = pool.request_cache("partial")
    assert len(table.block_ids) == 3
    assert pool.block_state(table.block_ids[0]) is PhysicalBlockState.SHARED
    assert pool.block_state(table.block_ids[1]) is PhysicalBlockState.SHARED
    assert pool.block_state(table.block_ids[2]) is PhysicalBlockState.PRIVATE
    starts = [
        event.detail
        for event in engine.events
        if event.event_type is EngineEventType.PREFILL_CHUNK_STARTED
    ]
    assert starts == ["start=0;end=2", "start=2;end=4", "start=4;end=5"]

    _run(engine, start=4)
    engine.verify_cache_invariants()


def test_chunked_prefill_interleaves_decode_with_long_prompt() -> None:
    # Given: one short prompt and one long prompt share the same active batch.
    engine = _engine(_model(), chunked=True)
    engine.submit(_request("short", (1, 2), max_new_tokens=3))
    engine.submit(_request("long", (3, 4, 5, 6, 7, 8), max_new_tokens=1))
    engine.tick(now=0.0)
    engine.tick(now=0.01)
    assert engine.request_state("short").status is RequestStatus.DECODING
    assert engine.request_state("long").status is RequestStatus.PREFILLING
    before = len(engine.events)

    # When: the next tick has both decode and remaining prefill work.
    engine.tick(now=0.02)

    # Then: decode is executed before the next long-prompt chunk in that tick.
    events = engine.events[before:]
    short_token = next(
        index
        for index, event in enumerate(events)
        if event.event_type is EngineEventType.TOKEN and event.request_id == "short"
    )
    long_chunk = next(
        index
        for index, event in enumerate(events)
        if event.event_type is EngineEventType.PREFILL_CHUNK_STARTED and event.request_id == "long"
    )
    assert short_token < long_chunk
    assert engine.request_state("long").cached_tokens == 4


def test_cancellation_between_chunks_releases_paged_resources() -> None:
    # Given: a long request has committed one intermediate chunk.
    engine = _engine(_model(), chunked=True, prefix_cache=True)
    engine.submit(_request("cancel", (1, 2, 3, 4, 5, 6), max_new_tokens=2))
    engine.tick(now=0.0)
    engine.tick(now=0.01)
    state = engine.request_state("cancel")
    assert state.status is RequestStatus.PREFILLING
    assert state.cached_tokens == 2
    assert len(state.prefill_logits_chunks) == 1

    # When: cancellation is requested before the next chunk.
    engine.cancel("cancel", at=0.015)
    engine.tick(now=0.02)

    # Then: request-private blocks, reservation capacity, and retained logits are released.
    assert state.status is RequestStatus.CANCELLED
    assert state.prefill_logits_chunks == []
    pool = engine.paged_cache_pool
    assert pool is not None
    metrics = pool.metrics()
    assert metrics.allocated_blocks == 0
    assert metrics.reserved_blocks == 0
    engine.verify_cache_invariants()


def test_model_failure_on_later_chunk_releases_paged_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the first chunk is committed and the request remains PREFILLING.
    model = _model()
    engine = _engine(model, chunked=True, prefix_cache=True)
    engine.submit(_request("fail", (1, 2, 3, 4, 5, 6), max_new_tokens=2))
    engine.tick(now=0.0)
    engine.tick(now=0.01)
    state = engine.request_state("fail")
    assert state.status is RequestStatus.PREFILLING
    assert state.cached_tokens == 2
    assert len(state.prefill_logits_chunks) == 1

    failure_message = "stage16 injected prefill failure"

    def fail_prefill(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError(failure_message)

    monkeypatch.setattr(model, "prefill_paged_batch", fail_prefill)

    # When: the next scheduled chunk reaches the failing model call.
    engine.tick(now=0.02)

    # Then: the request fails in isolation and all private capacity is released.
    assert state.status is RequestStatus.FAILED
    assert state.failure_reason == f"RuntimeError: {failure_message}"
    assert state.prefill_logits_chunks == []
    pool = engine.paged_cache_pool
    assert pool is not None
    metrics = pool.metrics()
    assert metrics.allocated_blocks == 0
    assert metrics.reserved_blocks == 0
    engine.verify_cache_invariants()


def test_chunked_prefill_rejects_batch_limit_smaller_than_one_block() -> None:
    # Given: the Stage 16 tensor batch limit cannot fit one physical KV block.
    model = _model()

    # When/Then: engine construction fails instead of creating a scheduler that cannot progress.
    with pytest.raises(InvalidServingConfigError, match="batch limit"):
        _ = _engine(model, chunked=True, prefill_batch_limit=1)


def test_apc_disabled_does_not_retain_intermediate_prefill_logits() -> None:
    # Given: chunked prefill runs without Automatic Prefix Caching.
    engine = _engine(_model(), chunked=True, prefix_cache=False)
    engine.submit(_request("no-apc", (1, 2, 3, 4, 5, 6), max_new_tokens=2))

    # When: the first intermediate chunk is committed.
    engine.tick(now=0.0)
    engine.tick(now=0.01)

    # Then: promotion-only logits are not retained when there is no APC consumer.
    state = engine.request_state("no-apc")
    assert state.status is RequestStatus.PREFILLING
    assert state.prefill_logits_chunks == []


def test_release_all_cache_resources_clears_intermediate_prefill_logits() -> None:
    # Given: an APC request has one retained intermediate promotion-logit tensor.
    engine = _engine(_model(), chunked=True, prefix_cache=True)
    engine.submit(_request("shutdown", (1, 2, 3, 4, 5, 6), max_new_tokens=2))
    engine.tick(now=0.0)
    engine.tick(now=0.01)
    state = engine.request_state("shutdown")
    assert len(state.prefill_logits_chunks) == 1

    # When: catastrophic/shutdown cleanup releases all request resources.
    engine.release_all_cache_resources()

    # Then: no request keeps a Tensor reachable through intermediate prefill state.
    assert state.prefill_logits_chunks == []


def test_normal_decode_uses_single_token_model_work() -> None:
    # Given: a seven-token prompt leaves exactly one normal paged decode before overflow.
    engine = _engine(_model(), chunked=True)
    engine.submit(_request("normal", (1, 2, 3, 4, 5, 6, 7), max_new_tokens=2))

    # When: prompt prefill and the first decode step complete.
    for tick in range(6):
        engine.tick(now=tick * 0.01)

    # Then: normal decode emits one token without the dense-overflow fallback.
    state = engine.request_state("normal")
    assert len(state.generated_tokens) == 2
    token_events = [event for event in engine.events if event.event_type is EngineEventType.TOKEN]
    assert token_events[-1].used_fallback is False


def test_overflow_decode_accounts_full_rebuild_context() -> None:
    # Given: one request reaches the learned-position window boundary.
    model = _model()
    engine = _engine(model, chunked=True)
    engine.submit(_request("overflow", (1, 2, 3, 4, 5, 6, 7), max_new_tokens=3))
    for tick in range(6):
        engine.tick(now=tick * 0.01)
    assert engine.request_state("overflow").cached_tokens == model.config.block_size

    # When: the next decode performs the dense sliding-window rebuild.
    engine.tick(now=0.06)

    # Then: telemetry records the actual full-window model-token work.
    overflow = next(
        item
        for item in engine.prefill_observations
        if item.execution_mode.value == "overflow_dense_rebuild"
    )
    assert overflow.useful_prompt_tokens == model.config.block_size
    token_events = [event for event in engine.events if event.event_type is EngineEventType.TOKEN]
    assert token_events[-1].used_fallback is True


def test_multiple_overflow_decodes_do_not_exceed_one_tick_budget() -> None:
    # Given: two FIFO requests reach overflow together with budget equal to one rebuild.
    engine = _engine(_model(), chunked=True)
    for request_id in ("first", "second"):
        engine.submit(_request(request_id, (1, 2, 3, 4, 5, 6, 7), max_new_tokens=3))
    for tick in range(6):
        engine.tick(now=tick * 0.01)
    assert all(engine.request_state(item).cached_tokens == 8 for item in ("first", "second"))

    # When: only eight model tokens are available for two eight-token rebuilds.
    engine.tick(now=0.06)

    # Then: FIFO schedules one fallback and defers the second without starvation.
    assert len(engine.request_state("first").generated_tokens) == 3
    assert len(engine.request_state("second").generated_tokens) == 2
    engine.tick(now=0.07)
    assert len(engine.request_state("second").generated_tokens) == 3


def test_overflow_and_prefill_do_not_silently_exceed_budget() -> None:
    # Given: an overflow decode and a newly admitted prefill share an eight-token budget.
    engine = _engine(_model(), chunked=True)
    engine.submit(_request("overflow", (1, 2, 3, 4, 5, 6, 7), max_new_tokens=4))
    for tick in range(6):
        engine.tick(now=tick * 0.01)
    engine.submit(_request("prefill", (8, 9, 10, 11, 12, 13), max_new_tokens=1))
    engine.tick(now=0.06)
    assert engine.request_state("prefill").cached_tokens == 0

    # When: the next tick has an eight-token overflow plus a two-token chunk ready.
    engine.tick(now=0.07)

    # Then: the older overflow consumes the budget and prefill is deferred, not overrun.
    assert engine.request_state("overflow").status is RequestStatus.FINISHED
    assert engine.request_state("prefill").cached_tokens == 0
    engine.tick(now=0.08)
    assert engine.request_state("prefill").cached_tokens == 2
