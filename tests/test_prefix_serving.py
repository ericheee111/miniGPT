from __future__ import annotations

from typing import final

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
    EngineConfig,
    EngineEventType,
    GenerationRequest,
    PagedAttentionExecutor,
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


def _engine(model: GPT, *, prefix_cache: bool, blocks: int = 16) -> ServingEngine:
    paged = PagedKVCacheConfig(block_tokens=2, num_blocks=blocks)
    pool = PagedKVCachePool.from_model(
        paged,
        model,
        prefix_cache_namespace=_namespace(model) if prefix_cache else None,
    )
    executor = PagedAttentionExecutor(
        model,
        pool,
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


def test_exact_repeated_prompt_skips_all_prefill_and_preserves_tokens_and_rng() -> None:
    # Given: identical direct-paged engines with APC disabled/enabled.
    model = _model()
    reference = _engine(model, prefix_cache=False)
    cached = _engine(model, prefix_cache=True)
    prompt = (1, 2, 3, 4)
    reference.submit(_request("reference", prompt))
    _run(reference)
    cached.submit(_request("prime", prompt, seed=3, max_new_tokens=2))
    _run(cached)

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
