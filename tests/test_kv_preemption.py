from __future__ import annotations

from typing import Never, final

import pytest
import torch

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
    InvalidServingConfigError,
    PagedAttentionExecutor,
    PrefillBatchConfig,
    PrefillExecutionMode,
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
    return GPT(
        GPTConfig(
            vocab_size=31,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
        )
    ).eval()


def _namespace(model: GPT) -> PrefixCacheNamespace:
    return PrefixCacheNamespace(
        model_checkpoint_identity="stage17-test-checkpoint",
        model_config_identity="stage17-test-config",
        dtype=str(model.token_embedding.weight.dtype),
        device=str(model.token_embedding.weight.device),
        block_tokens=2,
        cache_schema_version=1,
        position_embedding_semantics="learned_absolute_v1",
    )


def _engine(
    model: GPT,
    *,
    pool_blocks: int,
    preemption: bool,
    prefix_cache: bool = False,
    max_active: int = 2,
) -> ServingEngine:
    paged = PagedKVCacheConfig(2, pool_blocks)
    pool = PagedKVCachePool.from_model(
        paged,
        model,
        prefix_cache_namespace=_namespace(model) if prefix_cache else None,
    )
    scheduler = SchedulerConfig(
        max_active_requests=max_active,
        max_cached_tokens=pool_blocks * 2,
        max_scheduled_tokens=model.config.block_size,
        prefill_chunk_tokens=2,
        kv_preemption=preemption,
    )
    return ServingEngine(
        config=EngineConfig(
            scheduler=scheduler,
            block_size=model.config.block_size,
            kv_cache_backend=KVCacheBackend.PAGED,
            paged_kv_cache=paged,
        ),
        executor=PagedAttentionExecutor(
            model,
            pool,
            prefill_config=PrefillBatchConfig(max_active, model.config.block_size, 1.0),
            clock=StepClock(0.001),
            telemetry_clock=StepClock(0.0001),
        ),
        paged_cache_pool=pool,
        clock=lambda: 0.0,
    )


def _request(request_id: str, seed: int, *, max_new: int = 4) -> GenerationRequest:
    return GenerationRequest(request_id, tuple(range(1, 5)), max_new, seed=seed)


def _run(engine: ServingEngine, *, start: int = 0) -> None:
    for tick in range(start, start + 200):
        if engine.is_idle:
            return
        engine.tick(now=tick * 0.01)
    reason = "engine did not become idle"
    raise AssertionError(reason)


def test_pressure_preempts_decoding_request() -> None:
    engine = _engine(_model(), pool_blocks=4, preemption=True)
    engine.submit(_request("first", 17))
    engine.submit(_request("second", 19))
    for tick in range(4):
        engine.tick(now=tick * 0.01)

    first = engine.request_state("first")
    second = engine.request_state("second")
    assert first.status is RequestStatus.PREEMPTED
    assert second.status is RequestStatus.WAITING
    assert first.preemption_count == 1
    assert first.reserved_cache_blocks == 0
    pool = engine.paged_cache_pool
    assert pool is not None
    assert not pool.has_request("first")
    assert any(event.event_type is EngineEventType.PREEMPTED for event in engine.events)
    pool.verify_invariants()


def test_recompute_resume_preserves_rng_without_sampling() -> None:
    engine = _engine(_model(), pool_blocks=4, preemption=True)
    engine.submit(_request("first", 23))
    engine.submit(_request("second", 29))
    for tick in range(4):
        engine.tick(now=tick * 0.01)
    first = engine.request_state("first")
    rng_state = first.generator.get_state().clone()
    output = tuple(first.generated_tokens)

    for tick in range(4, 24):
        engine.tick(now=tick * 0.01)
        if first.status is RequestStatus.DECODING and first.resume_count == 1:
            break

    assert first.status is RequestStatus.DECODING
    assert first.resume_count == 1
    assert tuple(first.generated_tokens) == output
    assert torch.equal(first.generator.get_state(), rng_state)
    assert first.recompute_tokens > 0
    recompute = next(
        item
        for item in engine.prefill_observations
        if item.execution_mode is PrefillExecutionMode.PREEMPTION_RECOMPUTE
    )
    assert recompute.useful_prompt_tokens == first.recompute_tokens
    assert recompute.model_calls == 1
    assert any(event.event_type is EngineEventType.RECOMPUTE_STARTED for event in engine.events)
    assert any(event.event_type is EngineEventType.RESUMED for event in engine.events)


def test_preemption_matches_roomy_reference() -> None:
    model = _model()
    reference = _engine(model, pool_blocks=8, preemption=False)
    pressured = _engine(model, pool_blocks=4, preemption=True)
    requests = (_request("first", 31), _request("second", 37))
    for request in requests:
        reference.submit(request)
        pressured.submit(request)

    _run(reference)
    _run(pressured)

    for request in requests:
        expected = reference.request_state(request.request_id)
        actual = pressured.request_state(request.request_id)
        assert actual.status is RequestStatus.FINISHED
        assert actual.generated_tokens == expected.generated_tokens
        assert torch.equal(actual.generator.get_state(), expected.generator.get_state())
    assert pressured.metrics().preemptions > 0
    assert pressured.metrics().recompute_tokens > 0
    pool = pressured.paged_cache_pool
    assert pool is not None
    assert pool.metrics().reserved_blocks == 0
    pool.verify_invariants()


def test_cancelling_preempted_request_does_not_recompute() -> None:
    engine = _engine(_model(), pool_blocks=4, preemption=True)
    engine.submit(_request("first", 41))
    engine.submit(_request("second", 43))
    for tick in range(4):
        engine.tick(now=tick * 0.01)
    first = engine.request_state("first")
    assert first.status is RequestStatus.PREEMPTED

    engine.cancel("first", at=0.025)
    engine.tick(now=0.03)

    assert first.status is RequestStatus.CANCELLED
    assert first.resume_count == 0
    assert first.recompute_tokens == 0
    pool = engine.paged_cache_pool
    assert pool is not None
    assert not pool.has_request("first")
    pool.verify_invariants()


def test_recompute_failure_isolated_and_releases_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    engine = _engine(model, pool_blocks=4, preemption=True)
    engine.submit(_request("first", 47))
    engine.submit(_request("second", 53))
    for tick in range(9):
        engine.tick(now=tick * 0.01)
    first = engine.request_state("first")
    assert first.status is RequestStatus.RECOMPUTING

    failure_message = "stage17 injected recompute failure"

    def fail_prefill(_input: torch.Tensor) -> Never:
        raise RuntimeError(failure_message)

    monkeypatch.setattr(model, "prefill", fail_prefill)
    engine.tick(now=0.09)

    assert first.status is RequestStatus.FAILED
    assert "stage17 injected recompute failure" in (first.failure_reason or "")
    pool = engine.paged_cache_pool
    assert pool is not None
    assert not pool.has_request("first")
    pool.verify_invariants()


def test_preemption_releases_apc_refs_and_resume_uses_private_recompute() -> None:
    model = _model()
    engine = _engine(model, pool_blocks=4, preemption=True, prefix_cache=True)
    engine.submit(_request("first", 59))
    engine.submit(_request("second", 61))
    for tick in range(4):
        engine.tick(now=tick * 0.01)

    first = engine.request_state("first")
    pool = engine.paged_cache_pool
    assert pool is not None
    assert first.status is RequestStatus.PREEMPTED
    assert pool.metrics().prefix_cache_blocks > 0
    assert pool.metrics().active_shared_references == 0

    for tick in range(4, 12):
        engine.tick(now=tick * 0.01)
        current = engine.request_state("first")
        if current.status is RequestStatus.DECODING and current.resume_count == 1:
            break

    first = engine.request_state("first")
    assert first.status is RequestStatus.DECODING
    table = pool.request_cache("first")
    assert table.shared_blocks == 0
    pool.verify_invariants()


def test_preemption_preserves_overflow_sliding_window_correctness() -> None:
    model = _model()
    reference = _engine(model, pool_blocks=8, preemption=False)
    pressured = _engine(model, pool_blocks=4, preemption=True)
    requests = (
        _request("first", 67, max_new=7),
        _request("second", 71, max_new=7),
    )
    for request in requests:
        reference.submit(request)
        pressured.submit(request)

    _run(reference)
    _run(pressured)

    for request in requests:
        expected = reference.request_state(request.request_id)
        actual = pressured.request_state(request.request_id)
        assert actual.generated_tokens == expected.generated_tokens
        assert torch.equal(actual.generator.get_state(), expected.generator.get_state())
        assert actual.recompute_tokens >= model.config.block_size


def test_three_request_pressure_rotation_has_no_starvation() -> None:
    engine = _engine(_model(), pool_blocks=4, preemption=True)
    requests = tuple(_request(f"request-{index}", 80 + index) for index in range(3))
    for request in requests:
        engine.submit(request)

    _run(engine)

    for request in requests:
        state = engine.request_state(request.request_id)
        assert state.status is RequestStatus.FINISHED
        assert len(state.generated_tokens) == request.max_new_tokens
        assert state.preemption_count > 0
        assert state.resume_count > 0
    metrics = engine.metrics()
    assert metrics.preemptions >= 3
    assert metrics.resumes >= 3
    pool = engine.paged_cache_pool
    assert pool is not None
    assert pool.metrics().reserved_blocks == 0
    pool.verify_invariants()


def test_preemption_waits_for_decode_progress_after_prefill_first_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(_model(), pool_blocks=7, preemption=True, max_active=3)
    short = GenerationRequest("short", (1, 2), 4, seed=91)
    long = GenerationRequest("long", tuple(range(3, 11)), 1, seed=92)
    blocked = GenerationRequest("blocked", (9, 10, 11, 12), 1, seed=93)
    for request in (short, long, blocked):
        engine.submit(request)

    engine.tick(now=0.0)
    engine.tick(now=0.01)
    short_state = engine.request_state("short")
    long_state = engine.request_state("long")
    assert short_state.status is RequestStatus.DECODING
    assert long_state.status is RequestStatus.PREFILLING
    assert engine.request_state("blocked").status is RequestStatus.WAITING

    monkeypatch.setattr(engine, "_chunk_schedule_cursor", "long")
    engine.tick(now=0.02)

    short_state = engine.request_state("short")
    assert short_state.status is RequestStatus.DECODING
    assert short_state.preemption_count == 0

    engine.tick(now=0.03)
    short_state = engine.request_state("short")
    assert short_state.status is RequestStatus.PREEMPTED
    token_event = next(
        event
        for event in reversed(engine.events)
        if event.request_id == "short" and event.event_type is EngineEventType.TOKEN
    )
    preempt_event = next(
        event
        for event in reversed(engine.events)
        if event.request_id == "short" and event.event_type is EngineEventType.PREEMPTED
    )
    assert preempt_event.timestamp >= token_event.timestamp


def test_active_limit_alone_does_not_trigger_kv_preemption() -> None:
    model = _model()
    paged = PagedKVCacheConfig(2, 8)
    pool = PagedKVCachePool.from_model(paged, model)
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(
                max_active_requests=1,
                max_cached_tokens=16,
                max_scheduled_tokens=model.config.block_size,
                prefill_chunk_tokens=2,
                kv_preemption=True,
            ),
            block_size=model.config.block_size,
            kv_cache_backend=KVCacheBackend.PAGED,
            paged_kv_cache=paged,
        ),
        executor=PagedAttentionExecutor(model, pool),
        paged_cache_pool=pool,
        clock=lambda: 0.0,
    )
    engine.submit(_request("first", 101))
    engine.submit(_request("second", 103))

    for tick in range(4):
        engine.tick(now=tick * 0.01)

    assert engine.request_state("first").preemption_count == 0
    assert engine.request_state("second").status is RequestStatus.WAITING


def test_preemption_requires_stage16_token_budget_scheduler() -> None:
    model = _model()
    paged = PagedKVCacheConfig(2, 4)
    pool = PagedKVCachePool.from_model(paged, model)
    scheduler = SchedulerConfig(
        max_active_requests=2,
        max_cached_tokens=8,
        kv_preemption=True,
    )

    with pytest.raises(InvalidServingConfigError, match="Stage 16"):
        _ = ServingEngine(
            config=EngineConfig(
                scheduler=scheduler,
                block_size=model.config.block_size,
                kv_cache_backend=KVCacheBackend.PAGED,
                paged_kv_cache=paged,
            ),
            executor=PagedAttentionExecutor(model, pool),
            paged_cache_pool=pool,
        )
