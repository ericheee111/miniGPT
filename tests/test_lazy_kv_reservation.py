from __future__ import annotations

from typing import final

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
        _ = torch.default_generator.manual_seed(1818)
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
    finally:
        torch.set_rng_state(original)


def _namespace(model: GPT) -> PrefixCacheNamespace:
    return PrefixCacheNamespace(
        model_checkpoint_identity="stage18-test-checkpoint",
        model_config_identity="stage18-test-config",
        dtype=str(model.token_embedding.weight.dtype),
        device=str(model.token_embedding.weight.device),
        block_tokens=2,
        cache_schema_version=1,
        position_embedding_semantics="learned_absolute_v1",
    )


def _engine(  # noqa: PLR0913
    model: GPT,
    *,
    pool_blocks: int,
    lazy: bool,
    preemption: bool,
    max_active: int = 2,
    max_cached_tokens: int | None = None,
    overcommit_ratio: float = 1.0,
    prefix_cache: bool = False,
) -> ServingEngine:
    paged = PagedKVCacheConfig(block_tokens=2, num_blocks=pool_blocks)
    pool = PagedKVCachePool.from_model(
        paged,
        model,
        prefix_cache_namespace=_namespace(model) if prefix_cache else None,
    )
    scheduler = SchedulerConfig(
        max_active_requests=max_active,
        max_cached_tokens=pool_blocks * 2 if max_cached_tokens is None else max_cached_tokens,
        max_scheduled_tokens=8,
        prefill_chunk_tokens=2,
        kv_preemption=preemption,
        lazy_kv_reservation=lazy,
        kv_overcommit_ratio=overcommit_ratio,
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


def _request(request_id: str, seed: int, *, max_new: int = 5) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_tokens=(1, 2, 3, 4),
        max_new_tokens=max_new,
        seed=seed,
    )


def _run(engine: ServingEngine, *, start_tick: int = 0, max_ticks: int = 512) -> None:
    for tick in range(start_tick, start_tick + max_ticks):
        if engine.is_idle:
            return
        engine.tick(now=tick * 0.01)
    reason = "engine did not become idle"
    raise AssertionError(reason)


def test_lazy_config_requires_preemption_and_bounded_ratio() -> None:
    with pytest.raises(InvalidServingConfigError, match="requires kv_preemption"):
        _ = SchedulerConfig(
            max_active_requests=2,
            max_cached_tokens=8,
            max_scheduled_tokens=8,
            prefill_chunk_tokens=2,
            lazy_kv_reservation=True,
        )
    with pytest.raises(InvalidServingConfigError, match=r"must be 1\.0"):
        _ = SchedulerConfig(
            max_active_requests=2,
            max_cached_tokens=8,
            kv_overcommit_ratio=1.5,
        )
    with pytest.raises(InvalidServingConfigError, match=r"at least 1\.0"):
        _ = SchedulerConfig(
            max_active_requests=2,
            max_cached_tokens=8,
            max_scheduled_tokens=8,
            prefill_chunk_tokens=2,
            kv_preemption=True,
            lazy_kv_reservation=True,
            kv_overcommit_ratio=float("nan"),
        )


def test_lazy_admission_protects_prompt_but_tracks_lifetime_demand() -> None:
    engine = _engine(
        _model(),
        pool_blocks=4,
        lazy=True,
        preemption=True,
        overcommit_ratio=2.0,
    )
    for request in (_request("first", 1801), _request("second", 1802)):
        engine.submit(request)

    engine.tick(now=0.0)

    for request_id in ("first", "second"):
        state = engine.request_state(request_id)
        assert state.status is RequestStatus.PREFILLING
        assert state.reserved_cache_tokens == 4
        assert state.reserved_cache_blocks == 2
    metrics = engine.metrics()
    assert metrics.active_requests == 2
    assert metrics.reserved_cache_tokens == 8
    assert metrics.lifetime_reserved_cache_tokens == 16
    assert metrics.overcommitted_cache_tokens == 8
    assert metrics.lazy_kv_reservation_enabled is True
    assert metrics.kv_overcommit_ratio == 2.0


def test_legacy_full_reservation_admits_only_one_request_in_same_pool() -> None:
    engine = _engine(
        _model(),
        pool_blocks=4,
        lazy=False,
        preemption=False,
    )
    for request in (_request("first", 1803), _request("second", 1804)):
        engine.submit(request)

    engine.tick(now=0.0)

    assert engine.request_state("first").status is RequestStatus.PREFILLING
    assert engine.request_state("first").reserved_cache_tokens == 8
    assert engine.request_state("second").status is RequestStatus.WAITING
    metrics = engine.metrics()
    assert metrics.active_requests == 1
    assert metrics.overcommitted_cache_tokens == 0
    assert metrics.lazy_kv_reservation_enabled is False


def test_overcommit_ratio_caps_aggregate_lifetime_demand() -> None:
    constrained = _engine(
        _model(),
        pool_blocks=4,
        lazy=True,
        preemption=True,
        overcommit_ratio=1.5,
    )
    for request in (_request("first", 1805), _request("second", 1806)):
        constrained.submit(request)

    constrained.tick(now=0.0)

    assert constrained.request_state("first").status is RequestStatus.PREFILLING
    assert constrained.request_state("second").status is RequestStatus.WAITING
    assert constrained.metrics().lifetime_reserved_cache_tokens == 8


def test_lazy_intrinsically_impossible_head_fails_without_preemption() -> None:
    engine = _engine(
        _model(),
        pool_blocks=4,
        lazy=True,
        preemption=True,
        max_active=1,
        max_cached_tokens=4,
        overcommit_ratio=2.0,
    )
    engine.submit(
        GenerationRequest(
            request_id="viable",
            prompt_tokens=(1, 2),
            max_new_tokens=3,
            seed=1820,
        )
    )
    engine.submit(_request("oversized", 1821, max_new=5))

    engine.tick(now=0.0)

    oversized = engine.request_state("oversized")
    assert oversized.status is RequestStatus.FAILED
    assert "exceeds budget 4" in (oversized.failure_reason or "")
    assert engine.request_state("viable").preemption_count == 0
    assert engine.metrics().preemptions == 0
    assert not any(event.event_type is EngineEventType.PREEMPTED for event in engine.events)


def test_decode_reservation_grows_before_token_commit() -> None:
    engine = _engine(
        _model(),
        pool_blocks=8,
        lazy=True,
        preemption=True,
        overcommit_ratio=2.0,
    )
    engine.submit(_request("request", 1807))
    engine.tick(now=0.0)
    engine.tick(now=0.01)
    engine.tick(now=0.02)
    before_tokens = len(engine.request_state("request").generated_tokens)

    engine.tick(now=0.03)

    state = engine.request_state("request")
    assert len(state.generated_tokens) == before_tokens + 1
    assert state.reserved_cache_tokens == 5
    assert state.reservation_growth_count == 1
    assert state.reservation_growth_tokens == 1
    grown = next(
        event
        for event in engine.events
        if event.request_id == "request" and event.event_type is EngineEventType.RESERVATION_GROWN
    )
    token = next(
        event
        for event in engine.events
        if event.request_id == "request"
        and event.event_type is EngineEventType.TOKEN
        and event.sequence > grown.sequence
    )
    assert grown.sequence < token.sequence


def test_growth_pressure_preempts_other_decoder_and_retries_without_model_work() -> None:
    engine = _engine(
        _model(),
        pool_blocks=4,
        lazy=True,
        preemption=True,
        overcommit_ratio=2.0,
    )
    for request in (_request("blocked", 1808), _request("victim", 1809)):
        engine.submit(request)
    engine.tick(now=0.0)
    engine.tick(now=0.01)
    engine.tick(now=0.02)
    blocked = engine.request_state("blocked")
    before_tokens = tuple(blocked.generated_tokens)
    before_rng = blocked.generator.get_state().clone()
    before_decode_batches = len(engine.metrics().decode_batch_sizes)
    before_prefill_observations = len(engine.prefill_observations)

    engine.tick(now=0.03)

    victim = engine.request_state("victim")
    assert victim.status is RequestStatus.PREEMPTED
    assert blocked.status is RequestStatus.DECODING
    assert tuple(blocked.generated_tokens) == before_tokens
    assert torch.equal(blocked.generator.get_state(), before_rng)
    assert len(engine.metrics().decode_batch_sizes) == before_decode_batches
    assert len(engine.prefill_observations) == before_prefill_observations
    assert blocked.reserved_cache_tokens == 5
    assert blocked.reserved_cache_blocks == 3
    assert blocked.reservation_growth_blocked_count == 1
    assert blocked.reservation_growth_count == 1
    events = [event for event in engine.events if event.timestamp >= 0.03]
    blocked_event = next(
        event for event in events if event.event_type is EngineEventType.RESERVATION_GROWTH_BLOCKED
    )
    preempt_event = next(event for event in events if event.event_type is EngineEventType.PREEMPTED)
    grown_event = next(
        event for event in events if event.event_type is EngineEventType.RESERVATION_GROWN
    )
    assert "reservation_growth" in (preempt_event.detail or "")
    assert blocked_event.sequence < preempt_event.sequence < grown_event.sequence
    assert engine.metrics().growth_pressure_preemptions == 1


def test_lazy_pressure_matches_roomy_reference_tokens_and_rng() -> None:
    model = _model()
    reference = _engine(
        model,
        pool_blocks=8,
        lazy=False,
        preemption=False,
        max_cached_tokens=16,
    )
    pressured = _engine(
        model,
        pool_blocks=4,
        lazy=True,
        preemption=True,
        overcommit_ratio=2.0,
    )
    requests = (_request("first", 1810, max_new=7), _request("second", 1811, max_new=7))
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
    metrics = pressured.metrics()
    assert metrics.reservation_growths > 0
    assert metrics.growth_pressure_preemptions > 0
    assert metrics.preemptions > 0
    pool = pressured.paged_cache_pool
    assert pool is not None
    assert pool.metrics().reserved_blocks == 0
    pool.verify_invariants()


def test_lazy_apc_victim_resumes_with_private_recompute() -> None:
    engine = _engine(
        _model(),
        pool_blocks=4,
        lazy=True,
        preemption=True,
        overcommit_ratio=2.0,
        prefix_cache=True,
    )
    for request in (_request("first", 1812), _request("second", 1813)):
        engine.submit(request)
    for tick in range(4):
        engine.tick(now=tick * 0.01)
    victim_id = next(
        request_id
        for request_id in ("first", "second")
        if engine.request_state(request_id).status is RequestStatus.PREEMPTED
    )
    pool = engine.paged_cache_pool
    assert pool is not None
    assert not pool.has_request(victim_id)

    for tick in range(4, 64):
        engine.tick(now=tick * 0.01)
        state = engine.request_state(victim_id)
        if state.status is RequestStatus.DECODING and state.resume_count > 0:
            break

    state = engine.request_state(victim_id)
    assert state.status is RequestStatus.DECODING
    assert pool.request_cache(victim_id).shared_blocks == 0
    pool.verify_invariants()


def test_cancellation_and_catastrophic_cleanup_clear_growth_state() -> None:
    engine = _engine(
        _model(),
        pool_blocks=5,
        lazy=True,
        preemption=True,
        overcommit_ratio=2.0,
    )
    engine.submit(_request("request", 1814, max_new=5))
    engine.submit(
        GenerationRequest(
            request_id="prefilling",
            prompt_tokens=(1, 2, 3, 4, 5, 6),
            max_new_tokens=1,
            seed=1815,
        )
    )
    engine.tick(now=0.0)
    engine.tick(now=0.01)
    engine.tick(now=0.02)
    engine.tick(now=0.03)

    assert engine.growth_blocked_request_id == "request"
    assert engine.growth_blocked_target_tokens == 5
    assert engine.request_state("prefilling").status is RequestStatus.PREFILLING

    engine.cancel("request", at=0.04)
    engine.tick(now=0.04)

    assert engine.request_state("request").status is RequestStatus.CANCELLED
    assert engine.growth_blocked_request_id is None
    assert engine.growth_blocked_target_tokens == 0
    engine.release_all_cache_resources()
    pool = engine.paged_cache_pool
    assert pool is not None
    assert pool.metrics().reserved_blocks == 0
    pool.verify_invariants()
