from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from minigpt.engine_runner import EngineRunner, RunnerConfig, RunnerEventType, RunnerState
from minigpt.model import GPT
from minigpt.paged_kv_cache import KVCacheBackend, PagedKVCacheConfig, PagedKVCachePool
from minigpt.serving import (
    ContinuousDecodeExecutor,
    ContinuousExecutor,
    EngineConfig,
    EngineEvent,
    GenerationRequest,
    ReferenceExecutor,
    RequestMetrics,
    RequestStatus,
    SchedulerConfig,
    ServingEngine,
    ServingExecutor,
)
from minigpt.settings import GPTConfig

ExecutorFactory = Callable[[GPT, Callable[[], float], Callable[[], float]], ServingExecutor]


class StepClock:
    def __init__(self, step: float) -> None:
        self._step: float = step
        self._current: float = 0.0

    def __call__(self) -> float:
        current = self._current
        self._current += self._step
        return current


def _model() -> GPT:
    original_state = torch.get_rng_state()
    try:
        _ = torch.default_generator.manual_seed(20260807)
        model = GPT(
            GPTConfig(
                vocab_size=19,
                block_size=8,
                n_layer=2,
                n_head=2,
                n_embd=8,
                dropout=0.0,
                bias=False,
            )
        )
    finally:
        torch.set_rng_state(original_state)
    return model.eval()


def _reference(
    model: GPT,
    clock: Callable[[], float],
    telemetry_clock: Callable[[], float],
) -> ServingExecutor:
    return ReferenceExecutor(model, clock=clock, telemetry_clock=telemetry_clock)


def _continuous_decode(
    model: GPT,
    clock: Callable[[], float],
    telemetry_clock: Callable[[], float],
) -> ServingExecutor:
    return ContinuousDecodeExecutor(model, clock=clock, telemetry_clock=telemetry_clock)


def _continuous(
    model: GPT,
    clock: Callable[[], float],
    telemetry_clock: Callable[[], float],
) -> ServingExecutor:
    return ContinuousExecutor(model, clock=clock, telemetry_clock=telemetry_clock)


def _requests() -> tuple[GenerationRequest, ...]:
    return (
        GenerationRequest(
            request_id="short",
            prompt_tokens=(1, 2, 3),
            max_new_tokens=5,
            seed=11,
        ),
        GenerationRequest(
            request_id="long",
            prompt_tokens=(4, 5, 6, 7, 8),
            max_new_tokens=5,
            seed=29,
        ),
    )


def _engine(
    model: GPT,
    executor_factory: ExecutorFactory,
    *,
    backend: KVCacheBackend,
    num_blocks: int = 8,
    max_active_requests: int = 2,
) -> ServingEngine:
    executor = executor_factory(model, StepClock(0.001), StepClock(0.0001))
    paged_config = (
        PagedKVCacheConfig(block_tokens=2, num_blocks=num_blocks)
        if backend is KVCacheBackend.PAGED
        else None
    )
    pool = PagedKVCachePool.from_model(paged_config, model) if paged_config is not None else None
    return ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(
                max_active_requests=max_active_requests,
                max_cached_tokens=model.config.block_size * max_active_requests,
            ),
            block_size=model.config.block_size,
            kv_cache_backend=backend,
            paged_kv_cache=paged_config,
        ),
        executor=executor,
        paged_cache_pool=pool,
        clock=lambda: 0.0,
    )


def _run(engine: ServingEngine, requests: tuple[GenerationRequest, ...]) -> None:
    for request in requests:
        engine.submit(request)
    for tick in range(100):
        if engine.is_idle:
            break
        engine.tick(now=tick * 0.01)
    assert engine.is_idle


def _logical_events(events: tuple[EngineEvent, ...]) -> tuple[EngineEvent, ...]:
    return events


@pytest.mark.parametrize("executor_factory", [_reference, _continuous_decode, _continuous])
def test_dense_and_paged_backends_are_logically_equivalent(
    executor_factory: ExecutorFactory,
) -> None:
    # Given: identical weights, workload, seeds, scheduler, and executor.
    model = _model()
    dense = _engine(model, executor_factory, backend=KVCacheBackend.DENSE)
    paged = _engine(model, executor_factory, backend=KVCacheBackend.PAGED)
    requests = _requests()

    # When: both cache backends run through learned-position overflow re-prefill.
    _run(dense, requests)
    _run(paged, requests)

    # Then: tokens, terminal state, FIFO events, and request metrics are identical.
    for request in requests:
        dense_state = dense.request_state(request.request_id)
        paged_state = paged.request_state(request.request_id)
        assert paged_state.generated_tokens == dense_state.generated_tokens
        assert paged_state.status is dense_state.status is RequestStatus.FINISHED
        assert paged.request_metrics(request.request_id) == dense.request_metrics(
            request.request_id
        )
    assert _logical_events(paged.events) == _logical_events(dense.events)
    assert any(event.used_fallback for event in paged.events)
    metrics = paged.metrics()
    assert metrics.kv_cache_backend is KVCacheBackend.PAGED
    assert metrics.allocated_blocks == 0
    assert metrics.reserved_blocks == 0
    assert metrics.free_blocks == metrics.total_blocks == 8
    assert metrics.peak_allocated_blocks > 0
    assert metrics.peak_reserved_blocks > 0
    paged.verify_cache_invariants()


def test_paged_capacity_pressure_preserves_strict_fifo_admission() -> None:
    # Given: a two-block pool where only one two-block request can run at once.
    model = _model()
    engine = _engine(
        model,
        _continuous,
        backend=KVCacheBackend.PAGED,
        num_blocks=2,
        max_active_requests=3,
    )
    first = GenerationRequest("first", (1, 2), 3, seed=1)
    second = GenerationRequest("second", (2, 3), 3, seed=2)
    impossible = GenerationRequest("impossible", (3, 4, 5, 6, 7), 5, seed=3)

    # When: the head request occupies the pool and a later request cannot bypass it.
    _run(engine, (first, second, impossible))

    # Then: admission stays FIFO and the theoretically oversized request fails deterministically.
    admitted = tuple(
        event.request_id for event in engine.events if event.event_type.value == "admitted"
    )
    assert admitted == ("first", "second")
    assert engine.request_state("first").status is RequestStatus.FINISHED
    assert engine.request_state("second").status is RequestStatus.FINISHED
    failed = engine.request_state("impossible")
    assert failed.status is RequestStatus.FAILED
    assert failed.failure_reason is not None
    assert "exceeds pool 2" in failed.failure_reason
    assert engine.metrics().free_blocks == 2
    engine.verify_cache_invariants()


def test_paged_failure_isolation_and_mixed_lifecycle_cleanup() -> None:
    # Given: a valid decoder beside an invalid-vocabulary request.
    model = _model()
    engine = _engine(model, _continuous, backend=KVCacheBackend.PAGED, num_blocks=8)
    valid = GenerationRequest("valid", (1, 2, 3), 6, seed=7)
    invalid = GenerationRequest("invalid", (model.config.vocab_size,), 3, seed=8)
    engine.submit(valid)
    engine.submit(invalid)

    # When: invalid prefill fails while its peer keeps decoding.
    for tick in range(3):
        engine.tick(now=tick * 0.01)

    # Then: failure releases only its reservation and the peer remains healthy.
    assert engine.request_state("invalid").status is RequestStatus.FAILED
    assert engine.request_state("valid").status is RequestStatus.DECODING
    assert engine.metrics().reserved_blocks > 0
    engine.verify_cache_invariants()
    engine.cancel("valid", at=0.03)
    engine.tick(now=0.03)
    assert engine.request_state("valid").status is RequestStatus.CANCELLED
    metrics = engine.metrics()
    assert metrics.allocated_blocks == 0
    assert metrics.reserved_blocks == 0
    assert metrics.free_blocks == metrics.total_blocks
    engine.verify_cache_invariants()


def test_paged_waiting_prefilling_decoding_shutdown_has_zero_leaks() -> None:
    # Given: an engine advanced into simultaneous DECODING, PREFILLING, and WAITING states.
    model = _model()
    engine = _engine(
        model,
        _reference,
        backend=KVCacheBackend.PAGED,
        num_blocks=8,
        max_active_requests=2,
    )
    first, second, third = (
        GenerationRequest("decode", (1, 2), 6, seed=1),
        GenerationRequest("prefill", (2, 3), 6, seed=2),
        GenerationRequest("waiting", (3, 4), 6, seed=3),
    )
    engine.submit(first)
    engine.tick(now=0.0)
    engine.tick(now=0.01)
    engine.submit(second)
    engine.submit(third)
    engine.tick(now=0.02)
    assert engine.request_state("decode").status is RequestStatus.DECODING
    assert engine.request_state("prefill").status is RequestStatus.PREFILLING
    assert engine.request_state("waiting").status is RequestStatus.WAITING

    # When: graceful shutdown semantics cancel every live lifecycle state.
    for request_id in ("decode", "prefill", "waiting"):
        engine.cancel(request_id, at=0.03)
    engine.tick(now=0.03)

    # Then: ownership and reservations are completely released.
    assert all(
        engine.request_state(request_id).status is RequestStatus.CANCELLED
        for request_id in ("decode", "prefill", "waiting")
    )
    metrics = engine.metrics()
    assert metrics.allocated_blocks == 0
    assert metrics.reserved_blocks == 0
    assert metrics.free_blocks == metrics.total_blocks
    engine.verify_cache_invariants()


def test_engine_runner_graceful_shutdown_releases_paged_pool() -> None:
    # Given: several live requests owned by the dedicated runner thread.
    model = _model()
    engine = _engine(
        model,
        _reference,
        backend=KVCacheBackend.PAGED,
        num_blocks=8,
        max_active_requests=2,
    )
    pool = engine.paged_cache_pool
    assert pool is not None
    runner = EngineRunner(engine=engine, config=RunnerConfig())
    runner.start()
    requests = (
        GenerationRequest("runner-first", (1, 2), 500, seed=1),
        GenerationRequest("runner-second", (2, 3), 500, seed=2),
        GenerationRequest("runner-third", (3, 4), 500, seed=3),
    )
    handles = tuple(runner.submit(request, stream=False) for request in requests)

    # When: graceful shutdown is requested while submitted work is still live.
    runner.shutdown()

    # Then: every channel terminates and the physical pool has no ownership or reservation.
    assert runner.state is RunnerState.STOPPED
    assert all(handle.future.done() for handle in handles)
    pool.verify_invariants()
    metrics = pool.metrics()
    assert metrics.allocated_blocks == 0
    assert metrics.reserved_blocks == 0
    assert metrics.free_blocks == metrics.total_blocks


def test_engine_runner_stream_backpressure_releases_paged_pool() -> None:
    # Given: one paged request producing into a deliberately unconsumed one-token stream buffer.
    model = _model()
    engine = _engine(model, _reference, backend=KVCacheBackend.PAGED, num_blocks=4)
    pool = engine.paged_cache_pool
    assert pool is not None
    runner = EngineRunner(
        engine=engine,
        config=RunnerConfig(command_queue_size=8, stream_buffer_size=1),
    )
    runner.start()
    handle = runner.submit(
        GenerationRequest("backpressure", (1, 2), 20, seed=17),
        stream=True,
    )

    try:
        # When: the owner outruns the intentionally idle stream consumer.
        result = handle.future.result(timeout=5.0)

        # Then: bounded-buffer cancellation releases every physical block and reservation.
        assert result.status is RequestStatus.CANCELLED
        assert any(event.event_type is RunnerEventType.BACKPRESSURE for event in runner.events)
        pool.verify_invariants()
        metrics = pool.metrics()
        assert metrics.allocated_blocks == 0
        assert metrics.reserved_blocks == 0
        assert metrics.free_blocks == metrics.total_blocks
    finally:
        runner.shutdown()


def test_dense_backend_reports_zero_storage_specific_metrics() -> None:
    model = _model()
    engine = _engine(model, _reference, backend=KVCacheBackend.DENSE)
    _run(engine, (_requests()[0],))
    metrics = engine.metrics()
    assert metrics.kv_cache_backend is KVCacheBackend.DENSE
    assert metrics.total_blocks == 0
    assert metrics.allocated_blocks == 0
    assert metrics.internal_fragmentation_ratio == 0.0
    request_metrics: RequestMetrics = engine.request_metrics("short")
    assert request_metrics.status is RequestStatus.FINISHED
