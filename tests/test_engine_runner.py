from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from typing_extensions import override

from minigpt.engine_runner import (
    EngineRunner,
    RunnerConfig,
    RunnerEventType,
    RunnerInspectionTimeoutError,
    RunnerState,
    StreamEventType,
)
from minigpt.serving import (
    DecodeBatchObservation,
    EngineConfig,
    ExecutionResult,
    GenerationRequest,
    PrefillBatchEvent,
    PrefillBatchObservation,
    RequestState,
    RequestStatus,
    SchedulerConfig,
    ServingEngine,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(slots=True)
class RecordingExecutor:
    """Return deterministic tokens while recording the calling thread."""

    block_size: int = 8
    failing_ids: set[str] = field(default_factory=set)
    owner_threads: list[int] = field(default_factory=list)
    calls: int = 0

    @property
    def decode_observations(self) -> tuple[DecodeBatchObservation, ...]:
        return ()

    @property
    def prefill_observations(self) -> tuple[PrefillBatchObservation, ...]:
        return ()

    @property
    def prefill_events(self) -> tuple[PrefillBatchEvent, ...]:
        return ()

    def prefill(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        self._record_call()
        return tuple(self._result(state) for state in requests)

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        self._record_call()
        return tuple(self._result(state) for state in active_requests)

    def _record_call(self) -> None:
        self.calls += 1
        thread_id = threading.get_ident()
        self.owner_threads.append(thread_id)

    def _result(self, state: RequestState) -> ExecutionResult:
        request_id = state.request.request_id
        if request_id in self.failing_ids:
            return ExecutionResult.failure(request_id, "injected failure", 0.0)
        generated_count = len(state.generated_tokens)
        token_id = (state.request.seed + generated_count) % 17
        cache_tokens = min(
            self.block_size,
            len(state.request.prompt_tokens) + generated_count,
        )
        return ExecutionResult.success(
            request_id=request_id,
            token_id=token_id,
            cache=(),
            cache_tokens=cache_tokens,
            latency_seconds=0.0,
            used_fallback=False,
        )


@dataclass(slots=True)
class BlockingExecutor(RecordingExecutor):
    """Block the first prefill so cancellation ordering can be observed."""

    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    _blocked: bool = False

    @override
    def prefill(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        if not self._blocked:
            self._blocked = True
            self.entered.set()
            assert self.release.wait(timeout=5.0)
        return RecordingExecutor.prefill(self, requests)


def _request(
    request_id: str,
    *,
    max_new_tokens: int = 3,
    seed: int = 7,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_tokens=(1, 2),
        max_new_tokens=max_new_tokens,
        seed=seed,
        arrival_time=time.perf_counter(),
    )


def _runner(
    executor: RecordingExecutor,
    *,
    max_active_requests: int = 2,
    max_cached_tokens: int = 32,
    stream_buffer_size: int = 8,
) -> EngineRunner:
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(
                max_active_requests=max_active_requests,
                max_cached_tokens=max_cached_tokens,
            ),
            block_size=executor.block_size,
        ),
        executor=executor,
    )
    return EngineRunner(
        engine=engine,
        config=RunnerConfig(
            command_queue_size=32,
            stream_buffer_size=stream_buffer_size,
        ),
    )


def test_runner_uses_one_execution_owner_and_blocks_while_idle() -> None:
    # Given: a started runner whose engine has no work.
    executor = RecordingExecutor()
    runner = _runner(executor)
    runner.start()
    time.sleep(0.02)

    # When: one request is submitted from the test thread.
    caller_thread = threading.get_ident()
    handle = runner.submit(_request("owned"), stream=False)
    result = handle.future.result(timeout=2.0)

    # Then: idle time made no executor calls and all model work used only the owner thread.
    assert result.status is RequestStatus.FINISHED
    assert executor.calls == 3
    assert set(executor.owner_threads) == {runner.owner_thread_id}
    assert runner.owner_thread_id != caller_thread
    runner.shutdown()
    assert runner.state is RunnerState.STOPPED


def test_concurrent_requests_keep_independent_deterministic_results() -> None:
    # Given: four requests with two seed groups submitted to one owner.
    runner = _runner(RecordingExecutor(), max_active_requests=4)
    runner.start()
    handles = [
        runner.submit(_request(f"request-{index}", seed=index % 2), stream=False)
        for index in range(4)
    ]

    # When: all terminal futures are collected independently.
    results = [handle.future.result(timeout=2.0) for handle in handles]

    # Then: equal seeds match and different seeds retain distinct token sequences.
    assert results[0].generated_tokens == results[2].generated_tokens
    assert results[1].generated_tokens == results[3].generated_tokens
    assert results[0].generated_tokens != results[1].generated_tokens
    runner.shutdown()


def test_waiting_request_cancellation_releases_cache_budget() -> None:
    # Given: one active slot, a blocked active request, and a FIFO waiting request.
    executor = BlockingExecutor()
    runner = _runner(executor, max_active_requests=1, max_cached_tokens=8)
    runner.start()
    first = runner.submit(_request("active", max_new_tokens=3), stream=False)
    waiting = runner.submit(_request("waiting", max_new_tokens=2), stream=False)
    assert executor.entered.wait(timeout=2.0)
    cancel_thread = threading.Thread(target=runner.cancel, args=("waiting",))

    # When: cancellation is queued while the owner is inside a synchronous tick.
    cancel_thread.start()
    executor.release.set()
    cancel_thread.join(timeout=2.0)
    waiting_result = waiting.future.result(timeout=2.0)
    first_result = first.future.result(timeout=2.0)

    # Then: the waiting request never executes and all terminal reservations are released.
    assert not cancel_thread.is_alive()
    assert waiting_result.status is RequestStatus.CANCELLED
    assert first_result.status is RequestStatus.FINISHED
    metrics = runner.metrics()
    assert metrics.cancelled_requests == 1
    assert metrics.cached_tokens == 0
    assert metrics.reserved_cache_tokens == 0
    runner.shutdown()


def test_active_request_cancellation_is_applied_between_ticks() -> None:
    # Given: an active request blocked inside its first prefill tick.
    executor = BlockingExecutor()
    runner = _runner(executor, max_active_requests=1)
    runner.start()
    handle = runner.submit(_request("active", max_new_tokens=5), stream=False)
    assert executor.entered.wait(timeout=2.0)
    cancel_thread = threading.Thread(target=runner.cancel, args=("active",))

    # When: cancellation waits for the synchronous tick to finish.
    cancel_thread.start()
    executor.release.set()
    cancel_thread.join(timeout=2.0)
    result = handle.future.result(timeout=2.0)

    # Then: the next tick cancels the request and clears its reservation.
    assert result.status is RequestStatus.CANCELLED
    assert runner.metrics().reserved_cache_tokens == 0
    runner.shutdown()


def test_stream_backpressure_cancels_without_blocking_the_owner() -> None:
    # Given: a one-token stream buffer that the consumer does not drain.
    runner = _runner(RecordingExecutor(), stream_buffer_size=1)
    runner.start()
    handle = runner.submit(_request("slow", max_new_tokens=5), stream=True)

    # When: the producer reaches the full bounded stream queue.
    result = handle.future.result(timeout=2.0)

    # Then: deterministic cancellation releases resources and records backpressure.
    assert result.status is RequestStatus.CANCELLED
    assert any(event.event_type is RunnerEventType.BACKPRESSURE for event in runner.events)
    assert runner.metrics().reserved_cache_tokens == 0
    assert handle.stream_queue is not None
    stream_events = [handle.stream_queue.get_nowait(), handle.stream_queue.get_nowait()]
    assert [event.event_type for event in stream_events] == [
        StreamEventType.TOKEN,
        StreamEventType.BACKPRESSURE,
    ]
    assert handle.stream_queue.empty()
    runner.shutdown()


def test_executor_failure_is_isolated_from_peer_request() -> None:
    # Given: one failing request and one healthy peer admitted together.
    runner = _runner(RecordingExecutor(failing_ids={"bad"}))
    runner.start()
    bad = runner.submit(_request("bad"), stream=False)
    good = runner.submit(_request("good"), stream=False)

    # When: both terminal results are awaited.
    bad_result = bad.future.result(timeout=2.0)
    good_result = good.future.result(timeout=2.0)

    # Then: only the injected request fails and its peer completes.
    assert bad_result.status is RequestStatus.FAILED
    assert bad_result.failure_reason == "injected failure"
    assert good_result.status is RequestStatus.FINISHED
    runner.shutdown()


def test_shutdown_cancels_active_and_waiting_requests() -> None:
    # Given: active and waiting work while the owner is blocked in prefill.
    executor = BlockingExecutor()
    runner = _runner(executor, max_active_requests=1)
    runner.start()
    active = runner.submit(_request("active", max_new_tokens=10), stream=False)
    waiting = runner.submit(_request("waiting", max_new_tokens=10), stream=False)
    assert executor.entered.wait(timeout=2.0)
    shutdown_thread = threading.Thread(target=runner.shutdown)

    # When: graceful shutdown is requested and the current CPU tick returns.
    shutdown_thread.start()
    executor.release.set()
    shutdown_thread.join(timeout=3.0)

    # Then: both channels terminate as cancelled and the worker joins.
    assert not shutdown_thread.is_alive()
    assert active.future.result(timeout=1.0).status is RequestStatus.CANCELLED
    assert waiting.future.result(timeout=1.0).status is RequestStatus.CANCELLED
    assert runner.state is RunnerState.STOPPED


def test_inspection_runs_on_owner_thread_and_propagates_result() -> None:
    # Given: an idle runner and a pure observation callable.
    runner = _runner(RecordingExecutor())
    runner.start()
    observed_thread: list[int] = []

    def observe() -> tuple[str, int]:
        observed_thread.append(threading.get_ident())
        return ("observed", 7)

    # When: the caller requests an owner-thread inspection.
    result = runner.inspect(observe)

    # Then: the value returns unchanged and the callable used only the owner thread.
    assert result == ("observed", 7)
    assert observed_thread == [runner.owner_thread_id]
    runner.shutdown()


def test_queued_inspection_timeout_never_executes_stale_callable() -> None:
    # Given: the owner blocked inside model work and an inspection queued behind it.
    executor = BlockingExecutor()
    runner = _runner(executor, max_active_requests=1)
    runner.start()
    request = runner.submit(_request("blocking", max_new_tokens=2), stream=False)
    assert executor.entered.wait(timeout=2.0)
    inspection_called = threading.Event()

    def observe() -> None:
        inspection_called.set()

    # When: the inspection deadline expires before the owner can dequeue it.
    with pytest.raises(RunnerInspectionTimeoutError, match="did not finish"):
        _ = runner.inspect(observe, timeout_seconds=0.01)
    executor.release.set()
    _ = request.future.result(timeout=2.0)
    _ = runner.metrics(timeout_seconds=2.0)

    # Then: the cancelled future makes the owner skip the stale callable.
    assert not inspection_called.is_set()
    runner.shutdown()


def test_started_inspection_timeout_recovers_owner_after_callable_returns() -> None:
    # Given: an inspection that has already started synchronously on the owner.
    runner = _runner(RecordingExecutor())
    runner.start()
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def observe() -> int:
        entered.set()
        assert release.wait(timeout=2.0)
        return 11

    def invoke() -> None:
        try:
            _ = runner.inspect(observe, timeout_seconds=0.02)
        except BaseException as error:  # noqa: BLE001 - test captures the caller outcome.
            errors.append(error)

    caller = threading.Thread(target=invoke)
    caller.start()
    assert entered.wait(timeout=1.0)
    caller.join(timeout=1.0)
    assert not caller.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RunnerInspectionTimeoutError)

    # When: the already-running synchronous observation is allowed to finish.
    release.set()
    metrics = runner.metrics(timeout_seconds=2.0)

    # Then: the owner remains healthy and continues processing commands.
    assert metrics.completed_requests == 0
    runner.shutdown()
