"""Single-owner worker for the synchronous serving engine."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypeAlias, final

from typing_extensions import override

from minigpt.serving import (
    EngineEventType,
    GenerationRequest,
    RequestMetrics,
    RequestStatus,
    ServingEngine,
    UnknownRequestError,
)

if TYPE_CHECKING:
    from minigpt.serving import EngineMetrics


class RunnerState(StrEnum):
    """Describe the worker lifecycle without exposing engine state."""

    NEW = "new"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class StreamEventType(StrEnum):
    """Identify per-request producer messages."""

    TOKEN = "token"  # noqa: S105
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BACKPRESSURE = "backpressure"


class RunnerEventType(StrEnum):
    """Identify runner-boundary lifecycle and flow-control events."""

    SUBMITTED = "submitted"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BACKPRESSURE = "backpressure"
    SHUTDOWN = "shutdown"
    WORKER_FAILED = "worker_failed"


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Bound command and per-stream memory usage."""

    command_queue_size: int = 256
    stream_buffer_size: int = 64

    def __post_init__(self) -> None:
        """Reject unusable queue capacities."""
        if isinstance(self.command_queue_size, bool) or self.command_queue_size <= 0:
            reason = "command_queue_size must be a positive integer"
            raise ValueError(reason)
        if isinstance(self.stream_buffer_size, bool) or self.stream_buffer_size <= 0:
            reason = "stream_buffer_size must be a positive integer"
            raise ValueError(reason)


@dataclass(frozen=True, slots=True)
class RunnerEvent:
    """Record an append-only cross-thread lifecycle event."""

    sequence: int
    timestamp: float
    event_type: RunnerEventType
    request_id: str | None
    detail: str | None


@dataclass(frozen=True, slots=True)
class RunnerResult:
    """Return immutable terminal request state to an HTTP consumer."""

    request_id: str
    status: RequestStatus
    generated_tokens: tuple[int, ...]
    metrics: RequestMetrics
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Carry one token or terminal signal through a bounded queue."""

    event_type: StreamEventType
    token_id: int | None = None
    result: RunnerResult | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RequestHandle:
    """Expose independent terminal and optional streaming channels."""

    request_id: str
    future: Future[RunnerResult]
    stream_queue: queue.Queue[StreamEvent] | None


@dataclass(frozen=True, slots=True)
class RunnerUnavailableError(RuntimeError):
    """Report submission to a worker that cannot accept work."""

    state: RunnerState

    @override
    def __str__(self) -> str:
        """Render the unavailable lifecycle state."""
        return f"engine runner is not accepting work in state {self.state.value!r}"


@dataclass(frozen=True, slots=True)
class RunnerQueueFullError(RuntimeError):
    """Report bounded command-queue saturation."""

    command: str

    @override
    def __str__(self) -> str:
        """Render the saturated command kind."""
        return f"engine runner command queue is full for {self.command}"


@dataclass(frozen=True, slots=True)
class RunnerShutdownTimeoutError(TimeoutError):
    """Report a worker that did not stop within the caller's bound."""

    timeout_seconds: float

    @override
    def __str__(self) -> str:
        """Render the shutdown timeout."""
        return f"engine runner did not stop within {self.timeout_seconds} seconds"


@dataclass(frozen=True, slots=True)
class RunnerWorkerError(RuntimeError):
    """Report an unexpected worker-level failure."""

    detail: str

    @override
    def __str__(self) -> str:
        """Render the isolated worker failure."""
        return f"engine runner worker failed: {self.detail}"


@dataclass(frozen=True, slots=True)
class _SubmitCommand:
    request: GenerationRequest
    handle: RequestHandle


@dataclass(frozen=True, slots=True)
class _CancelCommand:
    request_id: str
    acknowledged: Future[None]


@dataclass(frozen=True, slots=True)
class _MetricsCommand:
    result: Future[EngineMetrics]


@dataclass(frozen=True, slots=True)
class _ShutdownCommand:
    acknowledged: Future[None]


_RunnerCommand: TypeAlias = _SubmitCommand | _CancelCommand | _MetricsCommand | _ShutdownCommand


@dataclass(slots=True)
class _ChannelState:
    handle: RequestHandle
    stream_terminal_sent: bool = False
    backpressured: bool = False


@final
class EngineRunner:
    """Own every call to one synchronous ``ServingEngine`` from one worker thread."""

    def __init__(
        self,
        *,
        engine: ServingEngine,
        config: RunnerConfig | None = None,
    ) -> None:
        """Bind an engine without calling it before the worker starts."""
        self._engine = engine
        self.config = config or RunnerConfig()
        self._commands: queue.Queue[_RunnerCommand] = queue.Queue(
            maxsize=self.config.command_queue_size
        )
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._state = RunnerState.NEW
        self._state_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._events: list[RunnerEvent] = []
        self._channels: dict[str, _ChannelState] = {}
        self._engine_event_cursor = 0
        self._owner_thread_id: int | None = None

    @property
    def state(self) -> RunnerState:
        """Return the synchronized worker lifecycle state."""
        with self._state_lock:
            return self._state

    @property
    def is_running(self) -> bool:
        """Return whether new submissions may be accepted."""
        return self.state is RunnerState.RUNNING

    @property
    def owner_thread_id(self) -> int | None:
        """Expose the execution owner identity for diagnostics and tests."""
        return self._owner_thread_id

    @property
    def events(self) -> tuple[RunnerEvent, ...]:
        """Return an immutable snapshot of runner-boundary events."""
        with self._event_lock:
            return tuple(self._events)

    def start(self, *, timeout_seconds: float = 5.0) -> None:
        """Start the dedicated owner and wait until it is accepting work."""
        _validate_timeout(timeout_seconds)
        with self._state_lock:
            if self._state is RunnerState.RUNNING:
                return
            if self._state is not RunnerState.NEW:
                raise RunnerUnavailableError(self._state)
            self._thread = threading.Thread(
                target=self._run,
                name="minigpt-engine-runner",
                daemon=False,
            )
            self._thread.start()
        if not self._started.wait(timeout_seconds):
            raise RunnerShutdownTimeoutError(timeout_seconds)
        if self.state is not RunnerState.RUNNING:
            raise RunnerUnavailableError(self.state)

    def submit(self, request: GenerationRequest, *, stream: bool) -> RequestHandle:
        """Enqueue one request without calling the engine on the caller thread."""
        self._require_running()
        stream_queue = (
            queue.Queue[StreamEvent](maxsize=self.config.stream_buffer_size + 1) if stream else None
        )
        handle = RequestHandle(
            request_id=request.request_id,
            future=Future(),
            stream_queue=stream_queue,
        )
        self._put_nowait(_SubmitCommand(request=request, handle=handle), command_name="submit")
        return handle

    def cancel(self, request_id: str, *, timeout_seconds: float = 1.0) -> None:
        """Enqueue cancellation and wait only for command acceptance by the owner."""
        _validate_timeout(timeout_seconds)
        state = self.state
        if state not in {RunnerState.RUNNING, RunnerState.STOPPING}:
            raise RunnerUnavailableError(state)
        acknowledged: Future[None] = Future()
        try:
            self._commands.put(
                _CancelCommand(request_id=request_id, acknowledged=acknowledged),
                timeout=timeout_seconds,
            )
        except queue.Full:
            command_name = "cancel"
            raise RunnerQueueFullError(command_name) from None
        try:
            _ = acknowledged.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            command_name = "cancel acknowledgement"
            raise RunnerQueueFullError(command_name) from None

    def metrics(self, *, timeout_seconds: float = 5.0) -> EngineMetrics:
        """Request an owner-thread engine metrics snapshot."""
        self._require_running()
        result: Future[EngineMetrics] = Future()
        self._put_nowait(_MetricsCommand(result=result), command_name="metrics")
        try:
            return result.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            command_name = "metrics acknowledgement"
            raise RunnerQueueFullError(command_name) from None

    def shutdown(self, *, timeout_seconds: float = 10.0) -> None:
        """Cancel all live work, release reservations, and join the owner."""
        _validate_timeout(timeout_seconds)
        with self._state_lock:
            if self._state is RunnerState.NEW:
                self._state = RunnerState.STOPPED
                return
            if self._state is RunnerState.STOPPED:
                return
            if self._state is RunnerState.FAILED:
                thread = self._thread
            else:
                self._state = RunnerState.STOPPING
                thread = self._thread
        if thread is None:
            return
        if thread.ident == threading.get_ident():
            reason = "engine runner cannot join itself"
            raise RuntimeError(reason)
        if thread.is_alive() and self.state is RunnerState.STOPPING:
            acknowledged: Future[None] = Future()
            try:
                self._commands.put(
                    _ShutdownCommand(acknowledged=acknowledged),
                    timeout=timeout_seconds,
                )
            except queue.Full:
                raise RunnerShutdownTimeoutError(timeout_seconds) from None
            try:
                _ = acknowledged.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                raise RunnerShutdownTimeoutError(timeout_seconds) from None
        thread.join(timeout_seconds)
        if thread.is_alive():
            raise RunnerShutdownTimeoutError(timeout_seconds)

    def _run(self) -> None:
        self._owner_thread_id = threading.get_ident()
        with self._state_lock:
            self._state = RunnerState.RUNNING
        self._started.set()
        try:
            while True:
                if self._engine.is_idle:
                    command = self._commands.get()
                    if not self._process_command(command):
                        break
                if not self._drain_commands():
                    break
                if not self._engine.is_idle:
                    self._engine.tick()
                    self._publish_engine_events()
        except Exception as error:  # noqa: BLE001
            self._fail_worker(error)
            return
        with self._state_lock:
            self._state = RunnerState.STOPPED

    def _drain_commands(self) -> bool:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return True
            if not self._process_command(command):
                return False

    def _process_command(self, command: _RunnerCommand) -> bool:
        if isinstance(command, _SubmitCommand):
            self._process_submit(command)
            return True
        if isinstance(command, _CancelCommand):
            self._process_cancel(command)
            return True
        if isinstance(command, _MetricsCommand):
            command.result.set_result(self._engine.metrics())
            return True
        self._process_shutdown(command)
        return False

    def _process_submit(self, command: _SubmitCommand) -> None:
        state = self.state
        if state is not RunnerState.RUNNING:
            command.handle.future.set_exception(RunnerUnavailableError(state))
            return
        try:
            self._engine.submit(command.request)
        except Exception as error:  # noqa: BLE001
            command.handle.future.set_exception(error)
            self._put_stream_terminal(
                command.handle.stream_queue,
                StreamEvent(event_type=StreamEventType.FAILED, detail=str(error)),
            )
            return
        self._channels[command.request.request_id] = _ChannelState(handle=command.handle)
        self._emit(RunnerEventType.SUBMITTED, command.request.request_id)
        self._publish_engine_events()

    def _process_cancel(self, command: _CancelCommand) -> None:
        with suppress(UnknownRequestError):
            self._engine.cancel(command.request_id)
        self._emit(RunnerEventType.CANCEL_REQUESTED, command.request_id)
        command.acknowledged.set_result(None)
        self._publish_engine_events()

    def _process_shutdown(self, command: _ShutdownCommand) -> None:
        self._emit(RunnerEventType.SHUTDOWN, None)
        for request_id, channel in tuple(self._channels.items()):
            if channel.handle.future.done():
                continue
            try:
                self._engine.cancel(request_id)
            except UnknownRequestError:
                continue
        while not self._engine.is_idle:
            self._engine.tick()
            self._publish_engine_events()
        self._reject_remaining_commands()
        command.acknowledged.set_result(None)

    def _reject_remaining_commands(self) -> None:
        unavailable = RunnerUnavailableError(RunnerState.STOPPING)
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            if isinstance(command, _SubmitCommand):
                command.handle.future.set_exception(unavailable)
                self._put_stream_terminal(
                    command.handle.stream_queue,
                    StreamEvent(event_type=StreamEventType.FAILED, detail=str(unavailable)),
                )
            elif isinstance(command, _CancelCommand):
                command.acknowledged.set_result(None)
            elif isinstance(command, _MetricsCommand):
                command.result.set_exception(unavailable)
            else:
                command.acknowledged.set_result(None)

    def _publish_engine_events(self) -> None:
        events = self._engine.events
        for event in events[self._engine_event_cursor :]:
            channel = self._channels.get(event.request_id)
            if channel is None:
                continue
            if event.event_type is EngineEventType.TOKEN and event.token_id is not None:
                self._publish_token(channel, event.token_id)
            elif event.event_type in {
                EngineEventType.FINISHED,
                EngineEventType.CANCELLED,
                EngineEventType.FAILED,
            }:
                self._publish_terminal(channel, event.event_type)
        self._engine_event_cursor = len(events)

    def _publish_token(self, channel: _ChannelState, token_id: int) -> None:
        stream_queue = channel.handle.stream_queue
        if stream_queue is None or channel.stream_terminal_sent or channel.backpressured:
            return
        if stream_queue.qsize() >= self.config.stream_buffer_size:
            channel.backpressured = True
            channel.stream_terminal_sent = True
            detail = f"stream buffer reached {self.config.stream_buffer_size} tokens"
            self._emit(RunnerEventType.BACKPRESSURE, channel.handle.request_id, detail)
            with suppress(UnknownRequestError):
                self._engine.cancel(channel.handle.request_id)
            self._put_stream_terminal(
                stream_queue,
                StreamEvent(event_type=StreamEventType.BACKPRESSURE, detail=detail),
            )
            return
        stream_queue.put_nowait(StreamEvent(event_type=StreamEventType.TOKEN, token_id=token_id))

    def _publish_terminal(
        self,
        channel: _ChannelState,
        event_type: EngineEventType,
    ) -> None:
        request_id = channel.handle.request_id
        state = self._engine.request_state(request_id)
        result = RunnerResult(
            request_id=request_id,
            status=state.status,
            generated_tokens=tuple(state.generated_tokens),
            metrics=self._engine.request_metrics(request_id),
            failure_reason=state.failure_reason,
        )
        if not channel.handle.future.done():
            channel.handle.future.set_result(result)
        if event_type is EngineEventType.FINISHED:
            runner_event = RunnerEventType.COMPLETED
            stream_event = StreamEventType.FINISHED
        elif event_type is EngineEventType.CANCELLED:
            runner_event = RunnerEventType.CANCELLED
            stream_event = StreamEventType.CANCELLED
        else:
            runner_event = RunnerEventType.FAILED
            stream_event = StreamEventType.FAILED
        self._emit(runner_event, request_id, result.failure_reason)
        if channel.handle.stream_queue is not None and not channel.stream_terminal_sent:
            channel.stream_terminal_sent = True
            self._put_stream_terminal(
                channel.handle.stream_queue,
                StreamEvent(
                    event_type=stream_event,
                    result=result,
                    detail=result.failure_reason,
                ),
            )

    def _fail_worker(self, error: BaseException) -> None:
        detail = f"{type(error).__name__}: {error}"
        try:
            self._engine.release_all_cache_resources()
        except Exception as cleanup_error:  # noqa: BLE001
            detail += f"; cache cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
        self._emit(RunnerEventType.WORKER_FAILED, None, detail)
        failure = RunnerWorkerError(detail)
        for channel in self._channels.values():
            if not channel.handle.future.done():
                channel.handle.future.set_exception(failure)
            if channel.handle.stream_queue is not None and not channel.stream_terminal_sent:
                channel.stream_terminal_sent = True
                self._put_stream_terminal(
                    channel.handle.stream_queue,
                    StreamEvent(event_type=StreamEventType.FAILED, detail=detail),
                )
        self._reject_remaining_commands()
        with self._state_lock:
            self._state = RunnerState.FAILED
        self._started.set()

    def _put_nowait(self, command: _RunnerCommand, *, command_name: str) -> None:
        try:
            self._commands.put_nowait(command)
        except queue.Full:
            raise RunnerQueueFullError(command_name) from None

    def _require_running(self) -> None:
        state = self.state
        if state is not RunnerState.RUNNING:
            raise RunnerUnavailableError(state)

    def _emit(
        self,
        event_type: RunnerEventType,
        request_id: str | None,
        detail: str | None = None,
    ) -> None:
        with self._event_lock:
            self._events.append(
                RunnerEvent(
                    sequence=len(self._events),
                    timestamp=time.perf_counter(),
                    event_type=event_type,
                    request_id=request_id,
                    detail=detail,
                )
            )

    @staticmethod
    def _put_stream_terminal(
        stream_queue: queue.Queue[StreamEvent] | None,
        event: StreamEvent,
    ) -> None:
        if stream_queue is None:
            return
        try:
            stream_queue.put_nowait(event)
        except queue.Full:
            with suppress(queue.Empty):
                _ = stream_queue.get_nowait()
            with suppress(queue.Full):
                stream_queue.put_nowait(event)


def _validate_timeout(timeout_seconds: float) -> None:
    if timeout_seconds <= 0.0:
        reason = "timeout_seconds must be positive"
        raise ValueError(reason)
