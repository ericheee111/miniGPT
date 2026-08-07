from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import httpx
import uvicorn
from httpx import ASGITransport

from minigpt.data import CharTokenizer, JsonValue
from minigpt.engine_runner import EngineRunner, RunnerConfig, RunnerEventType, RunnerState
from minigpt.http_server import MODEL_ID, create_app
from minigpt.serving import (
    DecodeBatchObservation,
    EngineConfig,
    ExecutionResult,
    PrefillBatchEvent,
    PrefillBatchObservation,
    RequestState,
    SchedulerConfig,
    ServingEngine,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from fastapi import FastAPI


@dataclass(slots=True)
class SlowExecutor:
    """Produce valid character IDs slowly enough to observe disconnects."""

    block_size: int = 8
    delay_seconds: float = 0.01
    failing_seeds: set[int] = field(default_factory=set)

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
        return tuple(self._result(state) for state in requests)

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        return tuple(self._result(state) for state in active_requests)

    def _result(self, state: RequestState) -> ExecutionResult:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        request_id = state.request.request_id
        if state.request.seed in self.failing_seeds:
            return ExecutionResult.failure(request_id, "injected HTTP failure", 0.0)
        generated = len(state.generated_tokens)
        cache_tokens = min(self.block_size, len(state.request.prompt_tokens) + generated)
        return ExecutionResult.success(
            request_id=request_id,
            token_id=(state.request.seed + generated) % 2,
            cache=(),
            cache_tokens=cache_tokens,
            latency_seconds=0.0,
            used_fallback=False,
        )


def _app(
    *,
    stream_buffer_size: int = 8,
    delay_seconds: float = 0.01,
    failing_seeds: set[int] | None = None,
) -> tuple[FastAPI, EngineRunner]:
    executor = SlowExecutor(
        delay_seconds=delay_seconds,
        failing_seeds=set() if failing_seeds is None else failing_seeds,
    )
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(max_active_requests=4, max_cached_tokens=64),
            block_size=executor.block_size,
        ),
        executor=executor,
    )
    runner = EngineRunner(
        engine=engine,
        config=RunnerConfig(command_queue_size=32, stream_buffer_size=stream_buffer_size),
    )
    return (
        create_app(
            runner=runner,
            tokenizer=CharTokenizer.from_text("AB"),
            model_id=MODEL_ID,
            block_size=executor.block_size,
        ),
        runner,
    )


@asynccontextmanager
async def _client(app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client


def _payload(*, seed: int, stream: bool, max_tokens: int) -> dict[str, object]:
    return {
        "model": MODEL_ID,
        "prompt": "A",
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "stream": stream,
        "seed": seed,
    }


def test_http_stream_backpressure_has_error_without_done() -> None:
    asyncio.run(_check_http_backpressure())


async def _check_http_backpressure() -> None:
    # Given: an in-process service with room for only one unconsumed token.
    app, runner = _app(stream_buffer_size=1, delay_seconds=0.0)

    # When: a long stream outruns its ASGI consumer.
    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json=_payload(seed=3, stream=True, max_tokens=100),
        )
        events = runner.events

    # Then: the stream reports bounded-buffer failure, omits [DONE], and records cancellation.
    assert response.status_code == 200
    assert "stream_backpressure" in response.text
    assert "data: [DONE]" not in response.text
    assert any(event.event_type is RunnerEventType.BACKPRESSURE for event in events)
    assert any(event.event_type is RunnerEventType.CANCELLED for event in events)


def test_http_generation_failure_does_not_affect_concurrent_peer() -> None:
    asyncio.run(_check_failure_isolation())


async def _check_failure_isolation() -> None:
    # Given: one seed configured to fail beside a healthy request.
    app, _runner = _app(delay_seconds=0.0, failing_seeds={13})

    # When: both requests enter the same HTTP service concurrently.
    async with _client(app) as client:
        failed, healthy = await asyncio.gather(
            client.post("/v1/completions", json=_payload(seed=13, stream=False, max_tokens=3)),
            client.post("/v1/completions", json=_payload(seed=14, stream=False, max_tokens=3)),
        )

    # Then: only the injected request maps to a stable internal-generation failure.
    assert failed.status_code == 500
    body = cast("JsonValue", failed.json())
    assert isinstance(body, dict)
    error = body.get("error")
    assert isinstance(error, dict)
    assert error.get("code") == "generation_failed"
    assert healthy.status_code == 200


def test_real_localhost_disconnect_cancels_engine_request() -> None:
    # Given: a real Uvicorn socket serving a deliberately slow long stream.
    app, runner = _app(stream_buffer_size=32, delay_seconds=0.01)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    server_thread = threading.Thread(target=server.run, name="uvicorn-test")
    server_thread.start()
    _wait_for_server(server, port)

    try:
        # When: the client reads one token event and closes the socket early.
        with (
            httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5.0) as client,
            client.stream(
                "POST",
                "/v1/completions",
                json=_payload(seed=21, stream=True, max_tokens=1000),
            ) as response,
        ):
            assert response.status_code == 200
            first_data = next(line for line in response.iter_lines() if line.startswith("data: {"))
            parsed = cast("JsonValue", json.loads(first_data.removeprefix("data: ")))
            assert isinstance(parsed, dict)

        # Then: generator cleanup becomes engine cancellation and releases its reservation.
        _wait_for_runner_event(runner, RunnerEventType.CANCELLED)
        metrics = runner.metrics()
        assert metrics.cancelled_requests == 1
        assert metrics.cached_tokens == 0
        assert metrics.reserved_cache_tokens == 0
    finally:
        server.should_exit = True
        server_thread.join(timeout=5.0)
    assert not server_thread.is_alive()
    assert runner.state is RunnerState.STOPPED


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return cast("int", listener.getsockname()[1])


def _wait_for_server(server: uvicorn.Server, port: int) -> None:
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        if server.started:
            response = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0)
            if response.status_code == 200:
                return
        time.sleep(0.01)
    message = "Uvicorn test server did not start"
    raise AssertionError(message)


def _wait_for_runner_event(runner: EngineRunner, event_type: RunnerEventType) -> None:
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        if any(event.event_type is event_type for event in runner.events):
            return
        time.sleep(0.01)
    message = f"runner did not emit {event_type.value}"
    raise AssertionError(message)
