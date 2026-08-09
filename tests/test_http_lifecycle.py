from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import httpx
import pytest
import torch
import uvicorn
from httpx import ASGITransport

from minigpt.data import CharTokenizer, JsonValue
from minigpt.engine_runner import EngineRunner, RunnerConfig, RunnerEventType, RunnerState
from minigpt.http_server import MODEL_ID, create_app
from minigpt.layers import LayerKVCache
from minigpt.paged_kv_cache import (
    KVCacheBackend,
    PagedKVCacheConfig,
    PagedKVCachePool,
    PrefixCacheNamespace,
)
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
        return tuple(self._prefill_result(state) for state in requests)

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        return tuple(
            self._result(state, used_fallback=state.cached_tokens >= self.block_size)
            for state in active_requests
        )

    def _result(self, state: RequestState, *, used_fallback: bool) -> ExecutionResult:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        request_id = state.request.request_id
        if state.request.seed in self.failing_seeds:
            return ExecutionResult.failure(request_id, "injected HTTP failure", 0.0)
        generated = len(state.generated_tokens)
        cache_tokens = min(self.block_size, len(state.request.prompt_tokens) + generated)
        cache_tensor = torch.full((1, 1, cache_tokens, 1), float(state.request.seed))
        return ExecutionResult.success(
            request_id=request_id,
            token_id=(state.request.seed + generated) % 2,
            cache=(LayerKVCache(key=cache_tensor, value=cache_tensor.clone()),),
            cache_tokens=cache_tokens,
            latency_seconds=0.0,
            used_fallback=used_fallback,
        )

    def _prefill_result(self, state: RequestState) -> ExecutionResult:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        request_id = state.request.request_id
        if state.request.seed in self.failing_seeds:
            return ExecutionResult.failure(request_id, "injected HTTP failure", 0.0)
        prompt_tokens = len(state.request.prompt_tokens)
        suffix_tokens = prompt_tokens - state.prefix_hit_tokens
        cache_tensor = torch.full((1, 1, suffix_tokens, 1), float(state.request.seed))
        cache = (
            (LayerKVCache(key=cache_tensor, value=cache_tensor.clone()),) if suffix_tokens else ()
        )
        return ExecutionResult.success(
            request_id=request_id,
            token_id=state.request.seed % 2,
            cache=cache,
            cache_tokens=prompt_tokens,
            latency_seconds=0.0,
            used_fallback=False,
            prefill_prefix_tokens=state.prefix_hit_tokens,
            prefill_logits=torch.zeros((suffix_tokens, 2)),
        )


def _app(
    *,
    stream_buffer_size: int = 8,
    delay_seconds: float = 0.01,
    failing_seeds: set[int] | None = None,
    kv_cache_backend: KVCacheBackend = KVCacheBackend.DENSE,
    prefix_cache_mode: Literal["disabled", "enabled"] = "disabled",
) -> tuple[FastAPI, EngineRunner]:
    executor = SlowExecutor(
        delay_seconds=delay_seconds,
        failing_seeds=set() if failing_seeds is None else failing_seeds,
    )
    paged_config = (
        PagedKVCacheConfig(block_tokens=2, num_blocks=32)
        if kv_cache_backend is KVCacheBackend.PAGED
        else None
    )
    paged_pool = (
        PagedKVCachePool(
            paged_config,
            n_layer=1,
            n_head=1,
            head_size=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
            prefix_cache_namespace=(
                PrefixCacheNamespace(
                    model_checkpoint_identity="http-lifecycle-checkpoint",
                    model_config_identity="http-lifecycle-model",
                    dtype="torch.float32",
                    device="cpu",
                    block_tokens=2,
                    cache_schema_version=1,
                    position_embedding_semantics="learned_absolute_v1",
                )
                if prefix_cache_mode == "enabled"
                else None
            ),
        )
        if paged_config is not None
        else None
    )
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(max_active_requests=4, max_cached_tokens=64),
            block_size=executor.block_size,
            kv_cache_backend=kv_cache_backend,
            paged_kv_cache=paged_config,
        ),
        executor=executor,
        paged_cache_pool=paged_pool,
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


def _payload(*, seed: int, stream: bool, max_tokens: int, prompt: str = "A") -> dict[str, object]:
    return {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "stream": stream,
        "seed": seed,
    }


def test_http_stream_backpressure_has_error_without_done() -> None:
    asyncio.run(_check_http_backpressure())


async def _check_http_backpressure() -> None:
    # Given: an in-process service with room for only one unconsumed token.
    app, runner = _app(
        stream_buffer_size=1,
        delay_seconds=0.0,
        kv_cache_backend=KVCacheBackend.DENSE,
    )

    # When: a long stream outruns its ASGI consumer.
    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json=_payload(seed=3, stream=True, max_tokens=100),
        )
        events = runner.events
        metrics = runner.metrics()

    # Then: the stream reports bounded-buffer failure, omits [DONE], and records cancellation.
    assert response.status_code == 200
    assert "stream_backpressure" in response.text
    assert "data: [DONE]" not in response.text
    assert any(event.event_type is RunnerEventType.BACKPRESSURE for event in events)
    assert any(event.event_type is RunnerEventType.CANCELLED for event in events)
    assert metrics.allocated_blocks == 0
    assert metrics.reserved_blocks == 0
    assert metrics.free_blocks == metrics.total_blocks


def test_prefix_cache_http_completion_and_failure_release_active_refs() -> None:
    asyncio.run(_check_prefix_cache_http_cleanup())


async def _check_prefix_cache_http_cleanup() -> None:
    # Given: an APC service primed with one immutable two-token prompt block.
    app, runner = _app(
        stream_buffer_size=128,
        delay_seconds=0.0,
        failing_seeds={13},
        kv_cache_backend=KVCacheBackend.PAGED,
        prefix_cache_mode="enabled",
    )

    async with _client(app) as client:
        prime = await client.post(
            "/v1/completions",
            json=_payload(seed=3, stream=False, max_tokens=1, prompt="AB"),
        )
        assert prime.status_code == 200

        # When: an exact-hit stream completes and another exact-hit request fails.
        streamed = await client.post(
            "/v1/completions",
            json=_payload(seed=5, stream=True, max_tokens=100, prompt="AB"),
        )
        failed = await client.post(
            "/v1/completions",
            json=_payload(seed=13, stream=False, max_tokens=3, prompt="AB"),
        )
        metrics = runner.metrics()

        # Then: both terminal paths decref without freeing the resident canonical block.
        assert "data: [DONE]" in streamed.text
        assert failed.status_code == 500
        assert metrics.prefix_hit_requests == 2
        assert metrics.active_shared_references == 0
        assert metrics.allocated_blocks == metrics.prefix_cache_blocks
        assert metrics.reserved_blocks == 0
        assert metrics.prefix_cache_blocks == 1


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


@pytest.mark.parametrize("kv_cache_backend", list(KVCacheBackend))
def test_real_localhost_disconnect_cancels_engine_request(
    kv_cache_backend: KVCacheBackend,
) -> None:
    # Given: a real Uvicorn socket serving a deliberately slow long stream.
    app, runner = _app(
        stream_buffer_size=32,
        delay_seconds=0.01,
        kv_cache_backend=kv_cache_backend,
    )
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
        assert metrics.allocated_blocks == 0
        assert metrics.reserved_blocks == 0
        assert metrics.free_blocks == metrics.total_blocks
    finally:
        server.should_exit = True
        server_thread.join(timeout=5.0)
    assert not server_thread.is_alive()
    assert runner.state is RunnerState.STOPPED


def test_prefix_cache_localhost_disconnect_decrefs_shared_blocks() -> None:
    # Given: a real APC service with one canonical two-token prompt block.
    app, runner = _app(
        stream_buffer_size=32,
        delay_seconds=0.01,
        kv_cache_backend=KVCacheBackend.PAGED,
        prefix_cache_mode="enabled",
    )
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    server_thread = threading.Thread(target=server.run, name="uvicorn-apc-test")
    server_thread.start()
    _wait_for_server(server, port)

    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5.0) as client:
            prime = client.post(
                "/v1/completions",
                json=_payload(seed=3, stream=False, max_tokens=1, prompt="AB"),
            )
            assert prime.status_code == 200

            # When: an exact-prefix-hit stream closes after its first token.
            with client.stream(
                "POST",
                "/v1/completions",
                json=_payload(seed=21, stream=True, max_tokens=1000, prompt="AB"),
            ) as response:
                assert response.status_code == 200
                _ = next(line for line in response.iter_lines() if line.startswith("data: {"))

        # Then: disconnect cancellation decrefs the shared prefix without evicting its cache entry.
        _wait_for_runner_event(runner, RunnerEventType.CANCELLED)
        metrics = runner.metrics()
        assert metrics.prefix_hit_requests == 1
        assert metrics.active_shared_references == 0
        assert metrics.allocated_blocks == metrics.prefix_cache_blocks
        assert metrics.reserved_blocks == 0
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
