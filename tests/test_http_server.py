from __future__ import annotations

import asyncio
import json
import logging
import math
import queue
import threading
import time
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast, final

import httpx
import pytest
import torch
from httpx import ASGITransport
from starlette.requests import ClientDisconnect

from minigpt import __version__
from minigpt import prediction as prediction_reference
from minigpt.data import CharTokenizer, JsonValue
from minigpt.engine_runner import (
    EngineRunner,
    RequestHandle,
    RunnerConfig,
    RunnerQueueFullError,
    RunnerResult,
    StreamEvent,
    StreamEventType,
)
from minigpt.http_server import MODEL_ID, create_app
from minigpt.model import GPT
from minigpt.public_demo import (
    InvalidPublicDemoConfigError,
    PublicDemoInfo,
    PublicDemoPolicy,
    create_public_demo_app,
    validate_bind_host,
)
from minigpt.serving import (
    ContinuousExecutor,
    EngineConfig,
    GenerationRequest,
    RequestMetrics,
    RequestStatus,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.settings import GPTConfig
from minigpt.tokenizer import BPETokenizer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from fastapi import FastAPI
    from starlette.types import Message, Scope

_BLOCK_SIZE = 8
_BASE_URL = "http://test"
_DATA_PREFIX = "data: "


def build_app() -> FastAPI:
    _ = torch.default_generator.manual_seed(1234)
    config = GPTConfig(
        vocab_size=2,
        block_size=_BLOCK_SIZE,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
    )
    model = GPT(config).eval()
    tokenizer = CharTokenizer.from_text("AB")
    executor = ContinuousExecutor(model)
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(
                max_active_requests=4,
                max_cached_tokens=64,
            ),
            block_size=_BLOCK_SIZE,
        ),
        executor=executor,
    )
    runner = EngineRunner(
        engine=engine,
        config=RunnerConfig(command_queue_size=64, stream_buffer_size=8),
    )
    return create_app(
        runner=runner,
        tokenizer=tokenizer,
        model_id=MODEL_ID,
        block_size=_BLOCK_SIZE,
    )


_PUBLIC_ORIGIN = "https://portfolio.example"
_PUBLIC_POLICY = PublicDemoPolicy(
    enabled=True,
    streaming_enabled=True,
    allowed_origins=(_PUBLIC_ORIGIN,),
)


@final
class _ControlledRunner:
    def __init__(
        self,
        *,
        failure: BaseException | None = None,
        submit_failure: BaseException | None = None,
        cancel_delay_seconds: float = 0.0,
    ) -> None:
        self.running = False
        self.failure = failure
        self.submit_failure = submit_failure
        self.cancel_delay_seconds = cancel_delay_seconds
        self.handles: list[RequestHandle] = []
        self.cancelled_request_ids: list[str] = []

    @property
    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True

    def shutdown(self) -> None:
        self.running = False
        for handle in self.handles:
            if not handle.future.done():
                _ = handle.future.cancel()

    def submit(self, request: GenerationRequest, *, stream: bool) -> RequestHandle:
        if self.submit_failure is not None:
            raise self.submit_failure
        future: Future[RunnerResult] = Future()
        if self.failure is not None:
            future.set_exception(self.failure)
        stream_queue: queue.Queue[StreamEvent] | None = queue.Queue() if stream else None
        handle = RequestHandle(
            request_id=request.request_id,
            future=future,
            stream_queue=stream_queue,
        )
        self.handles.append(handle)
        return handle

    def cancel(self, request_id: str, *, timeout_seconds: float = 1.0) -> None:
        del timeout_seconds
        if self.cancel_delay_seconds > 0:
            time.sleep(self.cancel_delay_seconds)
        self.cancelled_request_ids.append(request_id)
        for handle in self.handles:
            if handle.request_id == request_id and not handle.future.done():
                _ = handle.future.cancel()


def _public_info() -> PublicDemoInfo:
    return PublicDemoInfo(
        project_version=__version__,
        model_id=MODEL_ID,
        executor_name="continuous",
        kv_cache_backend="dense",
        prefix_cache_enabled=False,
    )


def _build_public_app(
    policy: PublicDemoPolicy = _PUBLIC_POLICY,
) -> tuple[FastAPI, EngineRunner]:
    _ = torch.default_generator.manual_seed(1234)
    model = GPT(
        GPTConfig(
            vocab_size=2,
            block_size=_BLOCK_SIZE,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
        )
    ).eval()
    runner = EngineRunner(
        engine=ServingEngine(
            config=EngineConfig(
                scheduler=SchedulerConfig(
                    max_active_requests=policy.max_concurrent_requests,
                    max_cached_tokens=128,
                ),
                block_size=_BLOCK_SIZE,
            ),
            executor=ContinuousExecutor(model),
        ),
        config=RunnerConfig(command_queue_size=64, stream_buffer_size=8),
    )
    app = create_public_demo_app(
        runner=runner,
        tokenizer=CharTokenizer.from_text("AB"),
        block_size=_BLOCK_SIZE,
        policy=policy,
        info=_public_info(),
    )
    return app, runner


def _build_controlled_public_app(
    *,
    policy: PublicDemoPolicy,
    failure: BaseException | None = None,
    cancel_delay_seconds: float = 0.0,
) -> tuple[FastAPI, _ControlledRunner]:
    runner = _ControlledRunner(
        failure=failure,
        cancel_delay_seconds=cancel_delay_seconds,
    )
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=CharTokenizer.from_text("AB"),
        block_size=_BLOCK_SIZE,
        policy=policy,
        info=_public_info(),
    )
    return app, runner


@asynccontextmanager
async def _test_client(app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Enter the FastAPI lifespan manually and yield an in-process HTTP client."""
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url=_BASE_URL,
        ) as client,
    ):
        yield client


def _completion_payload(
    *,
    prompt: str = "A",
    max_tokens: int = 3,
    stream: bool = False,
    seed: int = 42,
) -> dict[str, object]:
    return {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "stream": stream,
        "seed": seed,
    }


def _response_json(response: httpx.Response) -> JsonValue:
    return cast("JsonValue", response.json())


def _response_body(response: httpx.Response) -> dict[str, JsonValue]:
    body = _response_json(response)
    assert isinstance(body, dict)
    return body


def _error_code(body: JsonValue) -> str:
    assert isinstance(body, dict)
    error = body.get("error")
    assert isinstance(error, dict)
    code = error.get("code")
    assert isinstance(code, str)
    return code


def _parse_sse_data(lines: list[str]) -> list[dict[str, JsonValue] | None]:
    """Extract parsed JSON chunks and the [DONE] sentinel from SSE data lines."""
    chunks: list[dict[str, JsonValue] | None] = []
    for line in lines:
        if not line.startswith(_DATA_PREFIX):
            continue
        payload = line.removeprefix(_DATA_PREFIX)
        if payload == "[DONE]":
            chunks.append(None)
            continue
        decoded = cast("JsonValue", json.loads(payload))
        assert isinstance(decoded, dict)
        chunks.append(decoded)
    return chunks


def _choice_text(chunk: dict[str, JsonValue]) -> str:
    choices = chunk.get("choices")
    assert isinstance(choices, list)
    assert len(choices) >= 1
    choice = choices[0]
    assert isinstance(choice, dict)
    text = choice.get("text")
    assert isinstance(text, str)
    return text


def _choice_finish_reason(chunk: dict[str, JsonValue]) -> str | None:
    choices = chunk.get("choices")
    assert isinstance(choices, list)
    assert len(choices) >= 1
    choice = choices[0]
    assert isinstance(choice, dict)
    reason = choice.get("finish_reason")
    if reason is None:
        return None
    assert isinstance(reason, str)
    return reason


# --- Contract 1: GET /healthz ---


def test_healthz_returns_ok_status() -> None:
    asyncio.run(_check_healthz())


async def _check_healthz() -> None:
    # Given: a Stage 12 HTTP app built from a tiny CPU GPT.
    app = build_app()

    # When: a client requests GET /healthz.
    async with _test_client(app) as client:
        response = await client.get("/healthz")

    # Then: the response is 200 with a {"status": "ok"} body.
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- Contract 2: GET /v1/models ---


def test_models_endpoint_lists_minigpt_char_model() -> None:
    asyncio.run(_check_models())


async def _check_models() -> None:
    # Given: a Stage 12 HTTP app exposing the minigpt-char model.
    app = build_app()

    # When: a client requests GET /v1/models.
    async with _test_client(app) as client:
        response = await client.get("/v1/models")

    # Then: the response lists exactly one model with the expected id, object, and owner.
    assert response.status_code == 200
    body = _response_body(response)
    models = cast("list[dict[str, object]]", body.get("data"))
    assert isinstance(models, list)
    assert len(models) == 1
    model_entry = models[0]
    assert model_entry["id"] == MODEL_ID
    assert model_entry["object"] == "model"
    assert model_entry["owned_by"] == "minigpt"


# --- Contract 3: POST /v1/completions (non-stream) ---


def test_non_stream_completion_returns_expected_envelope() -> None:
    asyncio.run(_check_non_stream_completion())


async def _check_non_stream_completion() -> None:
    # Given: a Stage 12 app and a fixed completion request with seed 42.
    app = build_app()
    payload = _completion_payload(prompt="A", max_tokens=3, seed=42)

    # When: the client posts a non-streaming completion.
    async with _test_client(app) as client:
        response = await client.post("/v1/completions", json=payload)

    # Then: the response envelope matches the OpenAI-style text completion contract.
    assert response.status_code == 200
    body = _response_body(response)
    response_id = cast("str", body.get("id"))
    assert response_id.startswith("cmpl-")
    assert body.get("object") == "text_completion"
    assert body.get("model") == MODEL_ID
    assert isinstance(body.get("created"), int)
    choices = cast("list[dict[str, object]]", body.get("choices"))
    assert len(choices) == 1
    choice = choices[0]
    assert choice.get("index") == 0
    assert len(cast("str", choice.get("text"))) == 3
    assert choice.get("finish_reason") == "length"
    usage = cast("dict[str, object]", body.get("usage"))
    assert usage.get("prompt_tokens") == 1
    assert usage.get("completion_tokens") == 3
    assert usage.get("total_tokens") == 4


# --- Contract 4: Same request twice ---


def test_repeated_request_with_same_seed_produces_identical_text() -> None:
    asyncio.run(_check_repeated_request())


async def _check_repeated_request() -> None:
    # Given: a Stage 12 app and a fixed completion request.
    app = build_app()
    payload = _completion_payload(seed=42, max_tokens=3)

    # When: the same request is posted twice.
    async with _test_client(app) as client:
        first = await client.post("/v1/completions", json=payload)
        second = await client.post("/v1/completions", json=payload)

    # Then: both responses carry identical generated text.
    assert first.status_code == 200
    assert second.status_code == 200
    first_text = _choice_text(_response_body(first))
    second_text = _choice_text(_response_body(second))
    assert first_text == second_text


# --- Contract 5: Stream completion ---


def test_stream_completion_emits_token_chunks_then_done() -> None:
    asyncio.run(_check_stream_completion())


async def _check_stream_completion() -> None:
    # Given: a Stage 12 app and matching stream/non-stream requests with seed 42.
    app = build_app()

    # When: the client requests a streaming completion and a non-streaming one.
    async with _test_client(app) as client:
        non_stream = await client.post(
            "/v1/completions",
            json=_completion_payload(seed=42, max_tokens=3, stream=False),
        )
        assert non_stream.status_code == 200
        non_stream_body = _response_body(non_stream)
        expected_text = cast(
            "str",
            cast("list[dict[str, object]]", non_stream_body.get("choices"))[0].get("text"),
        )

        stream_lines: list[str] = []
        async with client.stream(
            "POST",
            "/v1/completions",
            json=_completion_payload(seed=42, max_tokens=3, stream=True),
        ) as response:
            assert response.status_code == 200
            stream_lines.extend([line async for line in response.aiter_lines()])

    # Then: SSE data lines contain token chunks, a final chunk, and exactly [DONE].
    data_chunks = _parse_sse_data(stream_lines)
    assert data_chunks.count(None) == 1
    json_chunks = [chunk for chunk in data_chunks if chunk is not None]
    token_chunks = [chunk for chunk in json_chunks if _choice_finish_reason(chunk) is None]
    final_chunks = [chunk for chunk in json_chunks if _choice_finish_reason(chunk) is not None]
    assert len(token_chunks) == 3
    assert len(final_chunks) == 1

    # Token chunks carry null finish_reason and no usage field.
    for chunk in token_chunks:
        assert _choice_finish_reason(chunk) is None
        assert "usage" not in chunk

    # The final chunk carries finish_reason "length" and usage totals.
    final_chunk = final_chunks[0]
    assert _choice_finish_reason(final_chunk) == "length"
    assert "usage" in final_chunk

    # Concatenated token text matches the non-stream result.
    stream_text = "".join(_choice_text(chunk) for chunk in token_chunks)
    assert stream_text == expected_text


# --- Contract 6: Unknown model ---


def test_unknown_model_returns_404_model_not_found() -> None:
    asyncio.run(_check_unknown_model())


async def _check_unknown_model() -> None:
    # Given: a Stage 12 app and a request for a model that is not served.
    app = build_app()
    payload = {**_completion_payload(), "model": "unknown-model"}

    # When: the client posts the completion request.
    async with _test_client(app) as client:
        response = await client.post("/v1/completions", json=payload)

    # Then: the response is 404 with a stable error envelope and model_not_found code.
    assert response.status_code == 404
    assert _error_code(_response_json(response)) == "model_not_found"


# --- Contract 7: Prompt too long ---


def test_prompt_exceeding_block_size_returns_400_prompt_too_long() -> None:
    asyncio.run(_check_prompt_too_long())


async def _check_prompt_too_long() -> None:
    # Given: a Stage 12 app with block_size 8 and a 9-character prompt.
    app = build_app()
    payload = _completion_payload(prompt="ABABABABA")

    # When: the client posts a prompt longer than the model block_size.
    async with _test_client(app) as client:
        response = await client.post("/v1/completions", json=payload)

    # Then: the response is 400 with a stable error envelope and prompt_too_long code.
    assert response.status_code == 400
    assert _error_code(_response_json(response)) == "prompt_too_long"


# --- Contract 8: Empty prompt ---


def test_empty_prompt_returns_422_invalid_request() -> None:
    asyncio.run(_check_empty_prompt())


async def _check_empty_prompt() -> None:
    # Given: a Stage 12 app and a request with an empty prompt.
    app = build_app()
    payload = _completion_payload(prompt="")

    # When: the client posts the empty-prompt completion request.
    async with _test_client(app) as client:
        response = await client.post("/v1/completions", json=payload)

    # Then: the response is 422 with a stable error envelope and invalid_request code.
    assert response.status_code == 422
    assert _error_code(_response_json(response)) == "invalid_request"


# --- Contract 9: Unsupported field ---


def test_unsupported_field_returns_422_invalid_request() -> None:
    asyncio.run(_check_unsupported_field())


async def _check_unsupported_field() -> None:
    # Given: a Stage 12 app and a request carrying an unsupported top_p field.
    app = build_app()
    payload = {**_completion_payload(), "top_p": 0.9}

    # When: the client posts the request with the unsupported field.
    async with _test_client(app) as client:
        response = await client.post("/v1/completions", json=payload)

    # Then: the response is 422 with a stable error envelope and invalid_request code.
    assert response.status_code == 422
    assert _error_code(_response_json(response)) == "invalid_request"


def test_http_result_matches_direct_serving_engine() -> None:
    asyncio.run(_check_direct_equivalence())


async def _check_direct_equivalence() -> None:
    # Given: identical tiny models behind a direct engine and the HTTP app.
    expected = _direct_completion(seed=73, max_tokens=5)
    app = build_app()

    # When: HTTP serves the same prompt, generation length, and request seed.
    async with _test_client(app) as client:
        response = await client.post(
            "/v1/completions",
            json=_completion_payload(seed=73, max_tokens=5),
        )

    # Then: the HTTP boundary preserves the direct engine's decoded tokens.
    assert response.status_code == 200
    assert _choice_text(_response_body(response)) == expected


def test_concurrent_http_requests_preserve_independent_rng() -> None:
    asyncio.run(_check_concurrent_requests())


async def _check_concurrent_requests() -> None:
    # Given: eight concurrent requests split across two fixed seeds.
    app = build_app()
    seeds = [100 + index % 2 for index in range(8)]

    # When: all requests are posted concurrently through the ASGI boundary.
    async with _test_client(app) as client:
        responses = await asyncio.gather(
            *(
                client.post(
                    "/v1/completions",
                    json=_completion_payload(seed=seed, max_tokens=5),
                )
                for seed in seeds
            )
        )

    # Then: every request succeeds and equal seeds match regardless of peers.
    assert all(response.status_code == 200 for response in responses)
    texts = [_choice_text(_response_body(response)) for response in responses]
    assert len(set(texts[::2])) == 1
    assert len(set(texts[1::2])) == 1
    assert texts[0] != texts[1]


def _direct_completion(*, seed: int, max_tokens: int) -> str:
    _ = torch.default_generator.manual_seed(1234)
    tokenizer = CharTokenizer.from_text("AB")
    model = GPT(
        GPTConfig(
            vocab_size=2,
            block_size=_BLOCK_SIZE,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
        )
    ).eval()
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(max_active_requests=4, max_cached_tokens=64),
            block_size=_BLOCK_SIZE,
        ),
        executor=ContinuousExecutor(model),
    )
    request_id = "direct"
    engine.submit(
        GenerationRequest(
            request_id=request_id,
            prompt_tokens=tuple(tokenizer.encode("A")),
            max_new_tokens=max_tokens,
            seed=seed,
            arrival_time=time.perf_counter(),
        )
    )
    while not engine.is_idle:
        engine.tick()
    return tokenizer.decode(engine.request_state(request_id).generated_tokens)


def test_public_demo_bind_and_origin_configuration_fail_closed() -> None:
    # Given: the default safe bind policy and explicit CORS origins.
    policy = PublicDemoPolicy()

    # When/Then: wildcard CORS and an implicit public bind are rejected.
    with pytest.raises(InvalidPublicDemoConfigError, match="non-wildcard"):
        _ = replace(policy, allowed_origins=("*",))
    with pytest.raises(InvalidPublicDemoConfigError, match="only bind loopback"):
        validate_bind_host(
            "0.0.0.0",  # noqa: S104 - deliberate rejected bind test
            unsafe_allow_non_loopback=False,
        )

    # And: loopback is accepted and the explicit unsafe escape hatch remains separate.
    validate_bind_host(
        "127.0.0.1",
        unsafe_allow_non_loopback=False,
    )
    validate_bind_host(
        "0.0.0.0",  # noqa: S104 - deliberate explicit unsafe bind test
        unsafe_allow_non_loopback=True,
    )


def test_public_demo_cors_allows_only_configured_origin_and_preflight() -> None:
    asyncio.run(_check_public_demo_cors())


async def _check_public_demo_cors() -> None:
    # Given: a public demo with one exact HTTPS browser origin.
    app, _runner = _build_public_app()

    # When: allowed, denied, and preflight requests cross the boundary.
    async with _test_client(app) as client:
        allowed = await client.get("/demo/info", headers={"Origin": _PUBLIC_ORIGIN})
        denied = await client.get(
            "/demo/info",
            headers={"Origin": "https://attacker.example"},
        )
        preflight = await client.options(
            "/v1/completions",
            headers={
                "Origin": _PUBLIC_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        legacy_header = await client.options(
            "/v1/completions",
            headers={
                "Origin": _PUBLIC_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type, ngrok-skip-browser-warning",
            },
        )

    # Then: only the configured origin receives CORS authorization.
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == _PUBLIC_ORIGIN
    assert denied.status_code == 403
    assert "access-control-allow-origin" not in denied.headers
    assert preflight.status_code == 204
    assert preflight.headers["access-control-allow-origin"] == _PUBLIC_ORIGIN
    assert preflight.headers["access-control-allow-methods"] == "GET, POST, OPTIONS"
    assert preflight.headers["access-control-allow-headers"] == "content-type"
    assert legacy_header.status_code == 403


def test_public_demo_rejects_body_prompt_and_generation_limit_violations() -> None:
    asyncio.run(_check_public_demo_request_limits())


async def _check_public_demo_request_limits() -> None:
    # Given: small independent body, character, token, generation, and temperature limits.
    body_app, _runner = _build_public_app(replace(_PUBLIC_POLICY, max_request_body_bytes=32))
    character_app, _runner = _build_public_app(replace(_PUBLIC_POLICY, max_prompt_characters=1))
    token_app, _runner = _build_public_app(
        replace(
            _PUBLIC_POLICY,
            max_prompt_characters=8,
            max_prompt_tokens=1,
        )
    )
    generation_app, _runner = _build_public_app(
        replace(
            _PUBLIC_POLICY,
            max_new_tokens=2,
            min_temperature=0.5,
            max_temperature=1.0,
        )
    )

    # When: each limit is exceeded at its public HTTP boundary.
    async with _test_client(body_app) as client:
        body_response = await client.post(
            "/v1/completions",
            content=b"{" + b"x" * 64,
            headers={"Content-Type": "application/json"},
        )
    async with _test_client(character_app) as client:
        character_response = await client.post(
            "/v1/completions",
            json=_completion_payload(prompt="AB", max_tokens=0),
        )
    async with _test_client(token_app) as client:
        token_response = await client.post(
            "/v1/completions",
            json=_completion_payload(prompt="AB", max_tokens=0),
        )
    async with _test_client(generation_app) as client:
        token_count_response = await client.post(
            "/v1/completions",
            json=_completion_payload(max_tokens=3),
        )
        cold_response = await client.post(
            "/v1/completions",
            json={**_completion_payload(max_tokens=0), "temperature": 0.4},
        )
        hot_response = await client.post(
            "/v1/completions",
            json={**_completion_payload(max_tokens=0), "temperature": 1.1},
        )

    # Then: each response fails with a bounded, non-traceback error.
    assert body_response.status_code == 413
    assert _error_code(_response_json(body_response)) == "request_body_too_large"
    assert character_response.status_code == 400
    assert _error_code(_response_json(character_response)) == "prompt_too_long"
    assert token_response.status_code == 400
    assert _error_code(_response_json(token_response)) == "prompt_too_long"
    assert token_count_response.status_code == 400
    assert cold_response.status_code == 400
    assert hot_response.status_code == 400


def test_public_demo_enforces_ip_independent_global_quotas() -> None:
    asyncio.run(_check_public_demo_global_quotas())


async def _check_public_demo_global_quotas() -> None:
    # Given: separate apps with strict hourly request and daily generated-token quotas.
    request_app, _runner = _build_public_app(
        replace(
            _PUBLIC_POLICY,
            global_requests_per_hour=2,
        )
    )
    token_app, _runner = _build_public_app(
        replace(
            _PUBLIC_POLICY,
            max_new_tokens=1,
            global_generated_tokens_per_day=1,
        )
    )

    # When: spoofed XFF values rotate and zero-token output precedes one generated token.
    async with _test_client(request_app) as client:
        request_responses = [
            await client.post(
                "/v1/completions",
                json=_completion_payload(max_tokens=0),
                headers={"X-Forwarded-For": f"192.0.2.{index}"},
            )
            for index in range(1, 4)
        ]
    async with _test_client(token_app) as client:
        zero_token = await client.post(
            "/v1/completions",
            json=_completion_payload(max_tokens=0),
        )
        one_token = await client.post(
            "/v1/completions",
            json=_completion_payload(max_tokens=1),
        )
        token_limited = await client.post(
            "/v1/completions",
            json=_completion_payload(max_tokens=1),
        )

    # Then: XFF cannot bypass the global bucket and quota counts actual generated output.
    assert [response.status_code for response in request_responses] == [200, 200, 429]
    assert [zero_token.status_code, one_token.status_code, token_limited.status_code] == [
        200,
        200,
        429,
    ]
    assert int(request_responses[-1].headers["retry-after"]) >= 1
    assert int(token_limited.headers["retry-after"]) >= 1


def test_public_demo_counts_only_valid_runner_accepted_requests() -> None:
    asyncio.run(_check_public_demo_accepted_request_accounting())


async def _check_public_demo_accepted_request_accounting() -> None:
    # Given: a one-request global quota.
    app, runner = _build_public_app(replace(_PUBLIC_POLICY, global_requests_per_hour=1))

    # When: an invalid Prompt is followed by two otherwise valid completions.
    async with _test_client(app) as client:
        invalid = await client.post(
            "/v1/completions",
            json=_completion_payload(prompt="Z", max_tokens=0),
        )
        accepted = await client.post(
            "/v1/completions",
            json=_completion_payload(max_tokens=0),
        )
        limited = await client.post(
            "/v1/completions",
            json=_completion_payload(max_tokens=0),
        )
        completed_requests = runner.metrics().completed_requests

    # Then: validation consumes no quota and only one request reaches the runner.
    assert [invalid.status_code, accepted.status_code, limited.status_code] == [400, 200, 429]
    assert completed_requests == 1


def test_public_demo_streaming_disabled_fails_closed_before_submission() -> None:
    asyncio.run(_check_public_demo_streaming_disabled())


async def _check_public_demo_streaming_disabled() -> None:
    # Given: an explicit non-streaming public policy.
    policy = replace(_PUBLIC_POLICY, streaming_enabled=False)
    app, runner = _build_controlled_public_app(policy=policy)

    # When: metadata is read and a direct client still asks for SSE.
    async with _test_client(app) as client:
        info = _response_body(await client.get("/demo/info"))
        response = await client.post(
            "/v1/completions",
            json=_completion_payload(stream=True),
        )

    # Then: the UI contract is false and no model request is submitted.
    assert info["streaming_enabled"] is False
    assert response.status_code == 400
    assert _error_code(_response_json(response)) == "streaming_disabled"
    assert runner.handles == []


async def _wait_until(predicate: object, *, timeout_seconds: float = 1.0) -> None:
    check = cast("Callable[[], bool]", predicate)
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while not check():
        if asyncio.get_running_loop().time() >= deadline:
            reason = "condition did not become true before the test deadline"
            raise AssertionError(reason)
        await asyncio.sleep(0.001)


def test_public_demo_queue_full_and_task_cancellation_release_capacity() -> None:
    asyncio.run(_check_public_demo_queue_capacity())


async def _check_public_demo_queue_capacity() -> None:
    # Given: one active slot, one waiter slot, and a runner that stays pending.
    policy = replace(
        _PUBLIC_POLICY,
        max_concurrent_requests=1,
        max_queue_size=1,
        request_timeout_seconds=10.0,
    )
    app, runner = _build_controlled_public_app(policy=policy)

    # When: three requests arrive before model work completes.
    async with _test_client(app) as client:
        first = asyncio.create_task(client.post("/v1/completions", json=_completion_payload()))
        await _wait_until(lambda: len(runner.handles) == 1)
        second = asyncio.create_task(client.post("/v1/completions", json=_completion_payload()))
        queued = 0
        queue_deadline = asyncio.get_running_loop().time() + 1.0
        while queued != 1:
            metrics_response = await client.get("/demo/metrics")
            queued = cast("int", _response_body(metrics_response)["queued_requests"])
            assert asyncio.get_running_loop().time() < queue_deadline
            await asyncio.sleep(0.001)
        third = await client.post("/v1/completions", json=_completion_payload())

        # Then: the third request is rejected, and cancelling both predecessors drains capacity.
        assert third.status_code == 429
        assert third.headers["retry-after"] == "1"
        _ = first.cancel()
        _ = await asyncio.gather(first, return_exceptions=True)
        await _wait_until(lambda: len(runner.handles) == 2)
        _ = second.cancel()
        _ = await asyncio.gather(second, return_exceptions=True)
        metrics = _response_body(await client.get("/demo/metrics"))
        assert metrics["active_requests"] == 0
        assert metrics["queued_requests"] == 0


def test_public_demo_timeout_cancels_runner_and_releases_capacity() -> None:
    asyncio.run(_check_public_demo_timeout())


async def _check_public_demo_timeout() -> None:
    # Given: a pending runner behind a short public request deadline.
    policy = replace(
        _PUBLIC_POLICY,
        max_new_tokens=3,
        global_generated_tokens_per_day=3,
        request_timeout_seconds=0.02,
    )
    app, runner = _build_controlled_public_app(policy=policy)

    # When: generation outlives that deadline and a second request uses the same quota window.
    async with _test_client(app) as client:
        response = await client.post("/v1/completions", json=_completion_payload())
        second = await client.post("/v1/completions", json=_completion_payload())
        metrics = _response_body(await client.get("/demo/metrics"))

    # Then: both requests reach the runner because timeout released capacity and token reservation.
    assert response.status_code == 504
    assert second.status_code == 504
    assert len(runner.handles) == 2
    assert len(runner.cancelled_request_ids) == 2
    assert metrics["timeout_requests"] == 2
    assert metrics["active_requests"] == 0
    assert metrics["queued_requests"] == 0


def test_public_demo_disconnect_cancels_runner_and_releases_capacity() -> None:
    asyncio.run(_check_public_demo_disconnect())


async def _check_public_demo_disconnect() -> None:
    # Given: a pending non-stream request with an explicit ASGI disconnect channel.
    app, runner = _build_controlled_public_app(policy=_PUBLIC_POLICY)
    body = json.dumps(_completion_payload()).encode("utf-8")
    receives: asyncio.Queue[Message] = asyncio.Queue()
    sent: list[Message] = []
    await receives.put(
        {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }
    )
    scope = cast(
        "Scope",
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/completions",
            "raw_path": b"/v1/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8000),
            "state": {},
        },
    )

    async def receive() -> Message:
        return await receives.get()

    async def send(message: Message) -> None:
        sent.append(message)

    # When: the client disconnects after submission but before completion.
    async with app.router.lifespan_context(app):
        request_task = asyncio.create_task(app(scope, receive, send))
        await _wait_until(lambda: len(runner.handles) == 1)
        await receives.put({"type": "http.disconnect"})
        await asyncio.wait_for(request_task, timeout=1.0)

    # Then: cancellation reaches the runner and the HTTP capacity is released.
    starts = [message for message in sent if message["type"] == "http.response.start"]
    assert starts
    assert starts[0]["status"] == 499
    assert len(runner.cancelled_request_ids) == 1


def test_public_demo_stream_completion_releases_slot_and_engine_resources() -> None:
    asyncio.run(_check_public_demo_stream_cleanup())


async def _check_public_demo_stream_cleanup() -> None:
    # Given: a real tiny EngineRunner and one streaming public completion.
    app, runner = _build_public_app()

    # When: the stream reaches [DONE] and all chunks are consumed.
    async with _test_client(app) as client:
        async with client.stream(
            "POST",
            "/v1/completions",
            json=_completion_payload(max_tokens=3, stream=True),
        ) as response:
            lines = [line async for line in response.aiter_lines()]
        public_metrics = _response_body(await client.get("/demo/metrics"))
        engine_metrics = runner.metrics()

    # Then: one stream completed and neither public nor KV capacity remains resident.
    assert response.status_code == 200
    assert any(line == "data: [DONE]" for line in lines)
    assert public_metrics["completed_requests"] == 1
    assert public_metrics["active_requests"] == 0
    assert public_metrics["queued_requests"] == 0
    assert engine_metrics.active_requests == 0
    assert engine_metrics.waiting_requests == 0
    assert engine_metrics.cached_tokens == 0
    assert engine_metrics.reserved_cache_tokens == 0


def test_public_demo_stream_disconnect_before_first_body_releases_capacity() -> None:
    asyncio.run(_check_public_demo_stream_start_disconnect())


async def _check_public_demo_stream_start_disconnect() -> None:
    # Given: a streaming request whose client vanishes while response headers are sent.
    policy = replace(_PUBLIC_POLICY, max_concurrent_requests=1, max_queue_size=0)
    app, runner = _build_controlled_public_app(policy=policy)
    body = json.dumps(_completion_payload(stream=True)).encode("utf-8")
    receives: asyncio.Queue[Message] = asyncio.Queue()
    await receives.put(
        {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }
    )
    scope = cast(
        "Scope",
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/completions",
            "raw_path": b"/v1/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8000),
            "state": {},
        },
    )

    async def receive() -> Message:
        return await receives.get()

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            raise ClientDisconnect

    # When: StreamingResponse cannot begin the body iterator after submission.
    async with app.router.lifespan_context(app):
        with pytest.raises(ClientDisconnect):
            await app(scope, receive, send)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url=_BASE_URL,
        ) as client:
            metrics = _response_body(await client.get("/demo/metrics"))

    # Then: the request is cancelled and no public slot survives the disconnect.
    assert len(runner.cancelled_request_ids) == 1
    assert metrics["active_requests"] == 0
    assert metrics["queued_requests"] == 0
    assert metrics["failed_requests"] == 1


def test_public_demo_stream_disconnect_after_token_records_one_failure() -> None:
    asyncio.run(_check_public_demo_stream_token_disconnect())


async def _check_public_demo_stream_token_disconnect() -> None:
    # Given: a streaming request whose client disconnects after receiving one token event.
    policy = replace(_PUBLIC_POLICY, max_concurrent_requests=1, max_queue_size=0)
    app, runner = _build_controlled_public_app(
        policy=policy,
        cancel_delay_seconds=0.05,
    )
    body = json.dumps(_completion_payload(stream=True)).encode("utf-8")
    receives: asyncio.Queue[Message] = asyncio.Queue()
    await receives.put(
        {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }
    )
    scope = cast(
        "Scope",
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/completions",
            "raw_path": b"/v1/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8000),
            "state": {},
        },
    )
    disconnected = False

    async def receive() -> Message:
        return await receives.get()

    async def send(message: Message) -> None:
        nonlocal disconnected
        if message["type"] == "http.response.body" and message.get("body") and not disconnected:
            disconnected = True
            await receives.put({"type": "http.disconnect"})

    # When: the runner publishes one token and the ASGI disconnect cancels the stream task.
    async with app.router.lifespan_context(app):
        request_task = asyncio.create_task(app(scope, receive, send))
        await _wait_until(lambda: len(runner.handles) == 1)
        stream_queue = runner.handles[0].stream_queue
        assert stream_queue is not None
        stream_queue.put_nowait(StreamEvent(event_type=StreamEventType.TOKEN, token_id=0))
        await asyncio.wait_for(request_task, timeout=1.0)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url=_BASE_URL,
        ) as client:
            metrics = _response_body(await client.get("/demo/metrics"))

    # Then: cancellation and failed-request accounting both occur exactly once.
    assert len(runner.cancelled_request_ids) == 1
    assert metrics["failed_requests"] == 1
    assert metrics["active_requests"] == 0
    assert metrics["queued_requests"] == 0


def test_public_demo_kill_switch_and_safe_read_only_documents() -> None:
    asyncio.run(_check_public_demo_kill_switch_and_documents())


async def _check_public_demo_kill_switch_and_documents() -> None:
    # Given: a disabled public demo whose model runner must not start.
    app, runner = _build_controlled_public_app(policy=replace(_PUBLIC_POLICY, enabled=False))

    # When: clients inspect health, info, metrics, and attempt generation.
    async with _test_client(app) as client:
        health = await client.get("/healthz")
        info = _response_body(await client.get("/demo/info"))
        metrics = _response_body(await client.get("/demo/metrics"))
        completion = await client.post(
            "/v1/completions",
            json=_completion_payload(prompt="AB"),
        )

    # Then: generation is off and documents contain only the explicit safe schema.
    assert health.status_code == 503
    assert completion.status_code == 503
    assert not runner.running
    assert set(info) == {
        "project_version",
        "model_id",
        "demo_mode",
        "limits",
        "executor",
        "kv_cache_backend",
        "prefix_cache_enabled",
        "streaming_enabled",
        "features",
    }
    features = info["features"]
    assert isinstance(features, dict)
    assert set(features) == {"story_forge", "prediction_lab", "systems_lab"}
    assert set(metrics) == {
        "online",
        "active_requests",
        "queued_requests",
        "completed_requests",
        "failed_requests",
        "rejected_requests",
        "rate_limited_requests",
        "timeout_requests",
        "generated_token_count",
        "average_latency_ms",
    }
    documents = json.dumps({"info": info, "metrics": metrics})
    assert "AB" not in documents
    assert "checkpoint" not in documents.lower()
    assert "hostname" not in documents.lower()
    assert ":\\" not in documents


def test_public_demo_model_failure_is_generic_and_never_logs_prompt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    asyncio.run(_check_public_demo_failure_privacy(caplog))


async def _check_public_demo_failure_privacy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: a model failure carrying sensitive-looking internal detail.
    prompt = "AB"
    app, _runner = _build_controlled_public_app(
        policy=_PUBLIC_POLICY,
        failure=RuntimeError("traceback sensitive-internal-marker"),
    )
    caplog.set_level(logging.INFO, logger="minigpt.public_demo")

    # When: the public completion fails after submission.
    async with _test_client(app) as client:
        response = await client.post(
            "/v1/completions",
            json=_completion_payload(prompt=prompt),
        )
        metrics = _response_body(await client.get("/demo/metrics"))

    # Then: the client and aggregate metrics omit traceback, prompt, and internal detail.
    assert response.status_code == 500
    assert _error_code(_response_json(response)) == "internal_error"
    public_text = response.text.lower()
    assert "traceback" not in public_text
    assert "sensitive-internal-marker" not in public_text
    assert prompt not in json.dumps(metrics)
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert prompt not in log_text
    assert "sensitive-internal-marker" not in log_text


# -- Story Forge and Prediction Lab public endpoints -----------------------------


def _story_branch_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "world": "space",
        "tone": "adventurous",
        "theme": "discovery",
        "opening": "The cargo shuttle drifted past the third beacon.",
        "seed": 20260901,
        "branch_count": 3,
        "max_tokens": 8,
        "stream": False,
    }
    payload.update(overrides)
    return payload


def _build_disabled_labs_app() -> tuple[FastAPI, _ControlledRunner]:
    runner = _ControlledRunner()
    info = PublicDemoInfo(
        project_version=__version__,
        model_id="minigpt-story-forge",
        executor_name="continuous",
        kv_cache_backend="dense",
        prefix_cache_enabled=False,
        story_forge_enabled=False,
        prediction_lab_enabled=False,
    )
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=CharTokenizer.from_text("AB"),
        block_size=_BLOCK_SIZE,
        policy=_PUBLIC_POLICY,
        info=info,
        model=None,
    )
    return app, runner


def test_story_branches_fails_closed_when_story_forge_disabled() -> None:
    # Given: a public demo whose Story Forge feature is disabled.
    app, _runner = _build_disabled_labs_app()

    # When: a three-branch request is submitted.
    async def _scenario() -> httpx.Response:
        async with _test_client(app) as client:
            return await client.post("/demo/story/branches", json=_story_branch_payload())

    response = asyncio.run(_scenario())

    # Then: it fails closed before touching the runner.
    assert response.status_code == 503
    assert _error_code(_response_json(response)) == "story_forge_unavailable"


def test_story_branches_rejects_branch_count_not_three() -> None:
    # Given: a branch request with an invalid branch count.
    from minigpt import public_demo  # noqa: PLC0415

    # When/Then: the parser rejects a non-public branch count.
    with pytest.raises(ValueError, match="branch_count"):
        _ = public_demo._parse_story_branch(  # pyright: ignore[reportPrivateUsage]
            _story_branch_payload(branch_count=4), _PUBLIC_POLICY
        )


def test_story_branches_rejects_invalid_world_tone_theme() -> None:
    # Given: invalid control selectors.
    from minigpt import public_demo  # noqa: PLC0415

    for field, value in (("world", "ocean"), ("tone", "scary"), ("theme", "revenge")):
        payload = _story_branch_payload(**{field: value})
        with pytest.raises(ValueError, match=field):
            _ = public_demo._parse_story_branch(  # pyright: ignore[reportPrivateUsage]
                payload, _PUBLIC_POLICY
            )


def test_story_branches_rejects_stream_when_public_streaming_is_disabled() -> None:
    # Given: a valid branch request that asks for SSE under a non-streaming policy.
    from minigpt import public_demo  # noqa: PLC0415

    policy = replace(_PUBLIC_POLICY, streaming_enabled=False)

    # When/Then: the parser rejects it before any branch is submitted.
    with pytest.raises(ValueError, match="streaming is disabled"):
        _ = public_demo._parse_story_branch(  # pyright: ignore[reportPrivateUsage]
            _story_branch_payload(stream=True),
            policy,
        )


def test_predict_endpoints_fail_closed_when_prediction_disabled() -> None:
    # Given: a public demo whose Prediction Lab feature is disabled.
    app, _runner = _build_disabled_labs_app()

    # When: both prediction endpoints are called.
    async def _scenario() -> tuple[httpx.Response, httpx.Response]:
        async with _test_client(app) as client:
            next_response = await client.post(
                "/demo/predict/next",
                json={"world": "space", "tone": "adventurous", "theme": "discovery", "text": "the"},
            )
            score_response = await client.post(
                "/demo/predict/score",
                json={"world": "space", "tone": "adventurous", "theme": "discovery", "text": "the"},
            )
            return next_response, score_response

    next_response, score_response = asyncio.run(_scenario())

    # Then: both fail closed before any model work.
    assert next_response.status_code == 503
    assert score_response.status_code == 503
    assert _error_code(_response_json(next_response)) == "prediction_unavailable"
    assert _error_code(_response_json(score_response)) == "prediction_unavailable"


# -- Story Forge exact HTTP lifecycle ------------------------------------------

_STORY_BLOCK_SIZE = 64
_STORY_CORPUS: tuple[str, ...] = (
    "hello world",
    "hello hello",
    "world of warcraft",
    "the quick brown fox",
    "jumps over the lazy dog",
    "café au lait",
    "你好 世界",
    "space  forest  robot  mystery",
)


def _story_bpe_tokenizer() -> BPETokenizer:
    """Train a deterministic in-memory Story Forge BPE tokenizer fixture."""
    return BPETokenizer.train_from_iterator(iter(_STORY_CORPUS), vocab_size=300)


def _story_result(
    request_id: str,
    generated_tokens: tuple[int, ...],
    *,
    status: RequestStatus = RequestStatus.FINISHED,
    failure_reason: str | None = None,
) -> RunnerResult:
    return RunnerResult(
        request_id=request_id,
        status=status,
        generated_tokens=generated_tokens,
        metrics=RequestMetrics(
            request_id=request_id,
            status=status,
            queue_time_seconds=0.0,
            prefill_latency_seconds=0.0,
            time_to_first_token_seconds=0.0,
            decode_latencies_seconds=(),
            time_per_output_token_seconds=None,
            end_to_end_latency_seconds=0.0,
            generated_tokens=len(generated_tokens),
            failure_reason=failure_reason,
            prefix_hit_blocks=0,
            prefix_hit_tokens=0,
            prefix_miss_tokens=0,
            prefill_tokens_computed=0,
            preemption_count=0,
            resume_count=0,
            recompute_tokens=0,
            reservation_growth_count=0,
            reservation_growth_tokens=0,
            reservation_growth_blocked_count=0,
        ),
        failure_reason=failure_reason,
    )


@dataclass(slots=True)
class _FakeBranchPlan:
    """One planned branch submission outcome for the deterministic fake runner."""

    result: RunnerResult | None = None
    exception: BaseException | None = None
    blocked: bool = False


@final
class _StoryFakeRunner:
    """Deterministic in-memory runner with planned outcomes and exact logs.

    Submissions consume plans in order; a one-shot ``fail_on_submit_index``
    raises ``submit_exception`` at the zero-based index of the next submit and
    then clears itself so a following request can succeed.
    """

    def __init__(
        self,
        *,
        fail_on_submit_index: int | None = None,
        submit_exception: BaseException | None = None,
    ) -> None:
        self.running = False
        self._fail_on_submit_index = fail_on_submit_index
        self._submit_exception = submit_exception
        self.plans: list[_FakeBranchPlan] = []
        self.handles: list[RequestHandle] = []
        self.futures: list[Future[RunnerResult]] = []
        self.submit_log: list[str] = []
        self.cancel_log: list[str] = []

    @property
    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True

    def shutdown(self) -> None:
        self.running = False
        for handle in self.handles:
            if not handle.future.done():
                _ = handle.future.cancel()

    def submit(self, request: GenerationRequest, *, stream: bool) -> RequestHandle:
        index = len(self.handles)
        if self._fail_on_submit_index == index:
            self._fail_on_submit_index = None
            assert self._submit_exception is not None
            raise self._submit_exception
        plan = self.plans[index] if index < len(self.plans) else _FakeBranchPlan()
        future: Future[RunnerResult] = Future()
        if plan.exception is not None:
            future.set_exception(plan.exception)
        elif not plan.blocked:
            result = plan.result or _story_result(request.request_id, ())
            future.set_result(result)
        stream_queue: queue.Queue[StreamEvent] | None = queue.Queue() if stream else None
        handle = RequestHandle(
            request_id=request.request_id,
            future=future,
            stream_queue=stream_queue,
        )
        self.handles.append(handle)
        self.futures.append(future)
        self.submit_log.append(request.request_id)
        return handle

    def cancel(self, request_id: str, *, timeout_seconds: float = 1.0) -> None:
        del timeout_seconds
        self.cancel_log.append(request_id)
        for handle in self.handles:
            if handle.request_id == request_id and not handle.future.done():
                _ = handle.future.cancel()


def _story_info() -> PublicDemoInfo:
    return PublicDemoInfo(
        project_version=__version__,
        model_id="minigpt-story-forge",
        executor_name="continuous",
        kv_cache_backend="dense",
        prefix_cache_enabled=False,
        story_forge_enabled=True,
        prediction_lab_enabled=False,
    )


def _build_story_app(
    *,
    policy: PublicDemoPolicy | None = None,
    runner: _StoryFakeRunner | None = None,
) -> tuple[FastAPI, _StoryFakeRunner]:
    effective_policy = policy or _PUBLIC_POLICY
    if runner is None:
        runner = _StoryFakeRunner()
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=effective_policy,
        info=_story_info(),
        model=None,
    )
    return app, runner


def _sse_branch_lines(lines: list[str]) -> list[dict[str, JsonValue]]:
    """Extract parsed Story Forge SSE ``data:`` JSON objects in order."""
    chunks: list[dict[str, JsonValue]] = []
    for line in lines:
        if not line.startswith(_DATA_PREFIX):
            continue
        payload = line.removeprefix(_DATA_PREFIX)
        if payload == "[DONE]":
            continue
        decoded = cast("JsonValue", json.loads(payload))
        assert isinstance(decoded, dict)
        chunks.append(decoded)
    return chunks


def _sse_event_kind(chunk: dict[str, JsonValue]) -> str:
    kind = chunk.get("type")
    assert isinstance(kind, str)
    return kind


def test_story_empty_opening_returns_three_branches_200() -> None:
    asyncio.run(_check_story_empty_opening())


async def _check_story_empty_opening() -> None:
    # Given: a Story Forge app with the in-memory BPE tokenizer and no opening.
    runner = _StoryFakeRunner()
    runner.start()
    tokenizer = _story_bpe_tokenizer()
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=tokenizer,
        block_size=_STORY_BLOCK_SIZE,
        policy=_PUBLIC_POLICY,
        info=_story_info(),
        model=None,
    )

    # When: an empty opening is submitted and all branches complete.
    async with _test_client(app) as client:
        response = await client.post(
            "/demo/story/branches",
            json=_story_branch_payload(opening=""),
        )

    # Then: exactly three branches succeed with stable, distinct seeds.
    assert response.status_code == 200
    body = _response_body(response)
    assert body["object"] == "story_branches"
    branches = cast("list[dict[str, object]]", body["branches"])
    assert len(branches) == 3
    assert len({cast("int", branch["seed"]) for branch in branches}) == 3


def test_story_rejects_opening_over_character_and_context_limit() -> None:
    asyncio.run(_check_story_context_limit())


async def _check_story_context_limit() -> None:
    # Given: a policy character limit and long opening plus a tiny block size.
    app, runner = _build_story_app()
    long_opening = "A" * (_PUBLIC_POLICY.max_prompt_characters + 1)

    # When: the opening exceeds the character limit, and then context is impossible.
    async with _test_client(app) as client:
        char_response = await client.post(
            "/demo/story/branches",
            json=_story_branch_payload(opening=long_opening),
        )
        context_response = await client.post(
            "/demo/story/branches",
            json=_story_branch_payload(opening="hello", max_tokens=64),
        )

    # Then: character limit returns opening_too_long; impossible context returns 400.
    assert char_response.status_code == 400
    assert _error_code(_response_json(char_response)) == "opening_too_long"
    assert context_response.status_code == 400
    assert _error_code(_response_json(context_response)) == "story_context_exhausted"
    assert runner.handles == []


def test_story_fail_on_submit_2_and_3_cancels_once_and_recovers() -> None:
    asyncio.run(_check_story_submit_failure())


async def _check_story_submit_failure() -> None:
    # Given: a runner that fails on the second and third submit, then succeeds.
    runner = _StoryFakeRunner(
        fail_on_submit_index=1,
        submit_exception=RunnerQueueFullError("submit"),
    )
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=_PUBLIC_POLICY,
        info=_story_info(),
        model=None,
    )

    # When: the request is submitted and only the first branch is accepted.
    async with _test_client(app) as client:
        response = await client.post(
            "/demo/story/branches",
            json=_story_branch_payload(),
        )

    # Then: it fails closed and the single accepted request is cancelled exactly once.
    assert response.status_code == 503
    assert _error_code(_response_json(response)) == "service_unavailable"
    assert len(runner.submit_log) == 1
    assert len(runner.cancel_log) == 1
    assert runner.cancel_log == runner.submit_log

    # And: a subsequent identical request succeeds with all three branches accepted.
    async with _test_client(app) as client:
        followup = await client.post("/demo/story/branches", json=_story_branch_payload())
    assert followup.status_code == 200
    assert len(runner.submit_log) == 4


def test_story_forced_timeout_returns_504_and_recovers() -> None:
    asyncio.run(_check_story_forced_timeout())


async def _check_story_forced_timeout() -> None:
    # Given: a runner whose three branch futures never resolve.
    policy = replace(
        _PUBLIC_POLICY,
        request_timeout_seconds=0.05,
        global_generated_tokens_per_day=10_000,
    )
    runner = _StoryFakeRunner()
    runner.plans = [_FakeBranchPlan(blocked=True) for _ in range(3)]
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=policy,
        info=_story_info(),
        model=None,
    )

    # When: generation exceeds the request deadline.
    async with _test_client(app) as client:
        response = await client.post("/demo/story/branches", json=_story_branch_payload())

    # Then: 504, all three branches cancelled exactly once, quota/capacity released.
    assert response.status_code == 504
    assert len(runner.submit_log) == 3
    assert sorted(runner.cancel_log) == sorted(runner.submit_log)
    assert len(runner.cancel_log) == 3

    # And: capacity is released, so the next request succeeds.
    runner.plans = [_FakeBranchPlan() for _ in range(3)]
    async with _test_client(app) as client:
        followup = await client.post("/demo/story/branches", json=_story_branch_payload())
    assert followup.status_code == 200


def test_story_nonstream_disconnect_cancels_and_releases() -> None:
    asyncio.run(_check_story_disconnect())


async def _check_story_disconnect() -> None:
    # Given: three blocked branch futures behind a raw ASGI request.
    runner = _StoryFakeRunner()
    runner.plans = [_FakeBranchPlan(blocked=True) for _ in range(3)]
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=_PUBLIC_POLICY,
        info=_story_info(),
        model=None,
    )
    body = json.dumps(_story_branch_payload()).encode("utf-8")
    receives: asyncio.Queue[Message] = asyncio.Queue()
    sent: list[Message] = []
    await receives.put({"type": "http.request", "body": body, "more_body": False})

    async def receive() -> Message:
        return await receives.get()

    async def send(message: Message) -> None:
        sent.append(message)

    # When: the client disconnects while all three branches are pending.
    async with app.router.lifespan_context(app):
        task = asyncio.create_task(app(_branch_http_scope(body), receive, send))
        await _wait_until(lambda: len(runner.handles) == 3)
        await receives.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, timeout=1.0)

    # Then: all three requests are cancelled exactly once and the reservation is released.
    assert sorted(runner.cancel_log) == sorted(runner.submit_log)
    assert len(runner.cancel_log) == 3


def test_story_one_branch_exception_keeps_healthy_branches() -> None:
    asyncio.run(_check_story_one_branch_exception())


async def _check_story_one_branch_exception() -> None:
    # Given: a runner with one failing branch future and two healthy branches.
    runner = _StoryFakeRunner()
    runner.plans = [
        _FakeBranchPlan(),
        _FakeBranchPlan(exception=RuntimeError("branch failure")),
        _FakeBranchPlan(),
    ]
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=_PUBLIC_POLICY,
        info=_story_info(),
        model=None,
    )

    # When: all three branches resolve with one error.
    async with _test_client(app) as client:
        response = await client.post("/demo/story/branches", json=_story_branch_payload())

    # Then: two healthy branches plus one error branch, all with correct finish reason.
    assert response.status_code == 200
    branches = cast("list[dict[str, object]]", _response_body(response)["branches"])
    assert len(branches) == 3
    finish_reasons = [cast("str", branch["finish_reason"]) for branch in branches]
    assert finish_reasons.count("length") == 2
    assert finish_reasons.count("error") == 1
    assert runner.cancel_log == []


def test_story_partial_failure_charges_conservative_aggregate_quota() -> None:
    asyncio.run(_check_story_partial_failure_quota())


async def _check_story_partial_failure_quota() -> None:
    # Given: one failing branch and a quota only slightly above one aggregate request.
    runner = _StoryFakeRunner()
    runner.plans = [
        _FakeBranchPlan(),
        _FakeBranchPlan(exception=RuntimeError("branch failure")),
        _FakeBranchPlan(),
    ]
    policy = replace(
        _PUBLIC_POLICY,
        max_new_tokens=64,
        global_requests_per_hour=100,
        global_generated_tokens_per_day=65,
    )
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=policy,
        info=_story_info(),
        model=None,
    )

    # When: the first 3x20-token reservation returns one error branch.
    async with _test_client(app) as client:
        first = await client.post(
            "/demo/story/branches",
            json=_story_branch_payload(max_tokens=20, seed=101),
        )
        second = await client.post(
            "/demo/story/branches",
            json=_story_branch_payload(max_tokens=20, seed=202),
        )
        metrics = _response_body(await client.get("/demo/metrics"))

    # Then: the unknown failed-branch work is charged conservatively.
    assert first.status_code == 200
    first_branches = cast("list[dict[str, object]]", _response_body(first)["branches"])
    assert [branch["finish_reason"] for branch in first_branches].count("error") == 1
    assert second.status_code == 429
    assert _error_code(_response_json(second)) == "rate_limit_exceeded"
    assert metrics["failed_requests"] == 1
    assert metrics["completed_requests"] == 0


def test_story_concurrent_gate_max1_queue0_is_exact() -> None:
    asyncio.run(_check_story_concurrent_gate())


async def _check_story_concurrent_gate() -> None:
    # Given: one active slot and no queue.
    policy = replace(
        _PUBLIC_POLICY,
        max_concurrent_requests=1,
        max_queue_size=0,
        global_requests_per_hour=100,
    )
    runner = _StoryFakeRunner()
    runner.plans = [_FakeBranchPlan(blocked=True) for _ in range(3)]
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=policy,
        info=_story_info(),
        model=None,
    )

    # When: the first request holds the only slot while a second arrives.
    async with _test_client(app) as client:
        first_task = asyncio.create_task(
            client.post("/demo/story/branches", json=_story_branch_payload(seed=1))
        )
        await _wait_until(lambda: len(runner.submit_log) == 3)
        second = await client.post("/demo/story/branches", json=_story_branch_payload(seed=2))

        # Then: the second is exactly 429 while the first is held.
        assert second.status_code == 429
        assert _error_code(_response_json(second)) == "queue_full"

        # And: releasing the first returns exactly 200.
        await _release_blocked_futures(runner)
        first = await first_task
        assert first.status_code == 200


async def _release_blocked_futures(runner: _StoryFakeRunner) -> None:
    for handle in runner.handles:
        if not handle.future.done():
            handle.future.set_result(_story_result(handle.request_id, (2,)))


def test_story_sse_true_interleaving_and_metadata() -> None:
    asyncio.run(_check_story_sse_interleaving())


async def _check_story_sse_interleaving() -> None:
    # Given: a raw ASGI streaming request and three manually driven branch queues.
    runner = _StoryFakeRunner()
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=replace(_PUBLIC_POLICY, streaming_enabled=True, request_timeout_seconds=1.0),
        info=_story_info(),
        model=None,
    )
    body = json.dumps(_story_branch_payload(stream=True, max_tokens=8)).encode("utf-8")
    receives: asyncio.Queue[Message] = asyncio.Queue()
    sent: list[Message] = []
    branch_one_token_sent = asyncio.Event()
    await receives.put({"type": "http.request", "body": body, "more_body": False})

    async def receive() -> Message:
        return await receives.get()

    async def send(message: Message) -> None:
        sent.append(message)
        if message["type"] != "http.response.body":
            return
        response_body = cast("bytes", message.get("body", b""))
        for line in response_body.decode("utf-8").splitlines():
            if not line.startswith(_DATA_PREFIX) or line == "data: [DONE]":
                continue
            payload = cast("JsonValue", json.loads(line.removeprefix(_DATA_PREFIX)))
            if (
                isinstance(payload, dict)
                and payload.get("type") == "token"
                and payload.get("branch_id") == 1
            ):
                branch_one_token_sent.set()

    async def produce() -> None:
        await _wait_until(lambda: len(runner.handles) == 3)
        queues = [handle.stream_queue for handle in runner.handles]
        assert all(queue_ref is not None for queue_ref in queues)
        branch_one_queue = queues[1]
        assert branch_one_queue is not None
        branch_one_queue.put_nowait(StreamEvent(StreamEventType.TOKEN, token_id=5))
        _ = await asyncio.wait_for(branch_one_token_sent.wait(), timeout=1.0)
        for branch_id, queue_ref in enumerate(queues):
            assert queue_ref is not None
            generated = (5,) if branch_id == 1 else ()
            queue_ref.put_nowait(
                StreamEvent(
                    StreamEventType.FINISHED,
                    result=_story_result(runner.handles[branch_id].request_id, generated),
                )
            )

    # When: branch 1 emits before branch 0 reaches its terminal event.
    async with app.router.lifespan_context(app):
        request_task = asyncio.create_task(app(_branch_http_scope(body), receive, send))
        producer_task = asyncio.create_task(produce())
        _ = await asyncio.wait_for(asyncio.gather(request_task, producer_task), timeout=2.0)

    lines = (
        b"".join(
            cast("bytes", message.get("body", b""))
            for message in sent
            if message["type"] == "http.response.body"
        )
        .decode("utf-8")
        .splitlines()
    )
    chunks = _sse_branch_lines(lines)
    seen = [
        (cast("int", chunk.get("branch_id")), _sse_event_kind(chunk))
        for chunk in chunks
        if chunk.get("branch_id") is not None
    ]

    # Then: the multiplexed stream preserves the observed cross-branch order and metadata.
    branch_one_token_index = seen.index((1, "token"))
    branch_zero_terminal_index = seen.index((0, "branch_finished"))
    assert branch_one_token_index < branch_zero_terminal_index
    started = [chunk for chunk in chunks if _sse_event_kind(chunk) == "branch_started"]
    assert len(started) == 3
    assert len({cast("int", chunk["seed"]) for chunk in started}) == 3
    assert all(isinstance(chunk.get("request_id"), str) for chunk in started)


def test_story_sse_done_and_empty_eos_piece_increments_quota() -> None:
    asyncio.run(_check_story_sse_done_and_quota())


async def _check_story_sse_done_and_quota() -> None:
    # Given: a raw ASGI stream under a strict token budget.
    eos_id = _story_bpe_tokenizer().eos_token_id
    runner = _StoryFakeRunner()
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=replace(
            _PUBLIC_POLICY,
            streaming_enabled=True,
            request_timeout_seconds=1.0,
            global_generated_tokens_per_day=100,
        ),
        info=_story_info(),
        model=None,
    )
    body = json.dumps(_story_branch_payload(stream=True, max_tokens=8)).encode("utf-8")
    receives: asyncio.Queue[Message] = asyncio.Queue()
    sent: list[Message] = []
    await receives.put({"type": "http.request", "body": body, "more_body": False})

    async def receive() -> Message:
        return await receives.get()

    async def send(message: Message) -> None:
        sent.append(message)

    async def produce() -> None:
        await _wait_until(lambda: len(runner.handles) == 3)
        for handle in runner.handles:
            stream_queue = handle.stream_queue
            assert stream_queue is not None
            stream_queue.put_nowait(StreamEvent(StreamEventType.TOKEN, token_id=eos_id))
            stream_queue.put_nowait(
                StreamEvent(
                    StreamEventType.FINISHED,
                    result=_story_result(handle.request_id, (eos_id,)),
                )
            )

    # When: every branch emits an EOS token and then finishes.
    async with app.router.lifespan_context(app):
        request_task = asyncio.create_task(app(_branch_http_scope(body), receive, send))
        producer_task = asyncio.create_task(produce())
        _ = await asyncio.wait_for(asyncio.gather(request_task, producer_task), timeout=2.0)

    lines = (
        b"".join(
            cast("bytes", message.get("body", b""))
            for message in sent
            if message["type"] == "http.response.body"
        )
        .decode("utf-8")
        .splitlines()
    )

    # Then: one done event, one sentinel, and three empty decoded EOS pieces still count.
    chunks = _sse_branch_lines(lines)
    kinds = [_sse_event_kind(chunk) for chunk in chunks]
    assert kinds.count("done") == 1
    assert lines.count("data: [DONE]") == 1
    token_events = [chunk for chunk in chunks if _sse_event_kind(chunk) == "token"]
    assert len(token_events) == 3
    assert all(chunk.get("text") == "" for chunk in token_events)
    async with _test_client(app) as client:
        metrics = _response_body(await client.get("/demo/metrics"))
    assert metrics["generated_token_count"] == 3


def test_story_sse_abort_cleanup_exact_once() -> None:
    asyncio.run(_check_story_sse_abort())


async def _check_story_sse_abort() -> None:
    # Given: a streaming Story Forge request whose client disconnects mid-stream.
    runner = _StoryFakeRunner()
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=replace(
            _PUBLIC_POLICY,
            streaming_enabled=True,
            max_concurrent_requests=1,
            max_queue_size=0,
        ),
        info=_story_info(),
        model=None,
    )
    body = json.dumps(_story_branch_payload(stream=True)).encode("utf-8")
    receives: asyncio.Queue[Message] = asyncio.Queue()
    await receives.put({"type": "http.request", "body": body, "more_body": False})

    async def receive() -> Message:
        return await receives.get()

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            await receives.put({"type": "http.disconnect"})

    # When: the client disconnects while the stream is active.
    async with app.router.lifespan_context(app):
        task = asyncio.create_task(app(_branch_http_scope(body), receive, send))
        await _wait_until(lambda: len(runner.handles) == 3)
        await asyncio.wait_for(task, timeout=2.0)

    # Then: all three branches cancelled exactly once and capacity released.
    assert sorted(runner.cancel_log) == sorted(runner.submit_log)
    assert len(runner.cancel_log) == 3

    async with _test_client(app) as client:
        metrics = _response_body(await client.get("/demo/metrics"))
    assert metrics["active_requests"] == 0
    assert metrics["queued_requests"] == 0


def _branch_http_scope(body: bytes) -> Scope:
    return cast(
        "Scope",
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/demo/story/branches",
            "raw_path": b"/demo/story/branches",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8000),
            "state": {},
        },
    )


def test_story_unequal_token_counts_charge_exact_quota() -> None:
    asyncio.run(_check_story_unequal_token_quota())


async def _check_story_unequal_token_quota() -> None:
    # Given: three branches with distinct generated-token counts of 1, 2, and 3.
    runner = _StoryFakeRunner()
    runner.plans = [
        _FakeBranchPlan(result=_story_result("r1-planned", (11,))),
        _FakeBranchPlan(result=_story_result("r2-planned", (21, 22))),
        _FakeBranchPlan(result=_story_result("r3-planned", (31, 32, 33))),
    ]
    policy = replace(_PUBLIC_POLICY, global_generated_tokens_per_day=100)
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=policy,
        info=_story_info(),
        model=None,
    )

    # When: three branches complete with unequal counts.
    async with _test_client(app) as client:
        response = await client.post(
            "/demo/story/branches", json=_story_branch_payload(max_tokens=3)
        )

    # Then: the aggregate metrics verify the exact total token count, not the reserve.
    assert response.status_code == 200
    branches = cast("list[dict[str, object]]", _response_body(response)["branches"])
    assert [cast("int", branch["token_count"]) for branch in branches] == [1, 2, 3]
    async with _test_client(app) as client:
        metrics = _response_body(await client.get("/demo/metrics"))
    assert metrics["generated_token_count"] == 6
    assert metrics["completed_requests"] == 1


def test_story_eos_and_length_terminal_finish_reasons() -> None:
    asyncio.run(_check_story_eos_length())


async def _check_story_eos_length() -> None:
    # Given: one branch hitting EOS (final token == eos id 3) and two hitting length.
    eos_id = _story_bpe_tokenizer().eos_token_id
    runner = _StoryFakeRunner()
    runner.plans = [
        _FakeBranchPlan(result=_story_result("r1-eos", (7, eos_id))),
        _FakeBranchPlan(result=_story_result("r2-len", (9, 10))),
        _FakeBranchPlan(result=_story_result("r3-len", (11,))),
    ]
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=_PUBLIC_POLICY,
        info=_story_info(),
        model=None,
    )

    # When: the branches finish.
    async with _test_client(app) as client:
        response = await client.post(
            "/demo/story/branches", json=_story_branch_payload(max_tokens=8)
        )

    # Then: EOS terminal yields stop; max-length terminals yield length.
    assert response.status_code == 200
    branches = cast("list[dict[str, object]]", _response_body(response)["branches"])
    reasons = [cast("str", branch["finish_reason"]) for branch in branches]
    assert reasons.count("stop") == 1
    assert reasons.count("length") == 2
    assert runner.cancel_log == []
    assert len(runner.submit_log) == 3


def test_story_fail_on_submit_3_no_cancel_duplicates() -> None:
    asyncio.run(_check_story_submit_failure_no_cancel_duplicates())


async def _check_story_submit_failure_no_cancel_duplicates() -> None:
    # Given: a runner that fails on submit index 2 (third branch).
    runner = _StoryFakeRunner(fail_on_submit_index=2, submit_exception=RuntimeError("boom"))
    app = create_public_demo_app(
        runner=cast("EngineRunner", cast("object", runner)),
        tokenizer=_story_bpe_tokenizer(),
        block_size=_STORY_BLOCK_SIZE,
        policy=_PUBLIC_POLICY,
        info=_story_info(),
        model=None,
    )

    # When: the third submit fails after two accepted handles.
    async with _test_client(app) as client:
        response = await client.post("/demo/story/branches", json=_story_branch_payload())

    # Then: the two accepted request IDs are cancelled exactly once each.
    assert response.status_code == 500
    assert len(runner.submit_log) == 2
    assert len(runner.cancel_log) == 2
    assert runner.cancel_log == runner.submit_log
    assert len(set(runner.cancel_log)) == len(runner.cancel_log)


def test_story_stream_decodes_complete_bpe_prefixes_without_replacement_text() -> None:
    # Given: a ByteLevel vocabulary with no merges for one multi-byte character.
    from minigpt import public_demo  # noqa: PLC0415

    tokenizer = BPETokenizer.train_from_iterator(iter(("plain ascii",)), vocab_size=273)
    token_ids = tokenizer.encode("你")
    assert len(token_ids) > 1
    assert any("\ufffd" in tokenizer.decode((token_id,)) for token_id in token_ids)

    # When: the Story Forge stream renders the complete generated prefix.
    rendered = public_demo._display_snapshot(  # pyright: ignore[reportPrivateUsage]
        tokenizer, token_ids
    )

    # Then: the browser snapshot is valid decoded text rather than per-byte replacement glyphs.
    assert rendered == "你"
    assert "\ufffd" not in rendered


def test_prediction_piece_labels_incomplete_byte_token_without_replacement_glyph() -> None:
    # Given: one ByteLevel token that is only part of a UTF-8 code point.
    from minigpt import public_demo  # noqa: PLC0415

    tokenizer = BPETokenizer.train_from_iterator(iter(("plain ascii",)), vocab_size=273)
    token_id = tokenizer.encode("你")[0]
    assert "\ufffd" in tokenizer.decode((token_id,))

    # When: Prediction Lab builds a display label for that candidate.
    piece, is_special = public_demo._prediction_piece(  # pyright: ignore[reportPrivateUsage]
        tokenizer, token_id
    )

    # Then: it exposes a stable token label, not invalid Unicode or a control token.
    assert piece == f"<token:{token_id}>"
    assert not is_special


def _build_prediction_app() -> tuple[FastAPI, EngineRunner, GPT]:
    tok = _story_bpe_tokenizer()
    model = GPT(
        GPTConfig(
            vocab_size=tok.vocab_size,
            block_size=_STORY_BLOCK_SIZE,
            n_layer=1,
            n_head=1,
            n_embd=16,
            dropout=0.0,
        )
    ).eval()
    executor = ContinuousExecutor(model)
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(max_active_requests=2, max_cached_tokens=256),
            block_size=_STORY_BLOCK_SIZE,
        ),
        executor=executor,
    )
    runner = EngineRunner(
        engine=engine,
        config=RunnerConfig(command_queue_size=16, stream_buffer_size=8),
    )
    app = create_public_demo_app(
        runner=runner,
        tokenizer=tok,
        block_size=_STORY_BLOCK_SIZE,
        policy=_PUBLIC_POLICY,
        info=replace(_story_info(), prediction_lab_enabled=True),
        model=model,
    )
    return app, runner, model


def test_prediction_endpoints_execute_on_owner_thread_and_return_bounded_math(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a real tiny model/runner and wrappers that record model-call threads.
    from minigpt import public_demo  # noqa: PLC0415

    app, runner, _model = _build_prediction_app()
    next_threads: list[int] = []
    score_threads: list[int] = []

    def recorded_next(model: GPT, prompt_ids: tuple[int, ...], *, top_k: int) -> object:
        next_threads.append(threading.get_ident())
        return prediction_reference.compute_next_token_distribution(model, prompt_ids, top_k=top_k)

    def recorded_score(
        model: GPT,
        token_ids: tuple[int, ...],
        *,
        control_prefix_length: int,
    ) -> object:
        score_threads.append(threading.get_ident())
        return prediction_reference.compute_sequence_surprisal(
            model,
            token_ids,
            control_prefix_length=control_prefix_length,
        )

    monkeypatch.setattr(public_demo, "compute_next_token_distribution", recorded_next)
    monkeypatch.setattr(public_demo, "compute_sequence_surprisal", recorded_score)

    async def scenario() -> tuple[dict[str, JsonValue], dict[str, JsonValue], dict[str, JsonValue]]:
        payload = {
            "world": "space",
            "tone": "adventurous",
            "theme": "discovery",
            "text": "hello world",
            "top_k": 5,
        }
        async with _test_client(app) as client:
            next_response = await client.post("/demo/predict/next", json=payload)
            score_response = await client.post("/demo/predict/score", json=payload)
            metrics_response = await client.get("/demo/metrics")
        assert next_response.status_code == 200
        assert score_response.status_code == 200
        return (
            _response_body(next_response),
            _response_body(score_response),
            _response_body(metrics_response),
        )

    # When: both public inspection endpoints are called.
    next_document, score_document, metrics = asyncio.run(scenario())

    # Then: all model work used the single owner thread and the transport is bounded/finite.
    assert next_threads == [runner.owner_thread_id]
    assert score_threads == [runner.owner_thread_id]
    assert next_document["object"] == "prediction_next"
    candidates = cast("list[JsonValue]", next_document["candidates"])
    assert len(candidates) == 5
    for raw_candidate in candidates:
        candidate = cast("dict[str, JsonValue]", raw_candidate)
        assert isinstance(candidate["token_id"], int)
        assert isinstance(candidate["piece"], str)
        assert math.isfinite(cast("float", candidate["logit"]))
        probability = cast("float", candidate["probability"])
        assert math.isfinite(probability)
        assert 0.0 <= probability <= 1.0
    assert score_document["object"] == "prediction_score"
    assert isinstance(score_document["per_token"], list)
    assert math.isfinite(cast("float", score_document["user_mean_nll"]))
    assert math.isfinite(cast("float", score_document["user_perplexity"]))
    assert metrics["completed_requests"] == 2
    assert metrics["generated_token_count"] == 0
