from __future__ import annotations

import asyncio
import json
import logging
import queue
import time
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, cast, final

import httpx
import pytest
import torch
from httpx import ASGITransport
from starlette.requests import ClientDisconnect

from minigpt import __version__
from minigpt.data import CharTokenizer, JsonValue
from minigpt.engine_runner import (
    EngineRunner,
    RequestHandle,
    RunnerConfig,
    RunnerResult,
    StreamEvent,
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
    SchedulerConfig,
    ServingEngine,
)
from minigpt.settings import GPTConfig

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
    allowed_origins=(_PUBLIC_ORIGIN,),
    per_client_requests=100,
    global_requests=100,
)


@final
class _ControlledRunner:
    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.running = False
        self.failure = failure
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
) -> tuple[FastAPI, _ControlledRunner]:
    runner = _ControlledRunner(failure=failure)
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
            trust_proxy=False,
        )

    # And: loopback is accepted while proxy trust cannot accompany a public bind.
    validate_bind_host(
        "127.0.0.1",
        unsafe_allow_non_loopback=False,
        trust_proxy=False,
    )
    with pytest.raises(InvalidPublicDemoConfigError, match="requires a loopback"):
        validate_bind_host(
            "0.0.0.0",  # noqa: S104 - deliberate rejected bind test
            unsafe_allow_non_loopback=True,
            trust_proxy=True,
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
                "Access-Control-Request-Headers": ("content-type, ngrok-skip-browser-warning"),
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
    assert (
        preflight.headers["access-control-allow-headers"]
        == "content-type, ngrok-skip-browser-warning"
    )


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


def test_public_demo_enforces_per_client_and_global_rate_limits() -> None:
    asyncio.run(_check_public_demo_rate_limits())


async def _check_public_demo_rate_limits() -> None:
    # Given: separate apps with a two-request client limit and a two-request global limit.
    client_app, _runner = _build_public_app(
        replace(
            _PUBLIC_POLICY,
            per_client_requests=2,
            global_requests=100,
        )
    )
    global_app, _runner = _build_public_app(
        replace(
            _PUBLIC_POLICY,
            trust_proxy=True,
            per_client_requests=100,
            global_requests=2,
        )
    )
    payload = _completion_payload(max_tokens=0)

    # When: one peer exceeds its limit and spoofed client keys try to bypass global capacity.
    async with _test_client(client_app) as client:
        client_responses = [
            await client.post("/v1/completions", json=payload) for _index in range(3)
        ]
    async with _test_client(global_app) as client:
        global_responses = [
            await client.post(
                "/v1/completions",
                json=payload,
                headers={"X-Forwarded-For": f"192.0.2.{index}"},
            )
            for index in range(1, 4)
        ]

    # Then: both third requests receive 429 and a bounded Retry-After value.
    assert [response.status_code for response in client_responses] == [200, 200, 429]
    assert [response.status_code for response in global_responses] == [200, 200, 429]
    assert int(client_responses[-1].headers["retry-after"]) >= 1
    assert int(global_responses[-1].headers["retry-after"]) >= 1


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
    policy = replace(_PUBLIC_POLICY, request_timeout_seconds=0.02)
    app, runner = _build_controlled_public_app(policy=policy)

    # When: generation outlives that deadline.
    async with _test_client(app) as client:
        response = await client.post("/v1/completions", json=_completion_payload())
        metrics = _response_body(await client.get("/demo/metrics"))

    # Then: the request is cancelled, counted as timed out, and holds no slot.
    assert response.status_code == 504
    assert len(runner.cancelled_request_ids) == 1
    assert metrics["timeout_requests"] == 1
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
        "streaming_available",
    }
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
