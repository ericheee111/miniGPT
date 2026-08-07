from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

import httpx
import torch
from httpx import ASGITransport
from minigpt.engine_runner import EngineRunner, RunnerConfig
from minigpt.http_server import MODEL_ID, create_app

from minigpt.data import CharTokenizer
from minigpt.model import GPT
from minigpt.serving import (
    ContinuousExecutor,
    EngineConfig,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI

_BLOCK_SIZE = 8
_BASE_URL = "http://test"
_DATA_PREFIX = "data: "


def build_app() -> FastAPI:
    torch.manual_seed(1234)
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


@asynccontextmanager
async def _test_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
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


def _response_body(response: httpx.Response) -> dict[str, object]:
    body: object = response.json()
    assert isinstance(body, dict)
    return cast("dict[str, object]", body)


def _error_code(body: object) -> str:
    assert isinstance(body, dict)
    error = body.get("error")
    assert isinstance(error, dict)
    code = error.get("code")
    assert isinstance(code, str)
    return code


def _parse_sse_data(lines: list[str]) -> list[dict[str, object] | None]:
    """Extract parsed JSON chunks and the [DONE] sentinel from SSE data lines."""
    chunks: list[dict[str, object] | None] = []
    for line in lines:
        if not line.startswith(_DATA_PREFIX):
            continue
        payload = line.removeprefix(_DATA_PREFIX)
        if payload == "[DONE]":
            chunks.append(None)
            continue
        decoded: object = json.loads(payload)
        assert isinstance(decoded, dict)
        chunks.append(cast("dict[str, object]", decoded))
    return chunks


def _choice_text(chunk: dict[str, object]) -> str:
    choices = chunk.get("choices")
    assert isinstance(choices, list)
    assert len(choices) >= 1
    choice = choices[0]
    assert isinstance(choice, dict)
    text = choice.get("text")
    assert isinstance(text, str)
    return text


def _choice_finish_reason(chunk: dict[str, object]) -> str | None:
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
    first_text = cast("str", _response_body(first).get("choices", [{}])[0].get("text"))
    second_text = cast("str", _response_body(second).get("choices", [{}])[0].get("text"))
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
    assert _error_code(response.json()) == "model_not_found"


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
    assert _error_code(response.json()) == "prompt_too_long"


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
    assert _error_code(response.json()) == "invalid_request"


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
    assert _error_code(response.json()) == "invalid_request"
