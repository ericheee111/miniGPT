"""Optional FastAPI boundary for OpenAI-compatible text completions."""

from __future__ import annotations

import asyncio
import json
import math
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Never, cast
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from minigpt.data import JsonValue, UnknownCharacterError
from minigpt.engine_runner import (
    EngineRunner,
    RequestHandle,
    RunnerQueueFullError,
    RunnerResult,
    RunnerUnavailableError,
    StreamEvent,
    StreamEventType,
)
from minigpt.serving import GenerationRequest, RequestStatus

MODEL_ID = "minigpt-char"
_MAX_SEED = 2**63
_COMPLETION_FIELDS = frozenset({"model", "prompt", "max_tokens", "temperature", "stream", "seed"})

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from minigpt.tokenizer import TokenizerProtocol


@dataclass(frozen=True, slots=True)
class _CompletionInput:
    model: str
    prompt: str
    max_tokens: int
    temperature: float
    stream: bool
    seed: int


@dataclass(frozen=True, slots=True)
class _RequestError(ValueError):
    message: str
    param: str | None
    status_code: int = 422


def create_app(  # noqa: C901
    *,
    runner: EngineRunner,
    tokenizer: TokenizerProtocol,
    model_id: str = MODEL_ID,
    block_size: int,
) -> FastAPI:
    """Create an ASGI app around one preloaded tokenizer and engine runner."""
    if not model_id:
        reason = "model_id must be non-empty"
        raise ValueError(reason)
    if isinstance(block_size, bool) or block_size <= 0:
        reason = "block_size must be a positive integer"
        raise ValueError(reason)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        runner.start()
        try:
            yield
        finally:
            await asyncio.to_thread(runner.shutdown)

    app = FastAPI(title="miniGPT HTTP Serving", version="1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        if not runner.is_running:
            return _error_response(
                status_code=503,
                message="engine runner is unavailable",
                error_type="service_unavailable_error",
                code="service_unavailable",
            )
        return JSONResponse({"status": "ok"})

    _ = healthz

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "minigpt",
                    }
                ],
            }
        )

    _ = models

    @app.post("/v1/completions", response_model=None)
    async def completions(  # noqa: C901, PLR0911
        request: Request,
    ) -> JSONResponse | StreamingResponse:
        try:
            document = cast("object", await request.json())
            completion = _parse_completion(document)
        except json.JSONDecodeError as error:
            return _error_response(
                status_code=400,
                message=str(error),
                error_type="invalid_request_error",
                code="invalid_json",
            )
        except _RequestError as error:
            return _error_response(
                status_code=error.status_code,
                message=error.message,
                error_type="invalid_request_error",
                code="invalid_request",
                param=error.param,
            )
        if completion.model != model_id:
            return _error_response(
                status_code=404,
                message=f"model {completion.model!r} is not served",
                error_type="invalid_request_error",
                code="model_not_found",
                param="model",
            )
        try:
            prompt_tokens = tuple(tokenizer.encode(completion.prompt))
        except UnknownCharacterError as error:
            return _error_response(
                status_code=400,
                message=str(error),
                error_type="invalid_request_error",
                code="invalid_prompt",
                param="prompt",
            )
        if len(prompt_tokens) > block_size:
            return _error_response(
                status_code=400,
                message=f"prompt has {len(prompt_tokens)} tokens but block_size is {block_size}",
                error_type="invalid_request_error",
                code="prompt_too_long",
                param="prompt",
            )

        completion_id = f"cmpl-{uuid4().hex}"
        created = int(time.time())
        generation = GenerationRequest(
            request_id=completion_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=completion.max_tokens,
            temperature=completion.temperature,
            seed=completion.seed,
            arrival_time=time.perf_counter(),
        )
        try:
            handle = runner.submit(generation, stream=completion.stream)
        except RunnerQueueFullError as error:
            return _error_response(
                status_code=429,
                message=str(error),
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
            )
        except RunnerUnavailableError as error:
            return _error_response(
                status_code=503,
                message=str(error),
                error_type="service_unavailable_error",
                code="service_unavailable",
            )

        if completion.stream:
            return StreamingResponse(
                _stream_completion(
                    runner=runner,
                    handle=handle,
                    tokenizer=tokenizer,
                    completion_id=completion_id,
                    created=created,
                    model_id=model_id,
                    prompt_tokens=len(prompt_tokens),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        try:
            result = await asyncio.wrap_future(handle.future)
        except asyncio.CancelledError:
            await _cancel_safely(runner, completion_id)
            raise
        except Exception as error:  # noqa: BLE001
            return _error_response(
                status_code=500,
                message=f"{type(error).__name__}: {error}",
                error_type="server_error",
                code="internal_error",
            )
        return _result_response(
            result=result,
            tokenizer=tokenizer,
            completion_id=completion_id,
            created=created,
            model_id=model_id,
            prompt_tokens=len(prompt_tokens),
        )

    _ = completions

    return app


async def _stream_completion(  # noqa: PLR0913
    *,
    runner: EngineRunner,
    handle: RequestHandle,
    tokenizer: TokenizerProtocol,
    completion_id: str,
    created: int,
    model_id: str,
    prompt_tokens: int,
) -> AsyncIterator[str]:
    stream_queue = handle.stream_queue
    if stream_queue is None:
        reason = "streaming request omitted its stream channel"
        raise RuntimeError(reason)
    completed_normally = False
    try:
        while True:
            event = await asyncio.to_thread(stream_queue.get)
            if event.event_type is StreamEventType.TOKEN:
                if event.token_id is None:
                    yield _sse_error("stream token omitted token_id", "internal_error")
                    return
                chunk = _completion_document(
                    completion_id=completion_id,
                    created=created,
                    model_id=model_id,
                    text=tokenizer.decode((event.token_id,)),
                    finish_reason=None,
                )
                yield _sse_data(chunk)
                continue
            if event.event_type is StreamEventType.FINISHED and event.result is not None:
                yield _sse_data(
                    _completion_document(
                        completion_id=completion_id,
                        created=created,
                        model_id=model_id,
                        text="",
                        finish_reason="length",
                        usage=_usage(prompt_tokens, len(event.result.generated_tokens)),
                    )
                )
                yield "data: [DONE]\n\n"
                completed_normally = True
                return
            yield _stream_failure(event)
            return
    finally:
        if not completed_normally and not handle.future.done():
            await _cancel_safely(runner, completion_id)


def _result_response(  # noqa: PLR0913
    *,
    result: RunnerResult,
    tokenizer: TokenizerProtocol,
    completion_id: str,
    created: int,
    model_id: str,
    prompt_tokens: int,
) -> JSONResponse:
    if result.status is RequestStatus.FINISHED:
        return JSONResponse(
            _completion_document(
                completion_id=completion_id,
                created=created,
                model_id=model_id,
                text=tokenizer.decode(result.generated_tokens),
                finish_reason="length",
                usage=_usage(prompt_tokens, len(result.generated_tokens)),
            )
        )
    if result.status is RequestStatus.CANCELLED:
        return _error_response(
            status_code=503,
            message="generation request was cancelled",
            error_type="service_unavailable_error",
            code="request_cancelled",
        )
    return _error_response(
        status_code=500,
        message=result.failure_reason or "generation failed",
        error_type="server_error",
        code="generation_failed",
    )


def _completion_document(  # noqa: PLR0913
    *,
    completion_id: str,
    created: int,
    model_id: str,
    text: str,
    finish_reason: Literal["length"] | None,
    usage: dict[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    document: dict[str, JsonValue] = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "text": text,
                "index": 0,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        document["usage"] = usage
    return document


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, JsonValue]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _error_response(
    *,
    status_code: int,
    message: str,
    error_type: str,
    code: str,
    param: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


def _sse_data(document: dict[str, JsonValue]) -> str:
    payload = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    return f"data: {payload}\n\n"


def _sse_error(message: str, code: str) -> str:
    return _sse_data(
        {
            "error": {
                "message": message,
                "type": "server_error",
                "param": None,
                "code": code,
            }
        }
    )


def _stream_failure(event: StreamEvent) -> str:
    if event.event_type is StreamEventType.BACKPRESSURE:
        return _sse_error(event.detail or "stream consumer is too slow", "stream_backpressure")
    if event.event_type is StreamEventType.CANCELLED:
        return _sse_error("generation request was cancelled", "request_cancelled")
    return _sse_error(event.detail or "generation failed", "generation_failed")


async def _cancel_safely(runner: EngineRunner, request_id: str) -> None:
    with suppress(RunnerQueueFullError, RunnerUnavailableError):
        await asyncio.to_thread(runner.cancel, request_id)


def _parse_completion(document: object) -> _CompletionInput:  # noqa: C901
    if not isinstance(document, dict):
        _invalid_request("request body must be a JSON object", None)
    raw_values = cast("dict[object, object]", document)
    values: dict[str, object] = {}
    for key, value in raw_values.items():
        if not isinstance(key, str):
            _invalid_request("request body keys must be strings", None)
        values[key] = value
    unsupported = sorted(set(values) - _COMPLETION_FIELDS)
    if unsupported:
        field = unsupported[0]
        _invalid_request(f"unsupported completion field {field!r}", field)

    model = values.get("model")
    if not isinstance(model, str) or not model:
        _invalid_request("model must be a non-empty string", "model")
    prompt = values.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        _invalid_request("prompt must be a non-empty string", "prompt")

    max_tokens = values.get("max_tokens", 16)
    if type(max_tokens) is not int or max_tokens < 0:
        _invalid_request("max_tokens must be a non-negative integer", "max_tokens")
    temperature = values.get("temperature", 1.0)
    if type(temperature) not in {int, float}:
        _invalid_request("temperature must be a finite positive number", "temperature")
    normalized_temperature = float(cast("int | float", temperature))
    if normalized_temperature <= 0.0 or not math.isfinite(normalized_temperature):
        _invalid_request("temperature must be a finite positive number", "temperature")
    stream = values.get("stream", False)
    if type(stream) is not bool:
        _invalid_request("stream must be a boolean", "stream")
    seed = values.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < _MAX_SEED:
        _invalid_request(f"seed must be an integer in [0, {_MAX_SEED})", "seed")
    return _CompletionInput(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=normalized_temperature,
        stream=stream,
        seed=seed,
    )


def _invalid_request(message: str, param: str | None) -> Never:
    raise _RequestError(message, param)
