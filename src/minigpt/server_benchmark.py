"""End-to-end HTTP load benchmark kept separate from executor benchmarks."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import httpx

if TYPE_CHECKING:
    from minigpt.data import JsonValue
    from minigpt.serving import EngineMetrics

PromptKind: TypeAlias = Literal["short", "mixed"]
_CONCURRENCIES = (1, 2, 4, 8)
_PROMPT_KINDS: tuple[PromptKind, ...] = ("short", "mixed")
_STREAM_MODES = (False, True)
_MIXED_PROMPTS = ("ROMEO:", "JULIET:\n", "First Citizen:")
_HTTP_OK = 200


@dataclass(frozen=True, slots=True)
class HTTPBenchmarkConfig:
    """Define one bounded localhost HTTP benchmark matrix."""

    base_url: str
    requests_per_case: int = 8
    max_tokens: int = 16
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Reject unusable workload and transport settings."""
        if not self.base_url:
            reason = "base_url must be non-empty"
            raise ValueError(reason)
        if isinstance(self.requests_per_case, bool) or self.requests_per_case <= 0:
            reason = "requests_per_case must be a positive integer"
            raise ValueError(reason)
        if isinstance(self.max_tokens, bool) or self.max_tokens <= 0:
            reason = "max_tokens must be a positive integer"
            raise ValueError(reason)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            reason = "timeout_seconds must be finite and positive"
            raise ValueError(reason)


@dataclass(frozen=True, slots=True)
class RequestMeasurement:
    """Record one client-observed HTTP request."""

    concurrency: int
    prompt_kind: PromptKind
    stream: bool
    request_index: int
    prompt: str
    http_status: int | None
    completion_tokens: int
    ttft_seconds: float | None
    tpot_seconds: float | None
    e2e_seconds: float
    error_code: str | None
    error_message: str | None
    cancelled: bool

    def to_document(self) -> dict[str, JsonValue]:
        """Serialize without losing optional timing fields."""
        return {
            "concurrency": self.concurrency,
            "prompt_kind": self.prompt_kind,
            "stream": self.stream,
            "request_index": self.request_index,
            "prompt": self.prompt,
            "http_status": self.http_status,
            "completion_tokens": self.completion_tokens,
            "ttft_seconds": self.ttft_seconds,
            "tpot_seconds": self.tpot_seconds,
            "e2e_seconds": self.e2e_seconds,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "cancelled": self.cancelled,
        }


@dataclass(frozen=True, slots=True)
class LatencyPercentiles:
    """Hold standard client-observed latency percentiles."""

    p50: float | None
    p95: float | None
    p99: float | None

    def to_document(self) -> dict[str, JsonValue]:
        """Serialize percentile values."""
        return {"p50": self.p50, "p95": self.p95, "p99": self.p99}


@dataclass(frozen=True, slots=True)
class CaseSummary:
    """Summarize one concurrency/prompt/stream workload case."""

    concurrency: int
    prompt_kind: PromptKind
    stream: bool
    requests: int
    successful_requests: int
    duration_seconds: float
    requests_per_second: float
    generated_tokens_per_second: float
    generated_tokens: int
    http_error_count: int
    cancellation_count: int
    ttft_seconds: LatencyPercentiles
    tpot_seconds: LatencyPercentiles
    e2e_seconds: LatencyPercentiles

    def to_document(self) -> dict[str, JsonValue]:
        """Serialize one matrix cell."""
        return {
            "concurrency": self.concurrency,
            "prompt_kind": self.prompt_kind,
            "stream": self.stream,
            "requests": self.requests,
            "successful_requests": self.successful_requests,
            "duration_seconds": self.duration_seconds,
            "requests_per_second": self.requests_per_second,
            "generated_tokens_per_second": self.generated_tokens_per_second,
            "generated_tokens": self.generated_tokens,
            "http_error_count": self.http_error_count,
            "cancellation_count": self.cancellation_count,
            "ttft_seconds": self.ttft_seconds.to_document(),
            "tpot_seconds": self.tpot_seconds.to_document(),
            "e2e_seconds": self.e2e_seconds.to_document(),
        }


@dataclass(frozen=True, slots=True)
class EngineBenchmarkSnapshot:
    """Capture scheduler and Stage 11 batch utilization after HTTP load."""

    total_requests: int
    completed_requests: int
    cancelled_requests: int
    failed_requests: int
    peak_active_requests: int
    peak_waiting_requests: int
    peak_reserved_cache_tokens: int
    decode_batch_sizes: tuple[int, ...]
    prefill_batch_sizes: tuple[int, ...]
    average_decode_batch_size: float
    average_prefill_batch_size: float

    @classmethod
    def from_metrics(cls, metrics: EngineMetrics) -> EngineBenchmarkSnapshot:
        """Select HTTP benchmark scheduler/batching evidence."""
        return cls(
            total_requests=metrics.total_requests,
            completed_requests=metrics.completed_requests,
            cancelled_requests=metrics.cancelled_requests,
            failed_requests=metrics.failed_requests,
            peak_active_requests=metrics.peak_active_requests,
            peak_waiting_requests=metrics.peak_waiting_requests,
            peak_reserved_cache_tokens=metrics.peak_reserved_cache_tokens,
            decode_batch_sizes=metrics.decode_batch_sizes,
            prefill_batch_sizes=metrics.prefill_batch_sizes,
            average_decode_batch_size=metrics.average_decode_batch_size,
            average_prefill_batch_size=metrics.average_prefill_batch_size,
        )

    def to_document(self) -> dict[str, JsonValue]:
        """Serialize the engine-side snapshot."""
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "cancelled_requests": self.cancelled_requests,
            "failed_requests": self.failed_requests,
            "peak_active_requests": self.peak_active_requests,
            "peak_waiting_requests": self.peak_waiting_requests,
            "peak_reserved_cache_tokens": self.peak_reserved_cache_tokens,
            "decode_batch_sizes": list(self.decode_batch_sizes),
            "prefill_batch_sizes": list(self.prefill_batch_sizes),
            "average_decode_batch_size": self.average_decode_batch_size,
            "average_prefill_batch_size": self.average_prefill_batch_size,
        }


@dataclass(frozen=True, slots=True)
class HTTPBenchmarkResult:
    """Contain the independent HTTP benchmark matrix and raw measurements."""

    started_at_utc: str
    duration_seconds: float
    config: HTTPBenchmarkConfig
    cases: tuple[CaseSummary, ...]
    measurements: tuple[RequestMeasurement, ...]
    engine: EngineBenchmarkSnapshot | None = None

    def with_engine_metrics(self, metrics: EngineMetrics) -> HTTPBenchmarkResult:
        """Attach an owner-thread snapshot after the HTTP workload finishes."""
        return HTTPBenchmarkResult(
            started_at_utc=self.started_at_utc,
            duration_seconds=self.duration_seconds,
            config=self.config,
            cases=self.cases,
            measurements=self.measurements,
            engine=EngineBenchmarkSnapshot.from_metrics(metrics),
        )

    def to_document(self) -> dict[str, JsonValue]:
        """Serialize the complete benchmark result."""
        config_document: dict[str, JsonValue] = {
            "base_url": self.config.base_url,
            "requests_per_case": self.config.requests_per_case,
            "max_tokens": self.config.max_tokens,
            "timeout_seconds": self.config.timeout_seconds,
        }
        return {
            "schema_version": 1,
            "benchmark_scope": "http_end_to_end",
            "started_at_utc": self.started_at_utc,
            "duration_seconds": self.duration_seconds,
            "config": config_document,
            "cases": [case.to_document() for case in self.cases],
            "measurements": [measurement.to_document() for measurement in self.measurements],
            "engine": None if self.engine is None else self.engine.to_document(),
        }


async def run_http_benchmark(config: HTTPBenchmarkConfig) -> HTTPBenchmarkResult:
    """Execute the fixed HTTP matrix against an already-running server."""
    started_at = datetime.now(UTC).isoformat()
    benchmark_start = time.perf_counter()
    measurements: list[RequestMeasurement] = []
    summaries: list[CaseSummary] = []
    max_connections = max(_CONCURRENCIES)
    async with httpx.AsyncClient(
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        ),
    ) as client:
        for concurrency in _CONCURRENCIES:
            for prompt_kind in _PROMPT_KINDS:
                for stream in _STREAM_MODES:
                    case_measurements, summary = await _run_case(
                        client=client,
                        config=config,
                        concurrency=concurrency,
                        prompt_kind=prompt_kind,
                        stream=stream,
                    )
                    measurements.extend(case_measurements)
                    summaries.append(summary)
    return HTTPBenchmarkResult(
        started_at_utc=started_at,
        duration_seconds=max(0.0, time.perf_counter() - benchmark_start),
        config=config,
        cases=tuple(summaries),
        measurements=tuple(measurements),
    )


async def _run_case(
    *,
    client: httpx.AsyncClient,
    config: HTTPBenchmarkConfig,
    concurrency: int,
    prompt_kind: PromptKind,
    stream: bool,
) -> tuple[tuple[RequestMeasurement, ...], CaseSummary]:
    request_count = max(config.requests_per_case, concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def measured(request_index: int) -> RequestMeasurement:
        async with semaphore:
            prompt = _prompt(prompt_kind, request_index)
            return await _measure_request(
                client=client,
                config=config,
                concurrency=concurrency,
                prompt_kind=prompt_kind,
                stream=stream,
                request_index=request_index,
                prompt=prompt,
            )

    case_start = time.perf_counter()
    values = tuple(await asyncio.gather(*(measured(index) for index in range(request_count))))
    duration = max(0.0, time.perf_counter() - case_start)
    return values, _summarize_case(
        values,
        concurrency=concurrency,
        prompt_kind=prompt_kind,
        stream=stream,
        duration_seconds=duration,
    )


async def _measure_request(  # noqa: PLR0913
    *,
    client: httpx.AsyncClient,
    config: HTTPBenchmarkConfig,
    concurrency: int,
    prompt_kind: PromptKind,
    stream: bool,
    request_index: int,
    prompt: str,
) -> RequestMeasurement:
    start = time.perf_counter()
    try:
        if stream:
            return await _measure_stream(
                client=client,
                config=config,
                concurrency=concurrency,
                prompt_kind=prompt_kind,
                request_index=request_index,
                prompt=prompt,
                start=start,
            )
        response = await client.post(
            "/v1/completions",
            json=_payload(prompt, config.max_tokens, request_index, stream=False),
        )
        e2e = max(0.0, time.perf_counter() - start)
        document = cast("JsonValue", response.json())
        completion_tokens = _completion_tokens(document)
        error_code, error_message = _error(document)
        return RequestMeasurement(
            concurrency=concurrency,
            prompt_kind=prompt_kind,
            stream=False,
            request_index=request_index,
            prompt=prompt,
            http_status=response.status_code,
            completion_tokens=completion_tokens,
            ttft_seconds=e2e if response.status_code == _HTTP_OK else None,
            tpot_seconds=None,
            e2e_seconds=e2e,
            error_code=error_code,
            error_message=error_message,
            cancelled=_is_cancelled(error_code),
        )
    except Exception as error:  # noqa: BLE001
        return _exception_measurement(
            concurrency=concurrency,
            prompt_kind=prompt_kind,
            stream=stream,
            request_index=request_index,
            prompt=prompt,
            start=start,
            error=error,
        )


async def _measure_stream(  # noqa: PLR0913
    *,
    client: httpx.AsyncClient,
    config: HTTPBenchmarkConfig,
    concurrency: int,
    prompt_kind: PromptKind,
    request_index: int,
    prompt: str,
    start: float,
) -> RequestMeasurement:
    token_times: list[float] = []
    completion_tokens = 0
    error_code: str | None = None
    error_message: str | None = None
    async with client.stream(
        "POST",
        "/v1/completions",
        json=_payload(prompt, config.max_tokens, request_index, stream=True),
    ) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            document = cast("JsonValue", json.loads(line.removeprefix("data: ")))
            chunk_text = _choice_text(document)
            if chunk_text:
                token_times.append(time.perf_counter())
                completion_tokens += len(chunk_text)
            chunk_error_code, chunk_error_message = _error(document)
            if chunk_error_code is not None:
                error_code = chunk_error_code
                error_message = chunk_error_message
    e2e = max(0.0, time.perf_counter() - start)
    ttft = None if not token_times else max(0.0, token_times[0] - start)
    tpot = _tpot(token_times)
    return RequestMeasurement(
        concurrency=concurrency,
        prompt_kind=prompt_kind,
        stream=True,
        request_index=request_index,
        prompt=prompt,
        http_status=response.status_code,
        completion_tokens=completion_tokens,
        ttft_seconds=ttft,
        tpot_seconds=tpot,
        e2e_seconds=e2e,
        error_code=error_code,
        error_message=error_message,
        cancelled=_is_cancelled(error_code),
    )


def _payload(
    prompt: str, max_tokens: int, request_index: int, *, stream: bool
) -> dict[str, object]:
    return {
        "model": "minigpt-char",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "stream": stream,
        "seed": request_index,
    }


def _prompt(kind: PromptKind, request_index: int) -> str:
    if kind == "short":
        return "ROMEO:"
    return _MIXED_PROMPTS[request_index % len(_MIXED_PROMPTS)]


def _completion_tokens(document: JsonValue) -> int:
    if not isinstance(document, dict):
        return 0
    usage = document.get("usage")
    if not isinstance(usage, dict):
        return 0
    value = usage.get("completion_tokens")
    return value if type(value) is int else 0


def _choice_text(document: JsonValue) -> str:
    if not isinstance(document, dict):
        return ""
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    text = choice.get("text")
    return text if isinstance(text, str) else ""


def _error(document: JsonValue) -> tuple[str | None, str | None]:
    if not isinstance(document, dict):
        return None, None
    error = document.get("error")
    if not isinstance(error, dict):
        return None, None
    code = error.get("code")
    message = error.get("message")
    return (
        code if isinstance(code, str) else None,
        message if isinstance(message, str) else None,
    )


def _is_cancelled(error_code: str | None) -> bool:
    return error_code in {"request_cancelled", "stream_backpressure"}


def _tpot(token_times: list[float]) -> float | None:
    if not token_times:
        return None
    if len(token_times) == 1:
        return 0.0
    return max(0.0, token_times[-1] - token_times[0]) / (len(token_times) - 1)


def _exception_measurement(  # noqa: PLR0913
    *,
    concurrency: int,
    prompt_kind: PromptKind,
    stream: bool,
    request_index: int,
    prompt: str,
    start: float,
    error: Exception,
) -> RequestMeasurement:
    return RequestMeasurement(
        concurrency=concurrency,
        prompt_kind=prompt_kind,
        stream=stream,
        request_index=request_index,
        prompt=prompt,
        http_status=None,
        completion_tokens=0,
        ttft_seconds=None,
        tpot_seconds=None,
        e2e_seconds=max(0.0, time.perf_counter() - start),
        error_code=type(error).__name__,
        error_message=str(error),
        cancelled=False,
    )


def _summarize_case(
    values: tuple[RequestMeasurement, ...],
    *,
    concurrency: int,
    prompt_kind: PromptKind,
    stream: bool,
    duration_seconds: float,
) -> CaseSummary:
    successful = tuple(
        value for value in values if value.http_status == _HTTP_OK and not value.error_code
    )
    generated_tokens = sum(value.completion_tokens for value in successful)
    return CaseSummary(
        concurrency=concurrency,
        prompt_kind=prompt_kind,
        stream=stream,
        requests=len(values),
        successful_requests=len(successful),
        duration_seconds=duration_seconds,
        requests_per_second=(len(successful) / duration_seconds if duration_seconds > 0.0 else 0.0),
        generated_tokens_per_second=(
            generated_tokens / duration_seconds if duration_seconds > 0.0 else 0.0
        ),
        generated_tokens=generated_tokens,
        http_error_count=len(values) - len(successful),
        cancellation_count=sum(value.cancelled for value in values),
        ttft_seconds=_percentiles(
            tuple(value.ttft_seconds for value in successful if value.ttft_seconds is not None)
        ),
        tpot_seconds=_percentiles(
            tuple(value.tpot_seconds for value in successful if value.tpot_seconds is not None)
        ),
        e2e_seconds=_percentiles(tuple(value.e2e_seconds for value in successful)),
    )


def _percentiles(values: tuple[float, ...]) -> LatencyPercentiles:
    if not values:
        return LatencyPercentiles(p50=None, p95=None, p99=None)
    ordered = tuple(sorted(values))
    return LatencyPercentiles(
        p50=_percentile(ordered, 0.50),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
    )


def _percentile(ordered: tuple[float, ...], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
