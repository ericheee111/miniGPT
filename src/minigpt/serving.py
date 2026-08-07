"""Deterministic multi-request serving control plane with a reference executor."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Never, Protocol, Self, cast, final

import torch
from torch import Tensor
from torch.nn import functional
from typing_extensions import override

from minigpt.layers import KVCache, LayerKVCache

if TYPE_CHECKING:
    from minigpt.model import GPT

Clock = Callable[[], float]
_LOGICAL_TIME_DECIMALS = 12


class RequestStatus(StrEnum):
    """Describe one request's serving lifecycle state."""

    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


class EngineEventType(StrEnum):
    """Identify deterministic control-plane transitions and outputs."""

    SUBMITTED = "submitted"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    ADMITTED = "admitted"
    PREFILL_STARTED = "prefill_started"
    TOKEN = "token"  # noqa: S105
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class InvalidServingConfigError(ValueError):
    """Report an invalid request, scheduler, or engine setting."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed serving constraint."""
        return f"invalid serving configuration: {self.reason}"


@dataclass(frozen=True, slots=True)
class DuplicateRequestError(ValueError):
    """Report a request ID that already belongs to the engine."""

    request_id: str

    @override
    def __str__(self) -> str:
        """Render the duplicate request ID."""
        return f"request ID {self.request_id!r} already exists"


@dataclass(frozen=True, slots=True)
class UnknownRequestError(KeyError):
    """Report a request ID unknown to the engine."""

    request_id: str

    @override
    def __str__(self) -> str:
        """Render the missing request ID."""
        return f"unknown request ID {self.request_id!r}"


@dataclass(frozen=True, slots=True)
class EngineRunawayError(RuntimeError):
    """Report an engine that did not become idle within its explicit bound."""

    max_ticks: int

    @override
    def __str__(self) -> str:
        """Render the exceeded tick limit."""
        return f"serving engine did not become idle within {self.max_ticks} ticks"


def _invalid(reason: str) -> Never:
    raise InvalidServingConfigError(reason)


def _finite_non_negative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        _invalid(f"{name} must be finite and non-negative")


def _logical_elapsed(clock: Clock, start: float) -> float:
    """Normalize injected logical-clock subtraction across executor call order."""
    return round(max(0.0, clock() - start), _LOGICAL_TIME_DECIMALS)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Define one independently sampled autoregressive request."""

    request_id: str
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int | None = None
    seed: int = 0
    arrival_time: float = 0.0
    cancellation_time: float | None = None

    def __post_init__(self) -> None:
        """Reject malformed requests before they enter scheduler state."""
        if not self.request_id:
            _invalid("request_id must be non-empty")
        if not self.prompt_tokens:
            _invalid("prompt_tokens must be non-empty")
        if any(type(token) is not int or token < 0 for token in self.prompt_tokens):
            _invalid("prompt_tokens must contain non-negative integers")
        if isinstance(self.max_new_tokens, bool) or self.max_new_tokens < 0:
            _invalid("max_new_tokens must be a non-negative integer")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            _invalid("temperature must be finite and positive")
        if self.top_k is not None and (isinstance(self.top_k, bool) or self.top_k <= 0):
            _invalid("top_k must be null or a positive integer")
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 2**63:
            _invalid("seed must be an integer in [0, 2**63)")
        _finite_non_negative(self.arrival_time, "arrival_time")
        if self.cancellation_time is not None:
            _finite_non_negative(self.cancellation_time, "cancellation_time")
            if self.cancellation_time < self.arrival_time:
                _invalid("cancellation_time must not precede arrival_time")


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Configure strict FIFO admission and KV-cache reservations."""

    max_active_requests: int
    max_cached_tokens: int

    def __post_init__(self) -> None:
        """Require usable positive serving capacity."""
        if isinstance(self.max_active_requests, bool) or self.max_active_requests <= 0:
            _invalid("max_active_requests must be a positive integer")
        if isinstance(self.max_cached_tokens, bool) or self.max_cached_tokens <= 0:
            _invalid("max_cached_tokens must be a positive integer")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Bind a scheduler policy to the executor's learned-position capacity."""

    scheduler: SchedulerConfig
    block_size: int

    def __post_init__(self) -> None:
        """Reject an unusable learned-position window."""
        if isinstance(self.block_size, bool) or self.block_size <= 0:
            _invalid("block_size must be a positive integer")


@dataclass(slots=True)
class RequestState:
    """Hold mutable scheduler, cache, RNG, output, and timing state."""

    request: GenerationRequest
    generator: torch.Generator = field(repr=False)
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: list[int] = field(default_factory=list)
    kv_cache: KVCache | None = field(default=None, repr=False)
    reserved_cache_tokens: int = 0
    cached_tokens: int = 0
    admission_time: float | None = None
    prefill_start_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    prefill_latency_seconds: float | None = None
    decode_latencies_seconds: list[float] = field(default_factory=list)
    token_timestamps: list[float] = field(default_factory=list)
    failure_reason: str | None = None

    @property
    def all_tokens(self) -> tuple[int, ...]:
        """Return the immutable prompt followed by generated tokens."""
        return self.request.prompt_tokens + tuple(self.generated_tokens)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Return one isolated per-request executor outcome."""

    request_id: str
    token_id: int | None
    cache: KVCache | None
    cache_tokens: int
    latency_seconds: float
    error: str | None
    used_fallback: bool

    @classmethod
    def success(  # noqa: PLR0913
        cls,
        *,
        request_id: str,
        token_id: int,
        cache: KVCache,
        cache_tokens: int,
        latency_seconds: float,
        used_fallback: bool,
    ) -> Self:
        """Build a successful token result."""
        return cls(
            request_id=request_id,
            token_id=token_id,
            cache=cache,
            cache_tokens=cache_tokens,
            latency_seconds=latency_seconds,
            error=None,
            used_fallback=used_fallback,
        )

    @classmethod
    def failure(cls, request_id: str, error: str, latency_seconds: float) -> Self:
        """Build an isolated executor failure result."""
        return cls(
            request_id=request_id,
            token_id=None,
            cache=None,
            cache_tokens=0,
            latency_seconds=latency_seconds,
            error=error,
            used_fallback=False,
        )


@dataclass(frozen=True, slots=True)
class DecodeBatchObservation:
    """Separate decode batch utilization and wall-clock phase timings."""

    request_ids: tuple[str, ...]
    batch_size: int
    padded_cache_tokens: int
    useful_cache_tokens: int
    assembly_seconds: float
    model_seconds: float
    scatter_seconds: float
    executor_seconds: float
    batch_failed: bool


@dataclass(frozen=True, slots=True)
class PrefillBatchConfig:
    """Bound deterministic FIFO prompt grouping and dense padding."""

    max_batch_size: int = 8
    max_batch_tokens: int = 1024
    max_padding_ratio: float = 0.25

    def __post_init__(self) -> None:
        """Reject unusable batch and padding limits."""
        if isinstance(self.max_batch_size, bool) or self.max_batch_size <= 0:
            _invalid("prefill max_batch_size must be a positive integer")
        if isinstance(self.max_batch_tokens, bool) or self.max_batch_tokens <= 0:
            _invalid("prefill max_batch_tokens must be a positive integer")
        if not math.isfinite(self.max_padding_ratio) or not 0.0 <= self.max_padding_ratio <= 1.0:
            _invalid("prefill max_padding_ratio must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class PrefillBatchObservation:
    """Separate prefill utilization and wall-clock phase timings."""

    request_ids: tuple[str, ...]
    batch_size: int
    padded_prompt_tokens: int
    useful_prompt_tokens: int
    assembly_seconds: float
    model_seconds: float
    scatter_seconds: float
    executor_seconds: float
    started_at: float
    finished_at: float
    batch_failed: bool

    @property
    def padding_waste_ratio(self) -> float:
        """Return the fraction of dense prompt slots occupied only by padding."""
        if self.padded_prompt_tokens == 0:
            return 0.0
        return (self.padded_prompt_tokens - self.useful_prompt_tokens) / self.padded_prompt_tokens


class PrefillBatchEventType(StrEnum):
    """Identify executor-level prefill batch evidence without changing request events."""

    STARTED = "PREFILL_BATCH_STARTED"
    FINISHED = "PREFILL_BATCH_FINISHED"


@dataclass(frozen=True, slots=True)
class PrefillBatchEvent:
    """Record one prefill batch boundary and its utilization evidence."""

    sequence: int
    timestamp: float
    event_type: PrefillBatchEventType
    request_ids: tuple[str, ...]
    batch_size: int
    useful_prompt_tokens: int
    padded_prompt_tokens: int
    padding_waste_ratio: float
    batch_failed: bool


def _prefill_events(
    observation: PrefillBatchObservation,
    *,
    starting_sequence: int,
) -> tuple[PrefillBatchEvent, PrefillBatchEvent]:
    return (
        PrefillBatchEvent(
            sequence=starting_sequence,
            timestamp=observation.started_at,
            event_type=PrefillBatchEventType.STARTED,
            request_ids=observation.request_ids,
            batch_size=observation.batch_size,
            useful_prompt_tokens=observation.useful_prompt_tokens,
            padded_prompt_tokens=observation.padded_prompt_tokens,
            padding_waste_ratio=observation.padding_waste_ratio,
            batch_failed=observation.batch_failed,
        ),
        PrefillBatchEvent(
            sequence=starting_sequence + 1,
            timestamp=observation.finished_at,
            event_type=PrefillBatchEventType.FINISHED,
            request_ids=observation.request_ids,
            batch_size=observation.batch_size,
            useful_prompt_tokens=observation.useful_prompt_tokens,
            padded_prompt_tokens=observation.padded_prompt_tokens,
            padding_waste_ratio=observation.padding_waste_ratio,
            batch_failed=observation.batch_failed,
        ),
    )


class ServingExecutor(Protocol):
    """Advance request states without defining scheduler policy."""

    block_size: int

    @property
    def decode_observations(self) -> tuple[DecodeBatchObservation, ...]:
        """Return append-only model-batch telemetry."""
        ...

    @property
    def prefill_observations(self) -> tuple[PrefillBatchObservation, ...]:
        """Return append-only prompt-batch telemetry."""
        ...

    @property
    def prefill_events(self) -> tuple[PrefillBatchEvent, ...]:
        """Return executor-level prompt batch boundary evidence."""
        ...

    def prefill(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Evaluate prompts separately and return at most one token per request."""
        ...

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Evaluate one new token separately for each active request."""
        ...


@final
class ReferenceExecutor:
    """Call Stage 9 model interfaces once per request, without tensor batching."""

    def __init__(
        self,
        model: GPT,
        *,
        clock: Clock = time.perf_counter,
        telemetry_clock: Clock = time.perf_counter,
    ) -> None:
        """Bind a CPU model and injectable operation clock."""
        self._model = model
        self._clock = clock
        self._telemetry_clock = telemetry_clock
        self._decode_observations: list[DecodeBatchObservation] = []
        self._prefill_observations: list[PrefillBatchObservation] = []
        self._prefill_events: list[PrefillBatchEvent] = []
        self.block_size = model.config.block_size

    @property
    def decode_observations(self) -> tuple[DecodeBatchObservation, ...]:
        """Return per-request reference decode calls as size-one batches."""
        return tuple(self._decode_observations)

    @property
    def prefill_observations(self) -> tuple[PrefillBatchObservation, ...]:
        """Return one observation per reference prompt call."""
        return tuple(self._prefill_observations)

    @property
    def prefill_events(self) -> tuple[PrefillBatchEvent, ...]:
        """Return size-one reference prompt batch boundaries."""
        return tuple(self._prefill_events)

    def prefill(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Prefill each prompt independently and sample its first token."""
        return tuple(self._prefill_one(state, used_fallback=False) for state in requests)

    def prefill_fallback(self, state: RequestState) -> ExecutionResult:
        """Rebuild one full learned-position window for overflow decode."""
        return self._prefill_one(state, used_fallback=True)

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Decode once or rebuild a full learned-position window per request."""
        results: list[ExecutionResult] = []
        for state in active_requests:
            if state.cached_tokens >= self.block_size:
                results.append(self._prefill_one(state, used_fallback=True))
            else:
                results.append(self._decode_one(state))
        return tuple(results)

    def _prefill_one(self, state: RequestState, *, used_fallback: bool) -> ExecutionResult:
        logical_start = self._clock()
        executor_start = self._telemetry_clock()
        assembly_seconds = 0.0
        model_seconds = 0.0
        scatter_seconds = 0.0
        context = state.all_tokens[-self.block_size :]
        try:
            assembly_start = self._telemetry_clock()
            token_ids = torch.tensor(
                (context,),
                dtype=torch.long,
                device=self._model.token_embedding.weight.device,
            )
            assembly_seconds = max(0.0, self._telemetry_clock() - assembly_start)
            model_start = self._telemetry_clock()
            logits, cache = self._model.prefill(token_ids)
            model_seconds = max(0.0, self._telemetry_clock() - model_start)
            scatter_start = self._telemetry_clock()
            token_id = self._sample(state, logits[:, -1, :])
            scatter_seconds = max(0.0, self._telemetry_clock() - scatter_start)
        except Exception as error:  # noqa: BLE001
            elapsed = self._elapsed(logical_start)
            self._record_prefill(
                state,
                context_length=len(context),
                logical_start=logical_start,
                elapsed=elapsed,
                executor_start=executor_start,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=scatter_seconds,
                batch_failed=True,
            )
            return ExecutionResult.failure(
                state.request.request_id,
                f"{type(error).__name__}: {error}",
                elapsed,
            )
        elapsed = self._elapsed(logical_start)
        self._record_prefill(
            state,
            context_length=len(context),
            logical_start=logical_start,
            elapsed=elapsed,
            executor_start=executor_start,
            assembly_seconds=assembly_seconds,
            model_seconds=model_seconds,
            scatter_seconds=scatter_seconds,
            batch_failed=False,
        )
        return ExecutionResult.success(
            request_id=state.request.request_id,
            token_id=token_id,
            cache=cache,
            cache_tokens=len(context),
            latency_seconds=elapsed,
            used_fallback=used_fallback,
        )

    def _record_prefill(  # noqa: PLR0913
        self,
        state: RequestState,
        *,
        context_length: int,
        logical_start: float,
        elapsed: float,
        executor_start: float,
        assembly_seconds: float,
        model_seconds: float,
        scatter_seconds: float,
        batch_failed: bool,
    ) -> None:
        observation = PrefillBatchObservation(
            request_ids=(state.request.request_id,),
            batch_size=1,
            padded_prompt_tokens=context_length,
            useful_prompt_tokens=context_length,
            assembly_seconds=assembly_seconds,
            model_seconds=model_seconds,
            scatter_seconds=scatter_seconds,
            executor_seconds=max(0.0, self._telemetry_clock() - executor_start),
            started_at=logical_start,
            finished_at=logical_start + elapsed,
            batch_failed=batch_failed,
        )
        self._prefill_observations.append(observation)
        self._prefill_events.extend(
            _prefill_events(observation, starting_sequence=len(self._prefill_events))
        )

    def _decode_one(self, state: RequestState) -> ExecutionResult:
        start = self._clock()
        telemetry_start = self._telemetry_clock()
        if state.kv_cache is None or not state.generated_tokens:
            return ExecutionResult.failure(
                state.request.request_id,
                "decode requires a populated cache and generated token",
                self._elapsed(start),
            )
        try:
            token_ids = torch.tensor(
                ((state.generated_tokens[-1],),),
                dtype=torch.long,
                device=self._model.token_embedding.weight.device,
            )
            logits, cache = self._model.decode(token_ids, state.kv_cache)
            token_id = self._sample(state, logits[:, -1, :])
        except Exception as error:  # noqa: BLE001
            telemetry_elapsed = max(0.0, self._telemetry_clock() - telemetry_start)
            self._record_decode(state, telemetry_elapsed, batch_failed=True)
            return ExecutionResult.failure(
                state.request.request_id,
                f"{type(error).__name__}: {error}",
                self._elapsed(start),
            )
        telemetry_elapsed = max(0.0, self._telemetry_clock() - telemetry_start)
        self._record_decode(state, telemetry_elapsed, batch_failed=False)
        return ExecutionResult.success(
            request_id=state.request.request_id,
            token_id=token_id,
            cache=cache,
            cache_tokens=cache[0].length,
            latency_seconds=self._elapsed(start),
            used_fallback=False,
        )

    def _record_decode(
        self,
        state: RequestState,
        elapsed_seconds: float,
        *,
        batch_failed: bool,
    ) -> None:
        self._decode_observations.append(
            DecodeBatchObservation(
                request_ids=(state.request.request_id,),
                batch_size=1,
                padded_cache_tokens=state.cached_tokens,
                useful_cache_tokens=state.cached_tokens,
                assembly_seconds=0.0,
                model_seconds=elapsed_seconds,
                scatter_seconds=0.0,
                executor_seconds=elapsed_seconds,
                batch_failed=batch_failed,
            )
        )

    def _sample(self, state: RequestState, logits: Tensor) -> int:
        scaled = logits / state.request.temperature
        if state.request.top_k is not None:
            retained_count = min(state.request.top_k, self._model.config.vocab_size)
            retained = torch.topk(scaled, retained_count, dim=-1).values
            cutoff = retained[:, -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < cutoff, -torch.inf)
        probabilities = functional.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=state.generator)
        return int(sampled.item())

    def _elapsed(self, start: float) -> float:
        return _logical_elapsed(self._clock, start)


@final
class ContinuousDecodeExecutor:
    """Keep per-request prefill while batching valid single-token decode rows."""

    def __init__(
        self,
        model: GPT,
        *,
        clock: Clock = time.perf_counter,
        telemetry_clock: Clock = time.perf_counter,
    ) -> None:
        """Bind one model plus separate logical-result and telemetry clocks."""
        self._model = model
        self._clock = clock
        self._telemetry_clock = telemetry_clock
        self._reference = ReferenceExecutor(
            model,
            clock=clock,
            telemetry_clock=telemetry_clock,
        )
        self._decode_observations: list[DecodeBatchObservation] = []
        self.block_size = model.config.block_size

    @property
    def decode_observations(self) -> tuple[DecodeBatchObservation, ...]:
        """Return one observation for every actual continuous decode model call."""
        return tuple(self._decode_observations)

    @property
    def prefill_observations(self) -> tuple[PrefillBatchObservation, ...]:
        """Expose the delegated per-request prefill observations."""
        return self._reference.prefill_observations

    @property
    def prefill_events(self) -> tuple[PrefillBatchEvent, ...]:
        """Expose the delegated per-request prefill batch boundaries."""
        return self._reference.prefill_events

    def prefill(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Preserve Stage 10 per-request prompt execution."""
        return self._reference.prefill(requests)

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Isolate invalid/overflow requests and batch the remaining decode rows."""
        results: dict[str, ExecutionResult] = {}
        batch: list[RequestState] = []
        for state in active_requests:
            error = self._validate_decode_state(state)
            if error is not None:
                results[state.request.request_id] = ExecutionResult.failure(
                    state.request.request_id,
                    error,
                    0.0,
                )
            elif state.cached_tokens >= self.block_size:
                results[state.request.request_id] = self._reference.prefill_fallback(state)
            else:
                batch.append(state)
        if batch:
            for result in self.decode_batch(batch):
                results[result.request_id] = result
        return tuple(results[state.request.request_id] for state in active_requests)

    def decode_batch(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Assemble, execute, sample, and scatter one variable-length decode batch."""
        logical_start = self._clock()
        executor_start = self._telemetry_clock()
        request_ids = tuple(state.request.request_id for state in requests)
        try:
            assembly_start = self._telemetry_clock()
            token_ids, dense_cache, cache_lengths = self._assemble(requests)
            assembly_seconds = max(0.0, self._telemetry_clock() - assembly_start)
            model_start = self._telemetry_clock()
            logits, next_dense_cache = self._model.decode_batch(
                token_ids,
                dense_cache,
                cache_lengths,
            )
            model_seconds = max(0.0, self._telemetry_clock() - model_start)
            scatter_start = self._telemetry_clock()
            compact_caches = self._scatter(next_dense_cache, cache_lengths)
            scatter_seconds = max(0.0, self._telemetry_clock() - scatter_start)
        except Exception as error:  # noqa: BLE001
            elapsed = _logical_elapsed(self._clock, logical_start)
            executor_seconds = max(0.0, self._telemetry_clock() - executor_start)
            useful = sum(state.cached_tokens for state in requests)
            padded = len(requests) * max((state.cached_tokens for state in requests), default=0)
            self._decode_observations.append(
                DecodeBatchObservation(
                    request_ids=request_ids,
                    batch_size=len(requests),
                    padded_cache_tokens=padded,
                    useful_cache_tokens=useful,
                    assembly_seconds=0.0,
                    model_seconds=executor_seconds,
                    scatter_seconds=0.0,
                    executor_seconds=executor_seconds,
                    batch_failed=True,
                )
            )
            message = f"{type(error).__name__}: {error}"
            return tuple(
                ExecutionResult.failure(state.request.request_id, message, elapsed)
                for state in requests
            )

        elapsed = _logical_elapsed(self._clock, logical_start)
        results: list[ExecutionResult] = []
        for row, (state, cache) in enumerate(zip(requests, compact_caches, strict=True)):
            try:
                token_id = self._sample(state, logits[row : row + 1, -1, :])
            except Exception as error:  # noqa: BLE001
                results.append(
                    ExecutionResult.failure(
                        state.request.request_id,
                        f"{type(error).__name__}: {error}",
                        elapsed,
                    )
                )
                continue
            results.append(
                ExecutionResult.success(
                    request_id=state.request.request_id,
                    token_id=token_id,
                    cache=cache,
                    cache_tokens=state.cached_tokens + 1,
                    latency_seconds=elapsed,
                    used_fallback=False,
                )
            )
        executor_seconds = max(0.0, self._telemetry_clock() - executor_start)
        useful = int(cache_lengths.sum().item())
        padded = len(requests) * int(cache_lengths.max().item())
        self._decode_observations.append(
            DecodeBatchObservation(
                request_ids=request_ids,
                batch_size=len(requests),
                padded_cache_tokens=padded,
                useful_cache_tokens=useful,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=scatter_seconds,
                executor_seconds=executor_seconds,
                batch_failed=False,
            )
        )
        return tuple(results)

    def _validate_decode_state(  # noqa: C901, PLR0911
        self,
        state: RequestState,
    ) -> str | None:
        if state.kv_cache is None or not state.generated_tokens:
            return "decode requires a populated cache and generated token"
        cache = cast("tuple[object, ...]", state.kv_cache)
        if len(cache) != self._model.config.n_layer:
            return "invalid KV cache: layer count does not match model"
        expected_device = self._model.token_embedding.weight.device
        expected_dtype = self._model.token_embedding.weight.dtype
        head_size = self._model.config.n_embd // self._model.config.n_head
        for layer_index, layer_cache in enumerate(cache):
            if not isinstance(layer_cache, LayerKVCache):
                return f"invalid KV cache: layer {layer_index} is not LayerKVCache"
            if layer_cache.key.shape != layer_cache.value.shape:
                return f"invalid KV cache: layer {layer_index} key/value shapes differ"
            expected_shape = (
                1,
                self._model.config.n_head,
                state.cached_tokens,
                head_size,
            )
            if tuple(layer_cache.key.shape) != expected_shape:
                return (
                    f"invalid KV cache: layer {layer_index} shape "
                    f"{tuple(layer_cache.key.shape)} must equal {expected_shape}"
                )
            for tensor in (layer_cache.key, layer_cache.value):
                if tensor.dtype != expected_dtype:
                    return f"invalid KV cache: layer {layer_index} dtype must equal model dtype"
                if tensor.device != expected_device:
                    return f"invalid KV cache: layer {layer_index} device must equal model device"
                if tensor.requires_grad:
                    return f"invalid KV cache: layer {layer_index} cache must be detached"
        if state.cached_tokens <= 0 or state.cached_tokens > self.block_size:
            return "invalid KV cache: cached token length is outside model capacity"
        return None

    def _assemble(
        self,
        requests: Sequence[RequestState],
    ) -> tuple[Tensor, KVCache, Tensor]:
        batch_size = len(requests)
        padded_length = max(state.cached_tokens for state in requests)
        device = self._model.token_embedding.weight.device
        dtype = self._model.token_embedding.weight.dtype
        head_size = self._model.config.n_embd // self._model.config.n_head
        cache_lengths = torch.tensor(
            [state.cached_tokens for state in requests],
            dtype=torch.long,
            device=device,
        )
        token_ids = torch.tensor(
            [[state.generated_tokens[-1]] for state in requests],
            dtype=torch.long,
            device=device,
        )
        dense_layers: list[LayerKVCache] = []
        for layer_index in range(self._model.config.n_layer):
            key = torch.zeros(
                batch_size,
                self._model.config.n_head,
                padded_length,
                head_size,
                dtype=dtype,
                device=device,
            )
            value = torch.zeros_like(key)
            for row, state in enumerate(requests):
                cache = cast("KVCache", state.kv_cache)
                layer_cache = cache[layer_index]
                key[row, :, : state.cached_tokens, :] = layer_cache.key[0]
                value[row, :, : state.cached_tokens, :] = layer_cache.value[0]
            dense_layers.append(LayerKVCache(key=key.detach(), value=value.detach()))
        return token_ids, tuple(dense_layers), cache_lengths

    @staticmethod
    def _scatter(next_dense_cache: KVCache, cache_lengths: Tensor) -> tuple[KVCache, ...]:
        compact: list[KVCache] = []
        for row in range(cache_lengths.shape[0]):
            length = int(cache_lengths[row].item())
            layers: list[LayerKVCache] = []
            for dense_layer in next_dense_cache:
                key = torch.cat(
                    (
                        dense_layer.key[row : row + 1, :, :length, :],
                        dense_layer.key[row : row + 1, :, -1:, :],
                    ),
                    dim=2,
                ).clone()
                value = torch.cat(
                    (
                        dense_layer.value[row : row + 1, :, :length, :],
                        dense_layer.value[row : row + 1, :, -1:, :],
                    ),
                    dim=2,
                ).clone()
                layers.append(LayerKVCache(key=key.detach(), value=value.detach()))
            compact.append(tuple(layers))
        return tuple(compact)

    def _sample(self, state: RequestState, logits: Tensor) -> int:
        scaled = logits / state.request.temperature
        if state.request.top_k is not None:
            retained_count = min(state.request.top_k, self._model.config.vocab_size)
            retained = torch.topk(scaled, retained_count, dim=-1).values
            cutoff = retained[:, -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < cutoff, -torch.inf)
        probabilities = functional.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=state.generator)
        return int(sampled.item())


@final
class ContinuousExecutor:
    """Length-bucket prompt prefill and reuse Stage 11A continuous decode."""

    def __init__(
        self,
        model: GPT,
        *,
        prefill_config: PrefillBatchConfig | None = None,
        clock: Clock = time.perf_counter,
        telemetry_clock: Clock = time.perf_counter,
    ) -> None:
        """Bind deterministic prompt grouping and the existing decode executor."""
        self._model = model
        self._clock = clock
        self._telemetry_clock = telemetry_clock
        self.prefill_config = prefill_config or PrefillBatchConfig()
        self._decode = ContinuousDecodeExecutor(
            model,
            clock=clock,
            telemetry_clock=telemetry_clock,
        )
        self._prefill_observations: list[PrefillBatchObservation] = []
        self.block_size = model.config.block_size

    @property
    def decode_observations(self) -> tuple[DecodeBatchObservation, ...]:
        """Expose the unchanged Stage 11A decode telemetry."""
        return self._decode.decode_observations

    @property
    def prefill_observations(self) -> tuple[PrefillBatchObservation, ...]:
        """Return initial batched prefill plus Stage 9 overflow re-prefill calls."""
        return (*self._prefill_observations, *self._decode.prefill_observations)

    @property
    def prefill_events(self) -> tuple[PrefillBatchEvent, ...]:
        """Return resequenced boundaries for initial and overflow prefill calls."""
        events: list[PrefillBatchEvent] = []
        for observation in self.prefill_observations:
            events.extend(_prefill_events(observation, starting_sequence=len(events)))
        return tuple(events)

    def prefill(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Validate independently, group valid FIFO prefixes, and prefill each batch."""
        results: dict[str, ExecutionResult] = {}
        valid: list[RequestState] = []
        for state in requests:
            error = self._validate_prefill_state(state)
            if error is None:
                valid.append(state)
            else:
                results[state.request.request_id] = ExecutionResult.failure(
                    state.request.request_id,
                    error,
                    0.0,
                )
        for batch in self._batches(valid):
            for result in self.prefill_batch(batch):
                results[result.request_id] = result
        return tuple(results[state.request.request_id] for state in requests)

    def prefill_batch(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Assemble, execute, sample, and scatter one padded prompt batch."""
        logical_start = self._clock()
        executor_start = self._telemetry_clock()
        request_ids = tuple(state.request.request_id for state in requests)
        lengths = tuple(len(state.request.prompt_tokens) for state in requests)
        useful_tokens = sum(lengths)
        padded_tokens = len(requests) * max(lengths, default=0)
        assembly_seconds = 0.0
        model_seconds = 0.0
        scatter_seconds = 0.0
        try:
            assembly_start = self._telemetry_clock()
            token_ids, prompt_lengths = self._assemble_prefill(requests)
            assembly_seconds = max(0.0, self._telemetry_clock() - assembly_start)
            model_start = self._telemetry_clock()
            logits, dense_cache = self._model.prefill_batch(token_ids, prompt_lengths)
            model_seconds = max(0.0, self._telemetry_clock() - model_start)
            scatter_start = self._telemetry_clock()
            compact_caches = self._scatter_prefill(dense_cache, prompt_lengths)
            results: list[ExecutionResult] = []
            for row, (state, cache) in enumerate(zip(requests, compact_caches, strict=True)):
                try:
                    token_id = self._sample(state, logits[row : row + 1, -1, :])
                except Exception as error:  # noqa: BLE001
                    results.append(
                        ExecutionResult.failure(
                            state.request.request_id,
                            f"{type(error).__name__}: {error}",
                            0.0,
                        )
                    )
                    continue
                results.append(
                    ExecutionResult.success(
                        request_id=state.request.request_id,
                        token_id=token_id,
                        cache=cache,
                        cache_tokens=len(state.request.prompt_tokens),
                        latency_seconds=0.0,
                        used_fallback=False,
                    )
                )
            scatter_seconds = max(0.0, self._telemetry_clock() - scatter_start)
        except Exception as error:  # noqa: BLE001
            elapsed = _logical_elapsed(self._clock, logical_start)
            self._record_prefill_batch(
                request_ids=request_ids,
                useful_tokens=useful_tokens,
                padded_tokens=padded_tokens,
                logical_start=logical_start,
                elapsed=elapsed,
                executor_start=executor_start,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=scatter_seconds,
                batch_failed=True,
            )
            message = f"{type(error).__name__}: {error}"
            return tuple(
                ExecutionResult.failure(state.request.request_id, message, elapsed)
                for state in requests
            )

        elapsed = _logical_elapsed(self._clock, logical_start)
        results = [replace(result, latency_seconds=elapsed) for result in results]
        self._record_prefill_batch(
            request_ids=request_ids,
            useful_tokens=useful_tokens,
            padded_tokens=padded_tokens,
            logical_start=logical_start,
            elapsed=elapsed,
            executor_start=executor_start,
            assembly_seconds=assembly_seconds,
            model_seconds=model_seconds,
            scatter_seconds=scatter_seconds,
            batch_failed=False,
        )
        return tuple(results)

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Reuse Stage 11A validation, overflow fallback, batching, and sampling."""
        return self._decode.decode(active_requests)

    def _validate_prefill_state(self, state: RequestState) -> str | None:
        tokens = state.request.prompt_tokens
        if not tokens:
            return "prefill prompt must be non-empty"
        if len(tokens) > self.block_size:
            return f"prefill prompt length {len(tokens)} exceeds block_size {self.block_size}"
        if any(type(token) is not int for token in tokens):
            return "prefill prompt tokens must be integers"
        if any(token < 0 or token >= self._model.config.vocab_size for token in tokens):
            return "prefill prompt token is outside model vocabulary"
        return None

    def _batches(self, requests: Sequence[RequestState]) -> tuple[tuple[RequestState, ...], ...]:
        batches: list[tuple[RequestState, ...]] = []
        current: list[RequestState] = []
        for state in requests:
            if not current:
                current.append(state)
                continue
            candidate = [*current, state]
            lengths = [len(item.request.prompt_tokens) for item in candidate]
            padded = len(candidate) * max(lengths)
            useful = sum(lengths)
            waste_ratio = (padded - useful) / padded
            config = self.prefill_config
            fits = (
                len(candidate) <= config.max_batch_size
                and padded <= config.max_batch_tokens
                and waste_ratio <= config.max_padding_ratio
            )
            if fits:
                current.append(state)
            else:
                batches.append(tuple(current))
                current = [state]
        if current:
            batches.append(tuple(current))
        return tuple(batches)

    def _assemble_prefill(
        self,
        requests: Sequence[RequestState],
    ) -> tuple[Tensor, Tensor]:
        device = self._model.token_embedding.weight.device
        lengths = [len(state.request.prompt_tokens) for state in requests]
        token_ids = torch.zeros(
            (len(requests), max(lengths)),
            dtype=torch.long,
            device=device,
        )
        for row, state in enumerate(requests):
            row_tokens = torch.tensor(
                state.request.prompt_tokens,
                dtype=torch.long,
                device=device,
            )
            token_ids[row, : row_tokens.shape[0]] = row_tokens
        return token_ids, torch.tensor(lengths, dtype=torch.long, device=device)

    @staticmethod
    def _scatter_prefill(
        dense_cache: KVCache,
        prompt_lengths: Tensor,
    ) -> tuple[KVCache, ...]:
        compact: list[KVCache] = []
        for row, length_tensor in enumerate(prompt_lengths):
            length = int(length_tensor.item())
            layers = tuple(
                LayerKVCache(
                    key=layer.key[row : row + 1, :, :length, :].clone().detach(),
                    value=layer.value[row : row + 1, :, :length, :].clone().detach(),
                )
                for layer in dense_cache
            )
            compact.append(layers)
        return tuple(compact)

    def _sample(self, state: RequestState, logits: Tensor) -> int:
        scaled = logits / state.request.temperature
        if state.request.top_k is not None:
            retained_count = min(state.request.top_k, self._model.config.vocab_size)
            retained = torch.topk(scaled, retained_count, dim=-1).values
            cutoff = retained[:, -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < cutoff, -torch.inf)
        probabilities = functional.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=state.generator)
        return int(sampled.item())

    def _record_prefill_batch(  # noqa: PLR0913
        self,
        *,
        request_ids: tuple[str, ...],
        useful_tokens: int,
        padded_tokens: int,
        logical_start: float,
        elapsed: float,
        executor_start: float,
        assembly_seconds: float,
        model_seconds: float,
        scatter_seconds: float,
        batch_failed: bool,
    ) -> None:
        observation = PrefillBatchObservation(
            request_ids=request_ids,
            batch_size=len(request_ids),
            padded_prompt_tokens=padded_tokens,
            useful_prompt_tokens=useful_tokens,
            assembly_seconds=assembly_seconds,
            model_seconds=model_seconds,
            scatter_seconds=scatter_seconds,
            executor_seconds=max(0.0, self._telemetry_clock() - executor_start),
            started_at=logical_start,
            finished_at=logical_start + elapsed,
            batch_failed=batch_failed,
        )
        self._prefill_observations.append(observation)


@dataclass(frozen=True, slots=True)
class EngineEvent:
    """Record one ordered transition with a scheduler snapshot."""

    sequence: int
    timestamp: float
    event_type: EngineEventType
    request_id: str
    status: RequestStatus
    token_id: int | None
    detail: str | None
    used_fallback: bool
    active_requests: int
    waiting_requests: int
    cached_tokens: int
    reserved_cache_tokens: int


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    """Summarize one request's latency and output accounting."""

    request_id: str
    status: RequestStatus
    queue_time_seconds: float | None
    prefill_latency_seconds: float | None
    time_to_first_token_seconds: float | None
    decode_latencies_seconds: tuple[float, ...]
    time_per_output_token_seconds: float | None
    end_to_end_latency_seconds: float | None
    generated_tokens: int
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class EngineMetrics:
    """Summarize lifecycle counts, capacity, occupancy, and throughput."""

    total_requests: int
    completed_requests: int
    cancelled_requests: int
    failed_requests: int
    active_requests: int
    waiting_requests: int
    cached_tokens: int
    reserved_cache_tokens: int
    peak_active_requests: int
    peak_waiting_requests: int
    peak_cached_tokens: int
    peak_reserved_cache_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    request_throughput_per_second: float
    token_throughput_per_second: float
    decode_batch_sizes: tuple[int, ...]
    average_decode_batch_size: float
    max_decode_batch_size: int
    padded_cache_tokens: int
    useful_cache_tokens: int
    padding_waste_ratio: float
    executor_time_seconds: float
    model_execution_time_seconds: float
    batch_assembly_scatter_time_seconds: float
    prefill_batch_sizes: tuple[int, ...]
    average_prefill_batch_size: float
    max_prefill_batch_size: int
    padded_prompt_tokens: int
    useful_prompt_tokens: int
    prompt_padding_waste_ratio: float
    prefill_executor_time_seconds: float
    prefill_model_execution_time_seconds: float
    prefill_batch_assembly_scatter_time_seconds: float


@final
class ServingEngine:
    """Advance independent requests through a deterministic FIFO engine loop."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        executor: ServingExecutor,
        clock: Clock = time.perf_counter,
    ) -> None:
        """Bind deterministic scheduler state to an executor."""
        if executor.block_size != config.block_size:
            reason = "executor block_size must equal EngineConfig.block_size"
            _invalid(reason)
        self.config = config
        self._executor = executor
        self._clock = clock
        self._states: dict[str, RequestState] = {}
        self._waiting: deque[str] = deque()
        self._active: list[str] = []
        self._pending_cancellations: set[str] = set()
        self._events: list[EngineEvent] = []
        self._last_tick_time: float | None = None
        self._peak_active = 0
        self._peak_waiting = 0
        self._peak_cached = 0
        self._peak_reserved = 0

    @property
    def events(self) -> tuple[EngineEvent, ...]:
        """Return an immutable snapshot of the append-only event stream."""
        return tuple(self._events)

    @property
    def prefill_events(self) -> tuple[PrefillBatchEvent, ...]:
        """Return executor-level prompt batch evidence separately from request events."""
        return self._executor.prefill_events

    @property
    def is_idle(self) -> bool:
        """Return whether no waiting or active request remains."""
        return not self._waiting and not self._active

    def submit(self, request: GenerationRequest) -> None:
        """Append a validated request to the FIFO waiting queue."""
        if request.request_id in self._states:
            raise DuplicateRequestError(request.request_id)
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        state = RequestState(request=request, generator=generator)
        self._states[request.request_id] = state
        self._waiting.append(request.request_id)
        self._emit(
            event_type=EngineEventType.SUBMITTED,
            state=state,
            timestamp=request.arrival_time,
        )

    def cancel(self, request_id: str, *, at: float | None = None) -> None:
        """Queue an idempotent cancellation for the next tick's first phase."""
        state = self.request_state(request_id)
        if state.status in self._terminal_statuses():
            return
        self._pending_cancellations.add(request_id)
        timestamp = self._clock() if at is None else at
        _finite_non_negative(timestamp, "cancellation timestamp")
        self._emit(
            event_type=EngineEventType.CANCELLATION_REQUESTED,
            state=state,
            timestamp=timestamp,
        )

    def request_state(self, request_id: str) -> RequestState:
        """Return mutable state for inspection or executor input."""
        try:
            return self._states[request_id]
        except KeyError:
            raise UnknownRequestError(request_id) from None

    def tick(self, *, now: float | None = None) -> None:
        """Apply cancellation, admission, then at-most-once prefill/decode work."""
        tick_time = self._clock() if now is None else now
        _finite_non_negative(tick_time, "tick time")
        if self._last_tick_time is not None and tick_time < self._last_tick_time:
            _invalid("tick time must be monotonic")
        self._last_tick_time = tick_time
        prefill_ids = tuple(
            request_id
            for request_id in self._active
            if self._states[request_id].status is RequestStatus.PREFILLING
        )
        decode_ids = tuple(
            request_id
            for request_id in self._active
            if self._states[request_id].status is RequestStatus.DECODING
        )

        self._apply_cancellations(tick_time)
        self._admit(tick_time)
        self._execute(prefill_ids, tick_time=tick_time, prefill=True)
        self._execute(decode_ids, tick_time=tick_time, prefill=False)
        self._update_peaks()

    def run_until_idle(
        self,
        *,
        start_time: float,
        tick_seconds: float,
        max_ticks: int,
    ) -> None:
        """Advance a finite workload on a deterministic logical clock."""
        _finite_non_negative(start_time, "start_time")
        if not math.isfinite(tick_seconds) or tick_seconds <= 0.0:
            _invalid("tick_seconds must be finite and positive")
        if isinstance(max_ticks, bool) or max_ticks <= 0:
            _invalid("max_ticks must be a positive integer")
        for tick_index in range(max_ticks):
            if self.is_idle:
                return
            self.tick(now=start_time + tick_index * tick_seconds)
        if not self.is_idle:
            raise EngineRunawayError(max_ticks)

    def request_metrics(self, request_id: str) -> RequestMetrics:
        """Calculate stable latency definitions for one request."""
        state = self.request_state(request_id)
        arrival = state.request.arrival_time
        queue_time = (
            None if state.admission_time is None else max(0.0, state.admission_time - arrival)
        )
        ttft = None if state.first_token_time is None else state.first_token_time - arrival
        e2e = None if state.finish_time is None else state.finish_time - arrival
        decode_latencies = tuple(state.decode_latencies_seconds)
        tpot = None
        if state.first_token_time is not None:
            tpot = sum(decode_latencies) / len(decode_latencies) if decode_latencies else 0.0
        return RequestMetrics(
            request_id=request_id,
            status=state.status,
            queue_time_seconds=queue_time,
            prefill_latency_seconds=state.prefill_latency_seconds,
            time_to_first_token_seconds=ttft,
            decode_latencies_seconds=decode_latencies,
            time_per_output_token_seconds=tpot,
            end_to_end_latency_seconds=e2e,
            generated_tokens=len(state.generated_tokens),
            failure_reason=state.failure_reason,
        )

    def metrics(self) -> EngineMetrics:
        """Calculate engine totals and descriptive workload throughput."""
        states = tuple(self._states.values())
        completed = sum(state.status is RequestStatus.FINISHED for state in states)
        cancelled = sum(state.status is RequestStatus.CANCELLED for state in states)
        failed = sum(state.status is RequestStatus.FAILED for state in states)
        generated = sum(len(state.generated_tokens) for state in states)
        finishes = [state.finish_time for state in states if state.finish_time is not None]
        if states and finishes:
            elapsed = max(0.0, max(finishes) - min(state.request.arrival_time for state in states))
        else:
            elapsed = 0.0
        request_throughput = completed / elapsed if elapsed > 0.0 else 0.0
        token_throughput = generated / elapsed if elapsed > 0.0 else 0.0
        observations = self._executor.decode_observations
        batch_sizes = tuple(observation.batch_size for observation in observations)
        padded_cache_tokens = sum(observation.padded_cache_tokens for observation in observations)
        useful_cache_tokens = sum(observation.useful_cache_tokens for observation in observations)
        padding_waste = padded_cache_tokens - useful_cache_tokens
        padding_waste_ratio = (
            padding_waste / padded_cache_tokens if padded_cache_tokens > 0 else 0.0
        )
        prefill_observations = self._executor.prefill_observations
        prefill_batch_sizes = tuple(observation.batch_size for observation in prefill_observations)
        padded_prompt_tokens = sum(
            observation.padded_prompt_tokens for observation in prefill_observations
        )
        useful_prompt_tokens = sum(
            observation.useful_prompt_tokens for observation in prefill_observations
        )
        prompt_padding_waste = padded_prompt_tokens - useful_prompt_tokens
        prompt_padding_waste_ratio = (
            prompt_padding_waste / padded_prompt_tokens if padded_prompt_tokens > 0 else 0.0
        )
        return EngineMetrics(
            total_requests=len(states),
            completed_requests=completed,
            cancelled_requests=cancelled,
            failed_requests=failed,
            active_requests=len(self._active),
            waiting_requests=len(self._waiting),
            cached_tokens=self._cached_tokens(),
            reserved_cache_tokens=self._reserved_cache_tokens(),
            peak_active_requests=self._peak_active,
            peak_waiting_requests=self._peak_waiting,
            peak_cached_tokens=self._peak_cached,
            peak_reserved_cache_tokens=self._peak_reserved,
            generated_tokens=generated,
            elapsed_seconds=elapsed,
            request_throughput_per_second=request_throughput,
            token_throughput_per_second=token_throughput,
            decode_batch_sizes=batch_sizes,
            average_decode_batch_size=(sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0),
            max_decode_batch_size=max(batch_sizes, default=0),
            padded_cache_tokens=padded_cache_tokens,
            useful_cache_tokens=useful_cache_tokens,
            padding_waste_ratio=padding_waste_ratio,
            executor_time_seconds=sum(observation.executor_seconds for observation in observations),
            model_execution_time_seconds=sum(
                observation.model_seconds for observation in observations
            ),
            batch_assembly_scatter_time_seconds=sum(
                observation.assembly_seconds + observation.scatter_seconds
                for observation in observations
            ),
            prefill_batch_sizes=prefill_batch_sizes,
            average_prefill_batch_size=(
                sum(prefill_batch_sizes) / len(prefill_batch_sizes) if prefill_batch_sizes else 0.0
            ),
            max_prefill_batch_size=max(prefill_batch_sizes, default=0),
            padded_prompt_tokens=padded_prompt_tokens,
            useful_prompt_tokens=useful_prompt_tokens,
            prompt_padding_waste_ratio=prompt_padding_waste_ratio,
            prefill_executor_time_seconds=sum(
                observation.executor_seconds for observation in prefill_observations
            ),
            prefill_model_execution_time_seconds=sum(
                observation.model_seconds for observation in prefill_observations
            ),
            prefill_batch_assembly_scatter_time_seconds=sum(
                observation.assembly_seconds + observation.scatter_seconds
                for observation in prefill_observations
            ),
        )

    def _apply_cancellations(self, tick_time: float) -> None:
        for state in tuple(self._states.values()):
            scheduled = (
                state.request.cancellation_time is not None
                and state.request.cancellation_time <= tick_time
            )
            if state.request.request_id not in self._pending_cancellations and not scheduled:
                continue
            self._pending_cancellations.discard(state.request.request_id)
            if state.status in self._terminal_statuses():
                continue
            self._remove_live_state(state)
            state.status = RequestStatus.CANCELLED
            state.finish_time = tick_time
            self._emit(
                event_type=EngineEventType.CANCELLED,
                state=state,
                timestamp=tick_time,
            )

    def _admit(self, tick_time: float) -> None:
        scheduler = self.config.scheduler
        while self._waiting and len(self._active) < scheduler.max_active_requests:
            state = self._states[self._waiting[0]]
            required = self._reservation_for(state.request)
            if required > scheduler.max_cached_tokens:
                _ = self._waiting.popleft()
                state.failure_reason = (
                    f"required cache reservation {required} exceeds budget "
                    f"{scheduler.max_cached_tokens}"
                )
                state.status = RequestStatus.FAILED
                state.finish_time = tick_time
                self._emit(
                    event_type=EngineEventType.FAILED,
                    state=state,
                    timestamp=tick_time,
                    detail=state.failure_reason,
                )
                continue
            if self._reserved_cache_tokens() + required > scheduler.max_cached_tokens:
                break
            _ = self._waiting.popleft()
            state.admission_time = tick_time
            state.reserved_cache_tokens = required
            if state.request.max_new_tokens == 0:
                state.status = RequestStatus.FINISHED
                state.finish_time = tick_time
                state.reserved_cache_tokens = 0
                self._emit(
                    event_type=EngineEventType.ADMITTED,
                    state=state,
                    timestamp=tick_time,
                )
                self._emit(
                    event_type=EngineEventType.FINISHED,
                    state=state,
                    timestamp=tick_time,
                )
                continue
            state.status = RequestStatus.PREFILLING
            self._active.append(state.request.request_id)
            self._emit(
                event_type=EngineEventType.ADMITTED,
                state=state,
                timestamp=tick_time,
            )

    def _execute(self, request_ids: tuple[str, ...], *, tick_time: float, prefill: bool) -> None:
        expected_status = RequestStatus.PREFILLING if prefill else RequestStatus.DECODING
        states = tuple(
            self._states[request_id]
            for request_id in request_ids
            if self._states[request_id].status is expected_status
        )
        if not states:
            return
        if prefill:
            for state in states:
                state.prefill_start_time = tick_time
                self._emit(
                    event_type=EngineEventType.PREFILL_STARTED,
                    state=state,
                    timestamp=tick_time,
                )
        try:
            results = self._executor.prefill(states) if prefill else self._executor.decode(states)
        except Exception as error:  # noqa: BLE001
            message = f"{type(error).__name__}: {error}"
            results = tuple(
                ExecutionResult.failure(state.request.request_id, message, 0.0) for state in states
            )
        result_by_id = {result.request_id: result for result in results}
        duplicate_count = len(results) - len(result_by_id)
        for state in states:
            result = result_by_id.get(state.request.request_id)
            if result is None or duplicate_count:
                self._fail_state(state, tick_time, "executor returned malformed request results")
                continue
            self._apply_result(state, result, tick_time=tick_time, prefill=prefill)

    def _apply_result(
        self,
        state: RequestState,
        result: ExecutionResult,
        *,
        tick_time: float,
        prefill: bool,
    ) -> None:
        if result.error is not None:
            self._fail_state(state, tick_time + result.latency_seconds, result.error)
            return
        if result.token_id is None or result.cache is None:
            self._fail_state(state, tick_time, "executor success omitted token or cache")
            return
        if result.latency_seconds < 0.0 or not math.isfinite(result.latency_seconds):
            self._fail_state(state, tick_time, "executor returned invalid latency")
            return
        if result.cache_tokens < 0 or result.cache_tokens > state.reserved_cache_tokens:
            reason = (
                f"executor cache occupancy {result.cache_tokens} exceeds reservation "
                f"{state.reserved_cache_tokens}"
            )
            self._fail_state(state, tick_time + result.latency_seconds, reason)
            return
        token_time = tick_time + result.latency_seconds
        state.generated_tokens.append(result.token_id)
        state.kv_cache = result.cache
        state.cached_tokens = result.cache_tokens
        state.token_timestamps.append(token_time)
        if prefill:
            state.prefill_latency_seconds = result.latency_seconds
            state.first_token_time = token_time
        else:
            state.decode_latencies_seconds.append(result.latency_seconds)
        self._emit(
            event_type=EngineEventType.TOKEN,
            state=state,
            timestamp=token_time,
            token_id=result.token_id,
            used_fallback=result.used_fallback,
        )
        if len(state.generated_tokens) >= state.request.max_new_tokens:
            self._finish_state(state, token_time)
        else:
            state.status = RequestStatus.DECODING

    def _finish_state(self, state: RequestState, timestamp: float) -> None:
        self._remove_live_state(state)
        state.status = RequestStatus.FINISHED
        state.finish_time = timestamp
        self._emit(
            event_type=EngineEventType.FINISHED,
            state=state,
            timestamp=timestamp,
        )

    def _fail_state(self, state: RequestState, timestamp: float, reason: str) -> None:
        self._remove_live_state(state)
        state.status = RequestStatus.FAILED
        state.failure_reason = reason
        state.finish_time = timestamp
        self._emit(
            event_type=EngineEventType.FAILED,
            state=state,
            timestamp=timestamp,
            detail=reason,
        )

    def _remove_live_state(self, state: RequestState) -> None:
        request_id = state.request.request_id
        if request_id in self._waiting:
            self._waiting.remove(request_id)
        if request_id in self._active:
            self._active.remove(request_id)
        state.kv_cache = None
        state.cached_tokens = 0
        state.reserved_cache_tokens = 0

    def _reservation_for(self, request: GenerationRequest) -> int:
        if request.max_new_tokens == 0:
            return 0
        prompt_length = min(len(request.prompt_tokens), self.config.block_size)
        return min(
            self.config.block_size,
            prompt_length + max(request.max_new_tokens - 1, 0),
        )

    def _emit(  # noqa: PLR0913
        self,
        *,
        event_type: EngineEventType,
        state: RequestState,
        timestamp: float,
        token_id: int | None = None,
        detail: str | None = None,
        used_fallback: bool = False,
    ) -> None:
        self._update_peaks()
        self._events.append(
            EngineEvent(
                sequence=len(self._events),
                timestamp=timestamp,
                event_type=event_type,
                request_id=state.request.request_id,
                status=state.status,
                token_id=token_id,
                detail=detail,
                used_fallback=used_fallback,
                active_requests=len(self._active),
                waiting_requests=len(self._waiting),
                cached_tokens=self._cached_tokens(),
                reserved_cache_tokens=self._reserved_cache_tokens(),
            )
        )

    def _update_peaks(self) -> None:
        self._peak_active = max(self._peak_active, len(self._active))
        self._peak_waiting = max(self._peak_waiting, len(self._waiting))
        self._peak_cached = max(self._peak_cached, self._cached_tokens())
        self._peak_reserved = max(self._peak_reserved, self._reserved_cache_tokens())

    def _cached_tokens(self) -> int:
        return sum(self._states[request_id].cached_tokens for request_id in self._active)

    def _reserved_cache_tokens(self) -> int:
        return sum(self._states[request_id].reserved_cache_tokens for request_id in self._active)

    @staticmethod
    def _terminal_statuses() -> frozenset[RequestStatus]:
        return frozenset({RequestStatus.FINISHED, RequestStatus.CANCELLED, RequestStatus.FAILED})
