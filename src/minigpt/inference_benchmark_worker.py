"""Execute one isolated deterministic Stage 9 inference benchmark replicate."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Literal, TypeAlias, cast

import torch
from torch import Tensor

from minigpt.benchmark_v2_environment import (
    WorkerEnvironment,
    apply_cpu_affinity,
    capture_worker_environment,
    read_process_memory,
)
from minigpt.inference_benchmark_config import (
    InferenceBenchmarkConfig,
    InferenceCase,
    JsonValue,
)
from minigpt.model import GPT, kv_cache_nbytes
from minigpt.settings import GPTConfig

InferenceMode: TypeAlias = Literal["cached", "uncached"]
WORKER_PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class InferenceWorkerRequest:
    """Bind one mode/case replicate to all deterministic worker controls."""

    protocol_version: int
    case: InferenceCase
    mode: InferenceMode
    replicate_index: int
    benchmark_seed: int
    vocab_size: int
    model: GPTConfig
    warmup_iterations: int
    measurement_iterations: int
    torch_num_threads: int
    torch_num_interop_threads: int
    cpu_affinity: tuple[int, ...] | None
    relevant_environment_variables: tuple[str, ...]

    @classmethod
    def from_config(
        cls,
        config: InferenceBenchmarkConfig,
        case: InferenceCase,
        *,
        mode: str,
        replicate_index: int,
    ) -> InferenceWorkerRequest:
        """Create a validated request from one resolved configuration task."""
        if mode not in {"cached", "uncached"}:
            msg = f"unsupported inference mode {mode!r}"
            raise ValueError(msg)
        if replicate_index < 0:
            msg = "replicate_index must be non-negative"
            raise ValueError(msg)
        return cls(
            protocol_version=WORKER_PROTOCOL_VERSION,
            case=case,
            mode=cast("InferenceMode", mode),
            replicate_index=replicate_index,
            benchmark_seed=config.benchmark_seed,
            vocab_size=config.vocab_size,
            model=config.model,
            warmup_iterations=config.warmup_iterations,
            measurement_iterations=config.measurement_iterations,
            torch_num_threads=config.torch_num_threads,
            torch_num_interop_threads=config.torch_num_interop_threads,
            cpu_affinity=config.cpu_affinity,
            relevant_environment_variables=config.relevant_environment_variables,
        )


@dataclass(frozen=True, slots=True)
class InferenceWorkerResult:
    """Report one aggregate of untimed warmup and repeated unprofiled measurements."""

    protocol_version: int
    status: Literal["ok"]
    worker_pid: int
    started_at_utc: str
    ended_at_utc: str
    case_name: str
    mode: InferenceMode
    replicate_index: int
    warmup_iterations: int
    measurement_iterations: int
    prefill_time_ms: float
    time_to_first_token_ms: float
    median_decode_time_ms: float
    generated_tokens_per_second: float
    end_to_end_time_ms: float
    final_rss_mib: float
    peak_rss_mib: float
    peak_rss_method: str
    peak_rss_scope: Literal["worker_lifetime"]
    kv_cache_bytes: int
    environment_signature: str
    environment: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _IterationMetrics:
    """Keep all canonical metrics from one generation iteration."""

    prefill_time_ms: float
    median_decode_time_ms: float
    end_to_end_time_ms: float
    generated_tokens_per_second: float
    kv_cache_bytes: int


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _environment_document(environment: WorkerEnvironment) -> dict[str, JsonValue]:
    return {
        "platform": environment.platform,
        "python_version": environment.python_version,
        "torch_version": environment.torch_version,
        "torch_num_threads": environment.torch_num_threads,
        "torch_num_interop_threads": environment.torch_num_interop_threads,
        "logical_cpu_count": environment.logical_cpu_count,
        "requested_cpu_affinity": (
            list(environment.requested_cpu_affinity)
            if environment.requested_cpu_affinity is not None
            else None
        ),
        "effective_cpu_affinity": (
            list(environment.effective_cpu_affinity)
            if environment.effective_cpu_affinity is not None
            else None
        ),
        "relevant_environment_variables": dict(environment.relevant_environment_variables),
    }


def _environment_signature(environment: dict[str, JsonValue]) -> str:
    encoded = json.dumps(
        environment,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def _run_iteration(
    model: GPT,
    prompt: Tensor,
    forced_tokens: Tensor,
    *,
    mode: InferenceMode,
) -> _IterationMetrics:
    generated = prompt
    decode_times_ms: list[float] = []
    cache_bytes = 0
    end_to_end_start = perf_counter()
    prefill_start = perf_counter()
    if mode == "cached":
        _, cache = model.prefill(prompt)
    else:
        _ = cast("tuple[Tensor, Tensor | None]", model(prompt))
        cache = ()
    prefill_time_ms = (perf_counter() - prefill_start) * 1_000

    generated_length = forced_tokens.shape[1]
    for token_index in range(generated_length):
        next_token = forced_tokens[:, token_index : token_index + 1]
        generated = torch.cat((generated, next_token), dim=1)
        if token_index + 1 == generated_length:
            break
        decode_start = perf_counter()
        if mode == "cached":
            _, cache = model.decode(next_token, cache)
        else:
            _ = cast("tuple[Tensor, Tensor | None]", model(generated))
        decode_times_ms.append((perf_counter() - decode_start) * 1_000)

    end_to_end_time_ms = (perf_counter() - end_to_end_start) * 1_000
    if mode == "cached":
        cache_bytes = kv_cache_nbytes(cache)
    return _IterationMetrics(
        prefill_time_ms=prefill_time_ms,
        median_decode_time_ms=statistics.median(decode_times_ms),
        end_to_end_time_ms=end_to_end_time_ms,
        generated_tokens_per_second=generated_length / (end_to_end_time_ms / 1_000),
        kv_cache_bytes=cache_bytes,
    )


def run_worker_request(request: InferenceWorkerRequest) -> InferenceWorkerResult:
    """Execute deterministic forced-token generation after applying CPU controls."""
    if request.protocol_version != WORKER_PROTOCOL_VERSION:
        msg = f"protocol_version must be {WORKER_PROTOCOL_VERSION}"
        raise ValueError(msg)
    started_at_utc = _utc_now()
    effective_affinity = apply_cpu_affinity(request.cpu_affinity)
    torch.set_num_threads(request.torch_num_threads)
    if torch.get_num_interop_threads() != request.torch_num_interop_threads:
        torch.set_num_interop_threads(request.torch_num_interop_threads)
    _ = torch.default_generator.manual_seed(request.benchmark_seed)
    model = GPT(request.model)
    _ = model.eval()
    input_generator = torch.Generator(device="cpu")
    _ = input_generator.manual_seed(request.benchmark_seed + 1)
    prompt = torch.randint(
        request.vocab_size,
        (request.case.batch_size, request.case.prompt_length),
        generator=input_generator,
        dtype=torch.long,
    )
    forced_tokens = torch.randint(
        request.vocab_size,
        (request.case.batch_size, request.case.generated_length),
        generator=input_generator,
        dtype=torch.long,
    )
    for _ in range(request.warmup_iterations):
        _ = _run_iteration(model, prompt, forced_tokens, mode=request.mode)
    _ = gc.collect()
    measurements = tuple(
        _run_iteration(model, prompt, forced_tokens, mode=request.mode)
        for _ in range(request.measurement_iterations)
    )
    memory = read_process_memory()
    environment = _environment_document(
        capture_worker_environment(
            requested_cpu_affinity=request.cpu_affinity,
            effective_cpu_affinity=effective_affinity,
            relevant_environment_variables=request.relevant_environment_variables,
        )
    )
    prefill_time_ms = statistics.median(item.prefill_time_ms for item in measurements)
    return InferenceWorkerResult(
        protocol_version=WORKER_PROTOCOL_VERSION,
        status="ok",
        worker_pid=os.getpid(),
        started_at_utc=started_at_utc,
        ended_at_utc=_utc_now(),
        case_name=request.case.name,
        mode=request.mode,
        replicate_index=request.replicate_index,
        warmup_iterations=request.warmup_iterations,
        measurement_iterations=request.measurement_iterations,
        prefill_time_ms=prefill_time_ms,
        time_to_first_token_ms=prefill_time_ms,
        median_decode_time_ms=statistics.median(
            item.median_decode_time_ms for item in measurements
        ),
        generated_tokens_per_second=statistics.median(
            item.generated_tokens_per_second for item in measurements
        ),
        end_to_end_time_ms=statistics.median(item.end_to_end_time_ms for item in measurements),
        final_rss_mib=memory.final_rss_mib,
        peak_rss_mib=memory.peak_rss_mib,
        peak_rss_method=memory.peak_rss_method,
        peak_rss_scope="worker_lifetime",
        kv_cache_bytes=max(item.kv_cache_bytes for item in measurements),
        environment_signature=_environment_signature(environment),
        environment=environment,
    )


def worker_request_document(request: InferenceWorkerRequest) -> dict[str, JsonValue]:
    """Serialize an exact request for stdin transport to a fresh process."""
    return {
        "protocol_version": request.protocol_version,
        "case": cast("dict[str, JsonValue]", asdict(request.case)),
        "mode": request.mode,
        "replicate_index": request.replicate_index,
        "benchmark_seed": request.benchmark_seed,
        "vocab_size": request.vocab_size,
        "model": cast("dict[str, JsonValue]", asdict(request.model)),
        "warmup_iterations": request.warmup_iterations,
        "measurement_iterations": request.measurement_iterations,
        "torch_num_threads": request.torch_num_threads,
        "torch_num_interop_threads": request.torch_num_interop_threads,
        "cpu_affinity": list(request.cpu_affinity) if request.cpu_affinity is not None else None,
        "relevant_environment_variables": list(request.relevant_environment_variables),
    }


def _request_from_document(value: object) -> InferenceWorkerRequest:
    if not isinstance(value, dict):
        msg = "worker request must be an object"
        raise TypeError(msg)
    document = cast("dict[str, object]", value)
    case = cast("dict[str, object]", document["case"])
    model = cast("dict[str, object]", document["model"])
    mode = document["mode"]
    if mode not in {"cached", "uncached"}:
        msg = "worker mode must be cached or uncached"
        raise ValueError(msg)
    raw_affinity = document["cpu_affinity"]
    raw_variables = cast("list[object]", document["relevant_environment_variables"])
    return InferenceWorkerRequest(
        protocol_version=cast("int", document["protocol_version"]),
        case=InferenceCase(
            name=cast("str", case["name"]),
            batch_size=cast("int", case["batch_size"]),
            prompt_length=cast("int", case["prompt_length"]),
            generated_length=cast("int", case["generated_length"]),
        ),
        mode=cast("InferenceMode", mode),
        replicate_index=cast("int", document["replicate_index"]),
        benchmark_seed=cast("int", document["benchmark_seed"]),
        vocab_size=cast("int", document["vocab_size"]),
        model=GPTConfig(
            vocab_size=cast("int", model["vocab_size"]),
            block_size=cast("int", model["block_size"]),
            n_layer=cast("int", model["n_layer"]),
            n_head=cast("int", model["n_head"]),
            n_embd=cast("int", model["n_embd"]),
            dropout=cast("float", model["dropout"]),
            bias=cast("bool", model["bias"]),
        ),
        warmup_iterations=cast("int", document["warmup_iterations"]),
        measurement_iterations=cast("int", document["measurement_iterations"]),
        torch_num_threads=cast("int", document["torch_num_threads"]),
        torch_num_interop_threads=cast("int", document["torch_num_interop_threads"]),
        cpu_affinity=(
            None
            if raw_affinity is None
            else tuple(cast("int", item) for item in cast("list[object]", raw_affinity))
        ),
        relevant_environment_variables=tuple(cast("str", item) for item in raw_variables),
    )


def worker_response_document(result: InferenceWorkerResult) -> dict[str, JsonValue]:
    """Serialize one successful worker result without dropping metrics."""
    return cast("dict[str, JsonValue]", asdict(result))


def worker_main() -> int:
    """Read one JSON request from stdin and emit one JSON response."""
    try:
        request = _request_from_document(cast("object", json.loads(sys.stdin.read())))
        result = run_worker_request(request)
    except Exception as error:  # noqa: BLE001 - subprocess protocol retains ordinary failure.
        document: dict[str, JsonValue] = {
            "protocol_version": WORKER_PROTOCOL_VERSION,
            "status": "error",
            "worker_pid": os.getpid(),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        status = 1
    else:
        document = worker_response_document(result)
        status = 0
    _ = sys.stdout.write(
        json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    )
    return status


if __name__ == "__main__":
    raise SystemExit(worker_main())
