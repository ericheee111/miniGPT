"""Resolve the real HTTP serving policy and its portable runtime manifest."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Never, cast

from typing_extensions import override

from minigpt.engine_runner import EngineRunner, RunnerConfig
from minigpt.paged_kv_cache import (
    KVCacheBackend,
    PagedKVCacheConfig,
    PagedKVCachePool,
    PrefixCacheNamespace,
)
from minigpt.serving import (
    APCPrefillStrategy,
    ContinuousDecodeExecutor,
    ContinuousExecutor,
    EngineConfig,
    PagedAttentionExecutor,
    ReferenceExecutor,
    SchedulerConfig,
    ServingEngine,
    ServingExecutor,
)

if TYPE_CHECKING:
    from minigpt.data import JsonValue
    from minigpt.model import GPT

_RUNTIME_MANIFEST_SCHEMA = 1
_RUNTIME_FEATURE_STAGE = "19"
_SHA256_HEX_LENGTH = 64
_DISTRIBUTION_NAME = "minitrain-gpt"


@dataclass(frozen=True, slots=True)
class InvalidServingRuntimeError(ValueError):
    """Report an invalid real HTTP serving policy or manifest operation."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed runtime constraint."""
        return f"invalid serving runtime: {self.reason}"


def _invalid(reason: str) -> Never:
    raise InvalidServingRuntimeError(reason)


class ServingExecutorName(StrEnum):
    """Select the model-execution strategy used by the HTTP process."""

    REFERENCE = "reference"
    CONTINUOUS_DECODE = "continuous_decode"
    CONTINUOUS = "continuous"
    PAGED_ATTENTION = "paged_attention"


@dataclass(frozen=True, slots=True)
class ServingRuntimeConfig:
    """Hold one fully resolved, portable HTTP serving policy."""

    executor: ServingExecutorName
    max_active_requests: int
    max_cached_tokens: int
    kv_cache_backend: KVCacheBackend = KVCacheBackend.DENSE
    kv_block_tokens: int = 16
    kv_num_blocks: int | None = None
    prefix_cache: bool = False
    apc_prefill_strategy: APCPrefillStrategy = APCPrefillStrategy.SEQUENTIAL
    max_scheduled_tokens: int | None = None
    prefill_chunk_tokens: int | None = None
    kv_preemption: bool = False
    lazy_kv_reservation: bool = False
    kv_overcommit_ratio: float = 1.0
    command_queue_size: int = 256
    stream_buffer_size: int = 64

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Reject ambiguous or unreachable real-runtime combinations."""
        raw_executor = cast("object", self.executor)
        if not isinstance(raw_executor, ServingExecutorName):
            _invalid("executor must be a ServingExecutorName")
        raw_backend = cast("object", self.kv_cache_backend)
        if not isinstance(raw_backend, KVCacheBackend):
            _invalid("kv_cache_backend must be a KVCacheBackend")
        raw_strategy = cast("object", self.apc_prefill_strategy)
        if not isinstance(raw_strategy, APCPrefillStrategy):
            _invalid("apc_prefill_strategy must be an APCPrefillStrategy")
        for name, value in (
            ("max_active_requests", cast("object", self.max_active_requests)),
            ("max_cached_tokens", cast("object", self.max_cached_tokens)),
            ("kv_block_tokens", cast("object", self.kv_block_tokens)),
            ("command_queue_size", cast("object", self.command_queue_size)),
            ("stream_buffer_size", cast("object", self.stream_buffer_size)),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                _invalid(f"{name} must be a positive integer")
        raw_num_blocks = cast("object", self.kv_num_blocks)
        if raw_num_blocks is not None and (
            isinstance(raw_num_blocks, bool)
            or not isinstance(raw_num_blocks, int)
            or raw_num_blocks <= 0
        ):
            _invalid("kv_num_blocks must be null or a positive integer")
        for name, value in (
            ("prefix_cache", cast("object", self.prefix_cache)),
            ("kv_preemption", cast("object", self.kv_preemption)),
            ("lazy_kv_reservation", cast("object", self.lazy_kv_reservation)),
        ):
            if not isinstance(value, bool):
                _invalid(f"{name} must be a boolean")
        raw_ratio = cast("object", self.kv_overcommit_ratio)
        if (
            isinstance(raw_ratio, bool)
            or not isinstance(raw_ratio, (int, float))
            or not math.isfinite(raw_ratio)
            or raw_ratio < 1.0
        ):
            _invalid("kv_overcommit_ratio must be finite and at least 1.0")
        if not self.lazy_kv_reservation and raw_ratio != 1.0:
            _invalid("kv_overcommit_ratio must be 1.0 when lazy reservation is disabled")

        direct_paged = (
            self.executor is ServingExecutorName.PAGED_ATTENTION
            and self.kv_cache_backend is KVCacheBackend.PAGED
        )
        if self.executor is ServingExecutorName.PAGED_ATTENTION and not direct_paged:
            _invalid("paged_attention executor requires the paged KV-cache backend")
        if self.prefix_cache and not direct_paged:
            _invalid("prefix_cache requires direct paged_attention serving")
        if self.apc_prefill_strategy is APCPrefillStrategy.BATCHED and not self.prefix_cache:
            _invalid("batched APC prefill requires prefix_cache")

        stage16 = (self.max_scheduled_tokens, self.prefill_chunk_tokens)
        if any(item is None for item in stage16) and any(item is not None for item in stage16):
            _invalid("max_scheduled_tokens and prefill_chunk_tokens must be configured together")
        if self.max_scheduled_tokens is not None and not direct_paged:
            _invalid("token-budget scheduling requires direct paged_attention serving")
        if self.kv_preemption and self.max_scheduled_tokens is None:
            _invalid("kv_preemption requires token-budget scheduling")
        if self.kv_preemption and not direct_paged:
            _invalid("kv_preemption requires direct paged_attention serving")
        if self.lazy_kv_reservation and not self.kv_preemption:
            _invalid("lazy_kv_reservation requires kv_preemption")
        if self.lazy_kv_reservation and not direct_paged:
            _invalid("lazy_kv_reservation requires direct paged_attention serving")
        _ = self.scheduler()

    def scheduler(self) -> SchedulerConfig:
        """Build the scheduler contract shared with simulator and engine validation."""
        return SchedulerConfig(
            max_active_requests=self.max_active_requests,
            max_cached_tokens=self.max_cached_tokens,
            max_scheduled_tokens=self.max_scheduled_tokens,
            prefill_chunk_tokens=self.prefill_chunk_tokens,
            kv_preemption=self.kv_preemption,
            lazy_kv_reservation=self.lazy_kv_reservation,
            kv_overcommit_ratio=float(self.kv_overcommit_ratio),
        )

    def paged_cache(self) -> PagedKVCacheConfig | None:
        """Resolve physical pool dimensions without changing dense defaults."""
        if self.kv_cache_backend is KVCacheBackend.DENSE:
            return None
        num_blocks = self.kv_num_blocks
        if num_blocks is None:
            num_blocks = math.ceil(self.max_cached_tokens / self.kv_block_tokens)
        return PagedKVCacheConfig(block_tokens=self.kv_block_tokens, num_blocks=num_blocks)

    def runner(self) -> RunnerConfig:
        """Build bounded cross-thread queue settings."""
        return RunnerConfig(
            command_queue_size=self.command_queue_size,
            stream_buffer_size=self.stream_buffer_size,
        )


@dataclass(frozen=True, slots=True)
class ServingRuntime:
    """Return the owner-thread runtime and its deterministic manifest."""

    config: ServingRuntimeConfig
    executor: ServingExecutor
    engine: ServingEngine
    runner: EngineRunner
    manifest: dict[str, JsonValue]


def build_serving_runtime(  # noqa: PLR0913
    *,
    model: GPT,
    block_size: int,
    num_threads: int,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
    config: ServingRuntimeConfig,
    tokenizer_type: str | None = None,
    model_family: str | None = None,
) -> ServingRuntime:
    """Construct the real engine/runner path used by the HTTP process."""
    raw_block_size = cast("object", block_size)
    if (
        isinstance(raw_block_size, bool)
        or not isinstance(raw_block_size, int)
        or raw_block_size <= 0
    ):
        _invalid("block_size must be a positive integer")
    raw_num_threads = cast("object", num_threads)
    if (
        isinstance(raw_num_threads, bool)
        or not isinstance(raw_num_threads, int)
        or raw_num_threads <= 0
    ):
        _invalid("num_threads must be a positive integer")
    _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
    _validate_sha256(tokenizer_sha256, "tokenizer_sha256")

    paged_config = config.paged_cache()
    paged_pool: PagedKVCachePool | None = None
    if paged_config is not None:
        namespace = (
            PrefixCacheNamespace(
                model_checkpoint_identity=checkpoint_sha256,
                model_config_identity=_document_sha256(cast("JsonValue", asdict(model.config))),
                dtype=str(model.token_embedding.weight.dtype),
                device=str(model.token_embedding.weight.device),
                block_tokens=paged_config.block_tokens,
                cache_schema_version=1,
                position_embedding_semantics="learned_absolute_v1",
            )
            if config.prefix_cache
            else None
        )
        paged_pool = PagedKVCachePool.from_model(
            paged_config,
            model,
            prefix_cache_namespace=namespace,
        )

    executor = _build_executor(config, model=model, paged_pool=paged_pool)
    scheduler = config.scheduler()
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=scheduler,
            block_size=block_size,
            kv_cache_backend=config.kv_cache_backend,
            paged_kv_cache=paged_config,
        ),
        executor=executor,
        paged_cache_pool=paged_pool,
    )
    runner_config = config.runner()
    runner = EngineRunner(engine=engine, config=runner_config)
    manifest = runtime_manifest_document(
        model=model,
        block_size=block_size,
        num_threads=num_threads,
        checkpoint_sha256=checkpoint_sha256,
        tokenizer_sha256=tokenizer_sha256,
        config=config,
        scheduler=scheduler,
        paged_config=paged_config,
        runner_config=runner_config,
        tokenizer_type=tokenizer_type,
        model_family=model_family,
    )
    return ServingRuntime(
        config=config,
        executor=executor,
        engine=engine,
        runner=runner,
        manifest=manifest,
    )


def runtime_manifest_document(  # noqa: PLR0913
    *,
    model: GPT,
    block_size: int,
    num_threads: int,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
    config: ServingRuntimeConfig,
    scheduler: SchedulerConfig,
    paged_config: PagedKVCacheConfig | None,
    runner_config: RunnerConfig,
    tokenizer_type: str | None = None,
    model_family: str | None = None,
) -> dict[str, JsonValue]:
    """Build the portable, deterministic Stage 19 runtime document."""
    tokenizer: dict[str, JsonValue] = {
        "sha256": tokenizer_sha256,
    }
    if tokenizer_type is not None:
        tokenizer["type"] = tokenizer_type
    if model_family is not None:
        tokenizer["model_family"] = model_family
    return {
        "schema_version": _RUNTIME_MANIFEST_SCHEMA,
        "stage": _RUNTIME_FEATURE_STAGE,
        "project_version": installed_project_version(),
        "checkpoint_sha256": checkpoint_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer": tokenizer,
        "model": cast("JsonValue", asdict(model.config)),
        "block_size": block_size,
        "cpu_threads": num_threads,
        "executor": config.executor.value,
        "kv_cache_backend": config.kv_cache_backend.value,
        "paged_kv_cache": (
            cast("JsonValue", asdict(paged_config)) if paged_config is not None else None
        ),
        "prefix_cache": {
            "enabled": config.prefix_cache,
            "apc_prefill_strategy": config.apc_prefill_strategy.value,
        },
        "scheduler": cast("JsonValue", asdict(scheduler)),
        "runner": cast("JsonValue", asdict(runner_config)),
        "claim_policy": {
            "benchmark_strict_verdict": "descriptive_only",
            "wall_clock_performance_improvement": False,
            "public_production_security_readiness": False,
        },
    }


def render_runtime_manifest(document: dict[str, JsonValue]) -> str:
    """Render stable JSON bytes independent of platform newline defaults."""
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_runtime_manifest(path: Path, document: dict[str, JsonValue]) -> Path:
    """Atomically write a stable UTF-8/LF runtime manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render_runtime_manifest(document)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            _ = stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def file_sha256(path: Path) -> str:
    """Return the SHA-256 identity of one immutable runtime input."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def installed_project_version() -> str:
    """Return the installed distribution version used by this runtime."""
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        _invalid(f"distribution {_DISTRIBUTION_NAME!r} is not installed")


def _atomic_replace(source: Path, target: Path) -> None:
    _ = source.replace(target)


def _build_executor(
    config: ServingRuntimeConfig,
    *,
    model: GPT,
    paged_pool: PagedKVCachePool | None,
) -> ServingExecutor:
    if config.executor is ServingExecutorName.REFERENCE:
        return ReferenceExecutor(model)
    if config.executor is ServingExecutorName.CONTINUOUS_DECODE:
        return ContinuousDecodeExecutor(model)
    if config.executor is ServingExecutorName.CONTINUOUS:
        return ContinuousExecutor(model)
    if paged_pool is None:
        _invalid("paged_attention executor requires a paged cache pool")
    return PagedAttentionExecutor(
        model,
        paged_pool,
        prefix_prefill_strategy=config.apc_prefill_strategy,
    )


def _document_sha256(document: JsonValue) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    reason = f"{name} must be a lowercase SHA-256 hex digest"
    if len(value) != _SHA256_HEX_LENGTH or value != value.lower():
        _invalid(reason)
    try:
        decoded = bytes.fromhex(value)
    except ValueError:
        _invalid(reason)
    if len(decoded) * 2 != _SHA256_HEX_LENGTH:
        _invalid(reason)
