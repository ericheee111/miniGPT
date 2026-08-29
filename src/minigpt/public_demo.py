"""Expose the existing miniGPT runtime through a restricted public-demo boundary."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import logging
import math
import os
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Never, TypeAlias, cast, final
from urllib.parse import urlsplit
from uuid import uuid4

import torch
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect
from typing_extensions import override

from minigpt import __version__
from minigpt.checkpoint import load_checkpoint_config, load_model_state
from minigpt.data import CharTokenizer, JsonValue, UnknownCharacterError
from minigpt.engine_runner import (
    EngineRunner,
    RequestHandle,
    RunnerQueueFullError,
    RunnerResult,
    RunnerUnavailableError,
    StreamEventType,
)
from minigpt.http_server import MODEL_ID
from minigpt.model import GPT
from minigpt.paged_kv_cache import KVCacheBackend
from minigpt.serving import APCPrefillStrategy, GenerationRequest, RequestStatus
from minigpt.serving_runtime import (
    ServingExecutorName,
    ServingRuntimeConfig,
    build_serving_runtime,
    file_sha256,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence

    from starlette.types import ASGIApp, Message, Receive, Scope, Send

ConfigValue: TypeAlias = (
    str | int | float | bool | list["ConfigValue"] | dict[str, "ConfigValue"] | None
)
ConfigMapping: TypeAlias = dict[str, ConfigValue]

_LOGGER = logging.getLogger("minigpt.public_demo")
_MAX_SEED = 2**63
_MAX_PORT = 65535
_SERVER_ERROR_THRESHOLD = 500
_COMPLETION_FIELDS = frozenset({"model", "prompt", "max_tokens", "temperature", "stream", "seed"})
_SERVER_KEYS = frozenset({"host", "port", "log_level"})
_RUNTIME_KEYS = frozenset(
    {
        "executor",
        "max_active_requests",
        "max_cached_tokens",
        "kv_cache_backend",
        "kv_block_tokens",
        "kv_num_blocks",
        "prefix_cache",
        "apc_prefill_strategy",
        "max_scheduled_tokens",
        "prefill_chunk_tokens",
        "kv_preemption",
        "lazy_kv_reservation",
        "kv_overcommit_ratio",
        "command_queue_size",
        "stream_buffer_size",
    }
)
_POLICY_KEYS = frozenset(
    {
        "max_request_body_bytes",
        "max_prompt_characters",
        "max_prompt_tokens",
        "max_new_tokens",
        "min_temperature",
        "max_temperature",
        "request_timeout_seconds",
        "max_concurrent_requests",
        "max_queue_size",
        "per_client_requests",
        "per_client_window_seconds",
        "global_requests",
        "global_window_seconds",
        "enabled",
        "trust_proxy",
    }
)
_TOP_LEVEL_KEYS = frozenset({"schema_version", "server", "runtime", "policy", "allowed_origins"})
_LOG_LEVELS = frozenset({"critical", "error", "warning", "info"})
_CORS_METHODS = "GET, POST, OPTIONS"
_CORS_HEADERS = frozenset({"content-type", "ngrok-skip-browser-warning"})


@dataclass(frozen=True, slots=True)
class InvalidPublicDemoConfigError(ValueError):
    """Report a public-demo configuration that must fail before startup."""

    reason: str
    source: Path | None = None

    @override
    def __str__(self) -> str:
        """Render a stable local configuration error."""
        prefix = "invalid public demo configuration"
        return (
            f"{prefix} {self.source}: {self.reason}"
            if self.source is not None
            else f"{prefix}: {self.reason}"
        )


def _invalid(reason: str, source: Path | None = None) -> Never:
    raise InvalidPublicDemoConfigError(reason=reason, source=source)


@dataclass(frozen=True, slots=True)
class PublicDemoPolicy:
    """Hold fail-closed limits for one personal public-demo process."""

    max_request_body_bytes: int = 8192
    max_prompt_characters: int = 256
    max_prompt_tokens: int = 256
    max_new_tokens: int = 96
    min_temperature: float = 0.1
    max_temperature: float = 1.2
    request_timeout_seconds: float = 45.0
    max_concurrent_requests: int = 2
    max_queue_size: int = 8
    per_client_requests: int = 5
    per_client_window_seconds: float = 600.0
    global_requests: int = 60
    global_window_seconds: float = 3600.0
    enabled: bool = False
    trust_proxy: bool = False
    allowed_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject ambiguous limits and normalize exact CORS origins."""
        for name, value in (
            ("max_request_body_bytes", self.max_request_body_bytes),
            ("max_prompt_characters", self.max_prompt_characters),
            ("max_prompt_tokens", self.max_prompt_tokens),
            ("max_new_tokens", self.max_new_tokens),
            ("max_concurrent_requests", self.max_concurrent_requests),
            ("per_client_requests", self.per_client_requests),
            ("global_requests", self.global_requests),
        ):
            _require_positive_integer(name, value)
        _require_non_negative_integer("max_queue_size", self.max_queue_size)
        for name, value in (
            ("min_temperature", self.min_temperature),
            ("max_temperature", self.max_temperature),
            ("request_timeout_seconds", self.request_timeout_seconds),
            ("per_client_window_seconds", self.per_client_window_seconds),
            ("global_window_seconds", self.global_window_seconds),
        ):
            _require_positive_number(name, value)
        if self.min_temperature > self.max_temperature:
            _invalid("min_temperature must not exceed max_temperature")
        if type(self.enabled) is not bool:
            _invalid("enabled must be a boolean")
        if type(self.trust_proxy) is not bool:
            _invalid("trust_proxy must be a boolean")
        normalized = tuple(_normalize_origin(origin) for origin in self.allowed_origins)
        if len(normalized) != len(set(normalized)):
            _invalid("allowed_origins must not contain duplicates")
        object.__setattr__(self, "allowed_origins", normalized)

    def public_limits(self) -> dict[str, JsonValue]:
        """Return only the numeric limits safe for GET /demo/info."""
        return {
            "max_request_body_bytes": self.max_request_body_bytes,
            "max_prompt_characters": self.max_prompt_characters,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_new_tokens": self.max_new_tokens,
            "min_temperature": self.min_temperature,
            "max_temperature": self.max_temperature,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_queue_size": self.max_queue_size,
            "per_client_requests": self.per_client_requests,
            "per_client_window_seconds": self.per_client_window_seconds,
            "global_requests": self.global_requests,
            "global_window_seconds": self.global_window_seconds,
        }


@dataclass(frozen=True, slots=True)
class PublicDemoServerConfig:
    """Describe the local Uvicorn boundary without enabling public interfaces."""

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "warning"

    def __post_init__(self) -> None:
        """Reject malformed local server values."""
        if not self.host.strip():
            _invalid("server.host must be a non-empty string")
        if isinstance(self.port, bool) or not 1 <= self.port <= _MAX_PORT:
            _invalid("server.port must be an integer in [1, 65535]")
        if self.log_level not in _LOG_LEVELS:
            _invalid(f"server.log_level must be one of {sorted(_LOG_LEVELS)}")


@dataclass(frozen=True, slots=True)
class PublicDemoRuntimeOptions:
    """Resolve process-level ServingRuntimeConfig after model block size is known."""

    executor: ServingExecutorName = ServingExecutorName.CONTINUOUS
    max_active_requests: int = 2
    max_cached_tokens: int | None = None
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
    command_queue_size: int = 16
    stream_buffer_size: int = 16

    def __post_init__(self) -> None:  # noqa: C901
        """Reject invalid runtime combinations before local model loading."""
        for name, value in (
            ("max_active_requests", self.max_active_requests),
            ("kv_block_tokens", self.kv_block_tokens),
            ("command_queue_size", self.command_queue_size),
            ("stream_buffer_size", self.stream_buffer_size),
        ):
            _require_positive_integer(name, value)
        for name, value in (
            ("max_cached_tokens", self.max_cached_tokens),
            ("kv_num_blocks", self.kv_num_blocks),
            ("max_scheduled_tokens", self.max_scheduled_tokens),
            ("prefill_chunk_tokens", self.prefill_chunk_tokens),
        ):
            if value is not None:
                _require_positive_integer(name, value)
        _require_positive_number("kv_overcommit_ratio", self.kv_overcommit_ratio)
        for name, value in (
            ("prefix_cache", self.prefix_cache),
            ("kv_preemption", self.kv_preemption),
            ("lazy_kv_reservation", self.lazy_kv_reservation),
        ):
            if type(value) is not bool:
                _invalid(f"{name} must be a boolean")
        chunk_values = (self.max_scheduled_tokens, self.prefill_chunk_tokens)
        if any(value is None for value in chunk_values) and any(
            value is not None for value in chunk_values
        ):
            fields = "runtime.max_scheduled_tokens and runtime.prefill_chunk_tokens"
            reason = f"{fields} must be configured together"
            _invalid(reason)
        if self.kv_cache_backend is KVCacheBackend.DENSE and (
            self.kv_num_blocks is not None or self.prefix_cache
        ):
            _invalid("dense runtime must not configure paged-only cache options")
        if self.kv_cache_backend is KVCacheBackend.PAGED and self.kv_num_blocks is None:
            _invalid("paged runtime requires kv_num_blocks")
        if self.prefix_cache and self.kv_cache_backend is not KVCacheBackend.PAGED:
            _invalid("prefix_cache requires the paged KV backend")
        if self.lazy_kv_reservation and not self.kv_preemption:
            _invalid("lazy_kv_reservation requires kv_preemption")
        if not self.lazy_kv_reservation and self.kv_overcommit_ratio != 1.0:
            _invalid("kv_overcommit_ratio must be 1.0 without lazy_kv_reservation")

    def validate_policy(self, policy: PublicDemoPolicy) -> None:
        """Keep HTTP capacity and the existing scheduler capacity aligned."""
        if self.max_active_requests != policy.max_concurrent_requests:
            _invalid("runtime.max_active_requests must equal policy.max_concurrent_requests")
        minimum_commands = policy.max_concurrent_requests + policy.max_queue_size + 2
        if self.command_queue_size < minimum_commands:
            _invalid(
                "runtime.command_queue_size must leave capacity for queued submits and cleanup"
            )

    def to_serving_runtime(
        self,
        *,
        block_size: int,
        policy: PublicDemoPolicy,
    ) -> ServingRuntimeConfig:
        """Build the existing typed runtime while keeping public capacity aligned."""
        self.validate_policy(policy)
        max_cached_tokens = self.max_cached_tokens
        if max_cached_tokens is None:
            prompt_tokens = min(block_size, policy.max_prompt_tokens)
            max_cached_tokens = self.max_active_requests * (prompt_tokens + policy.max_new_tokens)
        return ServingRuntimeConfig(
            executor=self.executor,
            max_active_requests=self.max_active_requests,
            max_cached_tokens=max_cached_tokens,
            kv_cache_backend=self.kv_cache_backend,
            kv_block_tokens=self.kv_block_tokens,
            kv_num_blocks=self.kv_num_blocks,
            prefix_cache=self.prefix_cache,
            apc_prefill_strategy=self.apc_prefill_strategy,
            max_scheduled_tokens=self.max_scheduled_tokens,
            prefill_chunk_tokens=self.prefill_chunk_tokens,
            kv_preemption=self.kv_preemption,
            lazy_kv_reservation=self.lazy_kv_reservation,
            kv_overcommit_ratio=self.kv_overcommit_ratio,
            command_queue_size=self.command_queue_size,
            stream_buffer_size=self.stream_buffer_size,
        )


@dataclass(frozen=True, slots=True)
class PublicDemoSettings:
    """Combine the strict public policy with existing runtime selection."""

    server: PublicDemoServerConfig = field(default_factory=PublicDemoServerConfig)
    runtime: PublicDemoRuntimeOptions = field(default_factory=PublicDemoRuntimeOptions)
    policy: PublicDemoPolicy = field(default_factory=PublicDemoPolicy)


@dataclass(frozen=True, slots=True)
class PublicDemoInfo:
    """Describe only the portable runtime fields exposed by GET /demo/info."""

    project_version: str
    model_id: str
    executor_name: str
    kv_cache_backend: str
    prefix_cache_enabled: bool
    streaming_available: bool = True


def load_public_demo_settings(path: Path | None) -> PublicDemoSettings:
    """Load a strict schema-v1 YAML file or return fail-closed defaults."""
    if path is None:
        return PublicDemoSettings()
    try:
        document = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path, "document")
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        _invalid(str(error), path)
    _exact_keys(document, _TOP_LEVEL_KEYS, path, "document")
    if _integer(document, "schema_version", path, positive=True) != 1:
        _invalid("schema_version must equal 1", path)
    server = _parse_server(_mapping(document["server"], path, "server"), path)
    runtime = _parse_runtime(_mapping(document["runtime"], path, "runtime"), path)
    policy = _parse_policy(
        _mapping(document["policy"], path, "policy"),
        _string_sequence(document, "allowed_origins", path),
        path,
    )
    return PublicDemoSettings(server=server, runtime=runtime, policy=policy)


def resolve_environment(
    settings: PublicDemoSettings,
    environment: Mapping[str, str],
    *,
    trust_proxy: bool,
    origins: Sequence[str] | None,
) -> PublicDemoSettings:
    """Apply the small, explicit deployment environment surface."""
    policy = settings.policy
    enabled_value = environment.get("DEMO_ENABLED")
    if enabled_value is not None:
        if enabled_value not in {"0", "1"}:
            _invalid("DEMO_ENABLED must be exactly 0 or 1")
        policy = replace(policy, enabled=enabled_value == "1")
    configured_origins = origins
    if configured_origins is None and "PUBLIC_ORIGIN" in environment:
        public_origin = environment["PUBLIC_ORIGIN"].strip()
        configured_origins = [] if not public_origin else [public_origin]
    if configured_origins is not None:
        policy = replace(policy, allowed_origins=tuple(configured_origins))
    if trust_proxy:
        policy = replace(policy, trust_proxy=True)
    return replace(settings, policy=policy)


def validate_bind_host(
    host: str,
    *,
    unsafe_allow_non_loopback: bool,
    trust_proxy: bool,
) -> None:
    """Require loopback unless the operator names the unsafe exception."""
    if _is_loopback_host(host):
        return
    if trust_proxy:
        _invalid("trust_proxy requires a loopback server host")
    if not unsafe_allow_non_loopback:
        _invalid("public demo may only bind loopback; use --unsafe-allow-non-loopback to override")


def _parse_server(document: ConfigMapping, source: Path) -> PublicDemoServerConfig:
    _exact_keys(document, _SERVER_KEYS, source, "server")
    return PublicDemoServerConfig(
        host=_string(document, "host", source),
        port=_integer(document, "port", source, positive=True),
        log_level=_string(document, "log_level", source),
    )


def _parse_runtime(document: ConfigMapping, source: Path) -> PublicDemoRuntimeOptions:
    _exact_keys(document, _RUNTIME_KEYS, source, "runtime")
    try:
        executor = ServingExecutorName(_string(document, "executor", source))
        backend = KVCacheBackend(_string(document, "kv_cache_backend", source))
        strategy = APCPrefillStrategy(_string(document, "apc_prefill_strategy", source))
    except ValueError as error:
        _invalid(str(error), source)
    return PublicDemoRuntimeOptions(
        executor=executor,
        max_active_requests=_integer(document, "max_active_requests", source, positive=True),
        max_cached_tokens=_optional_integer(document, "max_cached_tokens", source, positive=True),
        kv_cache_backend=backend,
        kv_block_tokens=_integer(document, "kv_block_tokens", source, positive=True),
        kv_num_blocks=_optional_integer(document, "kv_num_blocks", source, positive=True),
        prefix_cache=_boolean(document, "prefix_cache", source),
        apc_prefill_strategy=strategy,
        max_scheduled_tokens=_optional_integer(
            document, "max_scheduled_tokens", source, positive=True
        ),
        prefill_chunk_tokens=_optional_integer(
            document, "prefill_chunk_tokens", source, positive=True
        ),
        kv_preemption=_boolean(document, "kv_preemption", source),
        lazy_kv_reservation=_boolean(document, "lazy_kv_reservation", source),
        kv_overcommit_ratio=_number(document, "kv_overcommit_ratio", source, positive=True),
        command_queue_size=_integer(document, "command_queue_size", source, positive=True),
        stream_buffer_size=_integer(document, "stream_buffer_size", source, positive=True),
    )


def _parse_policy(
    document: ConfigMapping,
    allowed_origins: tuple[str, ...],
    source: Path,
) -> PublicDemoPolicy:
    _exact_keys(document, _POLICY_KEYS, source, "policy")
    return PublicDemoPolicy(
        max_request_body_bytes=_integer(document, "max_request_body_bytes", source, positive=True),
        max_prompt_characters=_integer(document, "max_prompt_characters", source, positive=True),
        max_prompt_tokens=_integer(document, "max_prompt_tokens", source, positive=True),
        max_new_tokens=_integer(document, "max_new_tokens", source, positive=True),
        min_temperature=_number(document, "min_temperature", source, positive=True),
        max_temperature=_number(document, "max_temperature", source, positive=True),
        request_timeout_seconds=_number(document, "request_timeout_seconds", source, positive=True),
        max_concurrent_requests=_integer(
            document, "max_concurrent_requests", source, positive=True
        ),
        max_queue_size=_integer(document, "max_queue_size", source, non_negative=True),
        per_client_requests=_integer(document, "per_client_requests", source, positive=True),
        per_client_window_seconds=_number(
            document, "per_client_window_seconds", source, positive=True
        ),
        global_requests=_integer(document, "global_requests", source, positive=True),
        global_window_seconds=_number(document, "global_window_seconds", source, positive=True),
        enabled=_boolean(document, "enabled", source),
        trust_proxy=_boolean(document, "trust_proxy", source),
        allowed_origins=allowed_origins,
    )


def _mapping(value: object, source: Path, context: str) -> ConfigMapping:
    if not isinstance(value, dict):
        _invalid(f"{context} must be a mapping", source)
    raw = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in raw):
        _invalid(f"{context} keys must be strings", source)
    return cast("ConfigMapping", raw)


def _exact_keys(
    document: ConfigMapping,
    expected: frozenset[str],
    source: Path,
    context: str,
) -> None:
    missing = expected - set(document)
    unexpected = set(document) - expected
    if missing:
        _invalid(f"{context} missing key {min(missing)!r}", source)
    if unexpected:
        _invalid(f"{context} has unexpected key {min(unexpected)!r}", source)


def _integer(
    document: ConfigMapping,
    key: str,
    source: Path,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> int:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid(f"{key} must be an integer", source)
    if positive and value <= 0:
        _invalid(f"{key} must be positive", source)
    if non_negative and value < 0:
        _invalid(f"{key} must be non-negative", source)
    return value


def _optional_integer(
    document: ConfigMapping,
    key: str,
    source: Path,
    *,
    positive: bool,
) -> int | None:
    if document[key] is None:
        return None
    return _integer(document, key, source, positive=positive)


def _number(
    document: ConfigMapping,
    key: str,
    source: Path,
    *,
    positive: bool,
) -> float:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(f"{key} must be a number", source)
    normalized = float(value)
    if not math.isfinite(normalized) or (positive and normalized <= 0.0):
        _invalid(f"{key} must be finite{' and positive' if positive else ''}", source)
    return normalized


def _boolean(document: ConfigMapping, key: str, source: Path) -> bool:
    value = document[key]
    if type(value) is not bool:
        _invalid(f"{key} must be a boolean", source)
    return value


def _string(document: ConfigMapping, key: str, source: Path) -> str:
    value = document[key]
    if not isinstance(value, str) or not value:
        _invalid(f"{key} must be a non-empty string", source)
    return value


def _string_sequence(
    document: ConfigMapping,
    key: str,
    source: Path,
) -> tuple[str, ...]:
    value = document[key]
    if not isinstance(value, list):
        _invalid(f"{key} must be a list", source)
    items = cast("list[object]", value)
    if any(not isinstance(item, str) for item in items):
        _invalid(f"{key} must contain only strings", source)
    return cast("tuple[str, ...]", tuple(items))


def _require_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _invalid(f"{name} must be a positive integer")


def _require_non_negative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _invalid(f"{name} must be a non-negative integer")


def _require_positive_number(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        _invalid(f"{name} must be a finite positive number")


def _normalize_origin(origin: str) -> str:
    if not origin or "*" in origin:
        _invalid("allowed origins must be explicit non-wildcard URLs")
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError as error:
        _invalid(f"invalid allowed origin: {error}")
    if parsed.username is not None or parsed.password is not None:
        _invalid("allowed origins must not contain credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        _invalid("allowed origins must not contain path, query, or fragment")
    hostname = parsed.hostname
    if hostname is None:
        _invalid("allowed origins must contain a hostname")
    if parsed.scheme == "http" and not _is_loopback_host(hostname):
        _invalid("non-loopback allowed origins must use https")
    if parsed.scheme not in {"http", "https"}:
        _invalid("allowed origins must use http or https")
    host = hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = 80 if parsed.scheme == "http" else 443
    suffix = "" if port in {None, default_port} else f":{port}"
    return f"{parsed.scheme}://{host}{suffix}"


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


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
    status_code: int = 400
    code: str = "invalid_request"


def _request_error(
    message: str,
    param: str | None,
    *,
    status_code: int = 400,
    code: str = "invalid_request",
) -> Never:
    raise _RequestError(message, param, status_code=status_code, code=code)


class _CapacityFullError(RuntimeError):
    """Report a full public HTTP waiter queue."""


@dataclass(slots=True)
class _CapacityLease:
    gate: _CapacityGate
    released: bool = False

    async def release(self) -> None:
        """Return capacity at most once."""
        if self.released:
            return
        self.released = True
        await self.gate.release()


@final
class _CapacityGate:
    """Bound active requests and FIFO waiters without adding a model scheduler."""

    def __init__(self, *, maximum: int, max_queue_size: int) -> None:
        self._maximum = maximum
        self._max_queue_size = max_queue_size
        self._active = 0
        self._waiters: deque[asyncio.Future[None]] = deque()
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    @property
    def queued(self) -> int:
        return len(self._waiters)

    async def acquire(self) -> _CapacityLease:
        """Acquire now or join the bounded FIFO waiter queue."""
        future: asyncio.Future[None] | None = None
        async with self._lock:
            if self._active < self._maximum:
                self._active += 1
                return _CapacityLease(self)
            if len(self._waiters) >= self._max_queue_size:
                raise _CapacityFullError
            future = asyncio.get_running_loop().create_future()
            self._waiters.append(future)
        try:
            await future
        except BaseException:
            async with self._lock:
                if future in self._waiters:
                    self._waiters.remove(future)
                elif future.done() and not future.cancelled():
                    self._release_locked()
            raise
        return _CapacityLease(self)

    async def release(self) -> None:
        """Transfer one active slot to the oldest live waiter."""
        async with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if waiter.cancelled():
                continue
            waiter.set_result(None)
            return
        if self._active <= 0:
            reason = "public demo capacity released without an active lease"
            raise RuntimeError(reason)
        self._active -= 1


@dataclass(slots=True)
class _PublicMetrics:
    completed_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    rate_limited_requests: int = 0
    timeout_requests: int = 0
    generated_token_count: int = 0
    latency_total_seconds: float = 0.0
    latency_samples: int = 0

    def reject(self, *, rate_limited: bool = False) -> None:
        self.rejected_requests += 1
        if rate_limited:
            self.rate_limited_requests += 1

    def complete(self, *, generated_tokens: int, latency_seconds: float) -> None:
        self.completed_requests += 1
        self.generated_token_count += generated_tokens
        self._latency(latency_seconds)

    def fail(self, *, latency_seconds: float, timed_out: bool = False) -> None:
        self.failed_requests += 1
        if timed_out:
            self.timeout_requests += 1
        self._latency(latency_seconds)

    def _latency(self, value: float) -> None:
        self.latency_total_seconds += max(0.0, value)
        self.latency_samples += 1

    def document(
        self,
        *,
        online: bool,
        active: int,
        queued: int,
    ) -> dict[str, JsonValue]:
        average_latency_ms = (
            None
            if self.latency_samples == 0
            else round(self.latency_total_seconds * 1000 / self.latency_samples, 3)
        )
        return {
            "online": online,
            "active_requests": active,
            "queued_requests": queued,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "timeout_requests": self.timeout_requests,
            "generated_token_count": self.generated_token_count,
            "average_latency_ms": average_latency_ms,
        }


@final
class _SlidingWindowLimiter:
    """Apply one atomic global window and one in-memory per-client window."""

    def __init__(
        self,
        *,
        policy: PublicDemoPolicy,
        clock: Callable[[], float],
    ) -> None:
        self._policy = policy
        self._clock = clock
        self._global: deque[float] = deque()
        self._clients: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, client_key: str) -> int | None:
        """Return Retry-After seconds when either window is saturated."""
        now = self._clock()
        async with self._lock:
            _prune(self._global, now - self._policy.global_window_seconds)
            client = self._clients.setdefault(client_key, deque())
            _prune(client, now - self._policy.per_client_window_seconds)
            retries: list[float] = []
            if len(self._global) >= self._policy.global_requests:
                retries.append(self._global[0] + self._policy.global_window_seconds - now)
            if len(client) >= self._policy.per_client_requests:
                retries.append(client[0] + self._policy.per_client_window_seconds - now)
            if retries:
                return max(1, math.ceil(max(retries)))
            self._global.append(now)
            client.append(now)
            return None


def _prune(values: deque[float], minimum: float) -> None:
    while values and values[0] <= minimum:
        _ = values.popleft()


@final
class _PublicBoundaryMiddleware:
    """Enforce exact CORS and safe response headers without buffering streams."""

    def __init__(self, app: ASGIApp, *, allowed_origins: tuple[str, ...]) -> None:
        self._app = app
        self._allowed_origins = frozenset(allowed_origins)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        raw_headers = cast("Sequence[tuple[bytes, bytes]]", scope["headers"])
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1") for key, value in raw_headers
        }
        origin = headers.get("origin")
        if origin is not None and origin not in self._allowed_origins:
            await _public_error(
                status_code=403,
                message="origin is not allowed",
                code="origin_not_allowed",
            )(scope, receive, send)
            return
        requested_method = headers.get("access-control-request-method")
        method = cast("str", scope["method"])
        if method == "OPTIONS" and requested_method is not None:
            await self._preflight(
                scope,
                receive,
                send,
                origin=origin,
                requested_method=requested_method,
                requested_headers=headers.get("access-control-request-headers", ""),
            )
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                raw_response_headers = cast(
                    "Sequence[tuple[bytes, bytes]]",
                    message.get("headers", []),
                )
                response_headers = list(raw_response_headers)
                _append_security_headers(response_headers)
                if origin is not None:
                    response_headers.extend(
                        [
                            (b"access-control-allow-origin", origin.encode("latin-1")),
                            (b"vary", b"Origin"),
                        ]
                    )
                message["headers"] = response_headers
            await send(message)

        await self._app(scope, receive, send_with_headers)

    async def _preflight(  # noqa: PLR0913
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        origin: str | None,
        requested_method: str,
        requested_headers: str,
    ) -> None:
        requested = {item.strip().lower() for item in requested_headers.split(",") if item.strip()}
        if (
            origin is None
            or requested_method.upper() not in {"GET", "POST"}
            or not requested.issubset(_CORS_HEADERS)
        ):
            await _public_error(
                status_code=403,
                message="CORS preflight is not allowed",
                code="cors_preflight_rejected",
            )(scope, receive, send)
            return
        response = Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": _CORS_METHODS,
                "Access-Control-Allow-Headers": ", ".join(sorted(_CORS_HEADERS)),
                "Access-Control-Max-Age": "600",
                "Vary": "Origin",
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )
        await response(scope, receive, send)


def _append_security_headers(headers: list[tuple[bytes, bytes]]) -> None:
    headers.extend(
        [
            (b"cache-control", b"no-store"),
            (b"x-content-type-options", b"nosniff"),
            (b"referrer-policy", b"no-referrer"),
        ]
    )


def create_public_demo_app(  # noqa: C901, PLR0913, PLR0915
    *,
    runner: EngineRunner,
    tokenizer: CharTokenizer,
    block_size: int,
    policy: PublicDemoPolicy,
    info: PublicDemoInfo,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Create the restricted ASGI app without changing the ordinary serve app."""
    _require_positive_integer("block_size", block_size)
    gate = _CapacityGate(
        maximum=policy.max_concurrent_requests,
        max_queue_size=policy.max_queue_size,
    )
    limiter = _SlidingWindowLimiter(policy=policy, clock=clock)
    metrics = _PublicMetrics()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        started = False
        if policy.enabled:
            runner.start()
            started = True
        try:
            yield
        finally:
            if started:
                await asyncio.to_thread(runner.shutdown)

    app = FastAPI(
        title="miniGPT Public Demo",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        _PublicBoundaryMiddleware,
        allowed_origins=policy.allowed_origins,
    )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        if not policy.enabled or not runner.is_running:
            return _public_error(
                status_code=503,
                message="public demo is offline",
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
                        "id": info.model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "minigpt",
                    }
                ],
            }
        )

    _ = models

    @app.get("/demo/info")
    async def demo_info() -> JSONResponse:
        return JSONResponse(
            {
                "project_version": info.project_version,
                "model_id": info.model_id,
                "demo_mode": "public" if policy.enabled else "disabled",
                "limits": policy.public_limits(),
                "executor": info.executor_name,
                "kv_cache_backend": info.kv_cache_backend,
                "prefix_cache_enabled": info.prefix_cache_enabled,
                "streaming_available": info.streaming_available,
            }
        )

    _ = demo_info

    @app.get("/demo/metrics")
    async def demo_metrics() -> JSONResponse:
        return JSONResponse(
            metrics.document(
                online=policy.enabled and runner.is_running,
                active=gate.active,
                queued=gate.queued,
            )
        )

    _ = demo_metrics

    @app.post("/v1/completions", response_model=None)
    async def completions(  # noqa: C901, PLR0911, PLR0912, PLR0915
        request: Request,
    ) -> JSONResponse | StreamingResponse:
        request_id = f"cmpl-{uuid4().hex}"
        started_at = clock()
        if not policy.enabled:
            metrics.reject()
            return _public_error(
                status_code=503,
                message="public demo is offline",
                code="service_unavailable",
            )
        retry_after = await limiter.allow(_client_key(request, policy))
        if retry_after is not None:
            metrics.reject(rate_limited=True)
            return _public_error(
                status_code=429,
                message="request rate limit exceeded",
                code="rate_limit_exceeded",
                headers={"Retry-After": str(retry_after)},
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + policy.request_timeout_seconds
        lease: _CapacityLease | None = None
        handle: RequestHandle | None = None
        try:
            async with asyncio.timeout_at(deadline):
                document = await _read_json_document(request, policy.max_request_body_bytes)
                completion = _parse_completion(document, policy)
                prompt_tokens = _encode_prompt(
                    tokenizer=tokenizer,
                    prompt=completion.prompt,
                    block_size=block_size,
                    policy=policy,
                )
                lease = await gate.acquire()
                handle = runner.submit(
                    GenerationRequest(
                        request_id=request_id,
                        prompt_tokens=prompt_tokens,
                        max_new_tokens=completion.max_tokens,
                        temperature=completion.temperature,
                        seed=completion.seed,
                        arrival_time=time.perf_counter(),
                    ),
                    stream=completion.stream,
                )
        except _RequestError as error:
            metrics.reject()
            if lease is not None:
                await lease.release()
            return _public_error(
                status_code=error.status_code,
                message=error.message,
                code=error.code,
                param=error.param,
            )
        except _CapacityFullError:
            metrics.reject()
            return _public_error(
                status_code=429,
                message="public demo queue is full",
                code="queue_full",
                headers={"Retry-After": "1"},
            )
        except RunnerQueueFullError:
            metrics.reject()
            if lease is not None:
                await lease.release()
            return _public_error(
                status_code=429,
                message="public demo queue is full",
                code="queue_full",
                headers={"Retry-After": "1"},
            )
        except RunnerUnavailableError:
            metrics.reject()
            if lease is not None:
                await lease.release()
            return _public_error(
                status_code=503,
                message="public demo is offline",
                code="service_unavailable",
            )
        except TimeoutError:
            if handle is not None:
                await _cancel_safely(runner, request_id)
            if lease is not None:
                await lease.release()
            metrics.fail(
                latency_seconds=_elapsed(clock, started_at),
                timed_out=True,
            )
            return _public_error(
                status_code=504,
                message="generation request timed out",
                code="request_timeout",
            )
        except ClientDisconnect:
            if handle is not None:
                await _cancel_safely(runner, request_id)
            if lease is not None:
                await lease.release()
            metrics.fail(latency_seconds=_elapsed(clock, started_at))
            return _public_error(
                status_code=499,
                message="client disconnected",
                code="client_disconnected",
            )
        except asyncio.CancelledError:
            if handle is not None:
                await _cancel_safely(runner, request_id)
            if lease is not None:
                await lease.release()
            metrics.fail(latency_seconds=_elapsed(clock, started_at))
            raise
        except Exception:  # noqa: BLE001
            if handle is not None:
                await _cancel_safely(runner, request_id)
            if lease is not None:
                await lease.release()
            metrics.fail(latency_seconds=_elapsed(clock, started_at))
            _log_request_failure(request_id)
            return _public_error(
                status_code=500,
                message="generation failed",
                code="internal_error",
            )

        _LOGGER.info("public demo request submitted request_id=%s", request_id)
        created = int(time.time())
        if completion.stream:
            return StreamingResponse(
                _stream_completion(
                    runner=runner,
                    handle=handle,
                    lease=lease,
                    tokenizer=tokenizer,
                    request_id=request_id,
                    created=created,
                    model_id=info.model_id,
                    prompt_tokens=len(prompt_tokens),
                    deadline=deadline,
                    metrics=metrics,
                    clock=clock,
                    started_at=started_at,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-store",
                    "X-Accel-Buffering": "no",
                },
            )
        return await _non_stream_completion(
            request=request,
            runner=runner,
            handle=handle,
            lease=lease,
            tokenizer=tokenizer,
            request_id=request_id,
            created=created,
            model_id=info.model_id,
            prompt_tokens=len(prompt_tokens),
            deadline=deadline,
            metrics=metrics,
            clock=clock,
            started_at=started_at,
        )

    _ = completions
    return app


async def _non_stream_completion(  # noqa: PLR0913
    *,
    request: Request,
    runner: EngineRunner,
    handle: RequestHandle,
    lease: _CapacityLease,
    tokenizer: CharTokenizer,
    request_id: str,
    created: int,
    model_id: str,
    prompt_tokens: int,
    deadline: float,
    metrics: _PublicMetrics,
    clock: Callable[[], float],
    started_at: float,
) -> JSONResponse:
    try:
        async with asyncio.timeout_at(deadline):
            result = await _wait_for_result_or_disconnect(request, handle)
    except TimeoutError:
        await _cancel_safely(runner, request_id)
        metrics.fail(latency_seconds=_elapsed(clock, started_at), timed_out=True)
        _LOGGER.info("public demo request timed out request_id=%s", request_id)
        return _public_error(
            status_code=504,
            message="generation request timed out",
            code="request_timeout",
        )
    except ClientDisconnect:
        await _cancel_safely(runner, request_id)
        metrics.fail(latency_seconds=_elapsed(clock, started_at))
        return _public_error(
            status_code=499,
            message="client disconnected",
            code="client_disconnected",
        )
    except asyncio.CancelledError:
        await _cancel_safely(runner, request_id)
        metrics.fail(latency_seconds=_elapsed(clock, started_at))
        raise
    except Exception:  # noqa: BLE001
        await _cancel_safely(runner, request_id)
        metrics.fail(latency_seconds=_elapsed(clock, started_at))
        _log_request_failure(request_id)
        return _public_error(
            status_code=500,
            message="generation failed",
            code="internal_error",
        )
    finally:
        await lease.release()

    latency = _elapsed(clock, started_at)
    if result.status is RequestStatus.FINISHED:
        metrics.complete(
            generated_tokens=len(result.generated_tokens),
            latency_seconds=latency,
        )
        _LOGGER.info("public demo request completed request_id=%s", request_id)
        return JSONResponse(
            _completion_document(
                request_id=request_id,
                created=created,
                model_id=model_id,
                text=tokenizer.decode(result.generated_tokens),
                finish_reason="length",
                usage=_usage(prompt_tokens, len(result.generated_tokens)),
            )
        )
    metrics.fail(latency_seconds=latency)
    _LOGGER.error("public demo request failed request_id=%s", request_id)
    status_code = 503 if result.status is RequestStatus.CANCELLED else 500
    code = "request_cancelled" if result.status is RequestStatus.CANCELLED else "generation_failed"
    return _public_error(
        status_code=status_code,
        message="generation did not complete",
        code=code,
    )


async def _wait_for_result_or_disconnect(
    request: Request,
    handle: RequestHandle,
) -> RunnerResult:
    result_task = asyncio.wrap_future(handle.future)
    stop = asyncio.Event()
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request, stop))
    tasks = (result_task, disconnect_task)
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if result_task in done:
            return result_task.result()
        raise ClientDisconnect
    finally:
        _ = stop.set()
        if not result_task.done():
            _ = result_task.cancel()
        _ = await asyncio.gather(*tasks, return_exceptions=True)


async def _wait_for_disconnect(request: Request, stop: asyncio.Event) -> None:
    while not stop.is_set():
        if await request.is_disconnected():
            return
        try:
            _ = await asyncio.wait_for(stop.wait(), timeout=0.05)
        except TimeoutError:
            continue


async def _stream_completion(  # noqa: C901, PLR0913
    *,
    runner: EngineRunner,
    handle: RequestHandle,
    lease: _CapacityLease,
    tokenizer: CharTokenizer,
    request_id: str,
    created: int,
    model_id: str,
    prompt_tokens: int,
    deadline: float,
    metrics: _PublicMetrics,
    clock: Callable[[], float],
    started_at: float,
) -> AsyncIterator[str]:
    stream_queue = handle.stream_queue
    if stream_queue is None:
        await lease.release()
        reason = "streaming request omitted its stream channel"
        raise RuntimeError(reason)
    completed_normally = False
    terminal_recorded = False
    try:
        async with asyncio.timeout_at(deadline):
            while True:
                event = await asyncio.to_thread(stream_queue.get)
                if event.event_type is StreamEventType.TOKEN:
                    if event.token_id is None:
                        metrics.fail(latency_seconds=_elapsed(clock, started_at))
                        terminal_recorded = True
                        yield _sse_error("generation failed", "internal_error")
                        return
                    yield _sse_data(
                        _completion_document(
                            request_id=request_id,
                            created=created,
                            model_id=model_id,
                            text=tokenizer.decode((event.token_id,)),
                            finish_reason=None,
                        )
                    )
                    continue
                if event.event_type is StreamEventType.FINISHED and event.result is not None:
                    generated_tokens = len(event.result.generated_tokens)
                    metrics.complete(
                        generated_tokens=generated_tokens,
                        latency_seconds=_elapsed(clock, started_at),
                    )
                    terminal_recorded = True
                    yield _sse_data(
                        _completion_document(
                            request_id=request_id,
                            created=created,
                            model_id=model_id,
                            text="",
                            finish_reason="length",
                            usage=_usage(prompt_tokens, generated_tokens),
                        )
                    )
                    yield "data: [DONE]\n\n"
                    completed_normally = True
                    _LOGGER.info("public demo request completed request_id=%s", request_id)
                    return
                metrics.fail(latency_seconds=_elapsed(clock, started_at))
                terminal_recorded = True
                yield _sse_error("generation failed", "generation_failed")
                return
    except TimeoutError:
        await _cancel_safely(runner, request_id)
        metrics.fail(latency_seconds=_elapsed(clock, started_at), timed_out=True)
        terminal_recorded = True
        _LOGGER.info("public demo request timed out request_id=%s", request_id)
        yield _sse_error("generation request timed out", "request_timeout")
    except asyncio.CancelledError:
        if not terminal_recorded:
            metrics.fail(latency_seconds=_elapsed(clock, started_at))
        raise
    except Exception:  # noqa: BLE001
        if not terminal_recorded:
            metrics.fail(latency_seconds=_elapsed(clock, started_at))
        _log_request_failure(request_id)
        yield _sse_error("generation failed", "internal_error")
    finally:
        if not completed_normally:
            await _cancel_safely(runner, request_id)
        await lease.release()


async def _read_json_document(
    request: Request,
    maximum_bytes: int,
) -> object:
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip()
    if content_type.lower() != "application/json":
        _request_error(
            "content type must be application/json",
            None,
            status_code=415,
            code="unsupported_media_type",
        )
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            _request_error("invalid Content-Length", None)
        if declared_length < 0:
            _request_error("invalid Content-Length", None)
        if declared_length > maximum_bytes:
            _request_error(
                "request body is too large",
                None,
                status_code=413,
                code="request_body_too_large",
            )
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > maximum_bytes:
            _request_error(
                "request body is too large",
                None,
                status_code=413,
                code="request_body_too_large",
            )
    try:
        text = bytes(payload).decode("utf-8")
        return cast(
            "object",
            json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            ),
        )
    except _RequestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _request_error("request body must be strict UTF-8 JSON", None)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            _request_error(f"duplicate request field {key!r}", key)
        document[key] = value
    return document


def _reject_json_constant(value: str) -> Never:
    reason = f"invalid JSON constant {value}"
    raise ValueError(reason)


def _parse_completion(  # noqa: C901
    document: object,
    policy: PublicDemoPolicy,
) -> _CompletionInput:
    if not isinstance(document, dict):
        _request_error("request body must be a JSON object", None)
    raw = cast("dict[object, object]", document)
    if any(not isinstance(key, str) for key in raw):
        _request_error("request body keys must be strings", None)
    values = cast("dict[str, object]", raw)
    unsupported = sorted(set(values) - _COMPLETION_FIELDS)
    if unsupported:
        field_name = unsupported[0]
        _request_error(f"unsupported completion field {field_name!r}", field_name)

    model = values.get("model")
    if not isinstance(model, str) or not model:
        _request_error("model must be a non-empty string", "model")
    if model != MODEL_ID:
        _request_error(
            "requested model is not served",
            "model",
            status_code=404,
            code="model_not_found",
        )
    prompt = values.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        _request_error("prompt must be a non-empty string", "prompt")
    if len(prompt) > policy.max_prompt_characters:
        _request_error(
            "prompt exceeds the public character limit",
            "prompt",
            code="prompt_too_long",
        )
    max_tokens = values.get("max_tokens", 16)
    if type(max_tokens) is not int or not 0 <= max_tokens <= policy.max_new_tokens:
        _request_error(
            f"max_tokens must be an integer in [0, {policy.max_new_tokens}]",
            "max_tokens",
        )
    temperature = values.get("temperature", 1.0)
    if type(temperature) not in {int, float}:
        _request_error("temperature must be a finite number", "temperature")
    normalized_temperature = float(cast("int | float", temperature))
    if (
        not math.isfinite(normalized_temperature)
        or not policy.min_temperature <= normalized_temperature <= policy.max_temperature
    ):
        _request_error(
            (f"temperature must be in [{policy.min_temperature}, {policy.max_temperature}]"),
            "temperature",
        )
    stream = values.get("stream", False)
    if type(stream) is not bool:
        _request_error("stream must be a boolean", "stream")
    seed = values.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < _MAX_SEED:
        _request_error(f"seed must be an integer in [0, {_MAX_SEED})", "seed")
    return _CompletionInput(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=normalized_temperature,
        stream=stream,
        seed=seed,
    )


def _encode_prompt(
    *,
    tokenizer: CharTokenizer,
    prompt: str,
    block_size: int,
    policy: PublicDemoPolicy,
) -> tuple[int, ...]:
    try:
        tokens = tuple(tokenizer.encode(prompt))
    except UnknownCharacterError:
        _request_error(
            "prompt contains unsupported characters",
            "prompt",
            code="invalid_prompt",
        )
    maximum = min(block_size, policy.max_prompt_tokens)
    if len(tokens) > maximum:
        _request_error(
            "prompt exceeds the public token limit",
            "prompt",
            code="prompt_too_long",
        )
    return tokens


def _client_key(request: Request, policy: PublicDemoPolicy) -> str:
    peer = request.client.host if request.client is not None else "unknown"
    if not policy.trust_proxy:
        return peer
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded is None:
        return peer
    candidate = forwarded.rsplit(",", maxsplit=1)[-1].strip()
    try:
        return ipaddress.ip_address(candidate).compressed
    except ValueError:
        return peer


async def _cancel_safely(runner: EngineRunner, request_id: str) -> None:
    with suppress(RunnerQueueFullError, RunnerUnavailableError):
        await asyncio.to_thread(runner.cancel, request_id)


def _elapsed(clock: Callable[[], float], started_at: float) -> float:
    return max(0.0, clock() - started_at)


def _completion_document(  # noqa: PLR0913
    *,
    request_id: str,
    created: int,
    model_id: str,
    text: str,
    finish_reason: str | None,
    usage: dict[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    document: dict[str, JsonValue] = {
        "id": request_id,
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


def _public_error(
    *,
    status_code: int,
    message: str,
    code: str,
    param: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": (
                    "invalid_request_error"
                    if status_code < _SERVER_ERROR_THRESHOLD
                    else "server_error"
                ),
                "param": param,
                "code": code,
            }
        },
        headers=dict(headers) if headers is not None else None,
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


def _log_request_failure(request_id: str) -> None:
    _LOGGER.error("public demo request failed request_id=%s", request_id)


def build_parser() -> argparse.ArgumentParser:
    """Create the explicit public-demo command parser."""
    parser = argparse.ArgumentParser(
        description="Run the restricted miniGPT public portfolio demo."
    )
    _ = parser.add_argument("--config", type=Path)
    _ = parser.add_argument("--checkpoint", type=Path)
    _ = parser.add_argument("--tokenizer", type=Path)
    _ = parser.add_argument("--host")
    _ = parser.add_argument("--port", type=int)
    _ = parser.add_argument("--log-level", choices=tuple(sorted(_LOG_LEVELS)))
    _ = parser.add_argument("--origin", action="append", dest="origins")
    _ = parser.add_argument("--trust-proxy", action="store_true")
    _ = parser.add_argument("--unsafe-allow-non-loopback", action="store_true")
    return parser


def build_runtime(
    arguments: argparse.Namespace,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[FastAPI, EngineRunner, PublicDemoServerConfig]:
    """Load local model assets once and build the restricted public-demo runtime."""
    values = cast("dict[str, object]", vars(arguments))
    settings = load_public_demo_settings(cast("Path | None", values["config"]))
    environment_values = os.environ if environment is None else environment
    settings = resolve_environment(
        settings,
        environment_values,
        trust_proxy=cast("bool", values["trust_proxy"]),
        origins=cast("list[str] | None", values["origins"]),
    )
    host = cast("str | None", values["host"])
    port = cast("int | None", values["port"])
    log_level = cast("str | None", values["log_level"])
    server = replace(
        settings.server,
        host=settings.server.host if host is None else host,
        port=settings.server.port if port is None else port,
        log_level=settings.server.log_level if log_level is None else log_level,
    )
    validate_bind_host(
        server.host,
        unsafe_allow_non_loopback=cast("bool", values["unsafe_allow_non_loopback"]),
        trust_proxy=settings.policy.trust_proxy,
    )
    settings.runtime.validate_policy(settings.policy)
    checkpoint_path = cast("Path | None", values["checkpoint"])
    tokenizer_path = cast("Path | None", values["tokenizer"])
    if checkpoint_path is None:
        checkpoint_value = environment_values.get("MINIGPT_CHECKPOINT")
        checkpoint_path = Path(checkpoint_value) if checkpoint_value else None
    if tokenizer_path is None:
        tokenizer_value = environment_values.get("MINIGPT_TOKENIZER")
        tokenizer_path = Path(tokenizer_value) if tokenizer_value else None
    if checkpoint_path is None or not checkpoint_path.is_file():
        _invalid("checkpoint must name an existing local file")
    if tokenizer_path is None or not tokenizer_path.is_file():
        _invalid("tokenizer must name an existing local file")

    tokenizer = CharTokenizer.load(tokenizer_path)
    experiment = load_checkpoint_config(checkpoint_path).resolve_vocab_size(tokenizer.vocab_size)
    if experiment.runtime.device != "cpu":
        _invalid("public demo supports CPU checkpoints only")
    _ = torch.set_num_threads(experiment.runtime.num_threads)
    model = GPT(experiment.model.to_gpt_config(experiment.data.block_size))
    load_model_state(checkpoint_path, model)
    _ = model.eval()
    runtime_config = settings.runtime.to_serving_runtime(
        block_size=experiment.data.block_size,
        policy=settings.policy,
    )
    runtime = build_serving_runtime(
        model=model,
        block_size=experiment.data.block_size,
        num_threads=experiment.runtime.num_threads,
        checkpoint_sha256=file_sha256(checkpoint_path),
        tokenizer_sha256=file_sha256(tokenizer_path),
        config=runtime_config,
    )
    info = PublicDemoInfo(
        project_version=__version__,
        model_id=MODEL_ID,
        executor_name=runtime_config.executor.value,
        kv_cache_backend=runtime_config.kv_cache_backend.value,
        prefix_cache_enabled=runtime_config.prefix_cache,
    )
    app = create_public_demo_app(
        runner=runtime.runner,
        tokenizer=tokenizer,
        block_size=experiment.data.block_size,
        policy=settings.policy,
        info=info,
    )
    return app, runtime.runner, server


def main(argv: Sequence[str] | None = None) -> int:
    """Run Uvicorn with proxy rewriting and access logging disabled."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        app, _runner, server = build_runtime(arguments)
    except (InvalidPublicDemoConfigError, OSError, ValueError) as error:
        parser.error(str(error))
    _ = uvicorn.run(
        app,
        host=server.host,
        port=server.port,
        log_level=server.log_level,
        access_log=False,
        proxy_headers=False,
        server_header=False,
        date_header=False,
        timeout_keep_alive=5,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
