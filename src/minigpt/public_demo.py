"""Expose the existing miniGPT runtime through a restricted public-demo boundary."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import logging
import math
import os
import queue
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Never, Protocol, TypeAlias, cast, final
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
from minigpt.http_server import MODEL_ID
from minigpt.model import GPT
from minigpt.paged_kv_cache import KVCacheBackend
from minigpt.prediction import (
    MAX_TOP_K,
    NextTokenDistribution,
    SequenceSurprisal,
    compute_next_token_distribution,
    compute_sequence_surprisal,
)
from minigpt.serving import APCPrefillStrategy, GenerationRequest, RequestStatus
from minigpt.serving_runtime import (
    ServingExecutorName,
    ServingRuntimeConfig,
    build_serving_runtime,
    file_sha256,
)
from minigpt.story import (
    StoryControlError,
    StoryControls,
    StoryFramingError,
    frame_story_prompt,
    story_control_prefix_ids,
)
from minigpt.story_data import THEMES, TONES, WORLDS
from minigpt.story_forge_product import (
    DEFAULT_BRANCH_COUNT,
    MAX_BRANCH_TOKENS,
    MAX_STORY_SEED,
    branch_seed_for,
    build_story_history,
)
from minigpt.tokenizer import (
    BPE_MODEL_FAMILY,
    BPE_SPECIAL_TOKEN_IDS,
    TokenizerProtocol,
    load_tokenizer,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
    from queue import Queue

    from starlette.types import ASGIApp, Message, Receive, Scope, Send

ConfigValue: TypeAlias = (
    str | int | float | bool | list["ConfigValue"] | dict[str, "ConfigValue"] | None
)
ConfigMapping: TypeAlias = dict[str, ConfigValue]

_LOGGER = logging.getLogger("minigpt.public_demo")
_MAX_SEED = 2**63
_MAX_PORT = 65535
_SERVER_ERROR_THRESHOLD = 500
_HOUR_SECONDS = 3600.0
_DAY_SECONDS = 86400.0
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
        "global_requests_per_hour",
        "global_generated_tokens_per_day",
        "streaming_enabled",
        "enabled",
    }
)
_TOP_LEVEL_KEYS = frozenset({"schema_version", "server", "runtime", "policy", "allowed_origins"})
_LOG_LEVELS = frozenset({"critical", "error", "warning", "info"})
_CORS_METHODS = "GET, POST, OPTIONS"
_CORS_HEADERS = frozenset({"content-type"})

_STORY_MODEL_ID = "minigpt-story-forge"
_STORY_FIELDS = frozenset(
    {"world", "tone", "theme", "opening", "seed", "branch_count", "max_tokens", "stream"}
)
_PREDICT_FIELDS = frozenset({"world", "tone", "theme", "text", "top_k"})
_MAX_PREDICT_TEXT_CHARS = 10_000
_STORY_EOS_ID = BPE_SPECIAL_TOKEN_IDS["<eos>"]

# Control tokens that must never surface in branch/prediction text output. The
# whole registered special-token vocabulary is excluded from rendered text via
# ``decode(skip_special_tokens=True)``; this set additionally guards pieces that
# a partial decode might otherwise render empty.
_STORY_CONTROL_TOKEN_IDS = frozenset(BPE_SPECIAL_TOKEN_IDS.values())
_STORY_SPECIAL_TOKEN_LABELS: dict[int, str] = {
    token_id: token for token, token_id in BPE_SPECIAL_TOKEN_IDS.items()
}


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
    global_requests_per_hour: int = 60
    global_generated_tokens_per_day: int = 10_000
    streaming_enabled: bool = False
    enabled: bool = False
    allowed_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject ambiguous limits and normalize exact CORS origins."""
        for name, value in (
            ("max_request_body_bytes", self.max_request_body_bytes),
            ("max_prompt_characters", self.max_prompt_characters),
            ("max_prompt_tokens", self.max_prompt_tokens),
            ("max_new_tokens", self.max_new_tokens),
            ("max_concurrent_requests", self.max_concurrent_requests),
            ("global_requests_per_hour", self.global_requests_per_hour),
            ("global_generated_tokens_per_day", self.global_generated_tokens_per_day),
        ):
            _require_positive_integer(name, value)
        _require_non_negative_integer("max_queue_size", self.max_queue_size)
        for name, value in (
            ("min_temperature", self.min_temperature),
            ("max_temperature", self.max_temperature),
            ("request_timeout_seconds", self.request_timeout_seconds),
        ):
            _require_positive_number(name, value)
        if self.min_temperature > self.max_temperature:
            _invalid("min_temperature must not exceed max_temperature")
        if self.global_generated_tokens_per_day < self.max_new_tokens:
            _invalid("global_generated_tokens_per_day must be at least max_new_tokens")
        for name, value in (
            ("streaming_enabled", self.streaming_enabled),
            ("enabled", self.enabled),
        ):
            if type(value) is not bool:
                _invalid(f"{name} must be a boolean")
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
            "global_requests_per_hour": self.global_requests_per_hour,
            "global_generated_tokens_per_day": self.global_generated_tokens_per_day,
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
    story_forge_enabled: bool = False
    prediction_lab_enabled: bool = False


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
    return replace(settings, policy=policy)


def validate_bind_host(
    host: str,
    *,
    unsafe_allow_non_loopback: bool,
) -> None:
    """Require loopback unless the operator names the unsafe exception."""
    if _is_loopback_host(host):
        return
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
        global_requests_per_hour=_integer(
            document, "global_requests_per_hour", source, positive=True
        ),
        global_generated_tokens_per_day=_integer(
            document, "global_generated_tokens_per_day", source, positive=True
        ),
        streaming_enabled=_boolean(document, "streaming_enabled", source),
        enabled=_boolean(document, "enabled", source),
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
class _StoryBranchInput:
    world: str
    tone: str
    theme: str
    opening: str
    seed: int
    branch_count: int
    max_tokens: int
    stream: bool


@dataclass(frozen=True, slots=True)
class _PredictInput:
    controls: StoryControls
    text: str
    top_k: int | None


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


@dataclass(frozen=True, slots=True)
class _GlobalQuotaFullError(RuntimeError):
    """Report an exhausted process-wide request or generated-token quota."""

    retry_after: int


@dataclass(slots=True)
class _GlobalQuotaLease:
    quota: _GlobalQuota
    request_id: str
    reserved_tokens: int
    finalized: bool = False

    async def cancel(self) -> None:
        """Roll back a request that the model runner did not accept."""
        if self.finalized:
            return
        self.finalized = True
        await self.quota.cancel(self.request_id)

    async def settle(self, generated_tokens: int) -> None:
        """Replace the hard reservation with the actual generated-token count."""
        if self.finalized:
            return
        self.finalized = True
        await self.quota.settle(
            self.request_id,
            reserved_tokens=self.reserved_tokens,
            generated_tokens=generated_tokens,
        )


@final
class _GlobalQuota:
    """Atomically enforce IP-independent hourly requests and daily output tokens."""

    def __init__(
        self,
        *,
        policy: PublicDemoPolicy,
        clock: Callable[[], float],
    ) -> None:
        self._policy = policy
        self._clock = clock
        self._requests: deque[tuple[str, float]] = deque()
        self._generated_tokens: deque[tuple[float, int]] = deque()
        self._reservations: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, request_id: str, maximum_tokens: int) -> _GlobalQuotaLease:
        """Reserve quota immediately before a request is submitted to the runner."""
        now = self._clock()
        async with self._lock:
            self._prune(now)
            if len(self._requests) >= self._policy.global_requests_per_hour:
                retry_after = max(
                    1,
                    math.ceil(self._requests[0][1] + _HOUR_SECONDS - now),
                )
                raise _GlobalQuotaFullError(retry_after)
            generated = sum(count for _timestamp, count in self._generated_tokens)
            reserved = sum(self._reservations.values())
            if generated + reserved + maximum_tokens > self._policy.global_generated_tokens_per_day:
                raise _GlobalQuotaFullError(
                    self._token_retry_after(
                        now=now,
                        required=generated
                        + reserved
                        + maximum_tokens
                        - self._policy.global_generated_tokens_per_day,
                    )
                )
            self._requests.append((request_id, now))
            self._reservations[request_id] = maximum_tokens
        return _GlobalQuotaLease(self, request_id, maximum_tokens)

    async def cancel(self, request_id: str) -> None:
        """Remove both reservations for a submission rejected by the runner."""
        async with self._lock:
            _ = self._reservations.pop(request_id, None)
            for record in self._requests:
                if record[0] == request_id:
                    self._requests.remove(record)
                    break

    async def settle(
        self,
        request_id: str,
        *,
        reserved_tokens: int,
        generated_tokens: int,
    ) -> None:
        """Commit actual output tokens while retaining the accepted request record."""
        if not 0 <= generated_tokens <= reserved_tokens:
            reason = "generated-token settlement exceeds its public quota reservation"
            raise RuntimeError(reason)
        async with self._lock:
            reserved = self._reservations.pop(request_id, None)
            if reserved is None:
                reason = "public quota settled without an active reservation"
                raise RuntimeError(reason)
            if reserved != reserved_tokens:
                reason = "public quota reservation changed before settlement"
                raise RuntimeError(reason)
            if generated_tokens > 0:
                self._generated_tokens.append((self._clock(), generated_tokens))

    def _prune(self, now: float) -> None:
        request_minimum = now - _HOUR_SECONDS
        while self._requests and self._requests[0][1] <= request_minimum:
            _ = self._requests.popleft()
        token_minimum = now - _DAY_SECONDS
        while self._generated_tokens and self._generated_tokens[0][0] <= token_minimum:
            _ = self._generated_tokens.popleft()

    def _token_retry_after(self, *, now: float, required: int) -> int:
        released = 0
        for timestamp, count in self._generated_tokens:
            released += count
            if released >= required:
                return max(1, math.ceil(timestamp + _DAY_SECONDS - now))
        return 1


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


class _StreamLifecycleProtocol(Protocol):
    """Shared terminal/cleanup surface accepted by the SSE response wrapper."""

    def fail(self, *, latency_seconds: float, timed_out: bool = False) -> None: ...

    async def cleanup(self, *, cancel: bool) -> None: ...


@dataclass(slots=True)
class _StreamLifecycle:
    runner: EngineRunner
    request_id: str
    lease: _CapacityLease
    quota_lease: _GlobalQuotaLease
    metrics: _PublicMetrics
    generated_tokens: int = 0
    terminal_recorded: bool = False
    cleanup_task: asyncio.Task[None] | None = field(default=None, init=False)

    def record_token(self) -> None:
        self.generated_tokens += 1

    def complete(self, *, generated_tokens: int, latency_seconds: float) -> None:
        self.generated_tokens = generated_tokens
        if self.terminal_recorded:
            return
        self.terminal_recorded = True
        self.metrics.complete(
            generated_tokens=generated_tokens,
            latency_seconds=latency_seconds,
        )

    def fail(self, *, latency_seconds: float, timed_out: bool = False) -> None:
        if self.terminal_recorded:
            return
        self.terminal_recorded = True
        self.metrics.fail(latency_seconds=latency_seconds, timed_out=timed_out)

    async def cleanup(self, *, cancel: bool) -> None:
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_once(cancel=cancel))
        cleanup_task = self.cleanup_task
        interrupted = False
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                interrupted = True
        cleanup_task.result()
        if interrupted:
            raise asyncio.CancelledError

    async def _cleanup_once(self, *, cancel: bool) -> None:
        if cancel:
            await _cancel_safely(self.runner, self.request_id)
        await self.quota_lease.settle(self.generated_tokens)
        await self.lease.release()


@final
class _PublicStreamingResponse(StreamingResponse):
    def __init__(
        self,
        content: AsyncGenerator[str, None],
        *,
        lifecycle: _StreamLifecycleProtocol,
        clock: Callable[[], float],
        started_at: float,
        headers: Mapping[str, str],
    ) -> None:
        super().__init__(content, media_type="text/event-stream", headers=headers)
        self._content = content
        self._lifecycle = lifecycle
        self._clock = clock
        self._started_at = started_at

    @override
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self._content.aclose()
            finally:
                self._lifecycle.fail(latency_seconds=_elapsed(self._clock, self._started_at))
                await self._lifecycle.cleanup(cancel=True)


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
    tokenizer: TokenizerProtocol,
    block_size: int,
    policy: PublicDemoPolicy,
    info: PublicDemoInfo,
    model: GPT | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Create the restricted ASGI app without changing the ordinary serve app."""
    _require_positive_integer("block_size", block_size)
    gate = _CapacityGate(
        maximum=policy.max_concurrent_requests,
        max_queue_size=policy.max_queue_size,
    )
    quota = _GlobalQuota(policy=policy, clock=clock)
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
                "streaming_enabled": policy.streaming_enabled,
                "features": {
                    "story_forge": info.story_forge_enabled,
                    "prediction_lab": info.prediction_lab_enabled,
                    "systems_lab": True,
                },
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
        loop = asyncio.get_running_loop()
        deadline = loop.time() + policy.request_timeout_seconds
        lease: _CapacityLease | None = None
        quota_lease: _GlobalQuotaLease | None = None
        handle: RequestHandle | None = None
        try:
            async with asyncio.timeout_at(deadline):
                document = await _read_json_document(request, policy.max_request_body_bytes)
                completion = _parse_completion(document, policy, model_id=info.model_id)
                prompt_tokens = _encode_prompt(
                    tokenizer=tokenizer,
                    prompt=completion.prompt,
                    block_size=block_size,
                    policy=policy,
                )
                lease = await gate.acquire()
                quota_lease = await quota.acquire(request_id, completion.max_tokens)
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
            if quota_lease is not None:
                await quota_lease.cancel()
            if lease is not None:
                await lease.release()
            return _public_error(
                status_code=error.status_code,
                message=error.message,
                code=error.code,
                param=error.param,
            )
        except _GlobalQuotaFullError as error:
            metrics.reject(rate_limited=True)
            if lease is not None:
                await lease.release()
            return _public_error(
                status_code=429,
                message="global public demo quota exceeded",
                code="rate_limit_exceeded",
                headers={"Retry-After": str(error.retry_after)},
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
            if quota_lease is not None:
                await quota_lease.cancel()
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
            if quota_lease is not None:
                await quota_lease.cancel()
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
            if quota_lease is not None:
                if handle is None:
                    await quota_lease.cancel()
                else:
                    await quota_lease.settle(0)
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
            if quota_lease is not None:
                if handle is None:
                    await quota_lease.cancel()
                else:
                    await quota_lease.settle(0)
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
            if quota_lease is not None:
                if handle is None:
                    await quota_lease.cancel()
                else:
                    await quota_lease.settle(0)
            if lease is not None:
                await lease.release()
            metrics.fail(latency_seconds=_elapsed(clock, started_at))
            raise
        except Exception:  # noqa: BLE001
            if handle is not None:
                await _cancel_safely(runner, request_id)
            if quota_lease is not None:
                if handle is None:
                    await quota_lease.cancel()
                else:
                    await quota_lease.settle(0)
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
            lifecycle = _StreamLifecycle(
                runner=runner,
                request_id=request_id,
                lease=lease,
                quota_lease=quota_lease,
                metrics=metrics,
            )
            stream = _stream_completion(
                handle=handle,
                lifecycle=lifecycle,
                tokenizer=tokenizer,
                request_id=request_id,
                created=created,
                model_id=info.model_id,
                prompt_tokens=len(prompt_tokens),
                deadline=deadline,
                clock=clock,
                started_at=started_at,
            )
            return _PublicStreamingResponse(
                stream,
                lifecycle=lifecycle,
                clock=clock,
                started_at=started_at,
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
            quota_lease=quota_lease,
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

    @app.post("/demo/story/branches", response_model=None)
    async def story_branches(request: Request) -> JSONResponse | StreamingResponse:
        started_at = clock()
        if not policy.enabled or not info.story_forge_enabled:
            metrics.reject()
            return _public_error(
                status_code=503,
                message="Story Forge is unavailable",
                code="story_forge_unavailable",
            )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + policy.request_timeout_seconds
        try:
            async with asyncio.timeout_at(deadline):
                document = await _read_json_document(request, policy.max_request_body_bytes)
                branch_input = _parse_story_branch(document, policy)
        except _RequestError as error:
            metrics.reject()
            return _public_error(
                status_code=error.status_code,
                message=error.message,
                code=error.code,
                param=error.param,
            )
        except TimeoutError:
            metrics.fail(latency_seconds=_elapsed(clock, started_at), timed_out=True)
            return _public_error(
                status_code=504, message="request timed out", code="request_timeout"
            )
        except ClientDisconnect:
            metrics.fail(latency_seconds=_elapsed(clock, started_at))
            return _public_error(
                status_code=499, message="client disconnected", code="client_disconnected"
            )
        except asyncio.CancelledError:
            metrics.fail(latency_seconds=_elapsed(clock, started_at))
            raise
        except Exception:  # noqa: BLE001
            metrics.fail(latency_seconds=_elapsed(clock, started_at))
            return _public_error(
                status_code=500, message="generation failed", code="internal_error"
            )
        return await _submit_story_branches(
            runner=runner,
            tokenizer=tokenizer,
            block_size=block_size,
            gate=gate,
            quota=quota,
            metrics=metrics,
            policy=policy,
            info=info,
            clock=clock,
            started_at=started_at,
            deadline=deadline,
            branch_input=branch_input,
            request=request,
        )

    _ = story_branches

    @app.post("/demo/predict/next", response_model=None)
    async def predict_next(request: Request) -> JSONResponse:
        return await _handle_predict(
            request=request,
            runner=runner,
            tokenizer=tokenizer,
            model=model,
            policy=policy,
            info=info,
            metrics=metrics,
            gate=gate,
            quota=quota,
            clock=clock,
            kind="next",
        )

    _ = predict_next

    @app.post("/demo/predict/score", response_model=None)
    async def predict_score(request: Request) -> JSONResponse:
        return await _handle_predict(
            request=request,
            runner=runner,
            tokenizer=tokenizer,
            model=model,
            policy=policy,
            info=info,
            metrics=metrics,
            gate=gate,
            quota=quota,
            clock=clock,
            kind="score",
        )

    _ = predict_score
    return app


async def _non_stream_completion(  # noqa: PLR0913
    *,
    request: Request,
    runner: EngineRunner,
    handle: RequestHandle,
    lease: _CapacityLease,
    quota_lease: _GlobalQuotaLease,
    tokenizer: TokenizerProtocol,
    request_id: str,
    created: int,
    model_id: str,
    prompt_tokens: int,
    deadline: float,
    metrics: _PublicMetrics,
    clock: Callable[[], float],
    started_at: float,
) -> JSONResponse:
    result: RunnerResult | None = None
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
        generated_tokens = 0 if result is None else len(result.generated_tokens)
        await quota_lease.settle(generated_tokens)
        await lease.release()

    terminal_result = cast("RunnerResult", result)
    latency = _elapsed(clock, started_at)
    if terminal_result.status is RequestStatus.FINISHED:
        metrics.complete(
            generated_tokens=len(terminal_result.generated_tokens),
            latency_seconds=latency,
        )
        _LOGGER.info("public demo request completed request_id=%s", request_id)
        return JSONResponse(
            _completion_document(
                request_id=request_id,
                created=created,
                model_id=model_id,
                text=tokenizer.decode(terminal_result.generated_tokens),
                finish_reason="length",
                usage=_usage(prompt_tokens, len(terminal_result.generated_tokens)),
            )
        )
    metrics.fail(latency_seconds=latency)
    _LOGGER.error("public demo request failed request_id=%s", request_id)
    status_code = 503 if terminal_result.status is RequestStatus.CANCELLED else 500
    code = (
        "request_cancelled"
        if terminal_result.status is RequestStatus.CANCELLED
        else "generation_failed"
    )
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


async def _stream_completion(  # noqa: PLR0913
    *,
    handle: RequestHandle,
    lifecycle: _StreamLifecycle,
    tokenizer: TokenizerProtocol,
    request_id: str,
    created: int,
    model_id: str,
    prompt_tokens: int,
    deadline: float,
    clock: Callable[[], float],
    started_at: float,
) -> AsyncGenerator[str, None]:
    stream_queue = handle.stream_queue
    if stream_queue is None:
        lifecycle.fail(latency_seconds=_elapsed(clock, started_at))
        await lifecycle.cleanup(cancel=True)
        reason = "streaming request omitted its stream channel"
        raise RuntimeError(reason)
    completed_normally = False
    try:
        async with asyncio.timeout_at(deadline):
            while True:
                event = await asyncio.to_thread(stream_queue.get)
                if event.event_type is StreamEventType.TOKEN:
                    if event.token_id is None:
                        lifecycle.fail(latency_seconds=_elapsed(clock, started_at))
                        yield _sse_error("generation failed", "internal_error")
                        return
                    lifecycle.record_token()
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
                    lifecycle.complete(
                        generated_tokens=len(event.result.generated_tokens),
                        latency_seconds=_elapsed(clock, started_at),
                    )
                    yield _sse_data(
                        _completion_document(
                            request_id=request_id,
                            created=created,
                            model_id=model_id,
                            text="",
                            finish_reason="length",
                            usage=_usage(prompt_tokens, lifecycle.generated_tokens),
                        )
                    )
                    yield "data: [DONE]\n\n"
                    completed_normally = True
                    _LOGGER.info("public demo request completed request_id=%s", request_id)
                    return
                lifecycle.fail(latency_seconds=_elapsed(clock, started_at))
                yield _sse_error("generation failed", "generation_failed")
                return
    except TimeoutError:
        lifecycle.fail(
            latency_seconds=_elapsed(clock, started_at),
            timed_out=True,
        )
        _LOGGER.info("public demo request timed out request_id=%s", request_id)
        yield _sse_error("generation request timed out", "request_timeout")
    except (asyncio.CancelledError, GeneratorExit):
        lifecycle.fail(latency_seconds=_elapsed(clock, started_at))
        raise
    except Exception:  # noqa: BLE001
        lifecycle.fail(latency_seconds=_elapsed(clock, started_at))
        _log_request_failure(request_id)
        yield _sse_error("generation failed", "internal_error")
    finally:
        await lifecycle.cleanup(cancel=not completed_normally)


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


def _parse_completion(  # noqa: C901, PLR0912
    document: object,
    policy: PublicDemoPolicy,
    *,
    model_id: str = MODEL_ID,
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
    if model != model_id:
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
    if stream and not policy.streaming_enabled:
        _request_error(
            "streaming is disabled for this public deployment",
            "stream",
            code="streaming_disabled",
        )
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
    tokenizer: TokenizerProtocol,
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


def _parse_story_branch(  # noqa: C901, PLR0912
    document: object,
    policy: PublicDemoPolicy,
) -> _StoryBranchInput:
    """Validate the Story Forge branch request fields under public policy bounds."""
    if not isinstance(document, dict):
        _request_error("request body must be a JSON object", None)
    raw = cast("dict[object, object]", document)
    if any(not isinstance(key, str) for key in raw):
        _request_error("request body keys must be strings", None)
    values = cast("dict[str, object]", raw)
    unsupported = sorted(set(values) - _STORY_FIELDS)
    if unsupported:
        _request_error(f"unsupported story field {unsupported[0]!r}", unsupported[0])

    world = values.get("world")
    if not isinstance(world, str) or world not in WORLDS:
        _request_error(f"world must be one of: {', '.join(WORLDS)}", "world")
    tone = values.get("tone")
    if not isinstance(tone, str) or tone not in TONES:
        _request_error(f"tone must be one of: {', '.join(TONES)}", "tone")
    theme = values.get("theme")
    if not isinstance(theme, str) or theme not in THEMES:
        _request_error(f"theme must be one of: {', '.join(THEMES)}", "theme")

    opening = values.get("opening", "")
    if not isinstance(opening, str):
        _request_error("opening must be a string", "opening")
    if len(opening) > policy.max_prompt_characters:
        _request_error(
            f"opening exceeds {policy.max_prompt_characters} characters",
            "opening",
            code="opening_too_long",
        )
    seed = values.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < MAX_STORY_SEED:
        _request_error(f"seed must be an integer in [0, {MAX_STORY_SEED})", "seed")
    branch_count = values.get("branch_count", DEFAULT_BRANCH_COUNT)
    if type(branch_count) is not int or branch_count != DEFAULT_BRANCH_COUNT:
        _request_error("branch_count must be exactly 3 for the public demo", "branch_count")
    max_tokens = values.get("max_tokens", MAX_BRANCH_TOKENS)
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_BRANCH_TOKENS:
        _request_error(
            f"max_tokens must be an integer in [1, {MAX_BRANCH_TOKENS}]",
            "max_tokens",
        )
    stream = values.get("stream", False)
    if type(stream) is not bool:
        _request_error("stream must be a boolean", "stream")
    if stream and not policy.streaming_enabled:
        _request_error(
            "streaming is disabled for this public deployment",
            "stream",
            code="streaming_disabled",
        )
    return _StoryBranchInput(
        world=world,
        tone=tone,
        theme=theme,
        opening=opening,
        seed=seed,
        branch_count=branch_count,
        max_tokens=max_tokens,
        stream=stream,
    )


def _parse_predict(document: object) -> _PredictInput:  # noqa: C901
    """Validate the Prediction Lab request fields and reject extras."""
    if not isinstance(document, dict):
        _request_error("request body must be a JSON object", None)
    raw = cast("dict[object, object]", document)
    if any(not isinstance(key, str) for key in raw):
        _request_error("request body keys must be strings", None)
    values = cast("dict[str, object]", raw)
    unsupported = sorted(set(values) - _PREDICT_FIELDS)
    if unsupported:
        _request_error(f"unsupported prediction field {unsupported[0]!r}", unsupported[0])

    world = values.get("world")
    if not isinstance(world, str) or world not in WORLDS:
        _request_error(f"world must be one of: {', '.join(WORLDS)}", "world")
    tone = values.get("tone")
    if not isinstance(tone, str) or tone not in TONES:
        _request_error(f"tone must be one of: {', '.join(TONES)}", "tone")
    theme = values.get("theme")
    if not isinstance(theme, str) or theme not in THEMES:
        _request_error(f"theme must be one of: {', '.join(THEMES)}", "theme")

    text = values.get("text", "")
    if not isinstance(text, str):
        _request_error("text must be a string", "text")
    if len(text) > _MAX_PREDICT_TEXT_CHARS:
        _request_error(f"text exceeds {_MAX_PREDICT_TEXT_CHARS} characters", "text")
    top_k = values.get("top_k", None)
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K):
        _request_error(f"top_k must be an integer in [1, {MAX_TOP_K}]", "top_k")
    try:
        controls = StoryControls(world=world, tone=tone, theme=theme)
    except StoryControlError as error:
        _request_error(str(error), None)
    return _PredictInput(controls=controls, text=text, top_k=top_k)


@dataclass(frozen=True, slots=True)
class _FramedStoryTokens:
    """A framed Story Forge request plus its truncation/retention accounting."""

    token_ids: tuple[int, ...]
    control_prefix_length: int
    history_truncated: bool
    retained_history_tokens: int


def _frame_story_tokens(  # noqa: PLR0913
    *,
    tokenizer: TokenizerProtocol,
    controls: StoryControls,
    opening: str,
    block_size: int,
    reserved_generation_tokens: int = 0,
    history_tokens: Sequence[int] | None = None,
) -> _FramedStoryTokens:
    """Frame a Story Forge prompt or later-round history into control-prefixed tokens."""
    if tokenizer.model_family != BPE_MODEL_FAMILY:
        _request_error(
            "Story Forge requires the story_forge model family",
            None,
            status_code=503,
            code="story_forge_unavailable",
        )
    prefix = _story_control_prefix(tokenizer, controls)
    if len(prefix) + reserved_generation_tokens > block_size:
        _request_error(
            "story request leaves no room for branch generation",
            None,
            code="story_context_exhausted",
        )
    if history_tokens:
        history = build_story_history(
            control_prefix_ids=prefix,
            story_token_ids=tuple(history_tokens),
            max_context_tokens=block_size - reserved_generation_tokens,
        )
        return _FramedStoryTokens(
            token_ids=history.token_ids,
            control_prefix_length=len(prefix),
            history_truncated=history.truncated,
            retained_history_tokens=len(history.token_ids) - len(prefix),
        )
    if not opening:
        # An empty opening is a valid request: generate directly from the
        # canonical control prefix without invoking the empty-rejecting framer.
        return _FramedStoryTokens(
            token_ids=prefix,
            control_prefix_length=len(prefix),
            history_truncated=False,
            retained_history_tokens=0,
        )
    try:
        framed = frame_story_prompt(
            tokenizer,
            controls,
            opening,
            max_context_tokens=block_size,
            reserved_generation_tokens=reserved_generation_tokens,
        )
    except (StoryControlError, StoryFramingError) as error:
        _request_error(str(error), None, code="story_framing_rejected")
    return _FramedStoryTokens(
        token_ids=framed.token_ids,
        control_prefix_length=framed.control_prefix_length,
        history_truncated=framed.truncated,
        retained_history_tokens=framed.retained_history_tokens,
    )


def _story_control_prefix(tokenizer: TokenizerProtocol, controls: StoryControls) -> tuple[int, ...]:
    """Resolve the canonical control prefix without sampling or mutation."""
    prefix = _story_prefix_via_story(tokenizer, controls)
    if prefix is None:
        _request_error(
            "tokenizer is missing Story Forge control tokens",
            None,
            status_code=503,
            code="story_forge_unavailable",
        )
    return prefix


def _story_prefix_via_story(
    tokenizer: TokenizerProtocol,
    controls: StoryControls,
) -> tuple[int, ...] | None:
    """Build the five-token control prefix from registered special tokens."""
    try:
        return story_control_prefix_ids(tokenizer, controls)
    except (StoryControlError, StoryFramingError):
        return None


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


# -- Prediction Lab -------------------------------------------------------------


def _frame_prediction_prompt(
    *,
    tokenizer: TokenizerProtocol,
    controls: StoryControls,
    text: str,
    block_size: int,
) -> tuple[int, ...]:
    """Frame prediction text with the canonical Story Forge control prefix."""
    if tokenizer.model_family != BPE_MODEL_FAMILY:
        _request_error(
            "Prediction Lab requires the story_forge model family",
            None,
            status_code=503,
            code="prediction_unavailable",
        )
    prefix = _story_control_prefix(tokenizer, controls)
    if not text:
        return prefix
    text_tokens = tokenizer.encode(text)
    budget = block_size - 1 - len(prefix)
    retained = text_tokens[-budget:] if len(text_tokens) > budget else text_tokens
    return (*prefix, *retained)


def _prediction_piece(tokenizer: TokenizerProtocol, token_id: int) -> tuple[str, bool]:
    """Return a display-safe piece plus whether it is a registered special token."""
    label = _STORY_SPECIAL_TOKEN_LABELS.get(token_id)
    if label is not None:
        return label, True
    try:
        piece = tokenizer.decode((token_id,), skip_special_tokens=True)
    except Exception:  # noqa: BLE001 - a single bad token must not fail the row.
        piece = ""
    if not piece or "�" in piece:
        return f"<token:{token_id}>", False
    return piece, False


def _prediction_next_document(
    *,
    tokenizer: TokenizerProtocol,
    distribution: NextTokenDistribution,
) -> dict[str, JsonValue]:
    """Render a top-k next-token distribution with display-safe pieces."""
    candidates: list[JsonValue] = []
    for candidate in distribution.candidates:
        piece, is_special = _prediction_piece(tokenizer, candidate.token_id)
        candidates.append(
            {
                "token_id": candidate.token_id,
                "piece": piece,
                "is_special": is_special,
                "logit": candidate.logit,
                "probability": candidate.probability,
            }
        )
    return {
        "object": "prediction_next",
        "top_k": distribution.top_k,
        "candidates": candidates,
    }


def _prediction_score_document(
    *,
    tokenizer: TokenizerProtocol,
    surprisal: SequenceSurprisal,
) -> dict[str, JsonValue]:
    """Render per-token surprisal chips plus aggregate NLL/perplexity.

    Control-prefix positions and user-text positions are reported separately;
    the full-framed ``mean_nll``/``perplexity`` are labeled as framed scores,
    never as user-text or authorship scores.
    """
    rows: list[JsonValue] = []
    for entry in surprisal.per_token:
        piece, is_special = _prediction_piece(tokenizer, entry.token_id)
        rows.append(
            {
                "token_id": entry.token_id,
                "piece": piece,
                "is_special": is_special,
                "is_control": entry.is_control,
                "surprisal": entry.surprisal,
            }
        )
    return {
        "object": "prediction_score",
        "per_token": rows,
        "control_prefix_length": surprisal.control_prefix_length,
        "mean_nll": surprisal.mean_nll,
        "perplexity": surprisal.perplexity,
        "user_mean_nll": surprisal.user_mean_nll,
        "user_perplexity": surprisal.user_perplexity,
        "disclaimer": ("model likelihood only; not authorship detection or semantic understanding"),
    }


async def _handle_predict(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0915
    *,
    request: Request,
    runner: EngineRunner,
    tokenizer: TokenizerProtocol,
    model: GPT | None,
    policy: PublicDemoPolicy,
    info: PublicDemoInfo,
    metrics: _PublicMetrics,
    gate: _CapacityGate,
    quota: _GlobalQuota,
    clock: Callable[[], float],
    kind: str,
) -> JSONResponse:
    """Serve one read-only Prediction Lab inspection on the owner thread."""
    started_at = clock()
    if not policy.enabled or not info.prediction_lab_enabled or model is None:
        metrics.reject()
        return _public_error(
            status_code=503,
            message="Prediction Lab is unavailable",
            code="prediction_unavailable",
        )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + policy.request_timeout_seconds
    lease: _CapacityLease | None = None
    quota_lease: _GlobalQuotaLease | None = None
    try:
        async with asyncio.timeout_at(deadline):
            document = await _read_json_document(request, policy.max_request_body_bytes)
            parsed = _parse_predict(document)
            lease = await gate.acquire()
            quota_lease = await quota.acquire(f"pred-{uuid4().hex}", 0)
            prefix = _story_control_prefix(tokenizer, parsed.controls)
            prompt_tokens = _frame_prediction_prompt(
                tokenizer=tokenizer,
                controls=parsed.controls,
                text=parsed.text,
                block_size=model.config.block_size,
            )
            if kind == "next":
                top_k = parsed.top_k if parsed.top_k is not None else MAX_TOP_K
                inspection_timeout = max(0.001, deadline - loop.time())
                distribution = await asyncio.to_thread(
                    runner.inspect,
                    lambda: compute_next_token_distribution(model, prompt_tokens, top_k=top_k),
                    timeout_seconds=inspection_timeout,
                )
                result = _prediction_next_document(
                    tokenizer=tokenizer,
                    distribution=cast("NextTokenDistribution", distribution),
                )
            else:
                inspection_timeout = max(0.001, deadline - loop.time())
                surprisal = await asyncio.to_thread(
                    runner.inspect,
                    lambda: compute_sequence_surprisal(
                        model,
                        prompt_tokens,
                        control_prefix_length=len(prefix),
                    ),
                    timeout_seconds=inspection_timeout,
                )
                result = _prediction_score_document(
                    tokenizer=tokenizer,
                    surprisal=cast("SequenceSurprisal", surprisal),
                )
    except _RequestError as error:
        metrics.reject()
        if quota_lease is not None:
            await quota_lease.cancel()
        if lease is not None:
            await lease.release()
        return _public_error(
            status_code=error.status_code,
            message=error.message,
            code=error.code,
            param=error.param,
        )
    except _GlobalQuotaFullError as error:
        metrics.reject(rate_limited=True)
        if lease is not None:
            await lease.release()
        return _public_error(
            status_code=429,
            message="global public demo quota exceeded",
            code="rate_limit_exceeded",
            headers={"Retry-After": str(error.retry_after)},
        )
    except _CapacityFullError:
        metrics.reject()
        return _public_error(
            status_code=429, message="public demo queue is full", code="queue_full"
        )
    except (RunnerQueueFullError, RunnerUnavailableError):
        metrics.reject()
        if quota_lease is not None:
            await quota_lease.cancel()
        if lease is not None:
            await lease.release()
        return _public_error(
            status_code=503, message="public demo is offline", code="service_unavailable"
        )
    except TimeoutError:
        metrics.fail(latency_seconds=_elapsed(clock, started_at), timed_out=True)
        if quota_lease is not None:
            await quota_lease.settle(0)
        if lease is not None:
            await lease.release()
        return _public_error(status_code=504, message="request timed out", code="request_timeout")
    except ClientDisconnect:
        metrics.fail(latency_seconds=_elapsed(clock, started_at))
        if quota_lease is not None:
            await quota_lease.settle(0)
        if lease is not None:
            await lease.release()
        return _public_error(
            status_code=499, message="client disconnected", code="client_disconnected"
        )
    except asyncio.CancelledError:
        metrics.fail(latency_seconds=_elapsed(clock, started_at))
        if quota_lease is not None:
            await quota_lease.settle(0)
        if lease is not None:
            await lease.release()
        raise
    except Exception:  # noqa: BLE001
        metrics.fail(latency_seconds=_elapsed(clock, started_at))
        if quota_lease is not None:
            await quota_lease.settle(0)
        if lease is not None:
            await lease.release()
        return _public_error(status_code=500, message="inspection failed", code="internal_error")

    await quota_lease.settle(0)
    await lease.release()
    metrics.complete(generated_tokens=0, latency_seconds=_elapsed(clock, started_at))
    return JSONResponse(result)


# -- Story Forge branches -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BranchHandles:
    """Bundle three submitted branch handles plus their shared leases."""

    request_ids: tuple[str, ...]
    handles: tuple[RequestHandle, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SubmittedBranches:
    """Three submitted branch handles plus shared framing/truncation metadata."""

    handles: _BranchHandles
    control_prefix_length: int
    history_truncated: bool
    retained_history_tokens: int


@dataclass(frozen=True, slots=True)
class _BranchOutcome:
    """One finished branch's terminal description."""

    text: str
    token_count: int
    finish_reason: str
    request_id: str
    history_truncated: bool
    retained_history_tokens: int


def _branch_finish_reason(
    *,
    status: RequestStatus,
    generated_tokens: Sequence[int],
    eos_token_id: int | None,
) -> str:
    """Derive the finish reason from actual terminal state and EOS identity.

    ``stop`` means the model emitted EOS; ``length`` means max-token exhaustion;
    ``cancelled`` and ``error`` map terminal scheduler states verbatim.
    """
    if status is RequestStatus.FINISHED:
        if generated_tokens and eos_token_id is not None and generated_tokens[-1] == eos_token_id:
            return "stop"
        return "length"
    if status is RequestStatus.CANCELLED:
        return "cancelled"
    return "error"


def _display_snapshot(tokenizer: TokenizerProtocol, token_ids: Sequence[int]) -> str:
    """Decode one complete generated-token prefix without exposing control tokens.

    ByteLevel BPE token IDs can represent only part of a UTF-8 code point, so
    decoding each token independently can emit replacement characters and is
    not a valid streaming boundary. Story SSE therefore publishes the complete
    decoded branch snapshot after each model token; the browser replaces the
    branch text with this snapshot instead of concatenating token pieces.
    """
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def submit_three_branches(
    *,
    runner: EngineRunner,
    tokenizer: TokenizerProtocol,
    block_size: int,
    branch_input: _StoryBranchInput,
    temperature: float,
) -> _SubmittedBranches:
    """Frame once and submit three branches before awaiting any result.

    Each branch derives a stable seed from ``(base_seed, branch_index)``. The
    shared control-prefix framing (with ``reserved_generation_tokens``) is
    computed once and reused. All three submission calls are made before any
    handle is awaited so the owner thread receives them together. On any
    partial failure after at least one successful submission, every accepted
    request is cancelled exactly once before the error is re-raised.
    """
    controls = StoryControls(
        world=branch_input.world,
        tone=branch_input.tone,
        theme=branch_input.theme,
    )
    framed = _frame_story_tokens(
        tokenizer=tokenizer,
        controls=controls,
        opening=branch_input.opening,
        block_size=block_size,
        reserved_generation_tokens=branch_input.max_tokens,
    )
    request_ids: list[str] = []
    handles: list[RequestHandle] = []
    try:
        for branch_index in range(branch_input.branch_count):
            seed = branch_seed_for(branch_input.seed, branch_index)
            request_id = f"stry-{uuid4().hex}"
            handle = runner.submit(
                GenerationRequest(
                    request_id=request_id,
                    prompt_tokens=framed.token_ids,
                    max_new_tokens=branch_input.max_tokens,
                    temperature=temperature,
                    seed=seed,
                    arrival_time=time.perf_counter(),
                    eos_token_id=tokenizer.eos_token_id,
                ),
                stream=branch_input.stream,
            )
            request_ids.append(request_id)
            handles.append(handle)
    except BaseException:
        # Transactional rollback: cancel every accepted request exactly once so
        # partial submissions never leak engine/KV resources.
        for request_id in request_ids:
            with suppress(RunnerQueueFullError, RunnerUnavailableError):
                runner.cancel(request_id)
        raise

    branch_handles = _BranchHandles(
        request_ids=tuple(request_ids),
        handles=tuple(handles),
        seeds=tuple(branch_seed_for(branch_input.seed, index) for index in range(len(handles))),
    )
    return _SubmittedBranches(
        handles=branch_handles,
        control_prefix_length=framed.control_prefix_length,
        history_truncated=framed.history_truncated,
        retained_history_tokens=framed.retained_history_tokens,
    )


def _branch_document(  # noqa: PLR0913
    *,
    branch_id: int,
    seed: int,
    text: str,
    token_count: int,
    finish_reason: str,
    request_id: str,
    history_truncated: bool,
    retained_history_tokens: int,
) -> dict[str, JsonValue]:
    return {
        "branch_id": branch_id,
        "seed": seed,
        "text": text,
        "token_count": token_count,
        "finish_reason": finish_reason,
        "request_id": request_id,
        "history_truncated": history_truncated,
        "retained_history_tokens": retained_history_tokens,
    }


async def _submit_story_branches(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0915
    *,
    runner: EngineRunner,
    tokenizer: TokenizerProtocol,
    block_size: int,
    gate: _CapacityGate,
    quota: _GlobalQuota,
    metrics: _PublicMetrics,
    policy: PublicDemoPolicy,
    info: PublicDemoInfo,
    clock: Callable[[], float],
    started_at: float,
    deadline: float,
    branch_input: _StoryBranchInput,
    request: Request,
) -> JSONResponse | StreamingResponse:
    """Acquire one aggregate lease, submit three branches, return ordered output."""
    temperature = min(0.8, policy.max_temperature)
    aggregate_token_budget = branch_input.max_tokens * branch_input.branch_count
    lease: _CapacityLease | None = None
    quota_lease: _GlobalQuotaLease | None = None
    submitted: _SubmittedBranches | None = None
    try:
        async with asyncio.timeout_at(deadline):
            lease = await gate.acquire()
            quota_lease = await quota.acquire(f"stry-{uuid4().hex}", aggregate_token_budget)
            submitted = submit_three_branches(
                runner=runner,
                tokenizer=tokenizer,
                block_size=block_size,
                branch_input=branch_input,
                temperature=temperature,
            )
    except _RequestError as error:
        metrics.reject()
        await _release_aggregate_lease(quota_lease, lease, generated=0)
        return _public_error(
            status_code=error.status_code,
            message=error.message,
            code=error.code,
            param=error.param,
        )
    except _GlobalQuotaFullError as error:
        metrics.reject(rate_limited=True)
        await _release_aggregate_lease(quota_lease, lease, generated=0)
        return _public_error(
            status_code=429,
            message="global public demo quota exceeded",
            code="rate_limit_exceeded",
            headers={"Retry-After": str(error.retry_after)},
        )
    except _CapacityFullError:
        metrics.reject()
        await _release_aggregate_lease(quota_lease, lease, generated=0)
        return _public_error(
            status_code=429, message="public demo queue is full", code="queue_full"
        )
    except (RunnerQueueFullError, RunnerUnavailableError):
        metrics.reject()
        await _release_aggregate_lease(quota_lease, lease, generated=0)
        return _public_error(
            status_code=503, message="public demo is offline", code="service_unavailable"
        )
    except TimeoutError:
        metrics.fail(latency_seconds=_elapsed(clock, started_at), timed_out=True)
        if submitted is not None:
            await _cancel_handles(runner, submitted.handles)
        await _release_aggregate_lease(quota_lease, lease, generated=0)
        return _public_error(
            status_code=504, message="generation timed out", code="request_timeout"
        )
    except asyncio.CancelledError:
        metrics.fail(latency_seconds=_elapsed(clock, started_at))
        if submitted is not None:
            await _cancel_handles(runner, submitted.handles)
        await _release_aggregate_lease(quota_lease, lease, generated=0)
        raise
    except Exception:  # noqa: BLE001
        metrics.fail(latency_seconds=_elapsed(clock, started_at))
        if submitted is not None:
            await _cancel_handles(runner, submitted.handles)
        await _release_aggregate_lease(quota_lease, lease, generated=0)
        return _public_error(status_code=500, message="generation failed", code="internal_error")

    created = int(time.time())
    asserted = submitted
    asserted_handles = asserted.handles
    if branch_input.stream:
        lifecycle = _BranchStreamLifecycle(
            runner=runner,
            request_ids=asserted_handles.request_ids,
            lease=lease,
            quota_lease=quota_lease,
            metrics=metrics,
            reserved_tokens=aggregate_token_budget,
        )
        stream = _stream_branches(
            handles=asserted_handles,
            lifecycle=lifecycle,
            tokenizer=tokenizer,
            deadline=deadline,
            clock=clock,
            started_at=started_at,
            eos_token_id=tokenizer.eos_token_id,
            history_truncated=asserted.history_truncated,
            retained_history_tokens=asserted.retained_history_tokens,
        )
        return _PublicStreamingResponse(
            stream,
            lifecycle=lifecycle,
            clock=clock,
            started_at=started_at,
            headers={
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        results = await _await_all_branches(
            runner=runner,
            handles=asserted_handles,
            tokenizer=tokenizer,
            deadline=deadline,
            eos_token_id=tokenizer.eos_token_id,
            history_truncated=asserted.history_truncated,
            retained_history_tokens=asserted.retained_history_tokens,
            request=request,
        )
    except TimeoutError:
        metrics.fail(latency_seconds=_elapsed(clock, started_at), timed_out=True)
        await _release_aggregate_lease(quota_lease, lease, generated=aggregate_token_budget)
        return _public_error(
            status_code=504, message="generation timed out", code="request_timeout"
        )
    except ClientDisconnect:
        metrics.fail(latency_seconds=_elapsed(clock, started_at))
        await _release_aggregate_lease(quota_lease, lease, generated=aggregate_token_budget)
        return _public_error(
            status_code=499, message="client disconnected", code="client_disconnected"
        )
    except asyncio.CancelledError:
        metrics.fail(latency_seconds=_elapsed(clock, started_at))
        await _release_aggregate_lease(quota_lease, lease, generated=aggregate_token_budget)
        raise
    except Exception:  # noqa: BLE001
        metrics.fail(latency_seconds=_elapsed(clock, started_at))
        _log_request_failure(asserted_handles.request_ids[0])
        await _release_aggregate_lease(quota_lease, lease, generated=aggregate_token_budget)
        return _public_error(status_code=500, message="generation failed", code="internal_error")
    generated = sum(result.token_count for result in results)
    all_branches_succeeded = all(result.finish_reason in {"stop", "length"} for result in results)
    settled_tokens = generated if all_branches_succeeded else aggregate_token_budget
    await _release_aggregate_lease(quota_lease, lease, generated=settled_tokens)
    if all_branches_succeeded:
        metrics.complete(generated_tokens=generated, latency_seconds=_elapsed(clock, started_at))
    else:
        metrics.fail(latency_seconds=_elapsed(clock, started_at))
    latency_ms = round(_elapsed(clock, started_at) * 1000, 3)
    branches = [
        _branch_document(
            branch_id=index,
            seed=asserted_handles.seeds[index],
            text=result.text,
            token_count=result.token_count,
            finish_reason=result.finish_reason,
            request_id=result.request_id,
            history_truncated=result.history_truncated,
            retained_history_tokens=result.retained_history_tokens,
        )
        for index, result in enumerate(results)
    ]
    return JSONResponse(
        {
            "object": "story_branches",
            "created": created,
            "model": info.model_id,
            "request_id": asserted_handles.request_ids[0],
            "history_truncated": asserted.history_truncated,
            "retained_history_tokens": asserted.retained_history_tokens,
            "branches": branches,
            "timing": {"aggregate_ms": latency_ms},
        }
    )


async def _release_aggregate_lease(
    quota_lease: _GlobalQuotaLease | None,
    lease: _CapacityLease | None,
    *,
    generated: int,
) -> None:
    """Release the aggregate capacity lease and settle quota exactly once."""
    if quota_lease is not None:
        await quota_lease.settle(generated)
    if lease is not None:
        await lease.release()


async def _await_all_branches(  # noqa: PLR0913
    *,
    runner: EngineRunner,
    handles: _BranchHandles,
    tokenizer: TokenizerProtocol,
    deadline: float,
    eos_token_id: int | None,
    history_truncated: bool,
    retained_history_tokens: int,
    request: Request,
) -> list[_BranchOutcome]:
    """Await all three branch futures unless the client disconnects, times out, or cancels.

    Timeout, disconnect, cancellation, and unexpected failure all cancel the
    three outstanding handles exactly once and raise a bounded error; the
    caller's outer block is responsible for releasing the aggregate lease/quota.
    """
    futures = [asyncio.wrap_future(handle.future) for handle in handles.handles]
    outcomes: list[_BranchOutcome] = []

    async def _await_outcomes() -> list[object]:
        return await asyncio.gather(*futures, return_exceptions=True)

    stop = asyncio.Event()
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request, stop))
    await_task = asyncio.create_task(_await_outcomes())
    awaited: list[object]
    try:
        loop = asyncio.get_running_loop()
        done, _pending = await asyncio.wait(
            (await_task, disconnect_task),
            timeout=max(0.0, deadline - loop.time()),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect_task in done:
            await _cancel_handles(runner, handles)
            raise ClientDisconnect
        if await_task not in done:
            await _cancel_handles(runner, handles)
            raise TimeoutError
        awaited = await_task.result()
    except asyncio.CancelledError:
        await _cancel_handles(runner, handles)
        raise
    finally:
        _ = stop.set()
        if not await_task.done():
            _ = await_task.cancel()
        _ = await asyncio.gather(await_task, disconnect_task, return_exceptions=True)

    for index, outcome in enumerate(awaited):
        request_id = handles.request_ids[index]
        if isinstance(outcome, BaseException):
            outcomes.append(
                _BranchOutcome(
                    text="",
                    token_count=0,
                    finish_reason="error",
                    request_id=request_id,
                    history_truncated=history_truncated,
                    retained_history_tokens=retained_history_tokens,
                )
            )
            continue
        terminal = cast("RunnerResult", outcome)
        text = tokenizer.decode(terminal.generated_tokens, skip_special_tokens=True)
        finish = _branch_finish_reason(
            status=terminal.status,
            generated_tokens=terminal.generated_tokens,
            eos_token_id=eos_token_id,
        )
        outcomes.append(
            _BranchOutcome(
                text=text,
                token_count=len(terminal.generated_tokens),
                finish_reason=finish,
                request_id=request_id,
                history_truncated=history_truncated,
                retained_history_tokens=retained_history_tokens,
            )
        )
    return outcomes


async def _cancel_handles(runner: EngineRunner, handles: _BranchHandles) -> None:
    for request_id in handles.request_ids:
        await _cancel_safely(runner, request_id)


@dataclass(slots=True)
class _BranchStreamLifecycle:
    """Track three-branch SSE cleanup and release the aggregate lease once."""

    runner: EngineRunner
    request_ids: tuple[str, ...]
    lease: _CapacityLease
    quota_lease: _GlobalQuotaLease
    metrics: _PublicMetrics
    reserved_tokens: int = 0
    generated_tokens: int = 0
    terminal_recorded: bool = False
    completed_normally: bool = False
    cleanup_task: asyncio.Task[None] | None = field(default=None, init=False)

    def record_token(self) -> None:
        self.generated_tokens += 1

    def complete(self, *, generated_tokens: int, latency_seconds: float) -> None:
        self.generated_tokens = generated_tokens
        self.completed_normally = True
        if self.terminal_recorded:
            return
        self.terminal_recorded = True
        self.metrics.complete(generated_tokens=generated_tokens, latency_seconds=latency_seconds)

    def fail(self, *, latency_seconds: float, timed_out: bool = False) -> None:
        if self.terminal_recorded:
            return
        self.terminal_recorded = True
        self.metrics.fail(latency_seconds=latency_seconds, timed_out=timed_out)

    async def cleanup(self, *, cancel: bool) -> None:
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_once(cancel=cancel))
        cleanup_task = self.cleanup_task
        interrupted = False
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                interrupted = True
        cleanup_task.result()
        if interrupted:
            raise asyncio.CancelledError

    def settle_tokens(self) -> int:
        """Return the token count to settle, charging the reserve when truncated.

        A complete three-branch terminal describes every generated token, so it
        settles the exact consumed count. Any abnormal termination (timeout,
        disconnect, cancellation, or unknown error) may leave branches that
        never reported a terminal event; settle the conservative reserved
        aggregate budget rather than under-charging the daily output quota.
        """
        if self.completed_normally:
            return self.generated_tokens
        return self.reserved_tokens

    async def _cleanup_once(self, *, cancel: bool) -> None:
        if cancel:
            for request_id in self.request_ids:
                await _cancel_safely(self.runner, request_id)
        await self.quota_lease.settle(self.settle_tokens())
        await self.lease.release()


async def _stream_branches(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    handles: _BranchHandles,
    lifecycle: _BranchStreamLifecycle,
    tokenizer: TokenizerProtocol,
    deadline: float,
    clock: Callable[[], float],
    started_at: float,
    eos_token_id: int | None,
    history_truncated: bool,
    retained_history_tokens: int,
) -> AsyncGenerator[str, None]:
    """Emit typed, multiplexed SSE events across three branches.

    Each blocking per-request ``queue.Queue`` is drained by its own thread task
    and merged into one async queue, so a slow branch never hides tokens or
    terminal events from faster branches and the event loop is never blocked.
    One branch failure emits that branch's terminal and keeps draining the
    others; ``done`` is emitted exactly once after all branches are terminal.
    """
    branch_count = len(handles.handles)
    stream_queues = [handle.stream_queue for handle in handles.handles]
    if any(stream_queue is None for stream_queue in stream_queues):
        lifecycle.fail(latency_seconds=_elapsed(clock, started_at))
        yield _sse_error("generation failed", "internal_error")
        return

    for branch_id in range(branch_count):
        yield _sse_branch_event(
            "branch_started",
            branch_id=branch_id,
            seed=handles.seeds[branch_id],
            request_id=handles.request_ids[branch_id],
            history_truncated=history_truncated,
            retained_history_tokens=retained_history_tokens,
        )

    merged: asyncio.Queue[tuple[int, StreamEvent]] = asyncio.Queue()

    terminal_reasons: list[str | None] = [None] * branch_count
    terminal_counts: list[int] = [0] * branch_count
    generated_token_ids: list[list[int]] = [[] for _ in range(branch_count)]
    token_indices = [0] * branch_count
    terminal_reached = 0
    completed_normally = False

    async def _drain_branch(branch_id: int, stream_queue: Queue[StreamEvent]) -> None:
        while True:
            try:
                event = await asyncio.to_thread(stream_queue.get, block=True, timeout=0.1)
            except queue.Empty:
                continue
            await merged.put((branch_id, event))
            if event.event_type is not StreamEventType.TOKEN:
                return

    drainers = [
        asyncio.create_task(_drain_branch(branch_id, cast("Queue[StreamEvent]", stream_queue)))
        for branch_id, stream_queue in enumerate(stream_queues)
    ]

    try:
        async with asyncio.timeout_at(deadline):
            while terminal_reached < branch_count:
                branch_id, event = await merged.get()
                if event.event_type is StreamEventType.TOKEN:
                    if event.token_id is None:
                        continue
                    lifecycle.record_token()  # count every token event exactly once
                    generated_token_ids[branch_id].append(event.token_id)
                    snapshot = _display_snapshot(tokenizer, generated_token_ids[branch_id])
                    yield _sse_branch_event(
                        "token",
                        branch_id=branch_id,
                        text=snapshot,
                        token_index=token_indices[branch_id],
                    )
                    token_indices[branch_id] += 1
                    continue
                if event.event_type is StreamEventType.FINISHED and event.result is not None:
                    reason = _branch_finish_reason(
                        status=event.result.status,
                        generated_tokens=event.result.generated_tokens,
                        eos_token_id=eos_token_id,
                    )
                    terminal_counts[branch_id] = len(event.result.generated_tokens)
                elif event.event_type is StreamEventType.CANCELLED:
                    reason = "cancelled"
                else:
                    reason = "error"
                terminal_reasons[branch_id] = reason
                terminal_reached += 1
                yield _sse_branch_finished(
                    branch_id=branch_id,
                    finish_reason=reason,
                    token_count=terminal_counts[branch_id],
                    seed=handles.seeds[branch_id],
                    request_id=handles.request_ids[branch_id],
                    history_truncated=history_truncated,
                    retained_history_tokens=retained_history_tokens,
                )
            lifecycle.complete(
                generated_tokens=lifecycle.generated_tokens,
                latency_seconds=_elapsed(clock, started_at),
            )
            yield _sse_branch_done()
            yield "data: [DONE]\n\n"
            completed_normally = True
    except TimeoutError:
        lifecycle.fail(latency_seconds=_elapsed(clock, started_at), timed_out=True)
        yield _sse_error("generation request timed out", "request_timeout")
    except (asyncio.CancelledError, GeneratorExit):
        lifecycle.fail(latency_seconds=_elapsed(clock, started_at))
        raise
    except Exception:  # noqa: BLE001
        lifecycle.fail(latency_seconds=_elapsed(clock, started_at))
        yield _sse_error("generation failed", "internal_error")
    finally:
        for task in drainers:
            _ = task.cancel()
        _ = await asyncio.gather(*drainers, return_exceptions=True)
        await lifecycle.cleanup(cancel=not completed_normally)


def _sse_branch_event(  # noqa: PLR0913
    event_type: str,
    *,
    branch_id: int,
    text: str = "",
    seed: int | None = None,
    request_id: str | None = None,
    token_index: int | None = None,
    history_truncated: bool | None = None,
    retained_history_tokens: int | None = None,
) -> str:
    payload: dict[str, JsonValue] = {
        "type": event_type,
        "branch_id": branch_id,
        "text": text,
    }
    if seed is not None:
        payload["seed"] = seed
    if request_id is not None:
        payload["request_id"] = request_id
    if token_index is not None:
        payload["token_index"] = token_index
    if history_truncated is not None:
        payload["history_truncated"] = history_truncated
    if retained_history_tokens is not None:
        payload["retained_history_tokens"] = retained_history_tokens
    return _sse_data(payload)


def _sse_branch_finished(  # noqa: PLR0913
    *,
    branch_id: int,
    finish_reason: str,
    token_count: int | None = None,
    seed: int | None = None,
    request_id: str | None = None,
    history_truncated: bool | None = None,
    retained_history_tokens: int | None = None,
) -> str:
    payload: dict[str, JsonValue] = {
        "type": "branch_finished",
        "branch_id": branch_id,
        "finish_reason": finish_reason,
    }
    if token_count is not None:
        payload["token_count"] = token_count
    if seed is not None:
        payload["seed"] = seed
    if request_id is not None:
        payload["request_id"] = request_id
    if history_truncated is not None:
        payload["history_truncated"] = history_truncated
    if retained_history_tokens is not None:
        payload["retained_history_tokens"] = retained_history_tokens
    return _sse_data(payload)


def _sse_branch_done() -> str:
    return _sse_data({"type": "done"})


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

    tokenizer = load_tokenizer(tokenizer_path)
    bpe_tokenizer = tokenizer.model_family == BPE_MODEL_FAMILY
    if bpe_tokenizer:
        _ = tokenizer.vocab_size  # force lazy BPE backend unless char
        _ = tokenizer.special_token_id("<bos>")
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
        tokenizer_type=tokenizer.tokenizer_type,
        model_family=tokenizer.model_family,
    )
    info = PublicDemoInfo(
        project_version=__version__,
        model_id=_STORY_MODEL_ID if bpe_tokenizer else MODEL_ID,
        executor_name=runtime_config.executor.value,
        kv_cache_backend=runtime_config.kv_cache_backend.value,
        prefix_cache_enabled=runtime_config.prefix_cache,
        story_forge_enabled=bpe_tokenizer,
        prediction_lab_enabled=bpe_tokenizer,
    )
    app = create_public_demo_app(
        runner=runtime.runner,
        tokenizer=tokenizer,
        block_size=experiment.data.block_size,
        policy=settings.policy,
        info=info,
        model=model,
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
