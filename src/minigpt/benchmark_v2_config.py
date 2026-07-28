"""Parse strict YAML configuration and identities for CPU Benchmark v2."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

import yaml
from typing_extensions import override

from minigpt.benchmark_v2_types import BenchmarkV2Case, BenchmarkV2Config, ProfileV2Settings
from minigpt.benchmark_workload_methodology import workload_methodology_document
from minigpt.settings import GPTConfig, InvalidModelConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from minigpt.benchmark_config import ConfigMapping, ConfigValue

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None

_SCHEMA_VERSION = 2
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "experiment_name",
        "benchmark_seed",
        "vocab_size",
        "output_root",
        "worker_timeout_seconds",
        "warmup_steps",
        "measurement_steps",
        "replicates",
        "torch_num_interop_threads",
        "cpu_affinity",
        "max_cv_percent",
        "minimum_replicates",
        "regression_threshold_percent",
        "relevant_environment_variables",
        "models",
        "cases",
        "profile",
    }
)
_MODEL_KEYS = frozenset({"n_layer", "n_head", "n_embd"})
_CASE_KEYS = frozenset({"name", "model_name", "torch_num_threads", "block_size", "batch_size"})
_RESOLVED_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS - frozenset({"models"})
_RESOLVED_CASE_KEYS = _CASE_KEYS | _MODEL_KEYS
_PROFILE_KEYS = frozenset({"enabled", "case_name", "warmup_steps", "active_steps"})


@dataclass(frozen=True, slots=True)
class InvalidBenchmarkV2ConfigError(ValueError):
    """Report a malformed or semantically invalid Benchmark v2 config."""

    source: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the config source and validation reason."""
        return f"invalid Benchmark v2 config {self.source}: {self.reason}"


class _DuplicateKeySafeLoader(yaml.SafeLoader):
    """Reject duplicate YAML mapping keys before schema validation can lose them."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Construct one YAML mapping while rejecting duplicate and non-string keys."""
    if not isinstance(node, yaml.MappingNode):
        msg = "expected a YAML mapping node"
        raise yaml.YAMLError(msg)
    mapping: dict[str, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            msg = "YAML mapping keys must be strings"
            raise yaml.YAMLError(msg)
        if key in mapping:
            msg = f"duplicate YAML mapping key {key!r}"
            raise yaml.YAMLError(msg)
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_DuplicateKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _load_yaml_document(content: bytes, source: Path) -> ConfigMapping:
    """Decode one duplicate-safe YAML mapping from already-owned bytes."""
    try:
        raw_document = yaml.load(
            content.decode("utf-8"),
            Loader=_DuplicateKeySafeLoader,  # noqa: S506
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise InvalidBenchmarkV2ConfigError(source, str(error)) from error
    return _mapping(cast("ConfigValue", raw_document), "top-level YAML value", source)


def _require_exact_keys(
    document: ConfigMapping, keys: frozenset[str], context: str, source: Path
) -> None:
    """Reject missing or unexpected keys in one schema mapping."""
    actual_keys = set(document)
    missing = keys - actual_keys
    unexpected = actual_keys - keys
    if missing:
        raise InvalidBenchmarkV2ConfigError(source, f"{context} missing key {min(missing)!r}")
    if unexpected:
        raise InvalidBenchmarkV2ConfigError(
            source,
            f"{context} has unexpected key {min(unexpected)!r}",
        )


def _mapping(value: ConfigValue, context: str, source: Path) -> ConfigMapping:
    """Require a YAML mapping whose keys are strings."""
    if not isinstance(value, dict):
        raise InvalidBenchmarkV2ConfigError(source, f"{context} must be a mapping")
    return cast("ConfigMapping", value)


def _string(document: ConfigMapping, key: str, source: Path) -> str:
    """Read one required non-empty string field."""
    value = document[key]
    if not isinstance(value, str) or not value:
        raise InvalidBenchmarkV2ConfigError(source, f"{key} must be a non-empty string")
    return value


def _integer(
    document: ConfigMapping,
    key: str,
    source: Path,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> int:
    """Read one required integer with an optional lower bound."""
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidBenchmarkV2ConfigError(source, f"{key} must be an integer")
    if positive and value <= 0:
        raise InvalidBenchmarkV2ConfigError(source, f"{key} must be positive")
    if non_negative and value < 0:
        raise InvalidBenchmarkV2ConfigError(source, f"{key} must be non-negative")
    return value


def _number(document: ConfigMapping, key: str, source: Path, *, positive: bool = False) -> float:
    """Read one finite number, optionally requiring a positive value."""
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidBenchmarkV2ConfigError(source, f"{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise InvalidBenchmarkV2ConfigError(source, f"{key} must be finite")
    if positive and number <= 0.0:
        raise InvalidBenchmarkV2ConfigError(source, f"{key} must be positive")
    return number


def _boolean(document: ConfigMapping, key: str, source: Path) -> bool:
    """Read one required boolean field."""
    value = document[key]
    if not isinstance(value, bool):
        raise InvalidBenchmarkV2ConfigError(source, f"{key} must be a boolean")
    return value


def _string_tuple(document: ConfigMapping, key: str, source: Path) -> tuple[str, ...]:
    """Read a list of unique non-empty string values."""
    value = document[key]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise InvalidBenchmarkV2ConfigError(source, f"{key} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise InvalidBenchmarkV2ConfigError(source, f"{key} must not contain duplicates")
    return cast("tuple[str, ...]", tuple(value))


def _cpu_affinity(document: ConfigMapping, source: Path) -> tuple[int, ...] | None:
    """Read an optional non-empty list of distinct logical CPU IDs."""
    value = document["cpu_affinity"]
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise InvalidBenchmarkV2ConfigError(source, "cpu_affinity must be a non-empty list or null")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise InvalidBenchmarkV2ConfigError(
            source,
            "cpu_affinity items must be non-negative logical CPU IDs",
        )
    if len(value) != len(set(value)):
        raise InvalidBenchmarkV2ConfigError(source, "cpu_affinity must not contain duplicate IDs")
    return cast("tuple[int, ...]", tuple(value))


def _parse_models(document: ConfigMapping, source: Path) -> dict[str, tuple[int, int, int]]:
    """Parse named GPT dimensions used by explicit cases."""
    raw_models = _mapping(document["models"], "models", source)
    if not raw_models:
        raise InvalidBenchmarkV2ConfigError(source, "models must not be empty")
    models: dict[str, tuple[int, int, int]] = {}
    for name, raw_model in raw_models.items():
        if not name:
            raise InvalidBenchmarkV2ConfigError(source, "models names must be non-empty")
        model = _mapping(raw_model, f"models.{name}", source)
        _require_exact_keys(model, _MODEL_KEYS, f"models.{name}", source)
        models[name] = (
            _integer(model, "n_layer", source, positive=True),
            _integer(model, "n_head", source, positive=True),
            _integer(model, "n_embd", source, positive=True),
        )
    return models


def _parse_cases(
    document: ConfigMapping,
    models: dict[str, tuple[int, int, int]],
    vocab_size: int,
    source: Path,
) -> tuple[BenchmarkV2Case, ...]:
    """Parse explicit case declarations and bind their named model dimensions."""
    raw_cases = document["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise InvalidBenchmarkV2ConfigError(source, "cases must be a non-empty list")
    cases: list[BenchmarkV2Case] = []
    names: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case_document = _mapping(raw_case, f"cases[{index}]", source)
        _require_exact_keys(case_document, _CASE_KEYS, f"cases[{index}]", source)
        name = _string(case_document, "name", source)
        if name in names:
            raise InvalidBenchmarkV2ConfigError(source, f"duplicate case name {name!r}")
        names.add(name)
        model_name = _string(case_document, "model_name", source)
        if model_name not in models:
            raise InvalidBenchmarkV2ConfigError(
                source, f"cases[{index}].model_name is an unknown model"
            )
        n_layer, n_head, n_embd = models[model_name]
        try:
            _ = GPTConfig(
                vocab_size=vocab_size,
                block_size=_integer(case_document, "block_size", source, positive=True),
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
            )
        except InvalidModelConfigError as error:
            raise InvalidBenchmarkV2ConfigError(source, str(error)) from error
        cases.append(
            BenchmarkV2Case(
                name=name,
                model_name=model_name,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                torch_num_threads=_integer(
                    case_document, "torch_num_threads", source, positive=True
                ),
                block_size=_integer(case_document, "block_size", source, positive=True),
                batch_size=_integer(case_document, "batch_size", source, positive=True),
            )
        )
    return tuple(cases)


def _parse_profile(
    document: ConfigMapping, cases: tuple[BenchmarkV2Case, ...], source: Path
) -> ProfileV2Settings:
    """Parse the optional profile settings against the explicit cases."""
    profile = _mapping(document["profile"], "profile", source)
    _require_exact_keys(profile, _PROFILE_KEYS, "profile", source)
    case_name = _string(profile, "case_name", source)
    if case_name not in {case.name for case in cases}:
        raise InvalidBenchmarkV2ConfigError(source, "profile.case_name is an unknown case")
    return ProfileV2Settings(
        enabled=_boolean(profile, "enabled", source),
        case_name=case_name,
        warmup_steps=_integer(profile, "warmup_steps", source, positive=True),
        active_steps=_integer(profile, "active_steps", source, positive=True),
    )


def _parse_resolved_cases(
    document: ConfigMapping, vocab_size: int, source: Path
) -> tuple[BenchmarkV2Case, ...]:
    """Parse the fully expanded cases persisted in a resolved run config."""
    raw_cases = document["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise InvalidBenchmarkV2ConfigError(source, "cases must be a non-empty list")
    cases: list[BenchmarkV2Case] = []
    names: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case_document = _mapping(raw_case, f"cases[{index}]", source)
        _require_exact_keys(case_document, _RESOLVED_CASE_KEYS, f"cases[{index}]", source)
        name = _string(case_document, "name", source)
        if name in names:
            raise InvalidBenchmarkV2ConfigError(source, f"duplicate case name {name!r}")
        names.add(name)
        model_name = _string(case_document, "model_name", source)
        n_layer = _integer(case_document, "n_layer", source, positive=True)
        n_head = _integer(case_document, "n_head", source, positive=True)
        n_embd = _integer(case_document, "n_embd", source, positive=True)
        block_size = _integer(case_document, "block_size", source, positive=True)
        try:
            _ = GPTConfig(
                vocab_size=vocab_size,
                block_size=block_size,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
            )
        except InvalidModelConfigError as error:
            raise InvalidBenchmarkV2ConfigError(source, str(error)) from error
        cases.append(
            BenchmarkV2Case(
                name=name,
                model_name=model_name,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                torch_num_threads=_integer(
                    case_document, "torch_num_threads", source, positive=True
                ),
                block_size=block_size,
                batch_size=_integer(case_document, "batch_size", source, positive=True),
            )
        )
    return tuple(cases)


def _build_config(
    document: ConfigMapping, cases: tuple[BenchmarkV2Case, ...], source: Path
) -> BenchmarkV2Config:
    """Validate shared top-level fields and construct one complete immutable config."""
    schema_version = _integer(document, "schema_version", source)
    if schema_version != _SCHEMA_VERSION:
        raise InvalidBenchmarkV2ConfigError(source, f"schema_version must be {_SCHEMA_VERSION}")
    replicates = _integer(document, "replicates", source, positive=True)
    minimum_replicates = _integer(document, "minimum_replicates", source, positive=True)
    if minimum_replicates > replicates:
        raise InvalidBenchmarkV2ConfigError(
            source=source, reason="minimum_replicates must not exceed replicates"
        )
    return BenchmarkV2Config(
        schema_version=schema_version,
        experiment_name=_string(document, "experiment_name", source),
        benchmark_seed=_integer(document, "benchmark_seed", source),
        vocab_size=_integer(document, "vocab_size", source, positive=True),
        output_root=Path(_string(document, "output_root", source)),
        worker_timeout_seconds=_number(document, "worker_timeout_seconds", source, positive=True),
        warmup_steps=_integer(document, "warmup_steps", source, non_negative=True),
        measurement_steps=_integer(document, "measurement_steps", source, positive=True),
        replicates=replicates,
        torch_num_interop_threads=_integer(
            document, "torch_num_interop_threads", source, positive=True
        ),
        cpu_affinity=_cpu_affinity(document, source),
        max_cv_percent=_number(document, "max_cv_percent", source, positive=True),
        minimum_replicates=minimum_replicates,
        regression_threshold_percent=_number(
            document, "regression_threshold_percent", source, positive=True
        ),
        relevant_environment_variables=_string_tuple(
            document, "relevant_environment_variables", source
        ),
        cases=cases,
        profile=_parse_profile(document, cases, source),
    )


def load_benchmark_v2_config(path: Path) -> BenchmarkV2Config:
    """Load one strict schema-v2 benchmark YAML document."""
    try:
        content = path.read_bytes()
    except OSError as error:
        raise InvalidBenchmarkV2ConfigError(path, str(error)) from error
    document = _load_yaml_document(content, path)
    _require_exact_keys(document, _TOP_LEVEL_KEYS, "top-level YAML value", path)
    vocab_size = _integer(document, "vocab_size", path, positive=True)
    models = _parse_models(document, path)
    cases = _parse_cases(document, models, vocab_size, path)
    return _build_config(document, cases, path)


def load_resolved_benchmark_v2_config(content: bytes, source: Path) -> BenchmarkV2Config:
    """Parse a snapshotted fully resolved YAML config without reading any path again."""
    document = _load_yaml_document(content, source)
    _require_exact_keys(document, _RESOLVED_TOP_LEVEL_KEYS, "top-level YAML value", source)
    vocab_size = _integer(document, "vocab_size", source, positive=True)
    cases = _parse_resolved_cases(document, vocab_size, source)
    return _build_config(document, cases, source)


def resolved_config_document(config: BenchmarkV2Config) -> dict[str, JsonValue]:
    """Return the resolved config as JSON-safe values in a stable structure."""
    return {
        "schema_version": config.schema_version,
        "experiment_name": config.experiment_name,
        "benchmark_seed": config.benchmark_seed,
        "vocab_size": config.vocab_size,
        "output_root": config.output_root.as_posix(),
        "worker_timeout_seconds": config.worker_timeout_seconds,
        "warmup_steps": config.warmup_steps,
        "measurement_steps": config.measurement_steps,
        "replicates": config.replicates,
        "torch_num_interop_threads": config.torch_num_interop_threads,
        "cpu_affinity": list(config.cpu_affinity) if config.cpu_affinity is not None else None,
        "max_cv_percent": config.max_cv_percent,
        "minimum_replicates": config.minimum_replicates,
        "regression_threshold_percent": config.regression_threshold_percent,
        "relevant_environment_variables": list(config.relevant_environment_variables),
        "cases": [
            {
                "name": case.name,
                "model_name": case.model_name,
                "n_layer": case.n_layer,
                "n_head": case.n_head,
                "n_embd": case.n_embd,
                "torch_num_threads": case.torch_num_threads,
                "block_size": case.block_size,
                "batch_size": case.batch_size,
            }
            for case in config.cases
        ],
        "profile": {
            "enabled": config.profile.enabled,
            "case_name": config.profile.case_name,
            "warmup_steps": config.profile.warmup_steps,
            "active_steps": config.profile.active_steps,
        },
    }


def _canonical_json(document: Mapping[str, JsonValue]) -> bytes:
    """Encode a JSON document deterministically for SHA-256 identities."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def resolved_config_sha256(config: BenchmarkV2Config) -> str:
    """Hash the canonical resolved config document with SHA-256."""
    return hashlib.sha256(_canonical_json(resolved_config_document(config))).hexdigest()


def _workload_document(config: BenchmarkV2Config, case: BenchmarkV2Case) -> dict[str, JsonValue]:
    """Select only settings that define the work performed by one case."""
    return {
        "benchmark_seed": config.benchmark_seed,
        "vocab_size": config.vocab_size,
        "warmup_steps": config.warmup_steps,
        "measurement_steps": config.measurement_steps,
        "torch_num_interop_threads": config.torch_num_interop_threads,
        "cpu_affinity": list(config.cpu_affinity) if config.cpu_affinity is not None else None,
        "case": {
            "name": case.name,
            "model_name": case.model_name,
            "n_layer": case.n_layer,
            "n_head": case.n_head,
            "n_embd": case.n_embd,
            "torch_num_threads": case.torch_num_threads,
            "block_size": case.block_size,
            "batch_size": case.batch_size,
        },
        "workload_methodology": cast("dict[str, JsonValue]", workload_methodology_document()),
    }


def case_identity(config: BenchmarkV2Config, case: BenchmarkV2Case) -> str:
    """Return a stable SHA-256 identity for one workload, not its execution settings."""
    return hashlib.sha256(_canonical_json(_workload_document(config, case))).hexdigest()
