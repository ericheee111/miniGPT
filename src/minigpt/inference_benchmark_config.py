"""Load strict configuration for isolated Stage 9 inference benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias, cast

import yaml
from typing_extensions import override

from minigpt.settings import GPTConfig, InvalidModelConfigError

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None
ConfigValue: TypeAlias = (
    str | int | float | bool | list["ConfigValue"] | dict[str, "ConfigValue"] | None
)
ConfigMapping: TypeAlias = dict[str, ConfigValue]

_SCHEMA_VERSION = 1
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "experiment_name",
        "benchmark_seed",
        "vocab_size",
        "output_root",
        "worker_timeout_seconds",
        "warmup_iterations",
        "measurement_iterations",
        "replicates",
        "minimum_replicates",
        "max_cv_percent",
        "torch_num_threads",
        "torch_num_interop_threads",
        "cpu_affinity",
        "relevant_environment_variables",
        "model",
        "batch_size",
        "prompt_lengths",
        "generated_lengths",
    }
)
_MODEL_KEYS = frozenset({"block_size", "n_layer", "n_head", "n_embd", "dropout", "bias"})


@dataclass(frozen=True, slots=True)
class InferenceCase:
    """Describe one non-overflow prompt/generated-length pair."""

    name: str
    batch_size: int
    prompt_length: int
    generated_length: int


@dataclass(frozen=True, slots=True)
class InferenceBenchmarkConfig:
    """Define one deterministic fresh-process inference benchmark matrix."""

    schema_version: int
    experiment_name: str
    benchmark_seed: int
    vocab_size: int
    output_root: Path
    worker_timeout_seconds: float
    warmup_iterations: int
    measurement_iterations: int
    replicates: int
    minimum_replicates: int
    max_cv_percent: float
    torch_num_threads: int
    torch_num_interop_threads: int
    cpu_affinity: tuple[int, ...] | None
    relevant_environment_variables: tuple[str, ...]
    model: GPTConfig
    batch_size: int
    prompt_lengths: tuple[int, ...]
    generated_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class InvalidInferenceBenchmarkConfigError(ValueError):
    """Report a malformed or semantically invalid inference benchmark config."""

    source: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the source path and failed constraint."""
        return f"invalid inference benchmark config {self.source}: {self.reason}"


def _mapping(value: object, source: Path, context: str) -> ConfigMapping:
    if not isinstance(value, dict):
        raise InvalidInferenceBenchmarkConfigError(source, f"{context} must be a mapping")
    raw = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in raw):
        raise InvalidInferenceBenchmarkConfigError(source, f"{context} keys must be strings")
    return cast("ConfigMapping", raw)


def _exact_keys(
    document: ConfigMapping, expected: frozenset[str], source: Path, context: str
) -> None:
    missing = expected - set(document)
    unexpected = set(document) - expected
    if missing:
        raise InvalidInferenceBenchmarkConfigError(
            source, f"{context} missing key {min(missing)!r}"
        )
    if unexpected:
        reason = f"{context} has unexpected key {min(unexpected)!r}"
        raise InvalidInferenceBenchmarkConfigError(source, reason)


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
        raise InvalidInferenceBenchmarkConfigError(source, f"{key} must be an integer")
    if positive and value <= 0:
        raise InvalidInferenceBenchmarkConfigError(source, f"{key} must be positive")
    if non_negative and value < 0:
        raise InvalidInferenceBenchmarkConfigError(source, f"{key} must be non-negative")
    return value


def _number(document: ConfigMapping, key: str, source: Path, *, positive: bool = False) -> float:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInferenceBenchmarkConfigError(source, f"{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        requirement = "positive and finite" if positive else "finite"
        raise InvalidInferenceBenchmarkConfigError(source, f"{key} must be {requirement}")
    return number


def _string(document: ConfigMapping, key: str, source: Path) -> str:
    value = document[key]
    if not isinstance(value, str) or not value:
        raise InvalidInferenceBenchmarkConfigError(source, f"{key} must be a non-empty string")
    return value


def _integer_tuple(document: ConfigMapping, key: str, source: Path) -> tuple[int, ...]:
    value = document[key]
    if not isinstance(value, list) or not value:
        raise InvalidInferenceBenchmarkConfigError(source, f"{key} must be a non-empty list")
    raw_values = cast("list[object]", value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in raw_values):
        raise InvalidInferenceBenchmarkConfigError(source, f"{key} must contain positive integers")
    values = cast("tuple[int, ...]", tuple(raw_values))
    if len(values) != len(set(values)):
        raise InvalidInferenceBenchmarkConfigError(source, f"{key} must not contain duplicates")
    return values


def _string_tuple(document: ConfigMapping, key: str, source: Path) -> tuple[str, ...]:
    value = document[key]
    if not isinstance(value, list):
        raise InvalidInferenceBenchmarkConfigError(source, f"{key} must be a list")
    raw_values = cast("list[object]", value)
    if any(not isinstance(item, str) or not item for item in raw_values):
        raise InvalidInferenceBenchmarkConfigError(source, f"{key} must contain non-empty strings")
    values = cast("tuple[str, ...]", tuple(raw_values))
    if len(values) != len(set(values)):
        raise InvalidInferenceBenchmarkConfigError(source, f"{key} must not contain duplicates")
    return values


def _affinity(document: ConfigMapping, source: Path) -> tuple[int, ...] | None:
    value = document["cpu_affinity"]
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        reason = "cpu_affinity must be null or a non-empty list"
        raise InvalidInferenceBenchmarkConfigError(source, reason)
    raw_values = cast("list[object]", value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in raw_values):
        reason = "cpu_affinity must contain non-negative integers"
        raise InvalidInferenceBenchmarkConfigError(source, reason)
    values = cast("tuple[int, ...]", tuple(raw_values))
    if len(values) != len(set(values)):
        raise InvalidInferenceBenchmarkConfigError(
            source, "cpu_affinity must not contain duplicates"
        )
    return values


def _model(document: ConfigMapping, source: Path, *, vocab_size: int) -> GPTConfig:
    raw_model = _mapping(document["model"], source, "model")
    _exact_keys(raw_model, _MODEL_KEYS, source, "model")
    dropout = _number(raw_model, "dropout", source)
    bias = raw_model["bias"]
    if not isinstance(bias, bool):
        raise InvalidInferenceBenchmarkConfigError(source, "bias must be a boolean")
    try:
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=_integer(raw_model, "block_size", source, positive=True),
            n_layer=_integer(raw_model, "n_layer", source, positive=True),
            n_head=_integer(raw_model, "n_head", source, positive=True),
            n_embd=_integer(raw_model, "n_embd", source, positive=True),
            dropout=dropout,
            bias=bias,
        )
    except InvalidModelConfigError as error:
        raise InvalidInferenceBenchmarkConfigError(source, str(error)) from error


def load_inference_benchmark_config(path: Path) -> InferenceBenchmarkConfig:
    """Load one exact schema-v1 YAML benchmark configuration."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise InvalidInferenceBenchmarkConfigError(path, str(error)) from error
    document = _mapping(raw, path, "top-level YAML value")
    _exact_keys(document, _TOP_LEVEL_KEYS, path, "top-level YAML value")
    schema_version = _integer(document, "schema_version", path)
    if schema_version != _SCHEMA_VERSION:
        raise InvalidInferenceBenchmarkConfigError(path, "schema_version must be 1")
    vocab_size = _integer(document, "vocab_size", path, positive=True)
    model = _model(document, path, vocab_size=vocab_size)
    replicates = _integer(document, "replicates", path, positive=True)
    minimum_replicates = _integer(document, "minimum_replicates", path, positive=True)
    if minimum_replicates > replicates:
        reason = "minimum_replicates must not exceed replicates"
        raise InvalidInferenceBenchmarkConfigError(path, reason)
    prompt_lengths = _integer_tuple(document, "prompt_lengths", path)
    generated_lengths = _integer_tuple(document, "generated_lengths", path)
    for prompt_length in prompt_lengths:
        for generated_length in generated_lengths:
            if prompt_length + generated_length > model.block_size:
                reason = (
                    f"canonical prompt length {prompt_length} plus generated length "
                    f"{generated_length} exceeds block_size {model.block_size}"
                )
                raise InvalidInferenceBenchmarkConfigError(path, reason)
    return InferenceBenchmarkConfig(
        schema_version=schema_version,
        experiment_name=_string(document, "experiment_name", path),
        benchmark_seed=_integer(document, "benchmark_seed", path),
        vocab_size=vocab_size,
        output_root=Path(_string(document, "output_root", path)),
        worker_timeout_seconds=_number(document, "worker_timeout_seconds", path, positive=True),
        warmup_iterations=_integer(document, "warmup_iterations", path, non_negative=True),
        measurement_iterations=_integer(document, "measurement_iterations", path, positive=True),
        replicates=replicates,
        minimum_replicates=minimum_replicates,
        max_cv_percent=_number(document, "max_cv_percent", path, positive=True),
        torch_num_threads=_integer(document, "torch_num_threads", path, positive=True),
        torch_num_interop_threads=_integer(
            document, "torch_num_interop_threads", path, positive=True
        ),
        cpu_affinity=_affinity(document, path),
        relevant_environment_variables=_string_tuple(
            document, "relevant_environment_variables", path
        ),
        model=model,
        batch_size=_integer(document, "batch_size", path, positive=True),
        prompt_lengths=prompt_lengths,
        generated_lengths=generated_lengths,
    )


def expand_inference_cases(config: InferenceBenchmarkConfig) -> tuple[InferenceCase, ...]:
    """Expand the configured prompt/generated-length matrix in stable order."""
    return tuple(
        InferenceCase(
            name=f"p{prompt_length}-g{generated_length}",
            batch_size=config.batch_size,
            prompt_length=prompt_length,
            generated_length=generated_length,
        )
        for prompt_length in config.prompt_lengths
        for generated_length in config.generated_lengths
    )


def resolved_config_document(config: InferenceBenchmarkConfig) -> dict[str, JsonValue]:
    """Return a stable fully explicit JSON-safe configuration document."""
    model_document = cast("dict[str, JsonValue]", asdict(config.model))
    return {
        "schema_version": config.schema_version,
        "experiment_name": config.experiment_name,
        "benchmark_seed": config.benchmark_seed,
        "vocab_size": config.vocab_size,
        "output_root": config.output_root.as_posix(),
        "worker_timeout_seconds": config.worker_timeout_seconds,
        "warmup_iterations": config.warmup_iterations,
        "measurement_iterations": config.measurement_iterations,
        "replicates": config.replicates,
        "minimum_replicates": config.minimum_replicates,
        "max_cv_percent": config.max_cv_percent,
        "torch_num_threads": config.torch_num_threads,
        "torch_num_interop_threads": config.torch_num_interop_threads,
        "cpu_affinity": list(config.cpu_affinity) if config.cpu_affinity is not None else None,
        "relevant_environment_variables": list(config.relevant_environment_variables),
        "model": model_document,
        "batch_size": config.batch_size,
        "prompt_lengths": list(config.prompt_lengths),
        "generated_lengths": list(config.generated_lengths),
    }


def resolved_config_bytes(config: InferenceBenchmarkConfig) -> bytes:
    """Serialize the resolved config with stable UTF-8 YAML bytes."""
    rendered = yaml.safe_dump(
        resolved_config_document(config),
        sort_keys=False,
        allow_unicode=True,
    )
    return rendered.encode("utf-8")


def resolved_config_sha256(config: InferenceBenchmarkConfig) -> str:
    """Hash the exact stable resolved configuration semantics."""
    canonical = json.dumps(
        resolved_config_document(config),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
