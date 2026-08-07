"""Load strict Stage 11A fresh-process serving benchmark configuration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Never, TypeAlias, cast

import yaml
from typing_extensions import override

from minigpt.serving import PrefillBatchConfig
from minigpt.settings import GPTConfig, InvalidModelConfigError

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None
ConfigValue: TypeAlias = (
    str | int | float | bool | list["ConfigValue"] | dict[str, "ConfigValue"] | None
)
ConfigMapping: TypeAlias = dict[str, ConfigValue]

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "experiment_name",
        "benchmark_seed",
        "output_root",
        "worker_timeout_seconds",
        "warmup_iterations",
        "measurement_iterations",
        "replicates",
        "minimum_replicates",
        "max_cv_percent",
        "torch_num_threads",
        "torch_num_interop_threads",
        "vocab_size",
        "model",
        "prefill",
        "scenarios",
    }
)
_REQUIRED_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS - {"prefill"}
_MODEL_KEYS = frozenset({"block_size", "n_layer", "n_head", "n_embd", "dropout", "bias"})
_PREFILL_KEYS = frozenset({"max_batch_size", "max_batch_tokens", "max_padding_ratio"})
_SCENARIO_KEYS = frozenset(
    {"name", "arrival_ticks", "prompt_lengths", "generated_lengths", "cancellation_ticks"}
)


@dataclass(frozen=True, slots=True)
class ServingBenchmarkScenario:
    """Define arrivals, prompt/decode lengths, and optional cancellations."""

    name: str
    arrival_ticks: tuple[int, ...]
    prompt_lengths: tuple[int, ...]
    generated_lengths: tuple[int, ...]
    cancellation_ticks: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class ServingBenchmarkConfig:
    """Define the canonical fresh-process comparison matrix."""

    schema_version: int
    experiment_name: str
    benchmark_seed: int
    output_root: Path
    worker_timeout_seconds: float
    warmup_iterations: int
    measurement_iterations: int
    replicates: int
    minimum_replicates: int
    max_cv_percent: float
    torch_num_threads: int
    torch_num_interop_threads: int
    vocab_size: int
    model: GPTConfig
    prefill: PrefillBatchConfig | None
    scenarios: tuple[ServingBenchmarkScenario, ...]


@dataclass(frozen=True, slots=True)
class InvalidServingBenchmarkConfigError(ValueError):
    """Report one malformed Stage 11A benchmark input."""

    source: Path
    reason: str

    @override
    def __str__(self) -> str:
        return f"invalid serving benchmark config {self.source}: {self.reason}"


def _invalid(source: Path, reason: str) -> Never:
    raise InvalidServingBenchmarkConfigError(source, reason)


def _mapping(value: object, source: Path, context: str) -> ConfigMapping:
    if not isinstance(value, dict):
        _invalid(source, f"{context} must be a mapping")
    raw = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in raw):
        _invalid(source, f"{context} keys must be strings")
    return cast("ConfigMapping", raw)


def _exact_keys(
    document: ConfigMapping, expected: frozenset[str], source: Path, context: str
) -> None:
    missing = expected - set(document)
    unexpected = set(document) - expected
    if missing:
        _invalid(source, f"{context} missing key {min(missing)!r}")
    if unexpected:
        _invalid(source, f"{context} has unexpected key {min(unexpected)!r}")


def _top_level_keys(document: ConfigMapping, source: Path) -> None:
    missing = _REQUIRED_TOP_LEVEL_KEYS - set(document)
    unexpected = set(document) - _TOP_LEVEL_KEYS
    if missing:
        _invalid(source, f"document missing key {min(missing)!r}")
    if unexpected:
        _invalid(source, f"document has unexpected key {min(unexpected)!r}")


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
        _invalid(source, f"{key} must be an integer")
    if positive and value <= 0:
        _invalid(source, f"{key} must be positive")
    if non_negative and value < 0:
        _invalid(source, f"{key} must be non-negative")
    return value


def _number(document: ConfigMapping, key: str, source: Path, *, positive: bool = False) -> float:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(source, f"{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        _invalid(source, f"{key} must be finite{' and positive' if positive else ''}")
    return number


def _string(document: ConfigMapping, key: str, source: Path) -> str:
    value = document[key]
    if not isinstance(value, str) or not value:
        _invalid(source, f"{key} must be a non-empty string")
    return value


def _integer_sequence(
    document: ConfigMapping,
    key: str,
    source: Path,
    *,
    positive: bool = False,
) -> tuple[int, ...]:
    value = document[key]
    if not isinstance(value, list) or not value:
        _invalid(source, f"{key} must be a non-empty list")
    raw = cast("list[object]", value)
    minimum = 1 if positive else 0
    if any(isinstance(item, bool) or not isinstance(item, int) or item < minimum for item in raw):
        _invalid(source, f"{key} must contain integers >= {minimum}")
    return cast("tuple[int, ...]", tuple(raw))


def _optional_integer_sequence(
    document: ConfigMapping,
    key: str,
    source: Path,
) -> tuple[int | None, ...]:
    value = document[key]
    if not isinstance(value, list) or not value:
        _invalid(source, f"{key} must be a non-empty list")
    raw = cast("list[object]", value)
    if any(
        item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 0)
        for item in raw
    ):
        _invalid(source, f"{key} must contain null or non-negative integers")
    return cast("tuple[int | None, ...]", tuple(raw))


def _model(document: ConfigMapping, source: Path, vocab_size: int) -> GPTConfig:
    raw = _mapping(document["model"], source, "model")
    _exact_keys(raw, _MODEL_KEYS, source, "model")
    bias = raw["bias"]
    if not isinstance(bias, bool):
        _invalid(source, "model.bias must be a boolean")
    try:
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=_integer(raw, "block_size", source, positive=True),
            n_layer=_integer(raw, "n_layer", source, positive=True),
            n_head=_integer(raw, "n_head", source, positive=True),
            n_embd=_integer(raw, "n_embd", source, positive=True),
            dropout=_number(raw, "dropout", source),
            bias=bias,
        )
    except InvalidModelConfigError as error:
        _invalid(source, str(error))


def _prefill(document: ConfigMapping, source: Path) -> PrefillBatchConfig | None:
    if "prefill" not in document:
        return None
    raw = _mapping(document["prefill"], source, "prefill")
    _exact_keys(raw, _PREFILL_KEYS, source, "prefill")
    return PrefillBatchConfig(
        max_batch_size=_integer(raw, "max_batch_size", source, positive=True),
        max_batch_tokens=_integer(raw, "max_batch_tokens", source, positive=True),
        max_padding_ratio=_number(raw, "max_padding_ratio", source),
    )


def _scenarios(
    document: ConfigMapping,
    source: Path,
    *,
    block_size: int,
) -> tuple[ServingBenchmarkScenario, ...]:
    value = document["scenarios"]
    if not isinstance(value, list) or not value:
        _invalid(source, "scenarios must be a non-empty list")
    scenarios: list[ServingBenchmarkScenario] = []
    for index, item in enumerate(cast("list[object]", value)):
        context = f"scenarios[{index}]"
        raw = _mapping(item, source, context)
        _exact_keys(raw, _SCENARIO_KEYS, source, context)
        scenario = ServingBenchmarkScenario(
            name=_string(raw, "name", source),
            arrival_ticks=_integer_sequence(raw, "arrival_ticks", source),
            prompt_lengths=_integer_sequence(raw, "prompt_lengths", source, positive=True),
            generated_lengths=_integer_sequence(raw, "generated_lengths", source, positive=True),
            cancellation_ticks=_optional_integer_sequence(raw, "cancellation_ticks", source),
        )
        lengths = {
            len(scenario.arrival_ticks),
            len(scenario.prompt_lengths),
            len(scenario.generated_lengths),
            len(scenario.cancellation_ticks),
        }
        if len(lengths) != 1:
            _invalid(source, f"{context} request field lengths must match")
        if any(length > block_size for length in scenario.prompt_lengths):
            _invalid(source, f"{context} prompt length exceeds block_size")
        if any(
            cancellation is not None and cancellation < arrival
            for arrival, cancellation in zip(
                scenario.arrival_ticks,
                scenario.cancellation_ticks,
                strict=True,
            )
        ):
            _invalid(source, f"{context} cancellation must not precede arrival")
        scenarios.append(scenario)
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        _invalid(source, "scenario names must be unique")
    return tuple(scenarios)


def load_serving_benchmark_config(path: Path) -> ServingBenchmarkConfig:
    """Load one exact schema-v1 YAML benchmark document."""
    try:
        document = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path, "document")
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        _invalid(path, str(error))
    _top_level_keys(document, path)
    schema_version = _integer(document, "schema_version", path, positive=True)
    if schema_version != 1:
        _invalid(path, "schema_version must equal 1")
    vocab_size = _integer(document, "vocab_size", path, positive=True)
    model = _model(document, path, vocab_size)
    replicates = _integer(document, "replicates", path, positive=True)
    minimum_replicates = _integer(document, "minimum_replicates", path, positive=True)
    if minimum_replicates > replicates:
        _invalid(path, "minimum_replicates must not exceed replicates")
    return ServingBenchmarkConfig(
        schema_version=schema_version,
        experiment_name=_string(document, "experiment_name", path),
        benchmark_seed=_integer(document, "benchmark_seed", path, non_negative=True),
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
        vocab_size=vocab_size,
        model=model,
        prefill=_prefill(document, path),
        scenarios=_scenarios(document, path, block_size=model.block_size),
    )


def resolved_config_document(config: ServingBenchmarkConfig) -> dict[str, JsonValue]:
    """Return stable JSON-safe resolved benchmark semantics."""
    return {
        "schema_version": config.schema_version,
        "experiment_name": config.experiment_name,
        "benchmark_seed": config.benchmark_seed,
        "output_root": config.output_root.as_posix(),
        "worker_timeout_seconds": config.worker_timeout_seconds,
        "warmup_iterations": config.warmup_iterations,
        "measurement_iterations": config.measurement_iterations,
        "replicates": config.replicates,
        "minimum_replicates": config.minimum_replicates,
        "max_cv_percent": config.max_cv_percent,
        "torch_num_threads": config.torch_num_threads,
        "torch_num_interop_threads": config.torch_num_interop_threads,
        "vocab_size": config.vocab_size,
        "model": cast("dict[str, JsonValue]", asdict(config.model)),
        "prefill": (
            None if config.prefill is None else cast("dict[str, JsonValue]", asdict(config.prefill))
        ),
        "scenarios": [
            cast("dict[str, JsonValue]", asdict(scenario)) for scenario in config.scenarios
        ],
    }


def resolved_config_sha256(config: ServingBenchmarkConfig) -> str:
    """Hash canonical resolved semantics independently of YAML formatting."""
    payload = json.dumps(
        resolved_config_document(config),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
