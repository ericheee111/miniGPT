"""Parse and validate CPU benchmark matrix configuration."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast, override

import yaml

type ConfigValue = str | int | float | bool | None | list["ConfigValue"] | dict[str, "ConfigValue"]
type ConfigMapping = dict[str, ConfigValue]


@dataclass(frozen=True, slots=True)
class InvalidBenchmarkConfigError(ValueError):
    """Report malformed benchmark YAML or an invalid matrix value."""

    source: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the invalid source and validation reason."""
        return f"invalid benchmark config {self.source}: {self.reason}"


@dataclass(frozen=True, slots=True)
class ModelSize:
    """Name one reusable Transformer model scale."""

    name: str
    n_layer: int
    n_head: int
    n_embd: int


@dataclass(frozen=True, slots=True)
class ProfileSettings:
    """Select one representative configuration for operator profiling."""

    enabled: bool
    thread_count: int
    block_size: int
    batch_size: int
    model_size: str
    warmup_steps: int
    active_steps: int


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Define the reproducible CPU benchmark matrix and sampling method."""

    seed: int
    vocab_size: int
    thread_counts: tuple[int, ...]
    block_sizes: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    model_sizes: tuple[ModelSize, ...]
    warmup_steps: int
    measurement_steps: int
    repeats: int
    output_dir: Path
    profile: ProfileSettings

    def with_matrix(
        self,
        *,
        thread_counts: tuple[int, ...],
        block_sizes: tuple[int, ...],
        batch_sizes: tuple[int, ...],
        model_size_names: tuple[str, ...],
    ) -> BenchmarkConfig:
        """Return a narrowed matrix while preserving measurement methodology."""
        selected_models = tuple(
            model for model in self.model_sizes if model.name in model_size_names
        )
        if len(selected_models) != len(model_size_names):
            raise InvalidBenchmarkConfigError(
                Path("<memory>"),
                "model_size_names contains an unknown model",
            )
        return replace(
            self,
            thread_counts=thread_counts,
            block_sizes=block_sizes,
            batch_sizes=batch_sizes,
            model_sizes=selected_models,
        )


def _integer(document: ConfigMapping, key: str, source: Path) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidBenchmarkConfigError(source, f"{key} must be an integer")
    return value


def _string(document: ConfigMapping, key: str, source: Path) -> str:
    value = document.get(key)
    if not isinstance(value, str):
        raise InvalidBenchmarkConfigError(source, f"{key} must be a string")
    return value


def _boolean(document: ConfigMapping, key: str, source: Path) -> bool:
    value = document.get(key)
    if not isinstance(value, bool):
        raise InvalidBenchmarkConfigError(source, f"{key} must be a boolean")
    return value


def _mapping(document: ConfigMapping, key: str, source: Path) -> ConfigMapping:
    value = document.get(key)
    if not isinstance(value, dict):
        raise InvalidBenchmarkConfigError(source, f"{key} must be a mapping")
    return value


def _integer_tuple(document: ConfigMapping, key: str, source: Path) -> tuple[int, ...]:
    value = document.get(key)
    if not isinstance(value, list):
        raise InvalidBenchmarkConfigError(source, f"{key} must be a list")
    parsed: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise InvalidBenchmarkConfigError(source, f"{key} items must be integers")
        if item <= 0:
            raise InvalidBenchmarkConfigError(source, f"{key} items must be positive")
        parsed.append(item)
    if not parsed:
        raise InvalidBenchmarkConfigError(source, f"{key} must not be empty")
    return tuple(parsed)


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load and validate a benchmark matrix from YAML."""
    try:
        raw_document = cast("object", yaml.safe_load(path.read_text(encoding="utf-8")))
    except yaml.YAMLError as error:
        raise InvalidBenchmarkConfigError(path, str(error)) from error
    if not isinstance(raw_document, dict):
        raise InvalidBenchmarkConfigError(path, "top-level YAML value must be a mapping")
    document = cast("ConfigMapping", raw_document)

    raw_models = _mapping(document, "model_sizes", path)
    model_sizes: list[ModelSize] = []
    for name, raw_model in raw_models.items():
        if not isinstance(raw_model, dict):
            raise InvalidBenchmarkConfigError(path, f"model_sizes.{name} must be a mapping")
        model = ModelSize(
            name=name,
            n_layer=_integer(raw_model, "n_layer", path),
            n_head=_integer(raw_model, "n_head", path),
            n_embd=_integer(raw_model, "n_embd", path),
        )
        if model.n_embd % model.n_head != 0:
            raise InvalidBenchmarkConfigError(
                path,
                f"model_sizes.{name}.n_embd must be divisible by n_head",
            )
        model_sizes.append(model)
    if not model_sizes:
        raise InvalidBenchmarkConfigError(path, "model_sizes must not be empty")

    raw_profile = _mapping(document, "profile", path)
    config = BenchmarkConfig(
        seed=_integer(document, "seed", path),
        vocab_size=_integer(document, "vocab_size", path),
        thread_counts=_integer_tuple(document, "thread_counts", path),
        block_sizes=_integer_tuple(document, "block_sizes", path),
        batch_sizes=_integer_tuple(document, "batch_sizes", path),
        model_sizes=tuple(model_sizes),
        warmup_steps=_integer(document, "warmup_steps", path),
        measurement_steps=_integer(document, "measurement_steps", path),
        repeats=_integer(document, "repeats", path),
        output_dir=Path(_string(document, "output_dir", path)),
        profile=ProfileSettings(
            enabled=_boolean(raw_profile, "enabled", path),
            thread_count=_integer(raw_profile, "thread_count", path),
            block_size=_integer(raw_profile, "block_size", path),
            batch_size=_integer(raw_profile, "batch_size", path),
            model_size=_string(raw_profile, "model_size", path),
            warmup_steps=_integer(raw_profile, "warmup_steps", path),
            active_steps=_integer(raw_profile, "active_steps", path),
        ),
    )
    positive_values = (
        config.vocab_size,
        config.measurement_steps,
        config.repeats,
        config.profile.thread_count,
        config.profile.block_size,
        config.profile.batch_size,
        config.profile.active_steps,
    )
    if any(value <= 0 for value in positive_values):
        raise InvalidBenchmarkConfigError(path, "numeric measurement settings must be positive")
    if config.warmup_steps < 0 or config.profile.warmup_steps < 0:
        raise InvalidBenchmarkConfigError(path, "warmup steps must be non-negative")
    if config.profile.model_size not in {model.name for model in config.model_sizes}:
        raise InvalidBenchmarkConfigError(path, "profile.model_size is unknown")
    return config
