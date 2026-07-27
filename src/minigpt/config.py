"""Load and validate miniGPT experiment configuration files."""

from pathlib import Path
from typing import Final, TypeAlias, cast

import yaml

from minigpt.settings import (
    DataSettings,
    ExperimentConfig,
    InvalidExperimentConfigError,
    ModelSettings,
    OptimizerSettings,
    RuntimeSettings,
    TrainingSettings,
)

ConfigValue: TypeAlias = (
    str | int | float | bool | list["ConfigValue"] | dict[str, "ConfigValue"] | None
)
ConfigMapping: TypeAlias = dict[str, ConfigValue]

_MEMORY_SOURCE: Final = Path("<memory>")


def _section(document: ConfigMapping, name: str, source: Path) -> ConfigMapping:
    value = document.get(name)
    if not isinstance(value, dict):
        raise InvalidExperimentConfigError(source, f"{name} must be a mapping")
    return value


def _integer(section: ConfigMapping, key: str, source: Path) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidExperimentConfigError(source, f"{key} must be an integer")
    return value


def _number(section: ConfigMapping, key: str, source: Path) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidExperimentConfigError(source, f"{key} must be numeric")
    return float(value)


def _boolean(section: ConfigMapping, key: str, source: Path) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise InvalidExperimentConfigError(source, f"{key} must be a boolean")
    return value


def _string(section: ConfigMapping, key: str, source: Path) -> str:
    value = section.get(key)
    if not isinstance(value, str):
        raise InvalidExperimentConfigError(source, f"{key} must be a string")
    return value


def _positive(value: float, name: str, source: Path) -> None:
    if value <= 0:
        raise InvalidExperimentConfigError(source, f"{name} must be positive")


def parse_experiment_config(
    yaml_text: str,
    source: Path = _MEMORY_SOURCE,
) -> ExperimentConfig:
    """Parse and validate an experiment configuration from YAML text."""
    return _parse_experiment_config(yaml_text, source, legacy_defaults=False)


def parse_legacy_experiment_config(
    yaml_text: str,
    source: Path,
) -> ExperimentConfig:
    """Parse a v1 inference config with deterministic compatibility defaults."""
    return _parse_experiment_config(yaml_text, source, legacy_defaults=True)


def _parse_experiment_config(
    yaml_text: str,
    source: Path,
    *,
    legacy_defaults: bool,
) -> ExperimentConfig:
    try:
        raw_document = yaml.safe_load(yaml_text)
    except yaml.YAMLError as error:
        raise InvalidExperimentConfigError(source, str(error)) from error
    if not isinstance(raw_document, dict):
        raise InvalidExperimentConfigError(source, "top-level YAML value must be a mapping")
    document = cast("ConfigMapping", raw_document)

    runtime = _section(document, "runtime", source)
    data = _section(document, "data", source)
    model = _section(document, "model", source)
    optimizer = _section(document, "optimizer", source)
    training = _section(document, "training", source)
    raw_optimizer_type = optimizer.get("type")
    if legacy_defaults and raw_optimizer_type is None:
        optimizer_type = "adamw"
    else:
        optimizer_type = _string(optimizer, "type", source)
    if optimizer_type != "adamw":
        raise InvalidExperimentConfigError(source, "optimizer.type must be adamw")
    device = _string(runtime, "device", source)
    if device != "cpu":
        raise InvalidExperimentConfigError(source, "runtime.device must be cpu")
    raw_vocab_size = model.get("vocab_size")
    if raw_vocab_size is not None and (
        isinstance(raw_vocab_size, bool) or not isinstance(raw_vocab_size, int)
    ):
        raise InvalidExperimentConfigError(source, "model.vocab_size must be an integer or null")

    max_steps = _integer(training, "max_steps", source)
    warmup_steps = _integer(training, "warmup_steps", source)
    raw_lr_decay_steps = training.get("lr_decay_steps")
    if legacy_defaults and raw_lr_decay_steps is None:
        lr_decay_steps = max_steps
    else:
        lr_decay_steps = _integer(training, "lr_decay_steps", source)

    experiment = ExperimentConfig(
        runtime=RuntimeSettings(
            seed=_integer(runtime, "seed", source),
            num_threads=_integer(runtime, "num_threads", source),
            device="cpu",
        ),
        data=DataSettings(
            directory=Path(_string(data, "directory", source)),
            block_size=_integer(data, "block_size", source),
            batch_size=_integer(data, "batch_size", source),
        ),
        model=ModelSettings(
            vocab_size=raw_vocab_size,
            n_layer=_integer(model, "n_layer", source),
            n_head=_integer(model, "n_head", source),
            n_embd=_integer(model, "n_embd", source),
            dropout=_number(model, "dropout", source),
            bias=_boolean(model, "bias", source),
        ),
        optimizer=OptimizerSettings(
            optimizer_type="adamw",
            learning_rate=_number(optimizer, "learning_rate", source),
            min_learning_rate=_number(optimizer, "min_learning_rate", source),
            weight_decay=_number(optimizer, "weight_decay", source),
            beta1=_number(optimizer, "beta1", source),
            beta2=_number(optimizer, "beta2", source),
            grad_clip=_number(optimizer, "grad_clip", source),
        ),
        training=TrainingSettings(
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            lr_decay_steps=lr_decay_steps,
            eval_interval=_integer(training, "eval_interval", source),
            eval_batches=_integer(training, "eval_batches", source),
            log_interval=_integer(training, "log_interval", source),
            checkpoint_interval=_integer(training, "checkpoint_interval", source),
            sample_interval=_integer(training, "sample_interval", source),
            sample_tokens=_integer(training, "sample_tokens", source),
            sample_prompt=_string(training, "sample_prompt", source),
            output_dir=Path(_string(training, "output_dir", source)),
            checkpoint_dir=Path(_string(training, "checkpoint_dir", source)),
            tensorboard_dir=Path(_string(training, "tensorboard_dir", source)),
        ),
    )
    _positive(experiment.runtime.num_threads, "runtime.num_threads", source)
    _positive(experiment.data.block_size, "data.block_size", source)
    _positive(experiment.data.batch_size, "data.batch_size", source)
    _positive(experiment.training.max_steps, "training.max_steps", source)
    _positive(experiment.training.eval_interval, "training.eval_interval", source)
    _positive(experiment.training.eval_batches, "training.eval_batches", source)
    _positive(experiment.training.log_interval, "training.log_interval", source)
    _positive(experiment.training.checkpoint_interval, "training.checkpoint_interval", source)
    _positive(experiment.training.sample_interval, "training.sample_interval", source)
    _positive(experiment.training.sample_tokens, "training.sample_tokens", source)
    _positive(experiment.optimizer.learning_rate, "optimizer.learning_rate", source)
    _positive(experiment.optimizer.grad_clip, "optimizer.grad_clip", source)
    if not (
        0
        <= experiment.training.warmup_steps
        < experiment.training.lr_decay_steps
        <= experiment.training.max_steps
    ):
        reason = "training steps must satisfy 0 <= warmup_steps < lr_decay_steps <= max_steps"
        raise InvalidExperimentConfigError(source, reason)
    return experiment


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load an experiment configuration from a UTF-8 YAML file."""
    return parse_experiment_config(path.read_text(encoding="utf-8"), path)
