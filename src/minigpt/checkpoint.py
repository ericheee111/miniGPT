"""Persist and restore complete, reproducible training checkpoints."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypedDict, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from typing_extensions import override

from minigpt.config import parse_experiment_config, parse_legacy_experiment_config

if TYPE_CHECKING:
    from pathlib import Path

    from minigpt.batching import TokenBatcher
    from minigpt.settings import DataSettings, ExperimentConfig

_LEGACY_CHECKPOINT_FORMAT_VERSION: Final = 1
_CHECKPOINT_FORMAT_VERSION: Final = 2
_MAPPING_REASON: Final = "top-level value must be a mapping"
_VERSION_REASON: Final = "unsupported format version"
_STEP_REASON: Final = "step must be an integer"
_CONFIG_REASON: Final = "config_yaml must be a string"
_NUMPY_STATE_REASON: Final = "NumPy random state is malformed"
_FIELDS_REASON: Final = "checkpoint fields have invalid types"
_PYTHON_STATE_PARTS: Final = 3
_UNEXPECTED_NUMPY_STATE: Final = "legacy NumPy random state must be a tuple"
_HASH_CHUNK_BYTES: Final = 1024 * 1024

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None
StateValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | Tensor
    | list["StateValue"]
    | tuple["StateValue", ...]
    | dict[str | int, "StateValue"]
    | None
)
PythonRandomState: TypeAlias = tuple[int, tuple[int, ...], float | None]
NumpyRandomState: TypeAlias = tuple[str, npt.NDArray[np.uint32], int, int, float]
ConfigValue: TypeAlias = str | int | float | bool | None


class DatasetFingerprintsPayload(TypedDict):
    """Describe persisted dataset SHA-256 values."""

    tokenizer_sha256: str
    train_sha256: str
    val_sha256: str


class CheckpointV1Payload(TypedDict):
    """Describe legacy fields retained for inference compatibility."""

    format_version: Literal[1]
    step: int
    config_yaml: str
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, StateValue]
    python_random_state: PythonRandomState
    numpy_random_state_json: str
    torch_random_state: torch.Tensor
    train_batcher_random_state: str
    val_batcher_random_state: str


class CheckpointV2Payload(TypedDict):
    """Describe the complete state persisted for exact training resume."""

    format_version: Literal[2]
    completed_step: int
    config_yaml: str
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, StateValue]
    python_random_state: PythonRandomState
    numpy_random_state_json: str
    torch_random_state: Tensor
    train_batcher_random_state: str
    val_batcher_random_state: str
    sample_generator_random_state: Tensor
    dataset_fingerprints: DatasetFingerprintsPayload


VersionedCheckpointPayload: TypeAlias = CheckpointV1Payload | CheckpointV2Payload


@dataclass(frozen=True, slots=True)
class CheckpointFormatError(ValueError):
    """Report a checkpoint missing a required compatible field."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the checkpoint path and format failure."""
        return f"invalid checkpoint {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class LegacyCheckpointResumeError(RuntimeError):
    """Reject a v1 checkpoint that lacks exact-resume state."""

    path: Path

    @override
    def __str__(self) -> str:
        """Explain that legacy checkpoints are inference-only."""
        return f"checkpoint v1 at {self.path} cannot be used for training resume"


@dataclass(frozen=True, slots=True)
class ResumeConfigMismatch:
    """Describe one trajectory-defining value that differs at resume."""

    field: str
    checkpoint_value: str
    current_value: str


@dataclass(slots=True)
class IncompatibleResumeConfigError(ValueError):
    """Report all configuration and data-identity mismatches found before resume."""

    mismatches: tuple[ResumeConfigMismatch, ...]

    @override
    def __str__(self) -> str:
        """Render every incompatible dotted field and its two values."""
        details = "; ".join(
            (
                f"{mismatch.field}: checkpoint={mismatch.checkpoint_value}, "
                f"current={mismatch.current_value}"
            )
            for mismatch in self.mismatches
        )
        return f"incompatible resume configuration: {details}"


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Describe where the training loop continues after restoration."""

    next_step: int


@dataclass(frozen=True, slots=True)
class DatasetFingerprints:
    """Identify the persisted tokenizer, training data, and validation data."""

    tokenizer_sha256: str
    train_sha256: str
    val_sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Expose immutable v2 identity without restoring mutable training state."""

    format_version: int
    completed_step: int
    config: ExperimentConfig
    dataset_fingerprints: DatasetFingerprints


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def compute_dataset_fingerprints(config: DataSettings) -> DatasetFingerprints:
    """Hash the exact persisted artifacts that define token meaning and batches."""
    return DatasetFingerprints(
        tokenizer_sha256=_sha256_file(config.tokenizer_path),
        train_sha256=_sha256_file(config.train_path),
        val_sha256=_sha256_file(config.val_path),
    )


def _numpy_random_state_json() -> str:
    state = np.random.get_state(legacy=True)  # noqa: NPY002
    if isinstance(state, dict):
        raise TypeError(_UNEXPECTED_NUMPY_STATE)
    algorithm, keys, position, has_gauss, cached_gaussian = state
    serialized_keys: list[JsonValue] = [
        int(cast("np.uint32", keys[index])) for index in range(keys.size)
    ]
    document: dict[str, JsonValue] = {
        "algorithm": algorithm,
        "keys": serialized_keys,
        "position": position,
        "has_gauss": has_gauss,
        "cached_gaussian": cached_gaussian,
    }
    return json.dumps(document)


def _restore_numpy_random_state(state_json: str, path: Path) -> None:
    document = cast("JsonValue", json.loads(state_json))
    if not isinstance(document, dict):
        raise CheckpointFormatError(path, _NUMPY_STATE_REASON)
    algorithm = document.get("algorithm")
    raw_keys = document.get("keys")
    position = document.get("position")
    has_gauss = document.get("has_gauss")
    cached_gaussian = document.get("cached_gaussian")
    if (
        not isinstance(algorithm, str)
        or not isinstance(raw_keys, list)
        or isinstance(position, bool)
        or not isinstance(position, int)
        or isinstance(has_gauss, bool)
        or not isinstance(has_gauss, int)
        or isinstance(cached_gaussian, bool)
        or not isinstance(cached_gaussian, int | float)
    ):
        raise CheckpointFormatError(path, _NUMPY_STATE_REASON)
    keys: list[int] = []
    for value in raw_keys:
        if isinstance(value, bool) or not isinstance(value, int):
            raise CheckpointFormatError(path, _NUMPY_STATE_REASON)
        keys.append(value)
    np.random.set_state(  # noqa: NPY002
        (
            algorithm,
            np.asarray(keys, dtype=np.uint32),
            position,
            has_gauss,
            float(cached_gaussian),
        )
    )


def _validated_model_state(value: StateValue, path: Path) -> dict[str, Tensor]:
    if not isinstance(value, dict):
        raise CheckpointFormatError(path, _FIELDS_REASON)
    model_state: dict[str, Tensor] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, Tensor):
            raise CheckpointFormatError(path, _FIELDS_REASON)
        model_state[key] = item
    return model_state


def _validated_optimizer_state(value: StateValue, path: Path) -> dict[str, StateValue]:
    if not isinstance(value, dict):
        raise CheckpointFormatError(path, _FIELDS_REASON)
    optimizer_state: dict[str, StateValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise CheckpointFormatError(path, _FIELDS_REASON)
        optimizer_state[key] = item
    return optimizer_state


def _validated_python_state(value: StateValue, path: Path) -> PythonRandomState:
    if not isinstance(value, tuple) or len(value) != _PYTHON_STATE_PARTS:
        raise CheckpointFormatError(path, _FIELDS_REASON)
    version, raw_internal_state, raw_gauss = value
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or not isinstance(raw_internal_state, tuple)
        or (
            raw_gauss is not None
            and (isinstance(raw_gauss, bool) or not isinstance(raw_gauss, int | float))
        )
    ):
        raise CheckpointFormatError(path, _FIELDS_REASON)
    internal_state: list[int] = []
    for item in raw_internal_state:
        if isinstance(item, bool) or not isinstance(item, int):
            raise CheckpointFormatError(path, _FIELDS_REASON)
        internal_state.append(item)
    return (
        version,
        tuple(internal_state),
        None if raw_gauss is None else float(raw_gauss),
    )


@dataclass(frozen=True, slots=True)
class _CommonCheckpointFields:
    config_yaml: str
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, StateValue]
    python_random_state: PythonRandomState
    numpy_random_state_json: str
    torch_random_state: Tensor
    train_batcher_random_state: str
    val_batcher_random_state: str


def _validated_common_fields(
    loaded: dict[str | int, StateValue],
    path: Path,
) -> _CommonCheckpointFields:
    config_yaml = loaded.get("config_yaml")
    numpy_state = loaded.get("numpy_random_state_json")
    torch_state = loaded.get("torch_random_state")
    train_batcher_state = loaded.get("train_batcher_random_state")
    val_batcher_state = loaded.get("val_batcher_random_state")
    if (
        not isinstance(config_yaml, str)
        or not isinstance(numpy_state, str)
        or not isinstance(torch_state, Tensor)
        or not isinstance(train_batcher_state, str)
        or not isinstance(val_batcher_state, str)
    ):
        raise CheckpointFormatError(path, _FIELDS_REASON)
    return _CommonCheckpointFields(
        config_yaml=config_yaml,
        model_state=_validated_model_state(loaded.get("model_state"), path),
        optimizer_state=_validated_optimizer_state(loaded.get("optimizer_state"), path),
        python_random_state=_validated_python_state(loaded.get("python_random_state"), path),
        numpy_random_state_json=numpy_state,
        torch_random_state=torch_state,
        train_batcher_random_state=train_batcher_state,
        val_batcher_random_state=val_batcher_state,
    )


def _validated_dataset_fingerprints(
    value: StateValue,
    path: Path,
) -> DatasetFingerprintsPayload:
    if not isinstance(value, dict):
        raise CheckpointFormatError(path, _FIELDS_REASON)
    tokenizer_sha256 = value.get("tokenizer_sha256")
    train_sha256 = value.get("train_sha256")
    val_sha256 = value.get("val_sha256")
    if (
        not isinstance(tokenizer_sha256, str)
        or not isinstance(train_sha256, str)
        or not isinstance(val_sha256, str)
    ):
        raise CheckpointFormatError(path, _FIELDS_REASON)
    return DatasetFingerprintsPayload(
        tokenizer_sha256=tokenizer_sha256,
        train_sha256=train_sha256,
        val_sha256=val_sha256,
    )


def _load_v1_payload(
    loaded: dict[str | int, StateValue],
    path: Path,
) -> CheckpointV1Payload:
    step = loaded.get("step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise CheckpointFormatError(path, _STEP_REASON)
    common = _validated_common_fields(loaded, path)
    return CheckpointV1Payload(
        format_version=1,
        step=step,
        config_yaml=common.config_yaml,
        model_state=common.model_state,
        optimizer_state=common.optimizer_state,
        python_random_state=common.python_random_state,
        numpy_random_state_json=common.numpy_random_state_json,
        torch_random_state=common.torch_random_state,
        train_batcher_random_state=common.train_batcher_random_state,
        val_batcher_random_state=common.val_batcher_random_state,
    )


def _load_v2_payload(
    loaded: dict[str | int, StateValue],
    path: Path,
) -> CheckpointV2Payload:
    completed_step = loaded.get("completed_step")
    sample_generator_state = loaded.get("sample_generator_random_state")
    if (
        isinstance(completed_step, bool)
        or not isinstance(completed_step, int)
        or not isinstance(sample_generator_state, Tensor)
    ):
        raise CheckpointFormatError(path, _FIELDS_REASON)
    common = _validated_common_fields(loaded, path)
    return CheckpointV2Payload(
        format_version=2,
        completed_step=completed_step,
        config_yaml=common.config_yaml,
        model_state=common.model_state,
        optimizer_state=common.optimizer_state,
        python_random_state=common.python_random_state,
        numpy_random_state_json=common.numpy_random_state_json,
        torch_random_state=common.torch_random_state,
        train_batcher_random_state=common.train_batcher_random_state,
        val_batcher_random_state=common.val_batcher_random_state,
        sample_generator_random_state=sample_generator_state,
        dataset_fingerprints=_validated_dataset_fingerprints(
            loaded.get("dataset_fingerprints"),
            path,
        ),
    )


def _load_versioned_payload(path: Path) -> VersionedCheckpointPayload:
    loaded = cast(
        "StateValue",
        torch.load(path, map_location="cpu", weights_only=True),
    )
    if not isinstance(loaded, dict):
        raise CheckpointFormatError(path, _MAPPING_REASON)
    version = loaded.get("format_version")
    if version == _LEGACY_CHECKPOINT_FORMAT_VERSION:
        return _load_v1_payload(loaded, path)
    if version == _CHECKPOINT_FORMAT_VERSION:
        return _load_v2_payload(loaded, path)
    raise CheckpointFormatError(path, _VERSION_REASON)


@dataclass(frozen=True, slots=True)
class CheckpointResources:
    """Group mutable objects whose state participates in checkpointing."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_batcher: TokenBatcher
    val_batcher: TokenBatcher
    sample_generator: torch.Generator
    dataset_fingerprints: DatasetFingerprints


def _immutable_config_values(
    config: ExperimentConfig,
) -> tuple[tuple[str, ConfigValue], ...]:
    return (
        ("runtime.seed", config.runtime.seed),
        ("runtime.num_threads", config.runtime.num_threads),
        ("runtime.device", config.runtime.device),
        ("data.block_size", config.data.block_size),
        ("data.batch_size", config.data.batch_size),
        ("model.vocab_size", config.model.vocab_size),
        ("model.n_layer", config.model.n_layer),
        ("model.n_head", config.model.n_head),
        ("model.n_embd", config.model.n_embd),
        ("model.dropout", config.model.dropout),
        ("model.bias", config.model.bias),
        ("optimizer.type", config.optimizer.optimizer_type),
        ("optimizer.learning_rate", config.optimizer.learning_rate),
        ("optimizer.min_learning_rate", config.optimizer.min_learning_rate),
        ("optimizer.weight_decay", config.optimizer.weight_decay),
        ("optimizer.beta1", config.optimizer.beta1),
        ("optimizer.beta2", config.optimizer.beta2),
        ("optimizer.grad_clip", config.optimizer.grad_clip),
        ("training.max_steps", config.training.max_steps),
        ("training.warmup_steps", config.training.warmup_steps),
        ("training.lr_decay_steps", config.training.lr_decay_steps),
        ("training.eval_interval", config.training.eval_interval),
        ("training.eval_batches", config.training.eval_batches),
    )


def _fingerprint_values(
    fingerprints: DatasetFingerprints | DatasetFingerprintsPayload,
) -> tuple[tuple[str, str], ...]:
    if isinstance(fingerprints, DatasetFingerprints):
        return (
            ("dataset.tokenizer_sha256", fingerprints.tokenizer_sha256),
            ("dataset.train_sha256", fingerprints.train_sha256),
            ("dataset.val_sha256", fingerprints.val_sha256),
        )
    return (
        ("dataset.tokenizer_sha256", fingerprints["tokenizer_sha256"]),
        ("dataset.train_sha256", fingerprints["train_sha256"]),
        ("dataset.val_sha256", fingerprints["val_sha256"]),
    )


def _validate_resume_compatibility(
    payload: CheckpointV2Payload,
    current_config: ExperimentConfig,
    current_fingerprints: DatasetFingerprints,
    path: Path,
) -> None:
    checkpoint_config = parse_experiment_config(payload["config_yaml"], path)
    mismatches: list[ResumeConfigMismatch] = []
    current_config_values = dict(_immutable_config_values(current_config))
    for field, checkpoint_value in _immutable_config_values(checkpoint_config):
        current_value = current_config_values[field]
        if checkpoint_value != current_value:
            mismatches.append(
                ResumeConfigMismatch(
                    field=field,
                    checkpoint_value=str(checkpoint_value),
                    current_value=str(current_value),
                )
            )
    current_fingerprint_values = dict(_fingerprint_values(current_fingerprints))
    for field, checkpoint_value in _fingerprint_values(payload["dataset_fingerprints"]):
        current_value = current_fingerprint_values[field]
        if checkpoint_value != current_value:
            mismatches.append(
                ResumeConfigMismatch(
                    field=field,
                    checkpoint_value=checkpoint_value,
                    current_value=current_value,
                )
            )
    if mismatches:
        raise IncompatibleResumeConfigError(tuple(mismatches))


def save_checkpoint(
    path: Path,
    *,
    resources: CheckpointResources,
    step: int,
    config: ExperimentConfig,
) -> None:
    """Atomically persist training and random state needed for exact resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload: CheckpointV2Payload = {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "completed_step": step,
        "config_yaml": config.to_yaml(),
        "model_state": resources.model.state_dict(),
        "optimizer_state": cast(
            "dict[str, StateValue]",
            resources.optimizer.state_dict(),
        ),
        "python_random_state": cast("PythonRandomState", random.getstate()),
        "numpy_random_state_json": _numpy_random_state_json(),
        "torch_random_state": torch.get_rng_state(),
        "train_batcher_random_state": resources.train_batcher.capture_random_state(),
        "val_batcher_random_state": resources.val_batcher.capture_random_state(),
        "sample_generator_random_state": resources.sample_generator.get_state(),
        "dataset_fingerprints": DatasetFingerprintsPayload(
            tokenizer_sha256=resources.dataset_fingerprints.tokenizer_sha256,
            train_sha256=resources.dataset_fingerprints.train_sha256,
            val_sha256=resources.dataset_fingerprints.val_sha256,
        ),
    }
    torch.save(payload, temporary_path)
    _ = temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    *,
    resources: CheckpointResources,
    config: ExperimentConfig,
) -> ResumeState:
    """Restore model, optimizer, step, global RNGs, and batch samplers."""
    payload = _load_versioned_payload(path)
    if payload["format_version"] == _LEGACY_CHECKPOINT_FORMAT_VERSION:
        raise LegacyCheckpointResumeError(path)
    _validate_resume_compatibility(
        payload,
        config,
        resources.dataset_fingerprints,
        path,
    )
    _ = resources.model.load_state_dict(payload["model_state"])
    resources.optimizer.load_state_dict(payload["optimizer_state"])
    random.setstate(payload["python_random_state"])
    _restore_numpy_random_state(payload["numpy_random_state_json"], path)
    torch.set_rng_state(payload["torch_random_state"])
    resources.train_batcher.restore_random_state(payload["train_batcher_random_state"])
    resources.val_batcher.restore_random_state(payload["val_batcher_random_state"])
    _ = resources.sample_generator.set_state(payload["sample_generator_random_state"])
    return ResumeState(next_step=payload["completed_step"] + 1)


def load_checkpoint_config(path: Path) -> ExperimentConfig:
    """Load the resolved experiment configuration stored in a checkpoint."""
    payload = _load_versioned_payload(path)
    if payload["format_version"] == _LEGACY_CHECKPOINT_FORMAT_VERSION:
        return parse_legacy_experiment_config(payload["config_yaml"], path)
    return parse_experiment_config(payload["config_yaml"], path)


def load_checkpoint_metadata(path: Path) -> CheckpointMetadata:
    """Load validated v2 identity and completion metadata without restoring state."""
    payload = _load_versioned_payload(path)
    if payload["format_version"] == _LEGACY_CHECKPOINT_FORMAT_VERSION:
        raise LegacyCheckpointResumeError(path)
    fingerprints = payload["dataset_fingerprints"]
    return CheckpointMetadata(
        format_version=payload["format_version"],
        completed_step=payload["completed_step"],
        config=parse_experiment_config(payload["config_yaml"], path),
        dataset_fingerprints=DatasetFingerprints(
            tokenizer_sha256=fingerprints["tokenizer_sha256"],
            train_sha256=fingerprints["train_sha256"],
            val_sha256=fingerprints["val_sha256"],
        ),
    )


def load_model_state(path: Path, model: nn.Module) -> None:
    """Load only model weights for inference."""
    payload = _load_versioned_payload(path)
    _ = model.load_state_dict(payload["model_state"])
