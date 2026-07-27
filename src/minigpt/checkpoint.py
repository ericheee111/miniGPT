"""Persist and restore complete, reproducible training checkpoints."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias, TypedDict, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from typing_extensions import override

from minigpt.config import parse_experiment_config

if TYPE_CHECKING:
    from pathlib import Path

    from minigpt.batching import TokenBatcher
    from minigpt.settings import ExperimentConfig

_CHECKPOINT_FORMAT_VERSION: Final = 1
_MAPPING_REASON: Final = "top-level value must be a mapping"
_VERSION_REASON: Final = "unsupported format version"
_STEP_REASON: Final = "step must be an integer"
_CONFIG_REASON: Final = "config_yaml must be a string"
_NUMPY_STATE_REASON: Final = "NumPy random state is malformed"
_FIELDS_REASON: Final = "checkpoint fields have invalid types"
_PYTHON_STATE_PARTS: Final = 3
_UNEXPECTED_NUMPY_STATE: Final = "legacy NumPy random state must be a tuple"

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


class CheckpointPayload(TypedDict):
    """Describe the top-level fields persisted in a training checkpoint."""

    format_version: int
    step: int
    config_yaml: str
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, StateValue]
    python_random_state: PythonRandomState
    numpy_random_state_json: str
    torch_random_state: torch.Tensor
    train_batcher_random_state: str
    val_batcher_random_state: str


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
class ResumeState:
    """Describe where the training loop continues after restoration."""

    next_step: int


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


def _load_payload(path: Path) -> CheckpointPayload:
    loaded = cast(
        "StateValue",
        torch.load(path, map_location="cpu", weights_only=True),
    )
    if not isinstance(loaded, dict):
        raise CheckpointFormatError(path, _MAPPING_REASON)
    if loaded.get("format_version") != _CHECKPOINT_FORMAT_VERSION:
        raise CheckpointFormatError(path, _VERSION_REASON)
    step = loaded.get("step")
    config_yaml = loaded.get("config_yaml")
    numpy_state = loaded.get("numpy_random_state_json")
    torch_state = loaded.get("torch_random_state")
    train_batcher_state = loaded.get("train_batcher_random_state")
    val_batcher_state = loaded.get("val_batcher_random_state")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not isinstance(config_yaml, str)
        or not isinstance(numpy_state, str)
        or not isinstance(torch_state, Tensor)
        or not isinstance(train_batcher_state, str)
        or not isinstance(val_batcher_state, str)
    ):
        raise CheckpointFormatError(path, _FIELDS_REASON)
    return CheckpointPayload(
        format_version=_CHECKPOINT_FORMAT_VERSION,
        step=step,
        config_yaml=config_yaml,
        model_state=_validated_model_state(loaded.get("model_state"), path),
        optimizer_state=_validated_optimizer_state(loaded.get("optimizer_state"), path),
        python_random_state=_validated_python_state(loaded.get("python_random_state"), path),
        numpy_random_state_json=numpy_state,
        torch_random_state=torch_state,
        train_batcher_random_state=train_batcher_state,
        val_batcher_random_state=val_batcher_state,
    )


@dataclass(frozen=True, slots=True)
class CheckpointResources:
    """Group mutable objects whose state participates in checkpointing."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_batcher: TokenBatcher
    val_batcher: TokenBatcher


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
    payload: CheckpointPayload = {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "step": step,
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
    }
    torch.save(payload, temporary_path)
    _ = temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    *,
    resources: CheckpointResources,
) -> ResumeState:
    """Restore model, optimizer, step, global RNGs, and batch samplers."""
    payload = _load_payload(path)
    _ = resources.model.load_state_dict(payload["model_state"])
    resources.optimizer.load_state_dict(payload["optimizer_state"])
    random.setstate(payload["python_random_state"])
    _restore_numpy_random_state(payload["numpy_random_state_json"], path)
    torch.set_rng_state(payload["torch_random_state"])
    resources.train_batcher.restore_random_state(payload["train_batcher_random_state"])
    resources.val_batcher.restore_random_state(payload["val_batcher_random_state"])
    return ResumeState(next_step=payload["step"] + 1)


def load_checkpoint_config(path: Path) -> ExperimentConfig:
    """Load the resolved experiment configuration stored in a checkpoint."""
    payload = _load_payload(path)
    return parse_experiment_config(payload["config_yaml"], path)


def load_model_state(path: Path, model: nn.Module) -> None:
    """Load only model weights for inference."""
    payload = _load_payload(path)
    _ = model.load_state_dict(payload["model_state"])
