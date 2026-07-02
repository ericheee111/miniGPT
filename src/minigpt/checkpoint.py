from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import TypedDict

import numpy as np
import torch
from torch import nn

from minigpt.batching import TokenBatcher
from minigpt.config import ExperimentConfig, parse_experiment_config


class CheckpointPayload(TypedDict):
    """Describe the top-level fields persisted in a training checkpoint."""

    format_version: int
    step: int
    config_yaml: str
    model_state: dict
    optimizer_state: dict
    python_random_state: tuple
    numpy_random_state_json: str
    torch_random_state: torch.Tensor
    train_batcher_random_state: str
    val_batcher_random_state: str


@dataclass(frozen=True, slots=True)
class CheckpointFormatError(ValueError):
    """Report a checkpoint missing a required compatible field."""

    path: Path
    reason: str

    def __str__(self) -> str:
        return f"invalid checkpoint {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Describe where the training loop continues after restoration."""

    next_step: int


def _numpy_random_state_json() -> str:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    document = {
        "algorithm": algorithm,
        "keys": keys.tolist(),
        "position": position,
        "has_gauss": has_gauss,
        "cached_gaussian": cached_gaussian,
    }
    return json.dumps(document)


def _restore_numpy_random_state(state_json: str) -> None:
    document = json.loads(state_json)
    np.random.set_state(
        (
            document["algorithm"],
            np.asarray(document["keys"], dtype=np.uint32),
            document["position"],
            document["has_gauss"],
            document["cached_gaussian"],
        )
    )


def _load_payload(path: Path) -> CheckpointPayload:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise CheckpointFormatError(path, "top-level value must be a mapping")
    if payload.get("format_version") != 1:
        raise CheckpointFormatError(path, "unsupported format version")
    return payload


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: ExperimentConfig,
    train_batcher: TokenBatcher,
    val_batcher: TokenBatcher,
) -> None:
    """Atomically persist training and random state needed for exact resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "step": step,
        "config_yaml": config.to_yaml(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state_json": _numpy_random_state_json(),
        "torch_random_state": torch.get_rng_state(),
        "train_batcher_random_state": train_batcher.capture_random_state(),
        "val_batcher_random_state": val_batcher.capture_random_state(),
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_batcher: TokenBatcher,
    val_batcher: TokenBatcher,
) -> ResumeState:
    """Restore model, optimizer, step, global RNGs, and batch samplers."""
    payload = _load_payload(path)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    random.setstate(payload["python_random_state"])
    _restore_numpy_random_state(payload["numpy_random_state_json"])
    torch.set_rng_state(payload["torch_random_state"])
    train_batcher.restore_random_state(payload["train_batcher_random_state"])
    val_batcher.restore_random_state(payload["val_batcher_random_state"])
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise CheckpointFormatError(path, "step must be an integer")
    return ResumeState(next_step=step + 1)


def load_checkpoint_config(path: Path) -> ExperimentConfig:
    """Load the resolved experiment configuration stored in a checkpoint."""
    payload = _load_payload(path)
    config_yaml = payload.get("config_yaml")
    if not isinstance(config_yaml, str):
        raise CheckpointFormatError(path, "config_yaml must be a string")
    return parse_experiment_config(config_yaml, path)


def load_model_state(path: Path, model: nn.Module) -> None:
    """Load only model weights for inference."""
    payload = _load_payload(path)
    model.load_state_dict(payload["model_state"])
