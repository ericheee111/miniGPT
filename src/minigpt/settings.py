"""Define validated, serializable settings for miniGPT experiments."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, override

import yaml

_VOCAB_SIZE_FIELD: Final = "vocab_size"
_BLOCK_SIZE_FIELD: Final = "block_size"
_N_LAYER_FIELD: Final = "n_layer"
_N_HEAD_FIELD: Final = "n_head"
_N_EMBD_FIELD: Final = "n_embd"
_DROPOUT_FIELD: Final = "dropout"
_POSITIVE_REASON: Final = "must be positive"
_DIVISIBLE_HEADS_REASON: Final = "must be divisible by n_head"
_DROPOUT_RANGE_REASON: Final = "must be in [0.0, 1.0)"
_MEMORY_SOURCE: Final = Path("<memory>")
_UNRESOLVED_VOCAB_REASON: Final = "model.vocab_size must be resolved before model construction"


@dataclass(frozen=True, slots=True)
class InvalidModelConfigError(ValueError):
    """Report a model configuration that cannot define a valid GPT."""

    field: str
    reason: str

    @override
    def __str__(self) -> str:
        """Render the invalid field and its failed constraint."""
        return f"invalid model config field {self.field!r}: {self.reason}"


@dataclass(frozen=True, slots=True)
class GPTConfig:
    """Define the tensor dimensions and regularization used by a GPT model."""

    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self) -> None:
        """Reject dimensions and probabilities that cannot define a GPT."""
        if self.vocab_size <= 0:
            raise InvalidModelConfigError(_VOCAB_SIZE_FIELD, _POSITIVE_REASON)
        if self.block_size <= 0:
            raise InvalidModelConfigError(_BLOCK_SIZE_FIELD, _POSITIVE_REASON)
        if self.n_layer <= 0:
            raise InvalidModelConfigError(_N_LAYER_FIELD, _POSITIVE_REASON)
        if self.n_head <= 0:
            raise InvalidModelConfigError(_N_HEAD_FIELD, _POSITIVE_REASON)
        if self.n_embd <= 0:
            raise InvalidModelConfigError(_N_EMBD_FIELD, _POSITIVE_REASON)
        if self.n_embd % self.n_head != 0:
            raise InvalidModelConfigError(_N_EMBD_FIELD, _DIVISIBLE_HEADS_REASON)
        if not 0.0 <= self.dropout < 1.0:
            raise InvalidModelConfigError(_DROPOUT_FIELD, _DROPOUT_RANGE_REASON)


@dataclass(frozen=True, slots=True)
class InvalidExperimentConfigError(ValueError):
    """Report malformed YAML or an invalid experiment setting."""

    source: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the source path and configuration failure."""
        return f"invalid experiment config {self.source}: {self.reason}"


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Control reproducibility and CPU execution."""

    seed: int
    num_threads: int
    device: Literal["cpu"]


@dataclass(frozen=True, slots=True)
class DataSettings:
    """Locate token arrays and define autoregressive batch dimensions."""

    directory: Path
    block_size: int
    batch_size: int

    @property
    def train_path(self) -> Path:
        """Return the training-token array path."""
        return self.directory / "train.npy"

    @property
    def val_path(self) -> Path:
        """Return the validation-token array path."""
        return self.directory / "val.npy"

    @property
    def tokenizer_path(self) -> Path:
        """Return the tokenizer metadata path."""
        return self.directory / "tokenizer.json"


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Hold model dimensions before the dataset vocabulary is resolved."""

    vocab_size: int | None
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool

    def to_gpt_config(self, block_size: int) -> GPTConfig:
        """Build a concrete GPT configuration after vocabulary resolution."""
        if self.vocab_size is None:
            raise InvalidExperimentConfigError(
                _MEMORY_SOURCE,
                _UNRESOLVED_VOCAB_REASON,
            )
        return GPTConfig(
            vocab_size=self.vocab_size,
            block_size=block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias,
        )


@dataclass(frozen=True, slots=True)
class OptimizerSettings:
    """Configure AdamW, learning-rate decay, and gradient clipping."""

    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Configure the training horizon and observable side effects."""

    max_steps: int
    warmup_steps: int
    eval_interval: int
    eval_batches: int
    log_interval: int
    checkpoint_interval: int
    sample_interval: int
    sample_tokens: int
    sample_prompt: str
    output_dir: Path
    checkpoint_dir: Path
    tensorboard_dir: Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Aggregate all settings required to reproduce one training run."""

    runtime: RuntimeSettings
    data: DataSettings
    model: ModelSettings
    optimizer: OptimizerSettings
    training: TrainingSettings

    def resolve_vocab_size(self, actual_vocab_size: int) -> ExperimentConfig:
        """Bind or verify the dataset vocabulary size."""
        configured = self.model.vocab_size
        if configured is not None and configured != actual_vocab_size:
            reason = f"model vocab_size {configured} does not match tokenizer {actual_vocab_size}"
            raise InvalidExperimentConfigError(
                _MEMORY_SOURCE,
                reason,
            )
        return replace(
            self,
            model=replace(self.model, vocab_size=actual_vocab_size),
        )

    def to_yaml(self) -> str:
        """Serialize primitive settings for checkpoint provenance."""
        document = {
            "runtime": {
                "seed": self.runtime.seed,
                "num_threads": self.runtime.num_threads,
                "device": self.runtime.device,
            },
            "data": {
                "directory": str(self.data.directory),
                "block_size": self.data.block_size,
                "batch_size": self.data.batch_size,
            },
            "model": {
                "vocab_size": self.model.vocab_size,
                "n_layer": self.model.n_layer,
                "n_head": self.model.n_head,
                "n_embd": self.model.n_embd,
                "dropout": self.model.dropout,
                "bias": self.model.bias,
            },
            "optimizer": {
                "learning_rate": self.optimizer.learning_rate,
                "min_learning_rate": self.optimizer.min_learning_rate,
                "weight_decay": self.optimizer.weight_decay,
                "beta1": self.optimizer.beta1,
                "beta2": self.optimizer.beta2,
                "grad_clip": self.optimizer.grad_clip,
            },
            "training": {
                "max_steps": self.training.max_steps,
                "warmup_steps": self.training.warmup_steps,
                "eval_interval": self.training.eval_interval,
                "eval_batches": self.training.eval_batches,
                "log_interval": self.training.log_interval,
                "checkpoint_interval": self.training.checkpoint_interval,
                "sample_interval": self.training.sample_interval,
                "sample_tokens": self.training.sample_tokens,
                "sample_prompt": self.training.sample_prompt,
                "output_dir": str(self.training.output_dir),
                "checkpoint_dir": str(self.training.checkpoint_dir),
                "tensorboard_dir": str(self.training.tensorboard_dir),
            },
        }
        return yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
