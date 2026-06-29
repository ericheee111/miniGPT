from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InvalidModelConfigError(ValueError):
    """Report a model configuration that cannot define a valid GPT."""

    field: str
    reason: str

    def __str__(self) -> str:
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
        if self.vocab_size <= 0:
            raise InvalidModelConfigError("vocab_size", "must be positive")
        if self.block_size <= 0:
            raise InvalidModelConfigError("block_size", "must be positive")
        if self.n_layer <= 0:
            raise InvalidModelConfigError("n_layer", "must be positive")
        if self.n_head <= 0:
            raise InvalidModelConfigError("n_head", "must be positive")
        if self.n_embd <= 0:
            raise InvalidModelConfigError("n_embd", "must be positive")
        if self.n_embd % self.n_head != 0:
            raise InvalidModelConfigError("n_embd", "must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise InvalidModelConfigError("dropout", "must be in [0.0, 1.0)")
