"""Define the miniGPT language model and its validation errors."""

from dataclasses import dataclass
from typing import Final, Never, cast, final

import torch
from torch import Tensor, nn
from torch.nn import functional
from typing_extensions import override

from minigpt.layers import (
    MLP,
    CausalSelfAttention,
    KVCache,
    LayerKVCache,
    LayerNorm,
    TransformerBlock,
    kv_cache_nbytes,
)
from minigpt.settings import GPTConfig, InvalidModelConfigError

__all__ = (
    "GPT",
    "MLP",
    "CausalSelfAttention",
    "GPTConfig",
    "InvalidGenerationConfigError",
    "InvalidKVCacheError",
    "InvalidModelConfigError",
    "InvalidTokenTensorError",
    "KVCache",
    "LayerKVCache",
    "LayerNorm",
    "TokenIdOutOfRangeError",
    "TransformerBlock",
    "UnexpectedTransformerBlockError",
    "expected_gpt_parameter_count",
    "kv_cache_nbytes",
)

_INPUT_NAME: Final = "input"
_TARGET_NAME: Final = "target"
_PROMPT_NAME: Final = "prompt"
_SHAPE_REASON: Final = "expected shape [batch, time]"
_DTYPE_REASON: Final = "dtype must be torch.int64"
_NON_EMPTY_REASON: Final = "batch and time dimensions must be non-zero"
_GENERATION_LENGTH_REASON: Final = "max_new_tokens must be non-negative"
_TEMPERATURE_REASON: Final = "temperature must be positive"
_TOP_K_REASON: Final = "top_k must be positive when provided"
_TOKEN_TENSOR_DIMENSIONS: Final = 2
_CACHE_TENSOR_DIMENSIONS: Final = 4
_CACHE_LENGTH_DIMENSIONS: Final = 1


def expected_gpt_parameter_count(config: GPTConfig) -> int:
    """Return the exact trainable GPT parameter count without allocating a model."""
    embedding_dim = config.n_embd
    bias_parameters_per_block = 11 * embedding_dim if config.bias else 0
    block_parameters = 12 * embedding_dim**2 + 2 * embedding_dim + bias_parameters_per_block
    final_norm_parameters = embedding_dim * (2 if config.bias else 1)
    embedding_and_head_parameters = (
        2 * config.vocab_size * embedding_dim + config.block_size * embedding_dim
    )
    return embedding_and_head_parameters + config.n_layer * block_parameters + final_norm_parameters


@dataclass(frozen=True, slots=True)
class InvalidTokenTensorError(ValueError):
    """Report an input or target tensor with an invalid shape or dtype."""

    name: str
    reason: str

    @override
    def __str__(self) -> str:
        """Render the tensor name and its failed constraint."""
        return f"invalid {self.name} tensor: {self.reason}"


@dataclass(frozen=True, slots=True)
class TokenIdOutOfRangeError(ValueError):
    """Report token IDs that cannot index the configured vocabulary."""

    name: str
    minimum: int
    maximum: int
    vocab_size: int

    @override
    def __str__(self) -> str:
        """Render the observed token range and configured vocabulary."""
        return (
            f"{self.name} token IDs span [{self.minimum}, {self.maximum}], "
            f"but vocabulary size {self.vocab_size} accepts [0, {self.vocab_size})"
        )


@dataclass(frozen=True, slots=True)
class InvalidGenerationConfigError(ValueError):
    """Report sampling settings that do not define a valid distribution."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed generation constraint."""
        return f"invalid generation configuration: {self.reason}"


@dataclass(frozen=True, slots=True)
class InvalidKVCacheError(ValueError):
    """Report a caller-owned KV cache that cannot be used for decode."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed cache invariant."""
        return f"invalid KV cache: {self.reason}"


def _invalid_cache(reason: str) -> Never:
    raise InvalidKVCacheError(reason)


def _validate_cache_tensor(  # noqa: PLR0913
    tensor: Tensor,
    *,
    prefix: str,
    batch_size: int,
    n_head: int,
    head_size: int,
    dtype: torch.dtype,
    model_device: torch.device,
    input_device: torch.device,
) -> None:
    if tensor.ndim != _CACHE_TENSOR_DIMENSIONS:
        _invalid_cache(f"{prefix} must have rank {_CACHE_TENSOR_DIMENSIONS}")
    if tensor.shape[0] != batch_size:
        _invalid_cache(f"{prefix} batch {tensor.shape[0]} must equal input batch {batch_size}")
    if tensor.shape[1] != n_head:
        _invalid_cache(f"{prefix} head count {tensor.shape[1]} must equal n_head {n_head}")
    if tensor.shape[3] != head_size:
        _invalid_cache(f"{prefix} head size {tensor.shape[3]} must equal {head_size}")
    if tensor.dtype != dtype:
        _invalid_cache(f"{prefix} dtype {tensor.dtype} must equal model dtype {dtype}")
    if tensor.device != model_device or tensor.device != input_device:
        reason = f"{prefix} device {tensor.device} must equal model/input device {model_device}"
        _invalid_cache(reason)
    if tensor.requires_grad:
        _invalid_cache(f"{prefix} requires grad but cache must be detached")


@dataclass(frozen=True, slots=True)
class UnexpectedTransformerBlockError(RuntimeError):
    """Report an unexpected module in the Transformer stack."""

    index: int
    actual_type: str

    @override
    def __str__(self) -> str:
        """Render the invalid block position and type."""
        return (
            f"unexpected GPT block {self.index}: expected TransformerBlock, got {self.actual_type}"
        )


@final
class GPT(nn.Module):
    """Run autoregressive language modeling and token generation on GPT blocks."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize embeddings, Transformer blocks, and language-model head."""
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def _validate_token_tensor(self, token_ids: Tensor, *, name: str) -> None:
        if token_ids.ndim != _TOKEN_TENSOR_DIMENSIONS:
            raise InvalidTokenTensorError(name, _SHAPE_REASON)
        if token_ids.dtype != torch.long:
            raise InvalidTokenTensorError(name, _DTYPE_REASON)
        if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise InvalidTokenTensorError(name, _NON_EMPTY_REASON)
        minimum = int(token_ids.min().item())
        maximum = int(token_ids.max().item())
        if minimum < 0 or maximum >= self.config.vocab_size:
            raise TokenIdOutOfRangeError(
                name=name,
                minimum=minimum,
                maximum=maximum,
                vocab_size=self.config.vocab_size,
            )

    @override
    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return [B,T,V] logits and optional mean next-token cross entropy."""
        self._validate_token_tensor(token_ids, name=_INPUT_NAME)
        batch_size, time_steps = token_ids.shape
        if time_steps > self.config.block_size:
            reason = f"time dimension {time_steps} exceeds block_size {self.config.block_size}"
            raise InvalidTokenTensorError(
                _INPUT_NAME,
                reason,
            )
        if targets is not None:
            self._validate_token_tensor(targets, name=_TARGET_NAME)
            if targets.shape != token_ids.shape:
                reason = (
                    f"shape {tuple(targets.shape)} must equal input shape {tuple(token_ids.shape)}"
                )
                raise InvalidTokenTensorError(
                    _TARGET_NAME,
                    reason,
                )

        positions = torch.arange(time_steps, device=token_ids.device)
        token_embeddings = cast("Tensor", self.token_embedding(token_ids))
        position_embeddings = cast("Tensor", self.position_embedding(positions))
        hidden_states = cast(
            "Tensor",
            self.embedding_dropout(token_embeddings + position_embeddings),
        )
        for index, module in enumerate(self.blocks):
            if not isinstance(module, TransformerBlock):
                raise UnexpectedTransformerBlockError(index, type(module).__name__)
            hidden_states = cast("Tensor", module(hidden_states))
        normalized = cast("Tensor", self.final_norm(hidden_states))
        logits = cast("Tensor", self.lm_head(normalized))

        loss: Tensor | None = None
        if targets is not None:
            loss = functional.cross_entropy(
                logits.reshape(batch_size * time_steps, self.config.vocab_size),
                targets.reshape(batch_size * time_steps),
            )
        return logits, loss

    def parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_cache(self, token_ids: Tensor, cache: object) -> int:  # noqa: C901
        if not isinstance(cache, tuple):
            _invalid_cache("cache must be a tuple of LayerKVCache entries")
        raw_cache = cast("tuple[object, ...]", cache)
        if len(raw_cache) != self.config.n_layer:
            reason = (
                f"layer count {len(raw_cache)} must equal configured n_layer {self.config.n_layer}"
            )
            _invalid_cache(reason)

        batch_size = token_ids.shape[0]
        expected_dtype = self.token_embedding.weight.dtype
        expected_device = self.token_embedding.weight.device
        expected_length: int | None = None
        head_size = self.config.n_embd // self.config.n_head
        for layer_index, layer_cache in enumerate(raw_cache):
            if not isinstance(layer_cache, LayerKVCache):
                reason = f"layer {layer_index} must be a LayerKVCache"
                _invalid_cache(reason)
            for tensor_name, tensor in (("key", layer_cache.key), ("value", layer_cache.value)):
                prefix = f"layer {layer_index} {tensor_name}"
                _validate_cache_tensor(
                    tensor,
                    prefix=prefix,
                    batch_size=batch_size,
                    n_head=self.config.n_head,
                    head_size=head_size,
                    dtype=expected_dtype,
                    model_device=expected_device,
                    input_device=token_ids.device,
                )
            if layer_cache.key.shape != layer_cache.value.shape:
                reason = f"layer {layer_index} key/value shapes must match"
                _invalid_cache(reason)
            length = layer_cache.length
            if length <= 0 or length > self.config.block_size:
                reason = (
                    f"layer {layer_index} length {length} must be in [1, {self.config.block_size}]"
                )
                _invalid_cache(reason)
            if expected_length is None:
                expected_length = length
            elif length != expected_length:
                reason = (
                    f"layer {layer_index} length {length} must equal cache length {expected_length}"
                )
                _invalid_cache(reason)
        if expected_length is None:
            _invalid_cache("cache must contain at least one layer")
        return expected_length

    def _cached_forward(
        self,
        token_ids: Tensor,
        cache: KVCache | None,
        *,
        past_length: int,
    ) -> tuple[Tensor, KVCache]:
        time_steps = token_ids.shape[1]
        positions = torch.arange(
            past_length,
            past_length + time_steps,
            device=token_ids.device,
        )
        token_embeddings = cast("Tensor", self.token_embedding(token_ids))
        position_embeddings = cast("Tensor", self.position_embedding(positions))
        hidden_states = cast(
            "Tensor",
            self.embedding_dropout(token_embeddings + position_embeddings),
        )
        next_cache: list[LayerKVCache] = []
        for index, module in enumerate(self.blocks):
            if not isinstance(module, TransformerBlock):
                raise UnexpectedTransformerBlockError(index, type(module).__name__)
            layer_cache = None if cache is None else cache[index]
            hidden_states, next_layer_cache = module.forward_cached(hidden_states, layer_cache)
            next_cache.append(next_layer_cache)
        normalized = cast("Tensor", self.final_norm(hidden_states))
        logits = cast("Tensor", self.lm_head(normalized))
        return logits, tuple(next_cache)

    def _validate_batched_cache_lengths(
        self,
        token_ids: Tensor,
        cache: KVCache,
        cache_lengths: Tensor,
    ) -> int:
        padded_length = self._validate_cache(token_ids, cache)
        if cache_lengths.ndim != _CACHE_LENGTH_DIMENSIONS:
            _invalid_cache("cache_lengths must have shape [batch]")
        if cache_lengths.shape[0] != token_ids.shape[0]:
            _invalid_cache("cache_lengths batch must equal input batch")
        if cache_lengths.dtype != torch.long:
            _invalid_cache("cache_lengths dtype must be torch.int64")
        if cache_lengths.device != token_ids.device:
            _invalid_cache("cache_lengths device must equal input device")
        minimum = int(cache_lengths.min().item())
        maximum = int(cache_lengths.max().item())
        if minimum <= 0 or maximum > padded_length:
            _invalid_cache(f"cache_lengths must be in [1, padded cache length {padded_length}]")
        if maximum >= self.config.block_size:
            reason = (
                f"cache length {maximum} plus new length 1 exceeds "
                f"block_size {self.config.block_size}"
            )
            _invalid_cache(reason)
        return padded_length

    def _batched_cached_forward(
        self,
        token_ids: Tensor,
        cache: KVCache,
        cache_lengths: Tensor,
        *,
        padded_length: int,
    ) -> tuple[Tensor, KVCache]:
        positions = cache_lengths.view(-1, 1)
        token_embeddings = cast("Tensor", self.token_embedding(token_ids))
        position_embeddings = cast("Tensor", self.position_embedding(positions))
        hidden_states = cast(
            "Tensor",
            self.embedding_dropout(token_embeddings + position_embeddings),
        )
        cache_columns = torch.arange(padded_length + 1, device=token_ids.device)
        valid_history = cache_columns.view(1, -1) < cache_lengths.view(-1, 1)
        valid_current = cache_columns == padded_length
        cache_valid_mask = (valid_history | valid_current).view(
            token_ids.shape[0], 1, 1, padded_length + 1
        )
        next_cache: list[LayerKVCache] = []
        for index, module in enumerate(self.blocks):
            if not isinstance(module, TransformerBlock):
                raise UnexpectedTransformerBlockError(index, type(module).__name__)
            hidden_states, next_layer_cache = module.forward_cached_batch(
                hidden_states,
                cache[index],
                cache_valid_mask,
            )
            next_cache.append(next_layer_cache)
        normalized = cast("Tensor", self.final_norm(hidden_states))
        logits = cast("Tensor", self.lm_head(normalized))
        return logits, tuple(next_cache)

    @torch.no_grad()
    def prefill(self, token_ids: Tensor) -> tuple[Tensor, KVCache]:
        """Evaluate a complete prompt and return final logits plus detached K/V."""
        self._validate_token_tensor(token_ids, name=_PROMPT_NAME)
        time_steps = token_ids.shape[1]
        if time_steps > self.config.block_size:
            reason = f"time dimension {time_steps} exceeds block_size {self.config.block_size}"
            raise InvalidTokenTensorError(_PROMPT_NAME, reason)
        logits, cache = self._cached_forward(token_ids, None, past_length=0)
        return logits[:, -1:, :], cache

    @torch.no_grad()
    def decode(self, token_ids: Tensor, cache: KVCache) -> tuple[Tensor, KVCache]:
        """Evaluate only new tokens against validated historical K/V states."""
        self._validate_token_tensor(token_ids, name=_INPUT_NAME)
        past_length = self._validate_cache(token_ids, cache)
        new_length = token_ids.shape[1]
        if past_length + new_length > self.config.block_size:
            reason = (
                f"cached length {past_length} plus new length {new_length} exceeds "
                f"block_size {self.config.block_size}"
            )
            _invalid_cache(reason)
        return self._cached_forward(token_ids, cache, past_length=past_length)

    @torch.no_grad()
    def decode_batch(
        self,
        token_ids: Tensor,
        cache: KVCache,
        cache_lengths: Tensor,
    ) -> tuple[Tensor, KVCache]:
        """Decode one token per row against variable-length right-padded K/V."""
        self._validate_token_tensor(token_ids, name=_INPUT_NAME)
        if token_ids.shape[1] != 1:
            raise InvalidTokenTensorError(_INPUT_NAME, "batched decode requires one token per row")
        padded_length = self._validate_batched_cache_lengths(token_ids, cache, cache_lengths)
        return self._batched_cached_forward(
            token_ids,
            cache,
            cache_lengths,
            padded_length=padded_length,
        )

    @staticmethod
    def _validate_generation_config(
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
    ) -> None:
        if max_new_tokens < 0:
            raise InvalidGenerationConfigError(_GENERATION_LENGTH_REASON)
        if temperature <= 0.0:
            raise InvalidGenerationConfigError(_TEMPERATURE_REASON)
        if top_k is not None and top_k <= 0:
            raise InvalidGenerationConfigError(_TOP_K_REASON)

    def _sample_next_token(
        self,
        next_token_logits: Tensor,
        *,
        temperature: float,
        top_k: int | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        next_token_logits = next_token_logits / temperature
        if top_k is not None:
            retained_count = min(top_k, self.config.vocab_size)
            retained_logits = torch.topk(next_token_logits, retained_count, dim=-1).values
            cutoff = retained_logits[:, -1].unsqueeze(-1)
            next_token_logits = next_token_logits.masked_fill(
                next_token_logits < cutoff,
                -torch.inf,
            )
        probabilities = functional.softmax(next_token_logits, dim=-1)
        return torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        )

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressively append sampled token IDs to a prompt."""
        self._validate_token_tensor(token_ids, name=_PROMPT_NAME)
        self._validate_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        generated = token_ids
        for _ in range(max_new_tokens):
            model_input = generated[:, -self.config.block_size :]
            logits, _ = cast("tuple[Tensor, Tensor | None]", self(model_input))
            next_token = self._sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            generated = torch.cat((generated, next_token), dim=1)
        return generated

    @torch.no_grad()
    def generate_cached(
        self,
        token_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Append sampled tokens while reusing valid historical key/value states."""
        self._validate_token_tensor(token_ids, name=_PROMPT_NAME)
        self._validate_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        if max_new_tokens == 0:
            return token_ids

        generated = token_ids
        model_input = generated[:, -self.config.block_size :]
        logits, cache = self.prefill(model_input)
        for generated_index in range(max_new_tokens):
            next_token = self._sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            generated = torch.cat((generated, next_token), dim=1)
            if generated_index + 1 == max_new_tokens:
                break
            if cache[0].length < self.config.block_size:
                logits, cache = self.decode(next_token, cache)
            else:
                model_input = generated[:, -self.config.block_size :]
                logits, cache = self.prefill(model_input)
        return generated
