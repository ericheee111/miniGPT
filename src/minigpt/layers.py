"""Provide the neural-network building blocks used by the GPT model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast, final

import torch
from torch import Tensor, nn
from torch.nn import functional
from typing_extensions import override

if TYPE_CHECKING:
    from minigpt.settings import GPTConfig


@dataclass(frozen=True, slots=True)
class LayerKVCache:
    """Hold detached key/value states for one Transformer layer."""

    key: Tensor
    value: Tensor

    @property
    def length(self) -> int:
        """Return the number of cached token positions."""
        return self.key.shape[2]

    @property
    def nbytes(self) -> int:
        """Return the storage bytes represented by both cache tensors."""
        return (self.key.numel() * self.key.element_size()) + (
            self.value.numel() * self.value.element_size()
        )


KVCache: TypeAlias = tuple[LayerKVCache, ...]


def kv_cache_nbytes(cache: KVCache) -> int:
    """Return the total bytes represented by every layer cache."""
    return sum(layer_cache.nbytes for layer_cache in cache)


@final
class LayerNorm(nn.Module):
    """Normalize each token vector and optionally learn an additive bias."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        bias: bool,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize scale, optional bias, and numerical stability epsilon."""
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim)) if bias else None

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Normalize the final tensor dimension using population variance."""
        centered = hidden_states - hidden_states.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.epsilon)
        shifted = normalized * self.weight
        if self.bias is not None:
            shifted = shifted + self.bias
        return shifted


@final
class CausalSelfAttention(nn.Module):
    """Apply masked multi-head self-attention over a token sequence."""

    causal_mask: Tensor

    def __init__(self, config: GPTConfig) -> None:
        """Initialize projections, dropout, and the reusable causal mask."""
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.qkv_projection = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )
        self.output_projection = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("causal_mask", mask, persistent=False)
        self.causal_mask = mask

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Transform [B,T,C] states through scaled dot-product attention."""
        query, key, value = self._project_qkv(hidden_states)
        return self._attend(query, key, value, past_length=0)

    def forward_cached(
        self,
        hidden_states: Tensor,
        cache: LayerKVCache | None,
    ) -> tuple[Tensor, LayerKVCache]:
        """Attend new states over optional historical keys and values."""
        query, key, value = self._project_qkv(hidden_states)
        past_length = 0
        if cache is not None:
            past_length = cache.length
            key = torch.cat((cache.key, key), dim=2)
            value = torch.cat((cache.value, value), dim=2)
        output = self._attend(query, key, value, past_length=past_length)
        next_cache = LayerKVCache(key=key.detach(), value=value.detach())
        return output, next_cache

    def forward_cached_batch(
        self,
        hidden_states: Tensor,
        cache: LayerKVCache,
        cache_valid_mask: Tensor,
    ) -> tuple[Tensor, LayerKVCache]:
        """Decode one token per row against a right-padded dense cache."""
        query, key, value = self._project_qkv(hidden_states)
        key = torch.cat((cache.key, key), dim=2)
        value = torch.cat((cache.value, value), dim=2)
        output = self._attend(
            query,
            key,
            value,
            past_length=cache.length,
            allowed_positions=cache_valid_mask,
        )
        next_cache = LayerKVCache(key=key.detach(), value=value.detach())
        return output, next_cache

    def forward_prefill_batch(
        self,
        hidden_states: Tensor,
        allowed_positions: Tensor,
    ) -> tuple[Tensor, LayerKVCache]:
        """Attend right-padded prompts through the shared projection and attention path."""
        query, key, value = self._project_qkv(hidden_states)
        output = self._attend(
            query,
            key,
            value,
            past_length=0,
            allowed_positions=allowed_positions,
        )
        cache = LayerKVCache(key=key.detach(), value=value.detach())
        return output, cache

    def _project_qkv(
        self,
        hidden_states: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project [B,T,C] states and split them into per-head Q, K, and V."""
        batch_size, time_steps, _ = hidden_states.shape
        projected_qkv = cast("Tensor", self.qkv_projection(hidden_states))
        query = projected_qkv[..., : self.n_embd]
        key = projected_qkv[..., self.n_embd : 2 * self.n_embd]
        value = projected_qkv[..., 2 * self.n_embd :]
        query = query.view(batch_size, time_steps, self.n_head, self.head_size).transpose(1, 2)
        key = key.view(batch_size, time_steps, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(batch_size, time_steps, self.n_head, self.head_size).transpose(1, 2)
        return query, key, value

    def _attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        past_length: int,
        allowed_positions: Tensor | None = None,
    ) -> Tensor:
        """Apply offset-causal attention to already projected per-head tensors."""
        batch_size = query.shape[0]
        time_steps = query.shape[2]
        key_length = key.shape[2]
        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if allowed_positions is None:
            allowed_positions = self.causal_mask[
                :, :, past_length : past_length + time_steps, :key_length
            ]
        attention_scores = attention_scores.masked_fill(
            ~allowed_positions,
            torch.finfo(attention_scores.dtype).min,
        )
        attention_weights = functional.softmax(attention_scores, dim=-1)
        attention_weights = cast("Tensor", self.attention_dropout(attention_weights))

        context = attention_weights @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, time_steps, self.n_embd)
        projected = cast("Tensor", self.output_projection(context))
        return cast("Tensor", self.residual_dropout(projected))


@final
class MLP(nn.Module):
    """Expand, transform, and project each token independently."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize the position-wise expansion and projection layers."""
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.input_projection = nn.Linear(
            config.n_embd,
            hidden_dim,
            bias=config.bias,
        )
        self.activation = nn.GELU(approximate="tanh")
        self.output_projection = nn.Linear(
            hidden_dim,
            config.n_embd,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the position-wise feed-forward network."""
        hidden_states = cast("Tensor", self.input_projection(hidden_states))
        hidden_states = cast("Tensor", self.activation(hidden_states))
        projected = cast("Tensor", self.output_projection(hidden_states))
        return cast("Tensor", self.dropout(projected))


@final
class TransformerBlock(nn.Module):
    """Combine pre-normalized attention and MLP residual branches."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize pre-normalization and both residual branches."""
        super().__init__()
        self.attention_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply both residual branches without changing tensor shape."""
        attention_input = cast("Tensor", self.attention_norm(hidden_states))
        hidden_states = hidden_states + cast("Tensor", self.attention(attention_input))
        mlp_input = cast("Tensor", self.mlp_norm(hidden_states))
        return hidden_states + cast("Tensor", self.mlp(mlp_input))

    def forward_cached(
        self,
        hidden_states: Tensor,
        cache: LayerKVCache | None,
    ) -> tuple[Tensor, LayerKVCache]:
        """Apply the block to new states and return an extended layer cache."""
        attention_input = cast("Tensor", self.attention_norm(hidden_states))
        attention_output, next_cache = self.attention.forward_cached(attention_input, cache)
        hidden_states = hidden_states + attention_output
        mlp_input = cast("Tensor", self.mlp_norm(hidden_states))
        output = hidden_states + cast("Tensor", self.mlp(mlp_input))
        return output, next_cache

    def forward_cached_batch(
        self,
        hidden_states: Tensor,
        cache: LayerKVCache,
        cache_valid_mask: Tensor,
    ) -> tuple[Tensor, LayerKVCache]:
        """Apply one variable-cache-length decode step to every batch row."""
        attention_input = cast("Tensor", self.attention_norm(hidden_states))
        attention_output, next_cache = self.attention.forward_cached_batch(
            attention_input,
            cache,
            cache_valid_mask,
        )
        hidden_states = hidden_states + attention_output
        mlp_input = cast("Tensor", self.mlp_norm(hidden_states))
        output = hidden_states + cast("Tensor", self.mlp(mlp_input))
        return output, next_cache

    def forward_prefill_batch(
        self,
        hidden_states: Tensor,
        allowed_positions: Tensor,
    ) -> tuple[Tensor, LayerKVCache]:
        """Apply the block to right-padded prompts and retain dense temporary K/V."""
        attention_input = cast("Tensor", self.attention_norm(hidden_states))
        attention_output, cache = self.attention.forward_prefill_batch(
            attention_input,
            allowed_positions,
        )
        hidden_states = hidden_states + attention_output
        mlp_input = cast("Tensor", self.mlp_norm(hidden_states))
        output = hidden_states + cast("Tensor", self.mlp(mlp_input))
        return output, cache
