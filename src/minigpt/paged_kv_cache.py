"""Fixed-size paged KV-cache storage with transactional request ownership."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Never, Self, final

import torch
from torch import Tensor
from typing_extensions import override

from minigpt.layers import KVCache, LayerKVCache

if TYPE_CHECKING:
    from minigpt.model import GPT

_CACHE_RANK = 4


class KVCacheBackend(StrEnum):
    """Select the long-lived serving cache representation."""

    DENSE = "dense"
    PAGED = "paged"


@dataclass(frozen=True, slots=True)
class InvalidPagedKVCacheError(ValueError):
    """Report a malformed paged-cache configuration or operation."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed cache constraint."""
        return f"invalid paged KV cache: {self.reason}"


@dataclass(frozen=True, slots=True)
class PagedKVCacheCapacityError(RuntimeError):
    """Report deterministic reservation or allocation exhaustion."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed capacity operation."""
        return f"paged KV cache capacity exhausted: {self.reason}"


@dataclass(frozen=True, slots=True)
class PagedKVCacheOwnershipError(RuntimeError):
    """Report duplicate, missing, or inconsistent request ownership."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the ownership violation."""
        return f"paged KV cache ownership violation: {self.reason}"


def _invalid(reason: str) -> Never:
    raise InvalidPagedKVCacheError(reason)


def _capacity(reason: str) -> Never:
    raise PagedKVCacheCapacityError(reason)


def _ownership(reason: str) -> Never:
    raise PagedKVCacheOwnershipError(reason)


@dataclass(frozen=True, slots=True)
class PagedKVCacheConfig:
    """Configure the fixed physical block pool."""

    block_tokens: int
    num_blocks: int

    def __post_init__(self) -> None:
        """Require positive integral block dimensions."""
        if isinstance(self.block_tokens, bool) or self.block_tokens <= 0:
            _invalid("block_tokens must be a positive integer")
        if isinstance(self.num_blocks, bool) or self.num_blocks <= 0:
            _invalid("num_blocks must be a positive integer")


@dataclass(frozen=True, slots=True)
class PagedRequestCache:
    """Expose one request's logical-to-physical block mapping."""

    request_id: str
    block_ids: tuple[int, ...]
    cache_length: int
    reserved_blocks: int


@dataclass(slots=True)
class _MutableRequestCache:
    request_id: str
    reserved_blocks: int
    block_ids: list[int]
    cache_length: int = 0


@dataclass(frozen=True, slots=True)
class PagedKVCacheMetrics:
    """Summarize pool capacity, tail waste, and allocator activity."""

    total_blocks: int
    free_blocks: int
    allocated_blocks: int
    reserved_blocks: int
    peak_allocated_blocks: int
    peak_reserved_blocks: int
    used_token_slots: int
    allocated_token_slots: int
    internal_fragmentation_tokens: int
    internal_fragmentation_ratio: float
    allocation_count: int
    free_count: int
    block_reuse_count: int


@final
class PagedKVCachePool:
    """Own all physical K/V blocks and per-request block tables."""

    def __init__(  # noqa: PLR0913
        self,
        config: PagedKVCacheConfig,
        *,
        n_layer: int,
        n_head: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Allocate non-persistent K/V storage outside model state."""
        for name, value in (("n_layer", n_layer), ("n_head", n_head), ("head_size", head_size)):
            if isinstance(value, bool) or value <= 0:
                _invalid(f"{name} must be a positive integer")
        self.config = config
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_size = head_size
        self.dtype = dtype
        self.device = device
        shape = (
            n_layer,
            config.num_blocks,
            n_head,
            config.block_tokens,
            head_size,
        )
        self.key_blocks = torch.empty(shape, dtype=dtype, device=device)
        self.value_blocks = torch.empty_like(self.key_blocks)
        self._free_blocks = list(range(config.num_blocks))
        heapq.heapify(self._free_blocks)
        self._tables: dict[str, _MutableRequestCache] = {}
        self._ever_allocated: set[int] = set()
        self._peak_allocated_blocks = 0
        self._peak_reserved_blocks = 0
        self._allocation_count = 0
        self._free_count = 0
        self._block_reuse_count = 0

    @classmethod
    def from_model(cls, config: PagedKVCacheConfig, model: GPT) -> Self:
        """Allocate a pool matching one model's layer/head/dtype/device layout."""
        return cls(
            config,
            n_layer=model.config.n_layer,
            n_head=model.config.n_head,
            head_size=model.config.n_embd // model.config.n_head,
            dtype=model.token_embedding.weight.dtype,
            device=model.token_embedding.weight.device,
        )

    @property
    def storage_nbytes(self) -> int:
        """Return the physical bytes owned by the fixed K/V tensors."""
        return (
            self.key_blocks.numel() + self.value_blocks.numel()
        ) * self.key_blocks.element_size()

    def required_blocks(self, cache_tokens: int) -> int:
        """Round a non-negative logical cache length up to whole blocks."""
        if isinstance(cache_tokens, bool) or cache_tokens < 0:
            _invalid("cache_tokens must be a non-negative integer")
        return math.ceil(cache_tokens / self.config.block_tokens)

    def can_reserve(self, reserved_blocks: int) -> bool:
        """Return whether another reservation fits without overcommit."""
        if isinstance(reserved_blocks, bool) or reserved_blocks < 0:
            _invalid("reserved_blocks must be a non-negative integer")
        return self.metrics().reserved_blocks + reserved_blocks <= self.config.num_blocks

    def has_request(self, request_id: str) -> bool:
        """Return whether a request currently owns a reservation table."""
        return request_id in self._tables

    def reserve(self, request_id: str, reserved_blocks: int) -> None:
        """Create an empty request table while protecting future allocation capacity."""
        if not request_id:
            _invalid("request_id must be non-empty")
        if request_id in self._tables:
            _ownership(f"request {request_id!r} already has a reservation")
        if isinstance(reserved_blocks, bool) or reserved_blocks < 0:
            _invalid("reserved_blocks must be a non-negative integer")
        if reserved_blocks > self.config.num_blocks:
            reason = f"request {request_id!r} requires {reserved_blocks} blocks"
            reason += f" but pool has {self.config.num_blocks}"
            _capacity(reason)
        if not self.can_reserve(reserved_blocks):
            available = self.config.num_blocks - self.metrics().reserved_blocks
            reason = f"request {request_id!r} requires {reserved_blocks} blocks"
            reason += f" with {available} available"
            _capacity(reason)
        self._tables[request_id] = _MutableRequestCache(
            request_id=request_id,
            reserved_blocks=reserved_blocks,
            block_ids=[],
        )
        self._update_peaks()
        self.verify_invariants()

    def request_cache(self, request_id: str) -> PagedRequestCache:
        """Return an immutable snapshot of one request block table."""
        table = self._table(request_id)
        return PagedRequestCache(
            request_id=table.request_id,
            block_ids=tuple(table.block_ids),
            cache_length=table.cache_length,
            reserved_blocks=table.reserved_blocks,
        )

    def write_prefill(self, request_id: str, cache: KVCache) -> None:
        """Transactionally scatter a complete compact prefill cache into new blocks."""
        table = self._table(request_id)
        if table.cache_length != 0 or table.block_ids:
            _ownership(f"request {request_id!r} already has allocated cache; use rebuild")
        cache_length = self._validate_cache(cache)
        required = self.required_blocks(cache_length)
        self._require_within_reservation(table, required)
        new_blocks = self._acquire_blocks(required)
        try:
            self._write_cache(cache, new_blocks, cache_length)
        except Exception:
            self._return_blocks(new_blocks)
            self.verify_invariants()
            raise
        table.block_ids.extend(new_blocks)
        table.cache_length = cache_length
        self._update_peaks()
        self.verify_invariants()

    def append(self, request_id: str, cache: KVCache) -> None:
        """Transactionally write only the newly appended final token from a dense result."""
        table = self._table(request_id)
        cache_length = self._validate_cache(cache)
        expected = table.cache_length + 1
        if cache_length != expected:
            reason = f"append cache length {cache_length} for request {request_id!r}"
            _invalid(f"{reason} must equal {expected}")
        needs_block = table.cache_length % self.config.block_tokens == 0
        if needs_block:
            self._require_within_reservation(table, len(table.block_ids) + 1)
        new_blocks = self._acquire_blocks(1) if needs_block else []
        block_id = new_blocks[0] if needs_block else table.block_ids[-1]
        offset = table.cache_length % self.config.block_tokens
        try:
            self._write_token(cache, block_id=block_id, offset=offset, token_index=-1)
        except Exception:
            self._return_blocks(new_blocks)
            self.verify_invariants()
            raise
        if new_blocks:
            table.block_ids.extend(new_blocks)
        table.cache_length = cache_length
        self._update_peaks()
        self.verify_invariants()

    def rebuild(self, request_id: str, cache: KVCache) -> None:
        """Transactionally replace an overflow window while preserving rollback state."""
        table = self._table(request_id)
        cache_length = self._validate_cache(cache)
        required = self.required_blocks(cache_length)
        self._require_within_reservation(table, required)
        old_ids = list(table.block_ids)
        old_length = table.cache_length
        old_key = self.key_blocks[:, old_ids].clone() if old_ids else None
        old_value = self.value_blocks[:, old_ids].clone() if old_ids else None
        extra = max(0, required - len(old_ids))
        new_blocks = self._acquire_blocks(extra)
        candidate = [*old_ids, *new_blocks][:required]
        try:
            self._write_cache(cache, candidate, cache_length)
        except Exception:
            if old_ids:
                if old_key is None or old_value is None:
                    _ownership("overflow rollback snapshot is missing")
                self.key_blocks[:, old_ids] = old_key
                self.value_blocks[:, old_ids] = old_value
            self._return_blocks(new_blocks)
            table.block_ids = old_ids
            table.cache_length = old_length
            self.verify_invariants()
            raise
        released = old_ids[required:]
        self._return_blocks(released)
        table.block_ids = candidate
        table.cache_length = cache_length
        self._update_peaks()
        self.verify_invariants()

    def materialize(self, request_id: str) -> KVCache:
        """Gather one request into the compact dense Stage 11 executor representation."""
        table = self._table(request_id)
        if table.cache_length <= 0 or not table.block_ids:
            _ownership(f"request {request_id!r} has no materializable cache")
        layers: list[LayerKVCache] = []
        for layer_index in range(self.n_layer):
            remaining = table.cache_length
            keys: list[Tensor] = []
            values: list[Tensor] = []
            for block_id in table.block_ids:
                used = min(remaining, self.config.block_tokens)
                keys.append(self.key_blocks[layer_index, block_id, :, :used, :])
                values.append(self.value_blocks[layer_index, block_id, :, :used, :])
                remaining -= used
                if remaining == 0:
                    break
            key = torch.cat(keys, dim=1).unsqueeze(0).clone().detach()
            value = torch.cat(values, dim=1).unsqueeze(0).clone().detach()
            layers.append(LayerKVCache(key=key, value=value))
        return tuple(layers)

    def release(self, request_id: str) -> None:
        """Release all ownership and reservation for one terminal request exactly once."""
        try:
            table = self._tables.pop(request_id)
        except KeyError:
            _ownership(f"request {request_id!r} has no reservation")
        self._return_blocks(table.block_ids)
        self.verify_invariants()

    def release_all(self) -> None:
        """Release every table during failure or graceful shutdown cleanup."""
        for request_id in tuple(self._tables):
            self.release(request_id)

    def metrics(self) -> PagedKVCacheMetrics:
        """Return current capacity and lifetime allocator counters."""
        allocated = sum(len(table.block_ids) for table in self._tables.values())
        reserved = sum(table.reserved_blocks for table in self._tables.values())
        used_slots = sum(table.cache_length for table in self._tables.values())
        allocated_slots = allocated * self.config.block_tokens
        fragmentation = allocated_slots - used_slots
        ratio = fragmentation / allocated_slots if allocated_slots else 0.0
        return PagedKVCacheMetrics(
            total_blocks=self.config.num_blocks,
            free_blocks=len(self._free_blocks),
            allocated_blocks=allocated,
            reserved_blocks=reserved,
            peak_allocated_blocks=self._peak_allocated_blocks,
            peak_reserved_blocks=self._peak_reserved_blocks,
            used_token_slots=used_slots,
            allocated_token_slots=allocated_slots,
            internal_fragmentation_tokens=fragmentation,
            internal_fragmentation_ratio=ratio,
            allocation_count=self._allocation_count,
            free_count=self._free_count,
            block_reuse_count=self._block_reuse_count,
        )

    def verify_invariants(self) -> None:  # noqa: C901
        """Reject any free/owned/reserved accounting or block-table inconsistency."""
        free = set(self._free_blocks)
        if len(free) != len(self._free_blocks):
            _ownership("free block list contains duplicates")
        owned_list = [block_id for table in self._tables.values() for block_id in table.block_ids]
        owned = set(owned_list)
        if len(owned) != len(owned_list):
            _ownership("a physical block has multiple owners")
        if free & owned:
            _ownership("free and owned block sets overlap")
        expected = set(range(self.config.num_blocks))
        if free | owned != expected:
            _ownership("physical blocks must be exactly free XOR owned")
        for table in self._tables.values():
            if len(table.block_ids) != len(set(table.block_ids)):
                _ownership(f"request {table.request_id!r} block table contains duplicates")
            if not 0 <= table.cache_length <= len(table.block_ids) * self.config.block_tokens:
                _ownership(f"request {table.request_id!r} cache length exceeds allocated capacity")
            if table.reserved_blocks < len(table.block_ids):
                _ownership(f"request {table.request_id!r} allocation exceeds reservation")
        metrics = self.metrics()
        if metrics.free_blocks + metrics.allocated_blocks != self.config.num_blocks:
            _ownership("free plus owned block count must equal pool size")
        if metrics.reserved_blocks > self.config.num_blocks:
            _ownership("reservations exceed pool size")

    def _table(self, request_id: str) -> _MutableRequestCache:
        try:
            return self._tables[request_id]
        except KeyError:
            _ownership(f"request {request_id!r} has no reservation")

    def _require_within_reservation(
        self,
        table: _MutableRequestCache,
        required_blocks: int,
    ) -> None:
        if required_blocks > table.reserved_blocks:
            reason = f"request {table.request_id!r} requires {required_blocks} allocated blocks"
            _capacity(f"{reason} but reserved {table.reserved_blocks}")

    def _acquire_blocks(self, count: int) -> list[int]:
        if count == 0:
            return []
        if count > len(self._free_blocks):
            reason = f"allocation requires {count} blocks"
            _capacity(f"{reason} with {len(self._free_blocks)} physically free")
        acquired: list[int] = []
        try:
            for _ in range(count):
                block_id = heapq.heappop(self._free_blocks)
                acquired.append(block_id)
                self._allocation_count += 1
                if block_id in self._ever_allocated:
                    self._block_reuse_count += 1
                self._ever_allocated.add(block_id)
        except Exception:
            self._return_blocks(acquired)
            raise
        return acquired

    def _return_blocks(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            heapq.heappush(self._free_blocks, block_id)
            self._free_count += 1

    def _validate_cache(self, cache: KVCache) -> int:  # noqa: C901
        if len(cache) != self.n_layer:
            _invalid(f"cache layer count {len(cache)} must equal {self.n_layer}")
        cache_length: int | None = None
        for layer_index, layer in enumerate(cache):
            if layer.key.shape != layer.value.shape:
                _invalid(f"layer {layer_index} key/value shapes differ")
            shape = tuple(layer.key.shape)
            if (
                len(shape) != _CACHE_RANK
                or shape[0] != 1
                or shape[1] != self.n_head
                or shape[3] != self.head_size
            ):
                _invalid(f"layer {layer_index} shape {shape} must match the pool layout")
            if layer.key.dtype != self.dtype or layer.value.dtype != self.dtype:
                _invalid(f"layer {layer_index} dtype must equal pool dtype")
            if layer.key.device != self.device or layer.value.device != self.device:
                _invalid(f"layer {layer_index} device must equal pool device")
            if layer.key.requires_grad or layer.value.requires_grad:
                _invalid(f"layer {layer_index} cache must be detached")
            length = layer.length
            if cache_length is None:
                cache_length = length
            elif length != cache_length:
                _invalid("all layer cache lengths must match")
        if cache_length is None:
            _invalid("cache must contain at least one layer")
        if cache_length <= 0:
            _invalid("cache length must be positive")
        return cache_length

    def _write_cache(self, cache: KVCache, block_ids: list[int], cache_length: int) -> None:
        position = 0
        for block_id in block_ids:
            used = min(self.config.block_tokens, cache_length - position)
            for layer_index, layer in enumerate(cache):
                self.key_blocks[layer_index, block_id, :, :used, :] = layer.key[
                    0, :, position : position + used, :
                ]
                self.value_blocks[layer_index, block_id, :, :used, :] = layer.value[
                    0, :, position : position + used, :
                ]
            position += used
        if position != cache_length:
            _capacity(f"allocated blocks hold {position} of {cache_length} cache tokens")

    def _write_token(
        self,
        cache: KVCache,
        *,
        block_id: int,
        offset: int,
        token_index: int,
    ) -> None:
        for layer_index, layer in enumerate(cache):
            self.key_blocks[layer_index, block_id, :, offset, :] = layer.key[0, :, token_index, :]
            self.value_blocks[layer_index, block_id, :, offset, :] = layer.value[
                0, :, token_index, :
            ]

    def _update_peaks(self) -> None:
        metrics = self.metrics()
        self._peak_allocated_blocks = max(self._peak_allocated_blocks, metrics.allocated_blocks)
        self._peak_reserved_blocks = max(self._peak_reserved_blocks, metrics.reserved_blocks)
