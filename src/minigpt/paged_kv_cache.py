"""Fixed-size paged KV-cache storage with transactional request ownership."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Never, Self, cast, final

import torch
from torch import Tensor
from typing_extensions import override

from minigpt.layers import KVCache, LayerKVCache, PagedKVCacheView, PagedLayerKVCacheView

if TYPE_CHECKING:
    from minigpt.model import GPT

_CACHE_RANK = 4


class KVCacheBackend(StrEnum):
    """Select the long-lived serving cache representation."""

    DENSE = "dense"
    PAGED = "paged"


class PhysicalBlockState(StrEnum):
    """Identify the unique lifecycle state of one physical K/V block."""

    FREE = "free"
    PRIVATE = "private"
    SHARED = "shared"


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
class PrefixCacheNamespace:
    """Bind immutable prefix blocks to one exact model/cache identity."""

    model_checkpoint_identity: str
    model_config_identity: str
    dtype: str
    device: str
    block_tokens: int
    cache_schema_version: int
    position_embedding_semantics: str

    def __post_init__(self) -> None:
        """Reject incomplete or incompatible cache identity fields."""
        string_fields = (
            self.model_checkpoint_identity,
            self.model_config_identity,
            self.dtype,
            self.device,
            self.position_embedding_semantics,
        )
        if any(not value for value in string_fields):
            _invalid("prefix cache namespace string fields must be non-empty")
        if isinstance(self.block_tokens, bool) or self.block_tokens <= 0:
            _invalid("prefix cache namespace block_tokens must be positive")
        if isinstance(self.cache_schema_version, bool) or self.cache_schema_version <= 0:
            _invalid("prefix cache schema version must be positive")

    @property
    def digest(self) -> str:
        """Return the deterministic SHA-256 namespace identity."""
        document = {
            "block_tokens": self.block_tokens,
            "cache_schema_version": self.cache_schema_version,
            "device": self.device,
            "dtype": self.dtype,
            "model_checkpoint_identity": self.model_checkpoint_identity,
            "model_config_identity": self.model_config_identity,
            "position_embedding_semantics": self.position_embedding_semantics,
        }
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PagedRequestCache:
    """Expose one request's logical-to-physical block mapping."""

    request_id: str
    block_ids: tuple[int, ...]
    cache_length: int
    reserved_blocks: int
    max_blocks: int
    shared_blocks: int


@dataclass(slots=True)
class _MutableRequestCache:
    request_id: str
    reserved_blocks: int
    block_ids: list[int]
    cache_length: int = 0
    max_blocks: int = 0


@dataclass(slots=True)
class _PhysicalBlock:
    state: PhysicalBlockState = PhysicalBlockState.FREE
    owner_request_id: str | None = None
    prefix_hash: str | None = None
    active_refcount: int = 0
    last_used: int = 0
    token_fingerprint: str | None = None
    token_block: tuple[int, ...] | None = None
    boundary_logits: Tensor | None = None


@dataclass(frozen=True, slots=True)
class PrefixCacheLookup:
    """Describe one atomic longest-contiguous-prefix attachment."""

    prefix_hashes: tuple[str, ...]
    block_ids: tuple[int, ...]
    prefix_hit_blocks: int
    prefix_hit_tokens: int
    prefix_miss_tokens: int
    evicted_prefix_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrefixCachePromotion:
    """Summarize canonical block promotion and duplicate disposal."""

    promoted_blocks: int
    duplicate_private_blocks_released: int
    evicted_prefix_hashes: tuple[str, ...]


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
    private_blocks: int = 0
    prefix_cache_blocks: int = 0
    evictable_blocks: int = 0
    active_shared_blocks: int = 0
    active_shared_references: int = 0
    prefix_cache_evictions: int = 0
    prefix_lookup_requests: int = 0
    prefix_hit_requests: int = 0
    prefix_hit_blocks: int = 0
    prefix_hit_tokens: int = 0
    prefix_miss_tokens: int = 0
    reservation_growth_count: int = 0
    reservation_growth_blocks: int = 0


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
        prefix_cache_namespace: PrefixCacheNamespace | None = None,
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
        if (
            prefix_cache_namespace is not None
            and prefix_cache_namespace.block_tokens != config.block_tokens
        ):
            _invalid("prefix cache namespace block_tokens must equal pool block_tokens")
        if prefix_cache_namespace is not None and (
            prefix_cache_namespace.dtype != str(dtype)
            or prefix_cache_namespace.device != str(device)
        ):
            _invalid("prefix cache namespace dtype/device must equal pool storage")
        self.prefix_cache_namespace = prefix_cache_namespace
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
        self._blocks = [_PhysicalBlock() for _ in range(config.num_blocks)]
        self._tables: dict[str, _MutableRequestCache] = {}
        self._prefix_index: dict[str, int] = {}
        self._logical_clock = 0
        self._recent_evictions: list[str] = []
        self._ever_allocated: set[int] = set()
        self._peak_allocated_blocks = 0
        self._peak_reserved_blocks = 0
        self._allocation_count = 0
        self._free_count = 0
        self._block_reuse_count = 0
        self._prefix_cache_evictions = 0
        self._prefix_lookup_requests = 0
        self._prefix_hit_requests = 0
        self._prefix_hit_blocks = 0
        self._prefix_hit_tokens = 0
        self._prefix_miss_tokens = 0
        self._reservation_growth_count = 0
        self._reservation_growth_blocks = 0

    @classmethod
    def from_model(
        cls,
        config: PagedKVCacheConfig,
        model: GPT,
        *,
        prefix_cache_namespace: PrefixCacheNamespace | None = None,
    ) -> Self:
        """Allocate a pool matching one model's layer/head/dtype/device layout."""
        return cls(
            config,
            n_layer=model.config.n_layer,
            n_head=model.config.n_head,
            head_size=model.config.n_embd // model.config.n_head,
            dtype=model.token_embedding.weight.dtype,
            device=model.token_embedding.weight.device,
            prefix_cache_namespace=prefix_cache_namespace,
        )

    @property
    def prefix_cache_enabled(self) -> bool:
        """Return whether this pool has a model-bound APC namespace."""
        return self.prefix_cache_namespace is not None

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
        return self._protected_blocks() + reserved_blocks <= self.config.num_blocks

    def prefix_hash_chain(self, prompt_tokens: tuple[int, ...]) -> tuple[str, ...]:
        """Hash every complete token block while binding namespace and all history."""
        namespace = self._require_prefix_namespace()
        hashes: list[str] = []
        parent = ""
        for start in range(
            0, len(prompt_tokens) - self.config.block_tokens + 1, self.config.block_tokens
        ):
            token_block = prompt_tokens[start : start + self.config.block_tokens]
            encoded = json.dumps(token_block, separators=(",", ":")).encode()
            payload = b"\0".join((namespace.digest.encode(), parent.encode(), encoded))
            parent = hashlib.sha256(payload).hexdigest()
            hashes.append(parent)
        return tuple(hashes)

    def prefix_block_tokens(self, prefix_hash: str) -> tuple[int, ...] | None:
        """Return exact collision-defense metadata for one resident canonical hash."""
        block_id = self._prefix_index.get(prefix_hash)
        if block_id is None:
            return None
        return self._blocks[block_id].token_block

    def block_state(self, block_id: int) -> PhysicalBlockState:
        """Expose one physical state for deterministic tests and diagnostics."""
        if not 0 <= block_id < self.config.num_blocks:
            _invalid("block_id is outside the physical pool")
        return self._blocks[block_id].state

    def can_reserve_with_prefix(
        self,
        *,
        reserved_blocks: int,
        prompt_tokens: tuple[int, ...],
    ) -> bool:
        """Preview whether atomic shared attachment plus private growth fits."""
        lookup_ids, _hashes = self._longest_prefix(prompt_tokens)
        newly_active = sum(self._blocks[block_id].active_refcount == 0 for block_id in lookup_ids)
        private_reservation = reserved_blocks - len(lookup_ids)
        if private_reservation < 0:
            _invalid("shared prefix blocks exceed request reservation")
        return (
            self._protected_blocks() + newly_active + private_reservation <= self.config.num_blocks
        )

    def has_request(self, request_id: str) -> bool:
        """Return whether a request currently owns a reservation table."""
        return request_id in self._tables

    def _resolve_max_blocks(self, reserved_blocks: int, max_blocks: int | None) -> int:
        """Validate current protection and its immutable lifetime upper bound."""
        if isinstance(reserved_blocks, bool) or reserved_blocks < 0:
            _invalid("reserved_blocks must be a non-negative integer")
        resolved = reserved_blocks if max_blocks is None else max_blocks
        if isinstance(resolved, bool) or resolved < 0:
            _invalid("max_blocks must be a non-negative integer")
        if reserved_blocks > resolved:
            _invalid("reserved_blocks must not exceed max_blocks")
        if resolved > self.config.num_blocks:
            _capacity(f"request lifetime reservation {resolved} blocks exceeds pool size")
        return resolved

    def reserve(
        self,
        request_id: str,
        reserved_blocks: int,
        *,
        max_blocks: int | None = None,
    ) -> None:
        """Create a table with current protection and a lifetime allocation upper bound."""
        if not request_id:
            _invalid("request_id must be non-empty")
        if request_id in self._tables:
            _ownership(f"request {request_id!r} already has a reservation")
        resolved_max = self._resolve_max_blocks(reserved_blocks, max_blocks)
        if not self.can_reserve(reserved_blocks):
            available = self.config.num_blocks - self.metrics().reserved_blocks
            reason = f"request {request_id!r} requires {reserved_blocks} protected blocks"
            reason += f" with {available} available"
            _capacity(reason)
        self._tables[request_id] = _MutableRequestCache(
            request_id=request_id,
            reserved_blocks=reserved_blocks,
            block_ids=[],
            max_blocks=resolved_max,
        )
        self._update_peaks()
        self.verify_invariants()

    def reserve_with_prefix(
        self,
        request_id: str,
        *,
        reserved_blocks: int,
        prompt_tokens: tuple[int, ...],
        max_blocks: int | None = None,
    ) -> PrefixCacheLookup:
        """Attach cached prefix blocks under current and lifetime reservation bounds."""
        if not self.prefix_cache_enabled:
            _invalid("prefix cache lookup requires a configured namespace")
        if not request_id:
            _invalid("request_id must be non-empty")
        if request_id in self._tables:
            _ownership(f"request {request_id!r} already has a reservation")
        resolved_max = self._resolve_max_blocks(reserved_blocks, max_blocks)
        block_ids, prefix_hashes = self._longest_prefix(prompt_tokens)
        newly_active = sum(self._blocks[block_id].active_refcount == 0 for block_id in block_ids)
        private_reservation = reserved_blocks - len(block_ids)
        protected = self._protected_blocks() + newly_active + private_reservation
        if private_reservation < 0 or protected > self.config.num_blocks:
            _capacity(f"request {request_id!r} prefix attachment exceeds protected capacity")
        for block_id in block_ids:
            block = self._blocks[block_id]
            block.active_refcount += 1
            self._touch(block)
        hit_tokens = len(block_ids) * self.config.block_tokens
        self._tables[request_id] = _MutableRequestCache(
            request_id=request_id,
            reserved_blocks=private_reservation,
            block_ids=list(block_ids),
            cache_length=hit_tokens,
            max_blocks=resolved_max,
        )
        self._prefix_lookup_requests += 1
        if block_ids:
            self._prefix_hit_requests += 1
        self._prefix_hit_blocks += len(block_ids)
        self._prefix_hit_tokens += hit_tokens
        miss_tokens = len(prompt_tokens) - hit_tokens
        self._prefix_miss_tokens += miss_tokens
        self._update_peaks()
        self.verify_invariants()
        return PrefixCacheLookup(
            prefix_hashes=prefix_hashes,
            block_ids=block_ids,
            prefix_hit_blocks=len(block_ids),
            prefix_hit_tokens=hit_tokens,
            prefix_miss_tokens=miss_tokens,
            evicted_prefix_hashes=self.take_recent_evictions(),
        )

    def current_total_reserved_blocks(self, request_id: str) -> int:
        """Return current private protection plus active shared prefix blocks."""
        table = self._table(request_id)
        return table.reserved_blocks + self._shared_count(table)

    def can_grow_reservation(self, request_id: str, target_total_blocks: int) -> bool:
        """Return whether current protection can grow to a total-block target."""
        if isinstance(target_total_blocks, bool) or target_total_blocks < 0:
            _invalid("target_total_blocks must be a non-negative integer")
        table = self._table(request_id)
        current = table.reserved_blocks + self._shared_count(table)
        if target_total_blocks <= current:
            return True
        if target_total_blocks > table.max_blocks:
            return False
        return self._protected_blocks() + target_total_blocks - current <= self.config.num_blocks

    def grow_reservation(self, request_id: str, target_total_blocks: int) -> int:
        """Grow private protection transactionally without allocating physical cache blocks."""
        if isinstance(target_total_blocks, bool) or target_total_blocks < 0:
            _invalid("target_total_blocks must be a non-negative integer")
        table = self._table(request_id)
        current = table.reserved_blocks + self._shared_count(table)
        if target_total_blocks <= current:
            return 0
        if target_total_blocks > table.max_blocks:
            _capacity(
                "".join(
                    (
                        f"request {request_id!r} growth target {target_total_blocks}",
                        f" exceeds max {table.max_blocks}",
                    )
                )
            )
        delta = target_total_blocks - current
        if self._protected_blocks() + delta > self.config.num_blocks:
            available = self.config.num_blocks - self.metrics().reserved_blocks
            _capacity(
                f"request {request_id!r} growth requires {delta} blocks with {available} available"
            )
        table.reserved_blocks += delta
        self._reservation_growth_count += 1
        self._reservation_growth_blocks += delta
        self._update_peaks()
        self.verify_invariants()
        return delta

    def request_cache(self, request_id: str) -> PagedRequestCache:
        """Return an immutable snapshot of one request block table."""
        table = self._table(request_id)
        return PagedRequestCache(
            request_id=table.request_id,
            block_ids=tuple(table.block_ids),
            cache_length=table.cache_length,
            reserved_blocks=table.reserved_blocks,
            max_blocks=table.max_blocks,
            shared_blocks=sum(
                self._blocks[block_id].state is PhysicalBlockState.SHARED
                for block_id in table.block_ids
            ),
        )

    def write_prefill(self, request_id: str, cache: KVCache) -> None:
        """Transactionally scatter a complete compact prefill cache into new blocks."""
        table = self._table(request_id)
        if table.cache_length != 0 or table.block_ids:
            _ownership(f"request {request_id!r} already has allocated cache; use rebuild")
        cache_length = self._validate_cache(cache)
        required = self.required_blocks(cache_length)
        self._require_within_reservation(table, required)
        new_blocks = self._acquire_blocks(required, owner_request_id=request_id)
        try:
            self._write_cache(cache, new_blocks, cache_length)
        except Exception:
            self._return_blocks(new_blocks, expected_owner=request_id)
            self.verify_invariants()
            raise
        table.block_ids.extend(new_blocks)
        table.cache_length = cache_length
        self._update_peaks()
        self.verify_invariants()

    def write_prefill_suffix(self, request_id: str, cache_delta: KVCache) -> None:
        """Append only unmatched prompt K/V after an aligned shared-prefix attachment."""
        table = self._table(request_id)
        if table.cache_length % self.config.block_tokens != 0:
            _ownership("prefix-hit prefill must start at a complete block boundary")
        if not cache_delta:
            self.verify_invariants()
            return
        delta_length = self._validate_cache(cache_delta)
        new_length = table.cache_length + delta_length
        required = self.required_blocks(new_length)
        private_required = required - self._shared_count(table)
        self._require_within_reservation(table, private_required)
        new_count = required - len(table.block_ids)
        new_blocks = self._acquire_blocks(new_count, owner_request_id=request_id)
        try:
            self._write_cache(cache_delta, new_blocks, delta_length)
        except Exception:
            self._return_blocks(new_blocks, expected_owner=request_id)
            self.verify_invariants()
            raise
        table.block_ids.extend(new_blocks)
        table.cache_length = new_length
        self._update_peaks()
        self.verify_invariants()

    def promote_prompt_blocks(  # noqa: PLR0915
        self,
        request_id: str,
        *,
        prompt_tokens: tuple[int, ...],
        prefix_hit_tokens: int,
        suffix_logits: Tensor,
    ) -> PrefixCachePromotion:
        """Promote complete PRIVATE prompt blocks or attach an existing canonical copy."""
        _ = self._require_prefix_namespace()
        table = self._table(request_id)
        if prefix_hit_tokens < 0 or prefix_hit_tokens % self.config.block_tokens != 0:
            _invalid("prefix_hit_tokens must be a non-negative complete-block length")
        if table.cache_length != len(prompt_tokens):
            _ownership("prompt promotion requires a fully committed prompt cache")
        expected_suffix = len(prompt_tokens) - prefix_hit_tokens
        expected_logits_dimensions = 2
        if (
            suffix_logits.ndim != expected_logits_dimensions
            or suffix_logits.shape[0] != expected_suffix
        ):
            _invalid("suffix_logits must contain one row per computed prompt suffix token")
        hashes = self.prefix_hash_chain(prompt_tokens)
        hit_blocks = prefix_hit_tokens // self.config.block_tokens
        promoted = 0
        duplicates = 0
        metadata_snapshot = self._metadata_snapshot()
        table_snapshot = self._table_snapshot(table)
        free_snapshot = list(self._free_blocks)
        prefix_snapshot = dict(self._prefix_index)
        free_count_snapshot = self._free_count
        logical_clock_snapshot = self._logical_clock
        try:
            for logical_index, prefix_hash in enumerate(hashes):
                token_start = logical_index * self.config.block_tokens
                token_block = prompt_tokens[token_start : token_start + self.config.block_tokens]
                fingerprint = self._token_fingerprint(token_block)
                block_id = table.block_ids[logical_index]
                if logical_index < hit_blocks:
                    self._validate_canonical(block_id, prefix_hash, token_block, fingerprint)
                    continue
                boundary_position = token_start + self.config.block_tokens - 1
                suffix_position = boundary_position - prefix_hit_tokens
                boundary_logits = suffix_logits[suffix_position].clone().detach()
                canonical_id = self._prefix_index.get(prefix_hash)
                if canonical_id is None:
                    block = self._blocks[block_id]
                    self._require_private_owner(block_id, request_id)
                    block.state = PhysicalBlockState.SHARED
                    block.owner_request_id = None
                    block.prefix_hash = prefix_hash
                    block.active_refcount = 1
                    block.token_fingerprint = fingerprint
                    block.token_block = token_block
                    block.boundary_logits = boundary_logits
                    self._touch(block)
                    self._prefix_index[prefix_hash] = block_id
                    table.reserved_blocks -= 1
                    promoted += 1
                    continue
                self._validate_canonical(
                    canonical_id,
                    prefix_hash,
                    token_block,
                    fingerprint,
                )
                canonical = self._blocks[canonical_id]
                if canonical.boundary_logits is None:
                    _ownership("canonical prefix block is missing boundary logits")
                canonical.active_refcount += 1
                self._touch(canonical)
                table.block_ids[logical_index] = canonical_id
                self._return_blocks([block_id], expected_owner=request_id)
                table.reserved_blocks -= 1
                duplicates += 1
        except Exception:
            self._restore_metadata(metadata_snapshot)
            table.reserved_blocks = table_snapshot.reserved_blocks
            table.block_ids = list(table_snapshot.block_ids)
            table.cache_length = table_snapshot.cache_length
            table.max_blocks = table_snapshot.max_blocks
            self._free_blocks = free_snapshot
            self._prefix_index = prefix_snapshot
            self._free_count = free_count_snapshot
            self._logical_clock = logical_clock_snapshot
            self.verify_invariants()
            raise
        self.verify_invariants()
        return PrefixCachePromotion(
            promoted_blocks=promoted,
            duplicate_private_blocks_released=duplicates,
            evicted_prefix_hashes=self.take_recent_evictions(),
        )

    def prefix_boundary_logits_for_request(self, request_id: str) -> Tensor:
        """Return immutable next-token logits for an exact complete-block prefix hit."""
        table = self._table(request_id)
        if table.cache_length <= 0 or table.cache_length % self.config.block_tokens != 0:
            _ownership("exact prefix logits require a non-empty complete-block cache")
        block = self._blocks[table.block_ids[-1]]
        if block.state is not PhysicalBlockState.SHARED or block.boundary_logits is None:
            _ownership("exact prefix hit is missing canonical boundary logits")
        self._touch(block)
        return block.boundary_logits.clone().detach()

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
            self._require_within_reservation(table, self._private_count(table) + 1)
        new_blocks = self._acquire_blocks(1, owner_request_id=request_id) if needs_block else []
        block_id = new_blocks[0] if needs_block else table.block_ids[-1]
        offset = table.cache_length % self.config.block_tokens
        try:
            self._write_token(cache, block_id=block_id, offset=offset, token_index=-1)
        except Exception:
            self._return_blocks(new_blocks, expected_owner=request_id)
            self.verify_invariants()
            raise
        if new_blocks:
            table.block_ids.extend(new_blocks)
        table.cache_length = cache_length
        self._update_peaks()
        self.verify_invariants()

    def append_delta(self, request_id: str, cache_delta: KVCache) -> None:
        """Transactionally append a one-token K/V delta from block-aware attention."""
        delta_length = self._validate_cache(cache_delta)
        if delta_length != 1:
            _invalid("paged attention cache delta must contain exactly one token")
        table = self._table(request_id)
        needs_block = table.cache_length % self.config.block_tokens == 0
        if needs_block:
            self._require_within_reservation(table, self._private_count(table) + 1)
        new_blocks = self._acquire_blocks(1, owner_request_id=request_id) if needs_block else []
        block_id = new_blocks[0] if needs_block else table.block_ids[-1]
        offset = table.cache_length % self.config.block_tokens
        try:
            self._write_token(cache_delta, block_id=block_id, offset=offset, token_index=0)
        except Exception:
            self._return_blocks(new_blocks, expected_owner=request_id)
            self.verify_invariants()
            raise
        if new_blocks:
            table.block_ids.extend(new_blocks)
        table.cache_length += 1
        self._update_peaks()
        self.verify_invariants()

    def rebuild(self, request_id: str, cache: KVCache) -> None:
        """Transactionally replace an overflow window while preserving rollback state."""
        table = self._table(request_id)
        cache_length = self._validate_cache(cache)
        required = self.required_blocks(cache_length)
        if not self.prefix_cache_enabled:
            self._rebuild_private_cache(table, cache, cache_length=cache_length, required=required)
            return
        current_total = table.reserved_blocks + self._shared_count(table)
        if required > current_total:
            _capacity(
                "".join(
                    (
                        f"overflow rebuild requires {required} blocks",
                        f" but current protection is {current_total}",
                    )
                )
            )
        metadata_snapshot = self._metadata_snapshot()
        table_snapshot = self._table_snapshot(table)
        free_snapshot = list(self._free_blocks)
        prefix_snapshot = dict(self._prefix_index)
        key_snapshot = self.key_blocks.clone()
        value_snapshot = self.value_blocks.clone()
        try:
            self._release_table_blocks(table)
            table.block_ids = []
            table.cache_length = 0
            table.reserved_blocks = current_total
            candidate = self._acquire_blocks(required, owner_request_id=request_id)
            self._write_cache(cache, candidate, cache_length)
            table.block_ids = candidate
            table.cache_length = cache_length
        except Exception:
            _ = self.key_blocks.copy_(key_snapshot)
            _ = self.value_blocks.copy_(value_snapshot)
            self._restore_metadata(metadata_snapshot)
            table.reserved_blocks = table_snapshot.reserved_blocks
            table.block_ids = list(table_snapshot.block_ids)
            table.cache_length = table_snapshot.cache_length
            table.max_blocks = table_snapshot.max_blocks
            self._free_blocks = free_snapshot
            self._prefix_index = prefix_snapshot
            self.verify_invariants()
            raise
        self._update_peaks()
        self.verify_invariants()

    def _rebuild_private_cache(
        self,
        table: _MutableRequestCache,
        cache: KVCache,
        *,
        cache_length: int,
        required: int,
    ) -> None:
        """Preserve the Stage 13 in-place rebuild and allocator-counter contract."""
        self._require_within_reservation(table, required)
        old_ids = list(table.block_ids)
        old_length = table.cache_length
        old_key = self.key_blocks[:, old_ids].clone() if old_ids else None
        old_value = self.value_blocks[:, old_ids].clone() if old_ids else None
        extra = max(0, required - len(old_ids))
        new_blocks = self._acquire_blocks(extra, owner_request_id=table.request_id)
        candidate = [*old_ids, *new_blocks][:required]
        try:
            self._write_cache(cache, candidate, cache_length)
        except Exception:
            if old_ids:
                if old_key is None or old_value is None:
                    _ownership("overflow rollback snapshot is missing")
                self.key_blocks[:, old_ids] = old_key
                self.value_blocks[:, old_ids] = old_value
            self._return_blocks(new_blocks, expected_owner=table.request_id)
            table.block_ids = old_ids
            table.cache_length = old_length
            self.verify_invariants()
            raise
        released = old_ids[required:]
        self._return_blocks(released, expected_owner=table.request_id)
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

    def request_view(self, request_id: str) -> PagedKVCacheView:
        """Return ordered per-layer block aliases for one owner-thread decode call."""
        table = self._table(request_id)
        if table.cache_length <= 0 or not table.block_ids:
            _ownership(f"request {request_id!r} has no block-aware cache view")
        layers: list[PagedLayerKVCacheView] = []
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
            layers.append(
                PagedLayerKVCacheView(
                    key_blocks=tuple(keys),
                    value_blocks=tuple(values),
                    cache_length=table.cache_length,
                    block_tokens=self.config.block_tokens,
                )
            )
        return tuple(layers)

    def release(self, request_id: str) -> None:
        """Release all ownership and reservation for one terminal request exactly once."""
        try:
            table = self._tables.pop(request_id)
        except KeyError:
            _ownership(f"request {request_id!r} has no reservation")
        self._release_table_blocks(table)
        self.verify_invariants()

    def release_all(self) -> None:
        """Release every table during failure or graceful shutdown cleanup."""
        for request_id in tuple(self._tables):
            self.release(request_id)
        while self._evict_one():
            pass
        self.verify_invariants()

    def take_recent_evictions(self) -> tuple[str, ...]:
        """Drain deterministic cache-eviction notifications for engine events."""
        evictions = tuple(self._recent_evictions)
        self._recent_evictions.clear()
        return evictions

    def metrics(self) -> PagedKVCacheMetrics:
        """Return current capacity and lifetime allocator counters."""
        private_blocks = sum(block.state is PhysicalBlockState.PRIVATE for block in self._blocks)
        shared_blocks = sum(block.state is PhysicalBlockState.SHARED for block in self._blocks)
        active_shared_blocks = sum(
            block.state is PhysicalBlockState.SHARED and block.active_refcount > 0
            for block in self._blocks
        )
        active_shared_references = sum(
            block.active_refcount
            for block in self._blocks
            if block.state is PhysicalBlockState.SHARED
        )
        allocated = private_blocks + shared_blocks
        reserved = self._protected_blocks()
        private_used_slots = 0
        for table in self._tables.values():
            remaining = table.cache_length
            for block_id in table.block_ids:
                used = min(remaining, self.config.block_tokens)
                if self._blocks[block_id].state is PhysicalBlockState.PRIVATE:
                    private_used_slots += used
                remaining -= used
        used_slots = private_used_slots + shared_blocks * self.config.block_tokens
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
            private_blocks=private_blocks,
            prefix_cache_blocks=shared_blocks,
            evictable_blocks=shared_blocks - active_shared_blocks,
            active_shared_blocks=active_shared_blocks,
            active_shared_references=active_shared_references,
            prefix_cache_evictions=self._prefix_cache_evictions,
            prefix_lookup_requests=self._prefix_lookup_requests,
            prefix_hit_requests=self._prefix_hit_requests,
            prefix_hit_blocks=self._prefix_hit_blocks,
            prefix_hit_tokens=self._prefix_hit_tokens,
            prefix_miss_tokens=self._prefix_miss_tokens,
            reservation_growth_count=self._reservation_growth_count,
            reservation_growth_blocks=self._reservation_growth_blocks,
        )

    def verify_invariants(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Reject any free/owned/reserved accounting or block-table inconsistency."""
        free = set(self._free_blocks)
        if len(free) != len(self._free_blocks):
            _ownership("free block list contains duplicates")
        expected = set(range(self.config.num_blocks))
        resident = {
            block_id
            for block_id, block in enumerate(self._blocks)
            if block.state is not PhysicalBlockState.FREE
        }
        if free & resident or free | resident != expected:
            _ownership("physical blocks must be exactly free XOR resident")
        for block_id, block in enumerate(self._blocks):
            if block.state is PhysicalBlockState.FREE:
                if block_id not in free or any(
                    value is not None
                    for value in (
                        block.owner_request_id,
                        block.prefix_hash,
                        block.token_fingerprint,
                        block.token_block,
                        block.boundary_logits,
                    )
                ):
                    _ownership("FREE block retains resident metadata")
                if block.active_refcount != 0:
                    _ownership("FREE block has active references")
            elif block.state is PhysicalBlockState.PRIVATE:
                if block.owner_request_id is None or block.prefix_hash is not None:
                    _ownership("PRIVATE block must have exactly one owner and no prefix hash")
                if block.active_refcount != 0:
                    _ownership("PRIVATE block cannot have shared references")
            elif (
                block.owner_request_id is not None
                or block.prefix_hash is None
                or block.active_refcount < 0
                or block.token_block is None
                or block.token_fingerprint is None
                or block.boundary_logits is None
            ):
                _ownership("SHARED block metadata is incomplete")
        shared_occurrences: dict[int, int] = {}
        private_occurrences: dict[int, int] = {}
        for table in self._tables.values():
            if len(table.block_ids) != len(set(table.block_ids)):
                _ownership(f"request {table.request_id!r} block table contains duplicates")
            if len(table.block_ids) != self.required_blocks(table.cache_length):
                _ownership(f"request {table.request_id!r} block table length is not logical order")
            private_count = self._private_count(table)
            shared_count = self._shared_count(table)
            if private_count > table.reserved_blocks:
                _ownership(f"request {table.request_id!r} allocation exceeds reservation")
            if table.reserved_blocks + shared_count > table.max_blocks:
                _ownership(f"request {table.request_id!r} current reservation exceeds lifetime max")
            if (
                table.cache_length % self.config.block_tokens
                and table.block_ids
                and self._blocks[table.block_ids[-1]].state is not PhysicalBlockState.PRIVATE
            ):
                _ownership("partial tail block must remain PRIVATE")
            for block_id in table.block_ids:
                block = self._blocks[block_id]
                if block.state is PhysicalBlockState.PRIVATE:
                    if block.owner_request_id != table.request_id:
                        _ownership("PRIVATE block appears in a different request table")
                    private_occurrences[block_id] = private_occurrences.get(block_id, 0) + 1
                elif block.state is PhysicalBlockState.SHARED:
                    shared_occurrences[block_id] = shared_occurrences.get(block_id, 0) + 1
                else:
                    _ownership("request table contains a FREE block")
        if any(count != 1 for count in private_occurrences.values()):
            _ownership("PRIVATE block must appear in exactly one owner table")
        private_ids = {
            block_id
            for block_id, block in enumerate(self._blocks)
            if block.state is PhysicalBlockState.PRIVATE
        }
        if set(private_occurrences) != private_ids:
            _ownership("PRIVATE ownership and request tables differ")
        for block_id, block in enumerate(self._blocks):
            if block.state is PhysicalBlockState.SHARED:
                if shared_occurrences.get(block_id, 0) != block.active_refcount:
                    _ownership("SHARED active_refcount differs from request-table references")
                if self._prefix_index.get(cast("str", block.prefix_hash)) != block_id:
                    _ownership("canonical prefix index differs from SHARED block metadata")
        if len(self._prefix_index) != sum(
            block.state is PhysicalBlockState.SHARED for block in self._blocks
        ):
            _ownership("each prefix hash must have exactly one canonical physical block")
        metrics = self.metrics()
        if metrics.free_blocks + metrics.allocated_blocks != self.config.num_blocks:
            _ownership("free plus resident block count must equal pool size")
        if metrics.reserved_blocks > self.config.num_blocks:
            _ownership("active shared occupancy plus private reservations exceed pool size")

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

    def _acquire_blocks(self, count: int, *, owner_request_id: str) -> list[int]:
        if count == 0:
            return []
        while count > len(self._free_blocks) and self._evict_one():
            pass
        if count > len(self._free_blocks):
            reason = f"allocation requires {count} blocks"
            _capacity(f"{reason} with {len(self._free_blocks)} physically free")
        acquired: list[int] = []
        try:
            for _ in range(count):
                block_id = heapq.heappop(self._free_blocks)
                acquired.append(block_id)
                block = self._blocks[block_id]
                if block.state is not PhysicalBlockState.FREE:
                    _ownership("allocator selected a non-FREE physical block")
                block.state = PhysicalBlockState.PRIVATE
                block.owner_request_id = owner_request_id
                self._allocation_count += 1
                if block_id in self._ever_allocated:
                    self._block_reuse_count += 1
                self._ever_allocated.add(block_id)
        except Exception:
            self._return_blocks(acquired, expected_owner=owner_request_id)
            raise
        return acquired

    def _return_blocks(self, block_ids: list[int], *, expected_owner: str) -> None:
        for block_id in block_ids:
            self._require_private_owner(block_id, expected_owner)
            self._blocks[block_id] = _PhysicalBlock()
            heapq.heappush(self._free_blocks, block_id)
            self._free_count += 1

    def _release_table_blocks(self, table: _MutableRequestCache) -> None:
        for block_id in table.block_ids:
            block = self._blocks[block_id]
            if block.state is PhysicalBlockState.PRIVATE:
                self._return_blocks([block_id], expected_owner=table.request_id)
            elif block.state is PhysicalBlockState.SHARED:
                if block.active_refcount <= 0:
                    _ownership("SHARED block release would create a negative refcount")
                block.active_refcount -= 1
                self._touch(block)
            else:
                _ownership("request release encountered a FREE block")

    def _evict_one(self) -> bool:
        candidates = [
            (block.last_used, block_id)
            for block_id, block in enumerate(self._blocks)
            if block.state is PhysicalBlockState.SHARED and block.active_refcount == 0
        ]
        if not candidates:
            return False
        _last_used, block_id = min(candidates)
        block = self._blocks[block_id]
        prefix_hash = block.prefix_hash
        if prefix_hash is None or self._prefix_index.get(prefix_hash) != block_id:
            _ownership("eviction candidate is not the canonical prefix block")
        del self._prefix_index[prefix_hash]
        self._blocks[block_id] = _PhysicalBlock()
        heapq.heappush(self._free_blocks, block_id)
        self._free_count += 1
        self._prefix_cache_evictions += 1
        self._recent_evictions.append(prefix_hash)
        return True

    def _protected_blocks(self) -> int:
        active_shared = sum(
            block.state is PhysicalBlockState.SHARED and block.active_refcount > 0
            for block in self._blocks
        )
        return active_shared + sum(table.reserved_blocks for table in self._tables.values())

    def _private_count(self, table: _MutableRequestCache) -> int:
        return sum(
            self._blocks[block_id].state is PhysicalBlockState.PRIVATE
            for block_id in table.block_ids
        )

    def _shared_count(self, table: _MutableRequestCache) -> int:
        return sum(
            self._blocks[block_id].state is PhysicalBlockState.SHARED
            for block_id in table.block_ids
        )

    def _longest_prefix(
        self,
        prompt_tokens: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[str, ...]]:
        block_ids: list[int] = []
        hit_hashes: list[str] = []
        for logical_index, prefix_hash in enumerate(self.prefix_hash_chain(prompt_tokens)):
            block_id = self._prefix_index.get(prefix_hash)
            if block_id is None:
                break
            start = logical_index * self.config.block_tokens
            token_block = prompt_tokens[start : start + self.config.block_tokens]
            self._validate_canonical(
                block_id,
                prefix_hash,
                token_block,
                self._token_fingerprint(token_block),
            )
            block_ids.append(block_id)
            hit_hashes.append(prefix_hash)
        return tuple(block_ids), tuple(hit_hashes)

    def _validate_canonical(
        self,
        block_id: int,
        prefix_hash: str,
        token_block: tuple[int, ...],
        fingerprint: str,
    ) -> None:
        block = self._blocks[block_id]
        if (
            block.state is not PhysicalBlockState.SHARED
            or block.prefix_hash != prefix_hash
            or block.token_fingerprint != fingerprint
            or block.token_block != token_block
        ):
            _ownership("prefix hash collision or canonical metadata mismatch")

    def _require_private_owner(self, block_id: int, request_id: str) -> None:
        block = self._blocks[block_id]
        if block.state is not PhysicalBlockState.PRIVATE or block.owner_request_id != request_id:
            _ownership(f"physical block {block_id} is not PRIVATE for request {request_id!r}")

    def _require_prefix_namespace(self) -> PrefixCacheNamespace:
        namespace = self.prefix_cache_namespace
        if namespace is None:
            _invalid("prefix cache operation requires a configured namespace")
        return namespace

    @staticmethod
    def _token_fingerprint(token_block: tuple[int, ...]) -> str:
        encoded = json.dumps(token_block, separators=(",", ":")).encode()
        return hashlib.sha256(b"prefix-token-block\0" + encoded).hexdigest()

    def _touch(self, block: _PhysicalBlock) -> None:
        self._logical_clock += 1
        block.last_used = self._logical_clock

    def _metadata_snapshot(self) -> list[_PhysicalBlock]:
        return [
            _PhysicalBlock(
                state=block.state,
                owner_request_id=block.owner_request_id,
                prefix_hash=block.prefix_hash,
                active_refcount=block.active_refcount,
                last_used=block.last_used,
                token_fingerprint=block.token_fingerprint,
                token_block=block.token_block,
                boundary_logits=block.boundary_logits,
            )
            for block in self._blocks
        ]

    def _restore_metadata(self, snapshot: list[_PhysicalBlock]) -> None:
        self._blocks = [
            _PhysicalBlock(
                state=block.state,
                owner_request_id=block.owner_request_id,
                prefix_hash=block.prefix_hash,
                active_refcount=block.active_refcount,
                last_used=block.last_used,
                token_fingerprint=block.token_fingerprint,
                token_block=block.token_block,
                boundary_logits=block.boundary_logits,
            )
            for block in snapshot
        ]

    def _table_snapshot(self, table: _MutableRequestCache) -> PagedRequestCache:
        return PagedRequestCache(
            request_id=table.request_id,
            block_ids=tuple(table.block_ids),
            cache_length=table.cache_length,
            reserved_blocks=table.reserved_blocks,
            max_blocks=table.max_blocks,
            shared_blocks=self._shared_count(table),
        )

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
