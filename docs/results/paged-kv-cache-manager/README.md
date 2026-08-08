# Stage 13A — Paged KV Cache Memory Manager

## Outcome

Stage 13A keeps `dense` as the reference and adds a fixed physical K/V block pool,
per-request block tables, block reservations, and transactional prefill/append/rebuild.
Every terminal lifecycle path releases request ownership and reserved capacity.
The physical layout is `[layers, num_blocks, heads, block_tokens, head_size]` for both K
and V; one block ID addresses every layer for the same logical token interval.

Reservation protects worst-case request capacity while allocation tracks blocks actually
written. Fixed blocks do not have traditional contiguous external fragmentation; reported
fragmentation is unused token slots in the tail block.

## Correctness and capacity

| Scenario | Equivalent | Completed | Cancelled | Failed | Peak allocated | Peak reserved | Reuse |
|---|---:|---:|---:|---:|---:|---:|---:|
| paged-normal-burst | True | 4 | 0 | 0 | 8 | 9 | 4 |
| paged-tiny-pool | None | 2 | 0 | 1 | 2 | 2 | 2 |
| paged-repeated-reuse | True | 5 | 0 | 0 | 2 | 2 | 8 |
| paged-cancellation-churn | True | 1 | 3 | 0 | 3 | 7 | 2 |
| paged-capacity-failure-rollback | None | 2 | 0 | 1 | 2 | 2 | 2 |
| paged-block-size-overflow | True | 2 | 0 | 0 | 8 | 8 | 0 |
| paged-fragmentation-heavy | True | 4 | 1 | 0 | 2 | 6 | 4 |

Five capacity-fitting workloads produced identical logical correctness hashes across
dense and paged storage. All 7 scenarios ended with zero
owned blocks and reservations. Tiny-pool and capacity-failure runs are paged-only
because deliberate physical rejection is storage-specific.

## Allocator stress and rollback

The deterministic stress run executed 3000 operations; invariants were
checked after every mutation. It covered allocation, append, rebuild, free, reuse,
and tail waste. Its trace hash is `368f9be5a1ba61cc7c4e4432bcfe3dfa7460d06285953f30f5e2d6c31b4ede81`.
Tests inject partial allocation, prefill write, append write, and overflow rebuild
failures, and cover FIFO pressure, HTTP disconnect, and graceful shutdown cleanup.

## Performance boundary

Normal decode still materializes compact dense K/V for the Stage 11 executor, then
writes only the newly appended token back. This is not PagedAttention; no speedup is claimed.
A slower paged backend is expected until Stage 13B adds block-aware attention and
measures it independently.
