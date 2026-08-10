# Stage 16 — Chunked Prefill and Token-Budget Scheduling

Stage 16 bounds prompt work per tick while decode continues beside long prompts.
Prompts advance through block-aligned chunks. Intermediate chunks do not sample or
advance request RNG; only the final prompt chunk produces the first generated token.

The implementation reuses Stage 15 paged-history batched prefill and Stage 14 APC.
Normal paged decode is charged one work unit; learned-position overflow is charged
the actual dense rebuild context length. Work that does not fit the remaining budget
is deferred by the FIFO fairness cursor instead of executing over budget.
Stage 15 batched APC suffix prefill remains an explicit config opt-in,
not the production default.
Complete APC blocks remain immutable/shared; partial tails remain request-private.
Chunked APC promotion reuses the existing final prompt promotion transaction.

Correctness equivalent to unchunked Stage 15: `True`.
Observed chunk count: `10`.
Decode/prefill interleaving observed: `True`.
Overflow rebuild work observed and budgeted: `True`.
Per-tick budget respected: `True`.
Partial final chunk observed: `True`.

## Performance claim policy

This package records structural scheduling evidence only. The benchmark verdict is
`descriptive_only`: no fresh-process timing comparison is included, and no wall-clock
performance improvement is claimed.

## Scope boundaries

This remains a Python/PyTorch reference implementation. There is no partial-block
copy-on-write, KV-pressure preemption, speculative decoding, GPU/CUDA path, custom
fused kernel, or new HTTP API.

Source commit: `9ab44bef0114173ad55c7a0d3b39c7e94760e535`.
