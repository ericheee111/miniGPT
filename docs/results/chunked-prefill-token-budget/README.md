# Stage 16 — Chunked Prefill and Token-Budget Scheduling

Stage 16 bounds prompt work per tick while decode continues beside long prompts.
Prompts advance through block-aligned chunks. Intermediate chunks do not sample or
advance request RNG; only the final prompt chunk produces the first generated token.

The implementation reuses Stage 15 paged-history batched prefill and Stage 14 APC.
Complete APC blocks remain immutable/shared; partial tails remain request-private.
Chunked APC promotion reuses the existing final prompt promotion transaction.

Correctness equivalent to unchunked Stage 15: `True`.
Observed chunk count: `10`.
Decode/prefill interleaving observed: `True`.
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

Source commit: `d77117ad1b9dea3537a76097afdd48a2d8058dd1`.
