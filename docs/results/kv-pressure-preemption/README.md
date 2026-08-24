# Stage 17 — KV-Pressure Preemption + Recompute Resume

Stage 17 lets a DECODING request yield its whole resident paged KV reservation
under FIFO KV pressure, then later rebuild cache-only history and resume.
Recompute never samples and does not advance request-local RNG.

Roomy/pressure logical equivalence: True.
Per-request RNG equivalence: True.
Per-tick actual model-work budget respected: True.
Observed preemptions: 10.
Observed recompute tokens: 68.
Intrinsically impossible logical/physical FIFO heads are rejected without preemption.

APC shared references are released on preemption. Resume intentionally rebuilds
private KV instead of reusing original-position prefix blocks across a sliding
window.

## Performance claim policy

This package records structural scheduling evidence only. The benchmark verdict is
descriptive_only; no wall-clock performance improvement is claimed.

## Scope boundaries

Dynamic/lazy KV reservation, CPU swap, partial-block COW, GPU/CUDA, fused kernels,
and new HTTP request APIs remain outside Stage 17.

Source commit: 5cb003a97086ea9257ca91a79758b6a40eb697f3.
