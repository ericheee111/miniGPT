# Stage 18 — Lazy KV Growth Reservation + Controlled Overcommit

Stage 18 protects current KV capacity at admission while retaining a bounded
full-lifetime demand. Capacity grows before model work, and Stage 17
whole-request preemption breaks deterministic growth pressure.

Roomy/full logical equivalence: True.
Per-request RNG equivalence: True.
Intrinsic logical/physical admission failures rejected without preemption: True.
Same-pool initial active requests (lazy/full): 2/1.
Per-tick actual model-work budget respected: True.
Observed reservation growths: 8.
Observed growth-pressure preemptions: 1.

APC shared references are released on preemption. Resume rebuilds private
KV rather than reusing original-position shared prefix blocks.

## Performance claim policy

The benchmark verdict is descriptive_only. No wall-clock performance
improvement is claimed.

## Scope boundaries

CPU swap, partial-block COW, GPU/CUDA, fused kernels, scheduler priorities,
and new HTTP request APIs remain outside Stage 18.

Source commit: 58c919f8cacff48c6d94d24e18b76290c3de7a2e.
