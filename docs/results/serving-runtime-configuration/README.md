# Stage 19 — Production Serving Configuration + Runtime Manifest

Stage 19 exposes the Stage 15-18 scheduler and paged-cache controls on the
real HTTP CLI behind one typed package-level runtime builder, while the
legacy dense/continuous service keeps its previous defaults.

Legacy defaults preserved: True.
Invalid combinations rejected: 8.
Deterministic manifest bytes: True.
Manifest SHA-256: 7ed3fa4df355808d06c4a6622202d94b2e7f3810604d23575aca2bfaa78fefea.
Atomic replacement verified: True.
Checkpoint/tokenizer SHA-256 identities are bound in the manifest.
Idle runtime resources released: True.

The completion request schema is unchanged. serve.py remains a thin
parser/Uvicorn boundary over minigpt.serving_runtime.

## Performance claim policy

The benchmark verdict is descriptive_only. No wall-clock performance
improvement is claimed and no public-production security readiness is
claimed.

## Scope boundaries

New HTTP endpoints, authentication, GPU kernels, CPU swap, partial-block
COW, and speculative decoding remain outside Stage 19.

Source commit: 92032f14ed493967997bdf9ea779fb233679d65e.
