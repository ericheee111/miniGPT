# Changelog

All notable project changes are documented here. miniGPT follows semantic versioning from the v1.0 release onward.

> **Release status:** the v1.0.0 code, Evidence, packaging, independent review, and cross-platform CI baseline is complete on `main` as of 2026-08-27. The annotated `v1.0.0` Git tag has not yet been published.

## 1.0.0 — 2026-08-27

### Added

- Character-level data preparation and reversible tokenizer artifacts.
- A custom GPT implementation with learned absolute positions, causal attention, MLP blocks, loss, and autoregressive generation.
- YAML-driven CPU training with AdamW, warmup/cosine scheduling, validation, sampling, JSONL/TensorBoard metrics, and memory telemetry.
- Checkpoint format v2 with atomic writes, data/config identity, complete RNG state, and exact interrupted/resumed training equivalence.
- Fresh-process Benchmark v2 infrastructure, independent comparison policy, profiler separation, raw samples, environment identity, and hash-bound reports.
- Vectorized mmap-preserving token batching.
- Explicit per-layer KV cache, prompt prefill, incremental decode, and learned-position overflow rebuild.
- Deterministic multi-request serving control plane with per-request RNG, cancellation, failure isolation, metrics, and simulator.
- Decode and prompt continuous batching executors.
- OpenAI-compatible completions subset with HTTP/SSE, bounded backpressure, disconnect cancellation, and a single engine-owner thread.
- Transactional fixed-block paged KV cache manager and block-aware reference decode.
- Namespace-bound Automatic Prefix Caching with immutable shared blocks, refcounts, collision defense, promotion, and deterministic LRU eviction.
- Cache-aware batched APC suffix prefill.
- Block-aligned chunked prefill with actual model-token budget and deterministic fairness.
- Whole-request KV-pressure preemption with resource release and cache-only recompute resume.
- Lazy KV growth reservation with bounded lifetime overcommit and growth-pressure preemption.
- Real serving runtime configuration for Stage 15–18 controls and an atomic deterministic runtime manifest.
- Installable `minigpt` / `python -m minigpt` CLI and Stage 7A–20 project doctor.
- Stage 21 release closure with wheel/sdist fresh-install validation and v1 capstone evidence.

### Correctness and evidence

- Windows/Python 3.14 and Linux/Python 3.11 quality gates.
- Strict Ruff formatting/linting and basedpyright `all` mode.
- Deterministic unit, integration, lifecycle, stress, HTTP subprocess, allocator, simulator, and evidence tests.
- Explicit source-commit ancestry and exact artifact membership for modern evidence packages; historical squash exceptions bind both the reviewed source and merged commit.
- A documented legacy Stage 7A `sources.checkpoint` exception without weakening committed artifact paths, uniqueness, membership, or hashes.
- Fresh-wheel validation proves metadata/help/version agreement and that imports resolve inside the isolated wheel environment.
- Capstone evidence rejects empty `exit_code: 0` shells and validates command records, focused test membership, full-suite counts, and quality-gate coverage.
- Capstone verification binds its source to a non-shallow ancestor and requires exact, non-duplicated coverage of every committed test file; squash provenance rejects any source other than the explicitly reviewed SHA even when that SHA is otherwise an ancestor.
- Bounded claim policy separating semantic correctness, structural work reduction, descriptive timing, and strict performance verdicts.

### Performance claim boundaries

Several stages demonstrate structural reductions or configuration improvements, but Stage 13B–21 evidence remains `descriptive_only` where strict wall-clock improvement was not established. v1.0 does not claim production-scale throughput, GPU parity, or universal speedup.

### Known limitations

- Character tokenizer rather than BPE/SentencePiece.
- CPU/PyTorch reference kernels rather than fused CUDA PagedAttention.
- Complete-block APC only; no partial-block copy-on-write.
- Dense learned-position overflow rebuild.
- Whole-request recompute instead of swap/offload.
- Single-machine, small-model experimental workloads.
- Completions API subset without public-service authentication or multi-tenant policy.
