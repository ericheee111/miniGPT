# AGENTS.md

Compact guide for agents working in this repository. Treat the current Git tree, tests, and
generated evidence as the source of truth.

## Project status

miniGPT is a CPU-first, character-level GPT training and profiling lab. The project already
implements:

- Tiny Shakespeare preparation, character tokenization, and deterministic train/validation splits;
- a custom GPT stack with LayerNorm, causal multi-head attention, MLP, residual blocks, loss, and
  autoregressive generation;
- YAML-driven training, AdamW, warmup/cosine learning-rate scheduling, validation, sampling,
  JSONL/TensorBoard metrics, and CPU memory telemetry;
- checkpoint format v2 with atomic writes, experiment/data identity validation, complete RNG state,
  and exact training resume;
- configurable CPU benchmark and PyTorch Profiler entrypoints;
- isolated Benchmark v2 infrastructure with raw evidence, strict manifests, independent comparison
  policy, deterministic identities, and separate profiler evidence;
- reproducible Stage 7A reference-training evidence;
- a vectorized, mmap-preserving `TokenBatcher` path with Stage 8 batch-only and end-to-end evidence;
- explicit per-layer KV caches, prompt prefill, incremental decode, cached generation with
  learned-position overflow re-prefill, and Stage 9 isolated inference evidence;
- a deterministic FIFO serving control plane with per-request RNG/cache state, admission and cache
  reservations, cancellation/backpressure, metrics, three Stage 10/11 executors, batched prefill/
  decode evidence, and an optional Stage 12 OpenAI-compatible completions HTTP/SSE boundary with a
  single-owner engine thread;
- an optional fixed-block paged KV-cache manager with transactional ownership, block reservations,
  lifecycle cleanup, invariant stress evidence, and a Stage 13B block-aware decode executor that
  avoids normal-path dense historical K/V materialization;
- namespace-bound Automatic Prefix Caching for immutable complete prompt blocks, longest-prefix
  suffix prefill, active refcounts, deterministic zero-ref LRU eviction, and Stage 14 evidence;
- cache-aware batched paged suffix prefill with variable per-row history/suffix lengths, absolute
  learned positions, exact-hit zero-compute scatter, explicit sequential-reference mode, and Stage
  15 structural/performance evidence;
- optional Stage 16 block-aligned chunked prefill with a useful-token budget per tick, decode-first
  interleaving, intermediate chunks that do not sample or advance request RNG, partial final prompt
  tails, APC preservation, deterministic stress, and hash-bound structural evidence.

Do not recreate or replace the existing GPT, trainer, tokenizer, optimizer, checkpoint, or
exact-resume systems. Extend their public contracts only when the active stage requires it.

## Supported environment

- Python `>=3.11,<3.15`; the toolchain targets Python 3.11 syntax.
- PyTorch CPU is the canonical runtime. CUDA/GPU work is outside the current scope.
- Windows is the primary development environment.
- GitHub Actions runs the quality gates on Windows/Python 3.14 and Linux/Python 3.11.

Install from `pyproject.toml`; the empty `requirements.txt` is not an install source:

```powershell
python -m pip install -e ".[dev,report]"
python -m pip check
```

The distribution is `minitrain-gpt`; the import package is `minigpt`.

## Checkpoint and resume rules

- v2 checkpoints are the only supported training-resume source.
- v2 preserves model/optimizer state, completed step, resolved experiment config, Python/NumPy/
  PyTorch RNG, train/validation batcher RNG, sample generator RNG, tokenizer/data SHA-256, and
  format version.
- v1 checkpoints remain loadable for configuration, weights, and generation, but training resume
  raises `LegacyCheckpointResumeError`.
- `training.max_steps` defines the complete experiment, `training.lr_decay_steps` defines the
  scheduler horizon, and `--run-until-step` is only the current process boundary.
- Do not add inexact or best-effort resume paths without a new design decision.

## Reference training evidence

The committed Stage 7A evidence is under
`docs/results/reference-training/`. Its `artifact_manifest.json` binds the report to Git, resolved
config, tokenizer/data, raw metrics, provenance, samples, and the uncommitted checkpoint by
SHA-256. Do not hand-edit or silently regenerate these artifacts. The checkpoint, raw outputs, and
dataset remain gitignored.

## Benchmark and profiler

- `python benchmark.py --config configs/benchmark_smoke.yaml` runs benchmark v1.
- `python profile_model.py --config configs/benchmark_smoke.yaml` runs an explicitly separate
  operator profile.
- Profiler timings include instrumentation overhead and must never be used as benchmark throughput.
- Benchmark v2 infrastructure and Stage 8 batcher/mmap optimization are implemented. The committed
  evidence distinguishes isolated batch-path gains from end-to-end training noise.
- `python benchmark_inference.py --config configs/inference_benchmark_stage9.yaml` runs the separate
  fresh-process cached/uncached inference matrix. `python profile_inference.py ...` remains
  descriptive only, and `docs/results/kv-cache-generation/` contains the hash-bound Stage 9 package.
- `python simulate_serving.py --config configs/serving_single_request.yaml` runs the Stage 10 offline
  control-plane simulator. It advances requests at iteration level while the reference executor
  still calls the model per request; it is not tensor-level continuous batching.
- `python serve.py --checkpoint ... --tokenizer ... --executor continuous` loads one CPU model and
  runs the Stage 12 optional HTTP service. `python benchmark_server.py ...` is an end-to-end HTTP
  system benchmark and must remain separate from Stage 11 executor benchmarks.
- `python serve.py ... --executor paged_attention --kv-cache-backend paged` selects the Stage 13B
  block-aware decode path. Initial/overflow prefill remains dense; do not describe it as an all-path
  paged kernel or claim speedup from descriptive single-machine evidence.
- `--prefix-cache` additionally enables Stage 14 full-block APC. Partial tails remain private and
  there is no partial-block COW. Stage 15 batches paged-history suffix rows by computed suffix tokens
  while retaining a sequential APC reference mode; exact hits still use boundary logits and overflow
  rebuild stays dense. Historical K/V is not materialized on the normal suffix path. The Stage 15
  fresh-process benchmark has strict verdict `fail`, so model-call reduction and avoided work do not
  imply a wall-clock performance improvement claim.
- `configs/serving_chunked_prefill.yaml` opts into Stage 16 by setting `max_scheduled_tokens` and
  `prefill_chunk_tokens`. Decode rows consume one useful-token budget unit first; remaining budget is
  assigned FIFO to block-aligned prompt chunks. Stage 16 evidence is structural/descriptive only and
  makes no wall-clock speedup claim.
- Do not begin partial-block sharing/COW, KV-pressure preemption, speculative decoding, BPE, GPU,
  LoRA, distributed training, `torch.compile`, or another optimization without an explicit stage
  decision.

## Quality gates

Run all gates from an editable installation:

```powershell
python -m pip check
ruff format --check src tests
ruff check src tests
basedpyright
pytest
```

Ruff selects `ALL` with project-specific exclusions. basedpyright uses `typeCheckingMode = "all"`
for `src` and `tests`. Tests use Given/When/Then comments, strict pytest configuration, and no real
network where a local `file://` source suffices.

## Artifacts and commits

`data/raw/`, `data/processed/`, `checkpoints/`, `outputs/`, `reports/`, `runs/`, profiler traces,
model weights, virtual environments, and tool caches are gitignored. Do not commit machine-specific
benchmark runs or large traces unless a separately reviewed evidence package explicitly requires
them.

Commit source, configs, tests, schemas, design/implementation docs, and small fixed fixtures. Keep
logical changes in separate commits, preserve unrelated user changes, and follow the repository's
Chinese commit-message style unless the user requests otherwise.
