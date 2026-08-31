# AGENTS.md

Compact guide for agents working in this repository. Treat the current Git tree, tests, and
generated evidence as the source of truth.

## Project status

miniGPT is a CPU-first, character-level GPT systems reference lab spanning training, inference,
serving, evidence, and release verification. The project already implements:

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
  tails, APC preservation, deterministic stress, and hash-bound structural evidence;
- Stage 17 whole-request KV-pressure preemption with immediate ownership/refcount cleanup, FIFO
  requeue, cache-only recompute resume, request-local RNG preservation, learned-position overflow
  equivalence, and admission classification that rejects intrinsically impossible heads without
  needless preemption;
- optional Stage 18 lazy KV growth reservation with distinct current/lifetime demand, bounded
  overcommit, growth-before-model-work enforcement, deterministic growth-pressure preemption,
  simulator/evidence coverage, and the Stage 17 recompute path as its correctness fallback;
- Stage 19 production serving configuration that exposes APC strategy, token-budget chunking,
  preemption, lazy reservation, bounded overcommit, and a deterministic atomic runtime manifest on
  the real HTTP process while preserving all legacy defaults.

Stage 20 adds the installable unified CLI and explicit Project Doctor registry; Stage 21 closes v1.0.0 with a single version source, wheel/sdist fresh-install validation, non-self-referential capstone Evidence, independent review, and green Windows/Linux CI. The planned v1 scope is complete. Default work should now be maintenance, compatibility, documentation, or explicitly designed post-v1 research—not another implicit completion Stage.

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
  there is no partial-block COW. Stage 15 can batch paged-history suffix rows by computed suffix
  tokens, but SEQUENTIAL remains the production default because the aggregate strict benchmark is
  `fail`; Stage 15/16 evidence configs opt into BATCHED explicitly. Exact hits still use boundary
  logits and overflow rebuild stays dense. Historical K/V is not materialized on the normal suffix
  path, and model-call reduction alone does not imply a wall-clock performance improvement claim.
- `configs/serving_chunked_prefill.yaml` opts into Stage 16 by setting `max_scheduled_tokens` and
  `prefill_chunk_tokens`. Normal paged decode costs one model-work unit; learned-position overflow
  costs the actual dense rebuild context length. Work that does not fit the remaining tick budget is
  deferred rather than executed over budget, and the deterministic fairness cursor alternates
  deferred decode/prefill work to avoid starvation. Stage 16 evidence is structural/descriptive only
  and makes no wall-clock speedup claim.
- `configs/serving_kv_preemption.yaml` opts into Stage 17 whole-request KV-pressure preemption. Only
  DECODING requests may yield resident paged KV; PREEMPTED requests later reserve private capacity,
  rebuild cache-only history without sampling, and resume with request-local RNG intact. Recompute
  model-token work is charged to the Stage 16 budget. APC shared refs are released on preemption and
  are not reattached across sliding-position recompute. Stage 17 remains `descriptive_only`.
- `configs/serving_lazy_kv_reservation.yaml` opts into Stage 18 lazy KV growth reservation and bounded
  lifetime-demand overcommit. New requests protect prompt capacity, resumed requests protect their
  recompute history, and every decode/recompute operation grows logical and physical protection before
  model work. Growth blockage executes no model work and may trigger deterministic Stage 17
  whole-request preemption of another decoder followed by an immediate growth retry. Legacy full
  reservation remains the default; Stage 18 evidence is `descriptive_only` and makes no wall-clock
  speedup claim.
- Stage 19 exposes these policies on `serve.py` through strict process-level flags.
  `--apc-prefill-strategy batched` remains opt-in, Stage 16 fields must be configured together, and
  `--runtime-manifest` writes a deterministic SHA-bound JSON description without absolute input paths.
  Typed policy resolution, runtime construction, and manifest writing live in
  `src/minigpt/serving_runtime.py`; `serve.py` stays a thin parser/Uvicorn boundary and the HTTP
  request schema is unchanged. `configs/serving_http_lazy_kv.yaml` records the canonical flag example,
  `generate_stage19_evidence.py --source-commit <sha>` builds the hash-bound evidence package under
  `docs/results/serving-runtime-configuration/`, and the Stage 19 verdict is `descriptive_only` with
  no wall-clock improvement and no public-production security-readiness claim.
- Stage 20 unified installable CLI is available as `minigpt` and `python -m minigpt`. Root help/version
  must remain lazy and must not import FastAPI/Uvicorn. Existing command parsers remain authoritative
  behind typed forwarding subcommands.
- `minigpt verify --mode quick|ci` uses an explicit Stage 7A–20 registry. Modern packages retain their
  stage-specific verifiers; Stage 7A alone may declare its historical uncommitted checkpoint through
  `sources.checkpoint`. Artifact entries are always package-local and exact hash-bound members; absolute
  paths, parent traversal, duplicate entries, and unlisted files are rejected. Historical squash merges
  require an exact registry mapping from reviewed source SHA to the merged `main` commit; a stage-level
  merge SHA alone is insufficient and ancestry checks may not be silently disabled.
- Evidence provenance requires a non-shallow repository and full commit objects. GitHub Actions must
  fetch full history before the doctor gate. Stage 20 is release hardening and makes no wall-clock
  speedup claim.
- Do not begin CPU swap, partial-block sharing/COW, speculative decoding, BPE, GPU, LoRA, distributed
  training, `torch.compile`, scheduler priorities, or another optimization without an explicit stage
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

## Stage 21 v1.0 release closure

- Stage 21 is the planned v1.0.0 closure point. Do not add another model/scheduler feature as part of release hardening.
- `src/minigpt/_version.py` is the only authored version value. Setuptools metadata is dynamic; generated `*.egg-info` must remain ignored and untracked.
- `minigpt verify --mode release` builds one wheel and one sdist, inspects wheel membership, fresh-installs the wheel without inherited `PYTHONPATH`, proves distribution metadata and `minigpt.__file__` come from that venv, runs pip check plus module/console help/version, then runs the installed quick doctor against the reviewed checkout.
- The default doctor registry covers Stage 7A–20. Stage 21 capstone evidence is intentionally outside its own registry to avoid self-verification. Capstone gate documents must contain non-empty command records, exact-resume/lifecycle membership, valid full-suite counts, and the four required quality gates; an `exit_code: 0` shell is not sufficient.
- The Stage 21 capstone verifier must also run with the repository root in release generation and CI so it can require a non-shallow source ancestor and exact, non-duplicated coverage of every committed `tests/test_*.py` file.
- `CHANGELOG.md`, `docs/PROJECT_COMPLETION.md`, and `docs/RELEASE_CHECKLIST.md` are required release contracts.
- The annotated `v1.0.0` tag may be created only after the exact main commit passes Windows/Python 3.14 and Linux/Python 3.11 GitHub Actions.
- Post-v1 research features require a new explicit design and evidence policy. v1.0 makes no production-scale or universal wall-clock performance claim.

## Post-v1 Public Playground contracts

- This deployment extension is named **Post-v1 Public Playground**; it is not Stage 22 and must not change the meaning of historical Stage 7A–21 Evidence.
- `minigpt demo-serve` is the only restricted public-demo entrypoint. Ordinary `minigpt serve` keeps its existing defaults and contracts.
- The public demo binds loopback by default. A non-loopback bind requires the explicit unsafe flag; proxy trust and a non-loopback bind must never be combined.
- Preserve the existing `ServingRuntimeConfig`, `EngineRunner`, `ServingEngine`, tokenizer, checkpoint, and completion/SSE contracts. The HTTP policy layer may validate, queue, cancel, and aggregate; it must not become a second scheduler or model owner.
- Public limits, exact CORS origins, kill switch, queue capacity, request timeout, per-client limiter, and independent global limiter must fail closed at startup or request boundaries. CORS is never authentication.
- Do not log Prompt text or token sequences. Public errors, `/demo/info`, `/demo/metrics`, manifests, and logs must not expose traceback, hostname, PID, IP, absolute path, checkpoint path, tokenizer path, or secret.
- Every terminal path, timeout, disconnect, stream close, error, and shutdown must cancel outstanding work as applicable and release the public concurrency lease plus existing KV/request resources.
- `web/` remains build-chain-free native HTML/CSS/JavaScript. Model output uses `textContent`; the page uses no analytics, cookie, Prompt storage, third-party CDN, or user/query API Base override.
- GitHub Pages artifacts contain static assets and JSON-encoded `DEMO_API_BASE` only. An empty variable builds offline-only; checkpoint, tokenizer, Tailscale state, `.env`, local logs, and machine paths never enter the artifact.
- `scripts/start_public_demo_tailscale.ps1` and `scripts/stop_public_demo_tailscale.ps1` may use only loopback, exact PID identity, gitignored logs, and parsed Tailscale status JSON. They must not run `tailscale down`, affect another tunnel/process, modify firewall, or create router port mappings.
- `PublicDemoPolicy.streaming_enabled` defaults to false. The frontend must not request SSE until `/demo/info` returns true, and real Funnel acceptance must prove separate chunks, cancellation, and slot/KV cleanup before enabling it.
- Public abuse control uses only IP/XFF-independent process-wide request/hour and actual generated-token/day quotas. Do not add a trusted client-IP boundary.
- Keep post-v1 tests in existing `tests/test_*.py` files unless a separately reviewed Evidence decision changes the Stage 21 historical test-file contract. Never weaken the Stage 21 verifier to accommodate new tests.
- Public documentation must keep the character-level continuation, no-SLA, non-commercial, no-sensitive-input, free-quota, non-production-security, and no-universal-speedup limitations visible.
