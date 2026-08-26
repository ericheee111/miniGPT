# Stage 19 — Production Serving Configuration and Runtime Manifest

## Goal

Stage 19 closes the gap between the advanced serving scheduler implemented in Stages 15–18 and the real HTTP process started by `serve.py`.

Before this stage, Automatic Prefix Caching can be enabled by `serve.py`, but cache-aware APC batching, token-budget chunked prefill, KV-pressure preemption, lazy KV reservation, and bounded overcommit are primarily exercised through simulator or evidence configuration. Stage 19 makes those capabilities explicitly selectable in the real CPU HTTP runtime while preserving all legacy defaults.

Stage 19 is a configuration, validation, and provenance stage. It does not add a new HTTP request field and makes no wall-clock performance claim.

## Design principles

1. Existing `serve.py` invocations keep their current behavior.
2. Advanced scheduler features are opt-in and fail closed on ambiguous combinations.
3. CLI resolution lives in an importable package module so tests do not need to start Uvicorn.
4. The runtime manifest is deterministic, excludes machine-specific absolute paths, and is written atomically only after the runtime is valid.
5. Stage 15 batched APC prefill remains explicitly opt-in; sequential remains the default.
6. The same `SchedulerConfig`, `EngineConfig`, and `PagedAttentionExecutor` contracts remain the final source of scheduler truth.

## CLI surface

`serve.py` gains:

```text
--max-scheduled-tokens INT
--prefill-chunk-tokens INT
--kv-preemption
--lazy-kv-reservation
--kv-overcommit-ratio FLOAT
--apc-prefill-strategy {sequential,batched}
--runtime-manifest PATH
```

Defaults preserve the Stage 12–14 runtime:

```text
max_scheduled_tokens = null
prefill_chunk_tokens = null
kv_preemption = false
lazy_kv_reservation = false
kv_overcommit_ratio = 1.0
apc_prefill_strategy = sequential
runtime_manifest = null
```

## Resolved runtime options

A package-level immutable `ResolvedServingRuntime` value contains:

- executor name;
- KV backend;
- prefix-cache enablement;
- APC prefill strategy;
- scheduler configuration;
- runner configuration;
- optional paged KV configuration;
- optional runtime-manifest destination.

Resolution receives the model `block_size` because chunk alignment and minimum budget depend on the learned-position window.

## Validation matrix

The resolver rejects:

- only one of `max_scheduled_tokens` and `prefill_chunk_tokens`;
- chunked scheduling without `paged_attention` plus paged KV;
- chunk size not aligned to paged block tokens;
- chunk size greater than model block size;
- scheduled-token budget below the engine minimum;
- KV preemption without token-budget scheduling or without direct paged execution;
- lazy reservation without preemption or without direct paged execution;
- non-finite or sub-one overcommit ratio;
- a non-1.0 ratio when lazy reservation is disabled;
- prefix cache without direct paged execution;
- batched APC strategy when prefix cache is disabled;
- batched APC strategy on a non-paged executor.

`SchedulerConfig` and `ServingEngine` still perform their own validation; Stage 19 does not bypass those checks.

## Executor construction

`PagedAttentionExecutor` receives the resolved `APCPrefillStrategy`. Other executors reject non-default APC options through resolution rather than silently ignoring them.

## Runtime manifest

When `--runtime-manifest` is supplied, `build_runtime()` writes a stable JSON document only after model, allocator, executor, engine, and runner construction succeeds.

The manifest contains:

```text
schema_version
project_version
checkpoint_sha256
tokenizer_sha256
model_config
block_size
cpu_num_threads
executor
kv_cache_backend
paged_kv_cache
prefix_cache_enabled
apc_prefill_strategy
scheduler
runner
```

The manifest intentionally omits:

- checkpoint/tokenizer absolute paths;
- hostname, user name, PID, current time, or temporary directory;
- credentials or environment variables;
- benchmark or speedup claims.

JSON uses sorted keys, two-space indentation, UTF-8, and one trailing LF. Writing uses a temporary file in the destination directory followed by `os.replace`; any exception removes the temporary file and leaves the old destination untouched.

## HTTP compatibility

The Stage 12 endpoints and request schema are unchanged:

```text
GET /healthz
GET /v1/models
POST /v1/completions
```

All scheduler choices remain process-level policy rather than per-request input.

## Evidence

The committed Stage 19 package lives under:

```text
docs/results/serving-runtime-configuration/
```

It proves:

1. legacy CLI defaults resolve unchanged;
2. Stage 18 options reach the real `ServingEngine` and `PagedAttentionExecutor`;
3. the runtime manifest is deterministic and hash-bound;
4. invalid combinations are rejected;
5. localhost HTTP completion works with the lazy paged runtime;
6. no new HTTP request field is introduced;
7. the verdict remains `descriptive_only` and `wall_clock_performance_improvement=false`.

## Scope boundaries

Stage 19 does not implement:

- a new public HTTP endpoint;
- dynamic request priorities;
- API authentication or public-internet hardening;
- GPU, CUDA, fused kernels, swap, COW, or speculative decoding;
- a wall-clock performance claim.
