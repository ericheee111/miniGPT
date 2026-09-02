# Post-v1 Story Forge Product Design

**Date:** 2026-09-02
**Status:** implementation and review
**Boundary:** post-v1 product/research extension; not Stage 22

## 1. Product objective

The v1 miniGPT reference system is complete. Story Forge changes the public demonstration from an unconstrained Shakespeare character continuation into a controlled, inspectable micro-adventure product built on the same training, exact-resume, serving, KV-cache, release, and evidence infrastructure.

The product has three complementary experiences:

1. **Story Forge** — select a world, tone, and theme, generate three deterministic branches concurrently, choose one, and continue for at most four rounds.
2. **Prediction Lab** — inspect next-token distributions and sequence surprisal without sampling or mutating request state.
3. **Systems Lab** — replay deterministic scenarios derived from committed Stage 11, 14, 17, and 18 evidence while the live model is online or offline.

The product is not a general chat assistant. It does not claim factuality, semantic understanding, authorship detection, production readiness, continuous availability, GPU parity, or universal performance improvement.

## 2. Delivery phases

### Phase A — runtime and API closure

- Keep `/v1/completions` backward compatible.
- Add `POST /demo/story/branches` with exactly three request-local RNG streams.
- Add `POST /demo/predict/next` and `POST /demo/predict/score` through the EngineRunner owner thread.
- Add EOS-aware serving while preserving legacy requests with `eos_token_id=None`.
- Bound body size, prompt size, generated tokens, queueing, concurrency, time, hourly requests, and daily generated tokens.
- Make every partial-submit, timeout, disconnect, stream close, and shutdown path release capacity, quota, futures, and KV ownership exactly once.

### Phase B — public web product

- Replace Shakespeare-first copy and controls with Story Forge.
- Keep all user and model text in DOM text nodes; never interpolate them into HTML.
- Use one multiplexed SSE request for three branches. Token events carry the complete decoded branch snapshot because individual ByteLevel BPE tokens are not guaranteed UTF-8 boundaries.
- Keep story history only in the current page process and cap it at four rounds.
- Provide local copy/download without server persistence.
- Keep Prediction and Story controls disabled when the local backend is offline while Systems Lab remains usable.

### Phase C — deployment and model validation

- Validate model family, tokenizer schema, vocabulary, special IDs, configuration, parameter count, hashes, and cached/uncached deterministic generation before startup.
- Bind only to loopback.
- Keep the existing Tailscale Funnel, CodexPro, and ngrok boundaries independent.
- Validate on an alternate local port before any operator-controlled cutover.
- Keep model weights, prepared arrays, runtime state, logs, and screenshots untracked.

### Phase D — evidence and release

- Release as miniGPT 1.1.0 after source review.
- Preserve Stage 1–21 and v1.0 evidence semantics.
- Publish a separate exact-membership, SHA-256-bound Story Forge product evidence package.
- Verify the final source and evidence in a detached fresh worktree, feature-branch CI, and `main` CI before the public page is switched.

## 3. Story contract

### Controls

Worlds:

- `space` — Space Expedition
- `forest` — Enchanted Forest
- `robot` — Robot Workshop
- `mystery` — Cozy Mystery

Tones:

- `adventurous`
- `mysterious`
- `warm`
- `funny`

Themes:

- `discovery`
- `friendship`
- `logic`
- `courage`

### Request

A branch action includes controls, an optional opening/history string, a base seed, exactly three branches, a bounded output length, and an optional stream flag. The server frames the canonical BOS/control/story prefix once, retains recent whole BPE tokens when history exceeds the model window, derives three stable seeds, and submits all branches before awaiting any result.

### Response

Non-stream responses retain branch order and report each branch independently. Stream responses multiplex `branch_started`, `token`, and `branch_finished` events, followed by one `done` event and one `[DONE]` sentinel. The `text` field on each token event is a complete decoded branch snapshot, not a token delta.

## 4. Prediction contract

Prediction calls are read-only model observations. They execute between engine commands on the single owner thread and do not sample, advance request RNG, allocate request KV, or alter request lifecycle.

- Next-token output is bounded to top-k ≤ 10 and includes token ID, safe display label, logit, and probability.
- Sequence scoring reports per-token negative log-likelihood and surprisal plus user-text-only aggregates; control-prefix metrics are separated.
- Byte fragments that cannot be displayed independently use a stable `<token:N>` label.
- Likelihood and perplexity are explicitly described as measurements under this model distribution.

## 5. Systems Lab provenance

Only fixed scenarios generated from committed evidence are accepted. The browser cannot upload simulator YAML or choose arbitrary resource limits. Every asset records:

- source evidence path and source commit;
- claim level;
- normalized events, lanes, and resource snapshots;
- checked invariants;
- a link back to the committed evidence.

The replay is labeled **Recorded verified scenario** and remains distinct from live request telemetry.

## 6. Security and privacy

- The backend binds to `127.0.0.1` unless an explicit unsafe development override is supplied.
- CORS allows only configured origins and is not treated as authentication.
- Global quotas are independent of client IP and apply equally to stream and non-stream calls.
- Full prompts and stories are not logged by the public boundary.
- API documents and metrics omit paths, hostnames, PIDs, IP addresses, secrets, prompts, and hardware identity.
- OpenAPI, Swagger, and Redoc are disabled.
- Static assets contain no API credential or model weight.
- The browser does not use cookies, analytics, or persistent prompt storage.

## 7. Acceptance gates

The extension is ready to merge only when all of the following hold:

- legacy character-tokenizer, training, serving, and HTTP tests remain green;
- Story branch outputs, EOS, quotas, cancellation, and resource cleanup are covered;
- Prediction math matches independent reference computation and runs on the owner thread;
- ByteLevel BPE streaming has no replacement-character corruption;
- the online and offline static builds are deterministic and contain every required scenario asset;
- desktop and mobile browser smoke tests have no console errors;
- selected checkpoint/tokenizer validation and localhost Story/Prediction/SSE smoke pass;
- Ruff formatting/lint, BasedPyright, full pytest, Project Doctor, evidence verifiers, detached fresh checkout, feature CI, and `main` CI pass;
- no Blocker or Important review finding remains.
