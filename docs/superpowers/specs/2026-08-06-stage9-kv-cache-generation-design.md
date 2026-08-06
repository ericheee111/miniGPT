# Stage 9 — KV Cache Autoregressive Generation Design

Date: 2026-08-06
Status: implemented and verified
Scope: CPU autoregressive inference, KV cache correctness, and isolated inference evidence

## 1. Goal and non-goals

Stage 9 adds an explicit, caller-owned KV cache to generation. Training keeps using the existing
`GPT.forward(token_ids, targets)` path and identical mathematics. The uncached `GPT.generate()`
remains the behavioral baseline.

The stage implements per-layer caches, prompt prefill, incremental decode, cached generation,
overflow re-prefill, an isolated inference benchmark, and a compact evidence package. It does not
implement BPE, GPU kernels, FlashAttention, LoRA, distributed execution, serving, or
`torch.compile`.

## 2. Existing repeated work

`GPT.generate()` appends one sampled token per loop, slices the latest `block_size` tokens, and
calls the ordinary full-sequence forward. Every Transformer layer consequently recomputes Q, K,
and V for every retained historical token. Only the last-position logits are sampled, so repeated
historical K/V projection is avoidable while the window has not moved.

## 3. Public inference contracts

Training-facing `forward()` is unchanged. Three independent inference APIs are added:

```python
last_logits, cache = model.prefill(token_ids)
last_logits, next_cache = model.decode(new_token_ids, cache)
generated = model.generate_cached(...)
```

`prefill()` accepts a non-empty prompt of at most `block_size`, returns logits for only the final
prompt position with shape `[B,1,V]`, and returns one immutable cache entry per Transformer layer.
`decode()` accepts one or more new tokens, requires `past_length + new_length <= block_size`, and
returns logits for all new positions plus a new cache. `generate_cached()` preserves the public
sampling arguments and token results of `generate()`.

Both inference paths are intended for eval-mode use. Cache tensors are detached, do not require
gradients, are not parameters or buffers, and are held only by the caller.

## 4. Cache representation and validation

`LayerKVCache` is a frozen dataclass with:

- `key: [batch, heads, cached_time, head_size]`;
- `value: [batch, heads, cached_time, head_size]`.

`KVCache` is an immutable tuple with exactly `n_layer` entries. Validation rejects mismatched
layer count, rank, key/value shape, batch, heads, head size, cached length, inconsistent lengths,
dtype, device, tensors that require gradients, and lengths outside `1..block_size`.

For element size `S`, cache memory is:

```text
2 * n_layer * batch * n_head * cached_time * head_size * S bytes
```

Because `n_head * head_size == n_embd`, this is also
`2 * n_layer * batch * cached_time * n_embd * S`.

## 5. Attention semantics

The normal attention `forward()` remains unchanged. A separate cached method projects Q/K/V only
for the new hidden states. It concatenates detached historical K/V with new K/V and computes
scores shaped `[B,H,new_time,past_time+new_time]`.

The causal mask is selected with a query offset:

```text
mask[:, :, past_length:past_length+new_length, :past_length+new_length]
```

Thus new query `i` can attend to all past keys and new keys through absolute position
`past_length+i`, but not later new keys. Transformer blocks expose a separate cached method so the
ordinary module call protocol and training forward remain intact.

## 6. Prefill and decode

Prefill embeds prompt positions `0..prompt_length-1`, runs each block once while capturing its K/V,
then returns final-position logits and the complete cache. Its last logits must equal
`forward(prompt)[0][:,-1:,:]` under the same eval model.

Decode validates a coherent cache, embeds new tokens starting at `past_length`, and runs only the
new hidden states through every layer. Each layer reads its own historical K/V and returns a newly
concatenated cache entry. The input cache object and tensors are never modified in place.

K/V are reusable because later tokens need historical keys for similarity and historical values
for the weighted sum. A historical query is not needed again: generation consumes only logits for
new positions. Decode still attends over every cached key, so cache removes repeated projections
and historical MLP work but does not make attention independent of context length.

## 7. Generation and absolute-position overflow

For a prompt at or below `block_size`, cached generation begins with prefill. After sampling, it
uses incremental decode whenever the retained context plus the next input token fits. Once the
context is full and another sampled token makes the window slide, it discards the old cache and
prefills the latest `block_size` tokens.

This re-prefill is required by learned absolute position embeddings. The uncached baseline
renumbers each sliding window to `0..block_size-1`; simply dropping the oldest K/V would leave the
survivors encoded with positions `1..block_size-1`, which is numerically different. Prompts longer
than `block_size` similarly prefill only their latest window while preserving the full returned
prompt.

Sampling is factored through one private helper shared by cached and uncached generation so fixed
generator state, temperature, and top-k consume the same probability tensor and RNG sequence.
`max_new_tokens=0` returns the original tensor without prefill.

## 8. Numerical acceptance

CPU float32 tests use exact equality where the operation order is identical. Cached attention
changes matrix shapes and may select different low-level kernels, so comparisons that are not
bit-identical use a documented `rtol=1e-5, atol=1e-6`. Token sampling equivalence is exact.

Tests cover prefill, one-step and multi-step decode, batch sizes above one, prompt lengths, cache
metadata and byte count, clear invalid-cache errors, no-grad ownership, zero generation, fixed
sampling, near-full prompts, overflow fallback, ordinary training forward, and exact resume.

## 9. Inference benchmark

The benchmark is independent of training timers and Profiler. Canonical cases use batch size one,
prompt lengths 16/32/64/128, generated lengths 8/32/64, and cached/uncached modes, restricted to
`prompt_length + generated_length <= block_size` so they measure incremental decode rather than
overflow re-prefill.

Every replicate runs in a fresh process with deterministic model weights, prompt, and forced-token
workload. Warmup is untimed. Raw evidence records execution order, configuration, Git/environment
identity, worker-lifetime peak RSS, cache bytes, TTFT/prefill time, median decode latency per token,
generated tokens/s, end-to-end time, and every artifact SHA-256. Summaries report median, MAD,
population CV, and replicate count.

Cached prefill time is TTFT before selection of the first forced token. For uncached generation,
TTFT is the first full forward. TPOT is the median interval producing subsequent token logits.
Tokens/s is generated token count divided by end-to-end generation time. Small models and short
prompts may not improve because concatenation, validation, allocations, and small-kernel overhead
can outweigh avoided work.

Strict comparison requires complete compatible environments, equal successful replicate counts,
the configured minimum count, and CV within policy. Only a `pass` verdict permits a performance
improvement claim; otherwise results are `not_comparable` or descriptive only.

## 10. Profiler and evidence

Profiler output is descriptive only. It may explain repeated uncached QKV projection, cached
attention/MLP cost, and hotspot changes with context length, but its timings never feed the
benchmark verdict.

The committed package under `docs/results/kv-cache-generation/` separates numerical equivalence,
functional generation, canonical performance, overflow behavior, cache memory, and profiler
observations. Stage 8 and reference-training evidence are immutable inputs and must not change.
