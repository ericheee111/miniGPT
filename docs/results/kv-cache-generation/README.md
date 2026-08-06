# Stage 9 — KV Cache Autoregressive Generation Evidence

## Outcome

The overall strict comparison is `not_comparable`; five cases exceeded the 5% CV limit, so this package does not claim a strict overall performance improvement.
All 168 fresh-process workers completed, and the descriptive cached median was lower in all 12 canonical cases. Seven cases were individually comparable.

## Correctness and generation semantics

Prefill final logits are exactly equal to ordinary forward. Incremental decode is checked with `rtol=1e-5, atol=1e-6`, and fixed-generator cached/uncached sampling remains token- and RNG-state identical across a `block_size` overflow. Cache tensors are detached, caller-owned, absent from model state, and represented per layer as `[batch, heads, time, head_size]`.

K/V can be cached because every future query still attends to historical keys and consumes their values. Historical Q is not reused. Decode projects only new-token Q/K/V, but its attention still reads all cached keys; the optimization removes repeated historical projection and MLP work; attention is not constant-time.

## Canonical performance

| Case | Uncached E2E ms | Cached E2E ms | Change | Cache bytes | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| p16-g8 | 10.858 | 8.204 | -24.44% | 94208 | pass |
| p16-g32 | 52.682 | 31.919 | -39.41% | 192512 | pass |
| p16-g64 | 127.926 | 63.656 | -50.24% | 323584 | pass |
| p32-g8 | 13.565 | 8.757 | -35.44% | 159744 | not_comparable |
| p32-g32 | 63.741 | 31.857 | -50.02% | 258048 | pass |
| p32-g64 | 151.359 | 62.914 | -58.43% | 389120 | pass |
| p64-g8 | 19.667 | 9.506 | -51.67% | 290816 | pass |
| p64-g32 | 89.515 | 33.015 | -63.12% | 389120 | not_comparable |
| p64-g64 | 202.610 | 64.955 | -67.94% | 520192 | pass |
| p128-g8 | 33.326 | 11.379 | -65.86% | 552960 | not_comparable |
| p128-g32 | 143.432 | 34.905 | -75.66% | 651264 | not_comparable |
| p128-g64 | 318.247 | 68.182 | -78.58% | 782336 | not_comparable |

Negative change is faster. Each mode/case has seven fresh-process replicates, deterministic weights and forced tokens, three untimed warmups, seven measured iterations, batch size one, and no Profiler in canonical timers. Raw JSONL, execution order, config, environment, summaries, and their SHA-256 hashes are committed under `evidence/inference/`.

## Cache memory and overflow

For element size `S`, cache bytes are `2 * layers * batch * cached_time * embedding * S`. The canonical cache value is measured after the final logits-producing decode, so its length is `prompt_length + generated_length - 1`. KV cache trades this memory and concatenation cost for avoided historical computation.

When the learned absolute-position window is full, cached generation does not drop the oldest K/V. It re-prefills the newest `block_size` tokens because uncached sliding windows renumber positions to `0..block_size-1`; old-position K/V would be numerically different.

## TTFT, TPOT, throughput, and profiler

TTFT is the prompt forward/prefill latency before the first forced token. TPOT is represented by median subsequent decode latency. Tokens/s divides generated tokens by end-to-end time. Small models and short prompts may not benefit because cache concatenation, allocation, and small-kernel overhead can exceed avoided work.

The separate P128/G32 profiler is descriptive only. It shows matrix multiplication, batched attention, normalization, and cached concatenation costs; its instrumented times never feed the benchmark verdict.

## Reproduction

```powershell
python benchmark_inference.py --config configs/inference_benchmark_stage9.yaml
python profile_inference.py `
  --config configs/inference_benchmark_stage9.yaml `
  --output reports/inference-profile-stage9/profile-p128-g32.json
python generate_stage9_evidence.py `
  --run-manifest <run_manifest.json> --profile <profile.json>
python generate_stage9_evidence.py --verify
```
