# Stage 8 — TokenBatcher / mmap Optimization Evidence

## Outcome

Stage 8 removes the construction-time full-corpus `uint16` to `int64` copy and the Python row loop. The candidate keeps the read-only mmap, gathers selected windows, performs one local `int64` conversion, and returns overlapping read-only contract views from one tensor owner.

The candidate used `3d6ae83e5c0b37a423d60abfb73377cf02fab51f`. The new alternating batch-only rerun completed all 112 requested worker replicates, and both candidate runs had lower medians than both baseline runs in every case. However, at least one side of every comparison exceeded the 5% CV limit, so all six strict comparison verdicts are `not_comparable`. This package therefore supports only a descriptive same-host observation, not a stable performance claim.

The earlier B4/T128 `-10.10%` and B16/T256 candidate `4.97%` CV values are not carried forward as conclusions: their raw replicates were not committed and the new same-code runs did not reproduce that stability.

## Batch-only method and provenance

Runs used the host-specific i7-14700 config, a 1,000,000-token read-only `uint16` mmap, 500 warmup batches, 20,000 measured batches, 7 fresh worker processes per case, and the order `baseline A → candidate A → baseline B → candidate B`. Each run manifest binds raw JSONL, resolved config, environment, execution order, CSV and Markdown summaries by SHA-256 and byte size. The strict loader recomputes median, MAD, population CV, tokens/s, worker-lifetime peak RSS and stability before comparison.

Comparison policy SHA-256: `36ddd5faa00837d921a9e46f9085cf82d557e3a614f5c1108aa8fe936e92876c`. Environment mismatches: `[]` for baseline A versus candidate A; all other committed comparisons also report an empty mismatch list.

## Four-run batch-only results

Every numeric cell below is rendered from the committed comparison JSON. Times are medians in milliseconds; CV is population CV in percent.

| Case | Base A ms | Base A CV | Cand A ms | Cand A CV | Base B ms | Base B CV | Cand B ms | Cand B CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B32 / T128 | 0.026737 | 5.52 | 0.011929 | 3.21 | 0.027124 | 5.65 | 0.011830 | 7.15 |
| B16 / T128 | 0.018668 | 2.21 | 0.011055 | 4.62 | 0.020496 | 10.68 | 0.011198 | 2.33 |
| B4 / T128 | 0.012379 | 6.52 | 0.010003 | 7.35 | 0.014059 | 8.97 | 0.010232 | 5.81 |
| B16 / T256 | 0.018956 | 12.56 | 0.011924 | 10.50 | 0.021495 | 20.39 | 0.011467 | 5.89 |

| Comparison | Descriptive batch-time change range | Strict verdict |
| --- | ---: | --- |
| baseline-a → candidate-a | -55.38% to -19.19% | not_comparable |
| baseline-a → candidate-b | -55.75% to -17.34% | not_comparable |
| baseline-b → candidate-a | -56.02% to -28.85% | not_comparable |
| baseline-b → candidate-b | -56.39% to -27.22% | not_comparable |

Negative change is faster. All 16 candidate-versus-baseline case medians are lower, but `not_comparable` prevents converting that consistency into a pass or an improvement claim.

## Same-code cross-run drift

| Comparison | Case | Median change | CV change |
| --- | --- | ---: | ---: |
| baseline-a-vs-baseline-b | char-gpt-b32-t128 | +1.45% | 5.52% → 5.65% |
| baseline-a-vs-baseline-b | char-gpt-b16-t128 | +9.79% | 2.21% → 10.68% |
| baseline-a-vs-baseline-b | char-gpt-b4-t128 | +13.57% | 6.52% → 8.97% |
| baseline-a-vs-baseline-b | char-gpt-b16-t256 | +13.40% | 12.56% → 20.39% |
| candidate-a-vs-candidate-b | char-gpt-b32-t128 | -0.83% | 3.21% → 7.15% |
| candidate-a-vs-candidate-b | char-gpt-b16-t128 | +1.30% | 4.62% → 2.33% |
| candidate-a-vs-candidate-b | char-gpt-b4-t128 | +2.29% | 7.35% → 5.81% |
| candidate-a-vs-candidate-b | char-gpt-b16-t256 | -3.83% | 10.50% → 5.89% |

Baseline drift is materially larger than candidate drift in several cases, which confirms that this host/run method was not stable enough for a strict verdict.

## Reference training-step evidence and preconditioning

The retained historical training-step comparisons report `['pass', 'pass']` against the two baselines. Those runs predate Benchmark v2 schema 3: their resolved config did not bind the reported external 120-second preconditioning step. They are therefore retained as legacy descriptive evidence only, and this report no longer claims that their exact resolved config covered that manual operation.

Benchmark v2 schema 3 now records preconditioning in the resolved config, methodology identity and run manifest. The i7-14700 Stage 8 config enables a 120-second training-step precondition, so the reproduction command actually performs and records it.

## Reproduction

```powershell
python -m pip install -e ".[dev,report]"
python benchmark_batcher.py --config configs/batcher_benchmark_i7_14700_stage8.yaml
python compare_batcher_benchmarks.py `
  --baseline <baseline-run-manifest> `
  --candidate <candidate-run-manifest> `
  --policy configs/benchmark_v2_comparison.yaml
python benchmark_v2.py --config configs/benchmark_v2_i7_14700_stage8.yaml
python profile_benchmark_v2.py --config configs/benchmark_v2_stage8_profile.yaml
python generate_stage8_evidence.py
```

`benchmark_v2_reference.yaml` and `batcher_benchmark_reference.yaml` remain portable templates with no host affinity. The explicitly named i7-14700 configs carry this host's affinity and Stage 8 measurement settings.

## Evidence and limitations

The committed batch-only package contains all four small raw JSONL files and every artifact needed to independently recompute raw → summary → comparison → report. Worker-lifetime peak RSS includes imports, mmap construction, warmup and measurement. The microbenchmark is not a model benchmark, and no result establishes statistical significance, cross-machine superiority, or GPU behavior.

Stage 8 is not ready for a performance-evidence PR claim: the batch-only reruns failed the stability policy, and the historical end-to-end preconditioning was not bound by its resolved config. The implementation and evidence machinery can still be reviewed independently of that withdrawn claim.
