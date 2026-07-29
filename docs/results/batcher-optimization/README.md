# Stage 8 — TokenBatcher / mmap Optimization Evidence

## Outcome

Stage 8 removes the construction-time full-corpus `uint16` to `int64` copy, the Python row loop in
`next_batch()`, and the two overlapping NumPy-to-PyTorch copies. `TokenBatcher` now retains the
read-only mmap, exposes a zero-copy sliding-window view, gathers only the selected rows, converts
only that local batch to `int64`, creates one tensor owner, and returns shifted `x/y` views.

The isolated batch-only benchmark found a clear improvement in all four cases. The complete
training-step matrix passed the independent comparison policy against both accepted same-code
baselines, but its descriptive changes stayed inside the observed machine noise. The honest
conclusion is therefore:

- **batch-only path:** measurably faster;
- **end-to-end training:** no distinguishable improvement or regression;
- **batch-only worker memory:** about 7.5–8.0 MiB lower lifetime peak RSS;
- **end-to-end worker memory:** no reliable reduction because model/optimizer allocations dominate.

No claim in this report applies to another machine, GPU training, a different model, or a different
data format.

## Measurement method

The documented host was Windows with an Intel Core i7-14700 and CPU-only PyTorch. Accepted
reference runs used the exact resolved config SHA-256
`73f82ff3bbb339904babcf165d7d3d9e1f2483f394eba298b42c13fefbb3f6bb`:

- Windows High Performance power plan;
- logical CPU affinity `0..15` (the host's eight P-cores with SMT);
- 120 seconds of unmeasured training-step preconditioning;
- 15 warmup steps, 200 measured steps, and 7 fresh worker processes per case;
- no discarded numeric outliers;
- comparison policy SHA-256
  `36ddd5faa00837d921a9e46f9085cf82d557e3a614f5c1108aa8fe936e92876c`;
- policy requirements: at least 5 successful replicates, CV at most 5%, equal replicate counts, and
  regression only when step time is strictly more than 7.5% worse.

Both baselines used `18436725aee764660bfdb99938af98c04fb5a87c`. The candidate used
`3d6ae83e5c0b421954f1ecf067e41ccf45970904`. The two same-code baselines were 10/10 stable and
their comparison verdict was `pass`; their per-case step-time drift ranged from -3.29% to +6.58%.

## Isolated batch-only results

Each case used a 1,000,000-token read-only `uint16` mmap, 500 warmup batches, 20,000 measured
batches, and 7 fresh workers. Negative step-time change is faster.

| Case | Baseline ms | Candidate ms | Step time | Tokens/s | Base CV | Candidate CV | Peak RSS change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B4 / T128 | 0.011606 | 0.010434 | -10.10% | +11.24% | 2.37% | 2.60% | -7.66 MiB |
| B16 / T128 | 0.016679 | 0.010924 | -34.51% | +52.69% | 1.18% | 1.02% | -7.47 MiB |
| B32 / T128 | 0.023134 | 0.012022 | -48.04% | +92.44% | 1.81% | 2.78% | -8.03 MiB |
| B16 / T256 | 0.017165 | 0.011634 | -32.22% | +47.54% | 0.61% | 4.97% | -7.66 MiB |

All four cases were classified `stable`. B16/T256 is only 0.03 percentage points below the 5% CV
limit, so its stability margin is small even though its median change is large.

## End-to-end training results

Both comparisons returned `pass` (CLI exit code 0), with no environment mismatch and 10 aligned
stable cases. The table reports candidate CV and descriptive step-time change against each
baseline. None exceeded the 7.5% regression threshold.

| Case | Candidate CV | vs baseline 1 | vs baseline 2 | Stability |
| --- | ---: | ---: | ---: | --- |
| teaching_t12_s64_b16 | 1.91% | +0.12% | +1.81% | stable |
| teaching_t12_s256_b16 | 1.83% | -0.15% | -0.39% | stable |
| teaching_t12_s128_b8 | 2.81% | -0.75% | -2.49% | stable |
| teaching_t12_s128_b16 | 2.53% | -0.57% | +2.81% | stable |
| teaching_t8_s128_b16 | 4.50% | +3.65% | -2.74% | stable |
| teaching_t20_s128_b16 | 1.65% | -1.98% | -0.16% | stable |
| teaching_t1_s128_b16 | 2.26% | +1.34% | +2.15% | stable |
| teaching_t12_s128_b32 | 1.65% | -0.43% | -0.95% | stable |
| teaching_t4_s128_b16 | 2.95% | +0.12% | +2.09% | stable |
| teaching_t12_s128_b4 | 3.23% | +2.93% | +0.80% | stable |

The candidate range (-2.74% to +3.65%) is narrower than the observed same-code range. This
supports “no detected end-to-end regression,” not “training is X% faster.”

## Profiler interpretation

The separate 20-step CPU profile for `teaching_t12_s128_b16` reported:

| Scope | Baseline self / total | Candidate self / total | Descriptive total change |
| --- | ---: | ---: | ---: |
| data preparation | 2.424 / 3.103 ms | 2.210 / 2.527 ms | -18.6% |
| forward/backward | 120.507 / 1092.493 ms | 135.247 / 1153.558 ms | not a benchmark |
| optimizer step | 0.729 / 77.738 ms | 0.735 / 86.772 ms | not a benchmark |

Profiler overhead makes these values unsuitable for a regression verdict. They only show that data
preparation became smaller while remaining a tiny fraction of the complete training step.

## Rejected run and noise control

An initial candidate matrix was retained locally but rejected: a timed-out diagnostic script had
left an orphan `python.exe -` process consuming one logical core. The same 120-second precondition
fell from 985 baseline steps to 880 steps, and 8 of 10 cases appeared more than 7.5% slower. After
terminating only that task-owned process, precondition throughput recovered to 969 steps. A fresh
candidate matrix then completed 70/70 workers, 10/10 stable cases, and passed both comparisons.

This rejected run is not hidden or reclassified as an outlier; it is excluded because the
“stable machine environment” precondition was demonstrably violated before measurement.

## Reproduction

Install and run the committed measurement definitions:

```powershell
python -m pip install -e ".[dev,report]"
python benchmark_batcher.py --config configs/batcher_benchmark_reference.yaml
python benchmark_v2.py --config configs/benchmark_v2_reference.yaml
python profile_benchmark_v2.py --config configs/benchmark_v2_stage8_profile.yaml
python compare_benchmarks.py `
  --baseline <baseline-run-manifest> `
  --candidate <candidate-run-manifest> `
  --policy configs/benchmark_v2_comparison.yaml
```

The reference commands assume the documented i7-14700 CPU layout. Other hosts must calibrate
affinity and repeat same-code baseline runs before comparing candidates.

## Evidence and limitations

The [`evidence/`](evidence/) directory contains compact manifests, full 10-case comparison JSON,
and selected Profiler operator CSV files. Raw worker JSONL, traces, corpora, logs, and complete local
run directories remain gitignored. [`summary.json`](summary.json) provides the key identities and
machine-readable conclusions; [`artifact_manifest.json`](artifact_manifest.json) hashes every
committed evidence file.

The microbenchmark measures `TokenBatcher.next_batch()` in isolation and is not a model benchmark.
Worker-lifetime peak RSS includes imports, corpus mapping, construction, warmup, and measurement;
it is not measurement-only RSS. The report does not establish statistical significance or
cross-machine superiority.
