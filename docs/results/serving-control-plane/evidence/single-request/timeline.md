# single-request timeline

This is deterministic control-plane evidence from a per-request reference executor; it is
not a tensor-level continuous-batching throughput result.

| Seq | Time | Event | Request | Status | Token | Active | Waiting | Cache | Reserved |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | 0.000000 | submitted | baseline-001 | waiting |  | 0 | 1 | 0 | 0 |
| 1 | 0.000000 | admitted | baseline-001 | prefilling |  | 1 | 0 | 0 | 7 |
| 2 | 1.000000 | prefill_started | baseline-001 | prefilling |  | 1 | 0 | 0 | 7 |
| 3 | 1.001000 | token | baseline-001 | prefilling | 0 | 1 | 0 | 4 | 7 |
| 4 | 2.001000 | token | baseline-001 | decoding | 1 | 1 | 0 | 5 | 7 |
| 5 | 3.001000 | token | baseline-001 | decoding | 9 | 1 | 0 | 6 | 7 |
| 6 | 4.001000 | token | baseline-001 | decoding | 4 | 1 | 0 | 7 | 7 |
| 7 | 4.001000 | finished | baseline-001 | finished |  | 0 | 0 | 0 | 0 |
