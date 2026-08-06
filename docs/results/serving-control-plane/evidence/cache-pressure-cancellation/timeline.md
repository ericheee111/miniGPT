# cache-pressure-cancellation timeline

This is deterministic control-plane evidence from a per-request reference executor; it is
not a tensor-level continuous-batching throughput result.

| Seq | Time | Event | Request | Status | Token | Active | Waiting | Cache | Reserved |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | 0.000000 | submitted | pressure-001 | waiting |  | 0 | 1 | 0 | 0 |
| 1 | 0.000000 | submitted | pressure-002 | waiting |  | 0 | 2 | 0 | 0 |
| 2 | 0.000000 | admitted | pressure-001 | prefilling |  | 1 | 1 | 0 | 7 |
| 3 | 1.000000 | submitted | pressure-003 | waiting |  | 1 | 2 | 0 | 7 |
| 4 | 1.000000 | prefill_started | pressure-001 | prefilling |  | 1 | 2 | 0 | 7 |
| 5 | 1.001000 | token | pressure-001 | prefilling | 8 | 1 | 2 | 4 | 7 |
| 6 | 2.000000 | cancelled | pressure-001 | cancelled |  | 0 | 2 | 0 | 0 |
| 7 | 2.000000 | admitted | pressure-002 | prefilling |  | 1 | 1 | 0 | 6 |
| 8 | 3.000000 | prefill_started | pressure-002 | prefilling |  | 1 | 1 | 0 | 6 |
| 9 | 3.001000 | token | pressure-002 | prefilling | 2 | 1 | 1 | 4 | 6 |
| 10 | 4.001000 | token | pressure-002 | decoding | 3 | 1 | 1 | 5 | 6 |
| 11 | 5.001000 | token | pressure-002 | decoding | 15 | 1 | 1 | 6 | 6 |
| 12 | 5.001000 | finished | pressure-002 | finished |  | 0 | 1 | 0 | 0 |
| 13 | 6.000000 | admitted | pressure-003 | prefilling |  | 1 | 0 | 0 | 5 |
| 14 | 7.000000 | prefill_started | pressure-003 | prefilling |  | 1 | 0 | 0 | 5 |
| 15 | 7.001000 | token | pressure-003 | prefilling | 1 | 1 | 0 | 3 | 5 |
| 16 | 8.001000 | token | pressure-003 | decoding | 16 | 1 | 0 | 4 | 5 |
| 17 | 9.001000 | token | pressure-003 | decoding | 5 | 1 | 0 | 5 | 5 |
| 18 | 9.001000 | finished | pressure-003 | finished |  | 0 | 0 | 0 | 0 |
