# burst-arrivals timeline

This is deterministic control-plane evidence from a per-request reference executor; it is
not a tensor-level continuous-batching throughput result.

| Seq | Time | Event | Request | Status | Token | Active | Waiting | Cache | Reserved |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | 0.000000 | submitted | burst-001 | waiting |  | 0 | 1 | 0 | 0 |
| 1 | 0.000000 | submitted | burst-002 | waiting |  | 0 | 2 | 0 | 0 |
| 2 | 0.000000 | submitted | burst-003 | waiting |  | 0 | 3 | 0 | 0 |
| 3 | 0.000000 | submitted | burst-004 | waiting |  | 0 | 4 | 0 | 0 |
| 4 | 0.000000 | admitted | burst-001 | prefilling |  | 1 | 3 | 0 | 6 |
| 5 | 0.000000 | admitted | burst-002 | prefilling |  | 2 | 2 | 0 | 12 |
| 6 | 1.000000 | prefill_started | burst-001 | prefilling |  | 2 | 2 | 0 | 12 |
| 7 | 1.000000 | prefill_started | burst-002 | prefilling |  | 2 | 2 | 0 | 12 |
| 8 | 1.001000 | token | burst-001 | prefilling | 5 | 2 | 2 | 3 | 12 |
| 9 | 1.001000 | token | burst-002 | prefilling | 7 | 2 | 2 | 7 | 12 |
| 10 | 2.001000 | token | burst-001 | decoding | 14 | 2 | 2 | 8 | 12 |
| 11 | 2.001000 | token | burst-002 | decoding | 0 | 2 | 2 | 9 | 12 |
| 12 | 3.001000 | token | burst-001 | decoding | 0 | 2 | 2 | 10 | 12 |
| 13 | 3.001000 | token | burst-002 | decoding | 16 | 2 | 2 | 11 | 12 |
| 14 | 3.001000 | finished | burst-002 | finished |  | 1 | 2 | 5 | 6 |
| 15 | 4.000000 | admitted | burst-003 | prefilling |  | 2 | 1 | 5 | 9 |
| 16 | 4.001000 | token | burst-001 | decoding | 9 | 2 | 1 | 6 | 9 |
| 17 | 4.001000 | finished | burst-001 | finished |  | 1 | 1 | 0 | 3 |
| 18 | 5.000000 | admitted | burst-004 | prefilling |  | 2 | 0 | 0 | 9 |
| 19 | 5.000000 | prefill_started | burst-003 | prefilling |  | 2 | 0 | 0 | 9 |
| 20 | 5.001000 | token | burst-003 | prefilling | 10 | 2 | 0 | 2 | 9 |
| 21 | 6.000000 | prefill_started | burst-004 | prefilling |  | 2 | 0 | 2 | 9 |
| 22 | 6.001000 | token | burst-004 | prefilling | 10 | 2 | 0 | 5 | 9 |
| 23 | 6.001000 | token | burst-003 | decoding | 8 | 2 | 0 | 6 | 9 |
| 24 | 6.001000 | finished | burst-003 | finished |  | 1 | 0 | 3 | 6 |
| 25 | 7.001000 | token | burst-004 | decoding | 4 | 1 | 0 | 4 | 6 |
| 26 | 8.001000 | token | burst-004 | decoding | 4 | 1 | 0 | 5 | 6 |
| 27 | 9.001000 | token | burst-004 | decoding | 4 | 1 | 0 | 6 | 6 |
| 28 | 9.001000 | finished | burst-004 | finished |  | 0 | 0 | 0 | 0 |
