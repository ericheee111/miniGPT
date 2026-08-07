# stage11a-overflow timeline

This is deterministic logical serving evidence, not canonical wall-clock benchmark data.
Prefill remains per request; eligible decode rows are tensor-batched with dense padding.

| Seq | Time | Event | Request | Status | Token | Active | Waiting | Cache | Reserved |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | 0.000000 | submitted | full-window | waiting |  | 0 | 1 | 0 | 0 |
| 1 | 0.000000 | submitted | crosses-window | waiting |  | 0 | 2 | 0 | 0 |
| 2 | 0.000000 | admitted | full-window | prefilling |  | 1 | 1 | 0 | 8 |
| 3 | 0.000000 | admitted | crosses-window | prefilling |  | 2 | 0 | 0 | 16 |
| 4 | 1.000000 | prefill_started | full-window | prefilling |  | 2 | 0 | 0 | 16 |
| 5 | 1.000000 | prefill_started | crosses-window | prefilling |  | 2 | 0 | 0 | 16 |
| 6 | 1.010000 | token | full-window | prefilling | 1 | 2 | 0 | 8 | 16 |
| 7 | 1.010000 | token | crosses-window | prefilling | 12 | 2 | 0 | 14 | 16 |
| 8 | 2.010000 | token | full-window | decoding | 2 | 2 | 0 | 14 | 16 |
| 9 | 2.010000 | token | crosses-window | decoding | 0 | 2 | 0 | 15 | 16 |
| 10 | 3.010000 | token | full-window | decoding | 2 | 2 | 0 | 15 | 16 |
| 11 | 3.010000 | token | crosses-window | decoding | 1 | 2 | 0 | 16 | 16 |
| 12 | 4.010000 | token | full-window | decoding | 2 | 2 | 0 | 16 | 16 |
| 13 | 4.010000 | finished | full-window | finished |  | 1 | 0 | 8 | 8 |
| 14 | 4.010000 | token | crosses-window | decoding | 9 | 1 | 0 | 8 | 8 |
| 15 | 5.010000 | token | crosses-window | decoding | 16 | 1 | 0 | 8 | 8 |
| 16 | 5.010000 | finished | crosses-window | finished |  | 0 | 0 | 0 | 0 |
