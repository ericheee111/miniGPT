# stage11b-overflow timeline

This is deterministic logical serving evidence, not canonical wall-clock benchmark data.
Model prefill and decode calls both remain per request.

| Seq | Time | Event | Request | Status | Token | Active | Waiting | Cache | Reserved |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | 0.000000 | submitted | full-window | waiting |  | 0 | 1 | 0 | 0 |
| 1 | 0.000000 | submitted | partial-window | waiting |  | 0 | 2 | 0 | 0 |
| 2 | 0.000000 | admitted | full-window | prefilling |  | 1 | 1 | 0 | 8 |
| 3 | 0.000000 | admitted | partial-window | prefilling |  | 2 | 0 | 0 | 16 |
| 4 | 1.000000 | prefill_started | full-window | prefilling |  | 2 | 0 | 0 | 16 |
| 5 | 1.000000 | prefill_started | partial-window | prefilling |  | 2 | 0 | 0 | 16 |
| 6 | 1.010000 | token | full-window | prefilling | 52 | 2 | 0 | 8 | 16 |
| 7 | 1.010000 | token | partial-window | prefilling | 47 | 2 | 0 | 14 | 16 |
| 8 | 2.010000 | token | full-window | decoding | 24 | 2 | 0 | 14 | 16 |
| 9 | 2.010000 | token | partial-window | decoding | 10 | 2 | 0 | 15 | 16 |
| 10 | 3.010000 | token | full-window | decoding | 50 | 2 | 0 | 15 | 16 |
| 11 | 3.010000 | token | partial-window | decoding | 50 | 2 | 0 | 16 | 16 |
| 12 | 4.010000 | token | full-window | decoding | 4 | 2 | 0 | 16 | 16 |
| 13 | 4.010000 | finished | full-window | finished |  | 1 | 0 | 8 | 8 |
| 14 | 4.010000 | token | partial-window | decoding | 8 | 1 | 0 | 8 | 8 |
| 15 | 4.010000 | finished | partial-window | finished |  | 0 | 0 | 0 | 0 |
