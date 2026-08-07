# stage11b-mixed timeline

This is deterministic logical serving evidence, not canonical wall-clock benchmark data.
Model prefill and decode calls both remain per request.

| Seq | Time | Event | Request | Status | Token | Active | Waiting | Cache | Reserved |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | 0.000000 | submitted | short-a | waiting |  | 0 | 1 | 0 | 0 |
| 1 | 0.000000 | submitted | short-b | waiting |  | 0 | 2 | 0 | 0 |
| 2 | 0.000000 | submitted | medium | waiting |  | 0 | 3 | 0 | 0 |
| 3 | 0.000000 | submitted | long | waiting |  | 0 | 4 | 0 | 0 |
| 4 | 0.000000 | admitted | short-a | prefilling |  | 1 | 3 | 0 | 5 |
| 5 | 0.000000 | admitted | short-b | prefilling |  | 2 | 2 | 0 | 10 |
| 6 | 0.000000 | admitted | medium | prefilling |  | 3 | 1 | 0 | 19 |
| 7 | 0.000000 | admitted | long | prefilling |  | 4 | 0 | 0 | 33 |
| 8 | 1.000000 | submitted | cancelled | waiting |  | 4 | 1 | 0 | 33 |
| 9 | 1.000000 | admitted | cancelled | prefilling |  | 5 | 0 | 0 | 41 |
| 10 | 1.000000 | prefill_started | short-a | prefilling |  | 5 | 0 | 0 | 41 |
| 11 | 1.000000 | prefill_started | short-b | prefilling |  | 5 | 0 | 0 | 41 |
| 12 | 1.000000 | prefill_started | medium | prefilling |  | 5 | 0 | 0 | 41 |
| 13 | 1.000000 | prefill_started | long | prefilling |  | 5 | 0 | 0 | 41 |
| 14 | 1.010000 | token | short-a | prefilling | 14 | 5 | 0 | 2 | 41 |
| 15 | 1.010000 | token | short-b | prefilling | 10 | 5 | 0 | 5 | 41 |
| 16 | 1.010000 | token | medium | prefilling | 13 | 5 | 0 | 11 | 41 |
| 17 | 1.010000 | token | long | prefilling | 58 | 5 | 0 | 23 | 41 |
| 18 | 2.000000 | prefill_started | cancelled | prefilling |  | 5 | 0 | 23 | 41 |
| 19 | 2.010000 | token | cancelled | prefilling | 6 | 5 | 0 | 27 | 41 |
| 20 | 2.010000 | token | short-a | decoding | 45 | 5 | 0 | 28 | 41 |
| 21 | 2.010000 | token | short-b | decoding | 52 | 5 | 0 | 29 | 41 |
| 22 | 2.010000 | token | medium | decoding | 26 | 5 | 0 | 30 | 41 |
| 23 | 2.010000 | token | long | decoding | 64 | 5 | 0 | 31 | 41 |
| 24 | 3.000000 | cancelled | cancelled | cancelled |  | 4 | 0 | 27 | 33 |
| 25 | 3.010000 | token | short-a | decoding | 9 | 4 | 0 | 28 | 33 |
| 26 | 3.010000 | token | short-b | decoding | 7 | 4 | 0 | 29 | 33 |
| 27 | 3.010000 | finished | short-b | finished |  | 3 | 0 | 24 | 28 |
| 28 | 3.010000 | token | medium | decoding | 41 | 3 | 0 | 25 | 28 |
| 29 | 3.010000 | token | long | decoding | 12 | 3 | 0 | 26 | 28 |
| 30 | 3.010000 | finished | long | finished |  | 2 | 0 | 12 | 14 |
| 31 | 4.010000 | token | short-a | decoding | 48 | 2 | 0 | 13 | 14 |
| 32 | 4.010000 | finished | short-a | finished |  | 1 | 0 | 8 | 9 |
| 33 | 4.010000 | token | medium | decoding | 23 | 1 | 0 | 9 | 9 |
| 34 | 4.010000 | finished | medium | finished |  | 0 | 0 | 0 | 0 |
