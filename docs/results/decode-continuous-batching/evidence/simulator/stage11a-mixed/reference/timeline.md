# stage11a-mixed timeline

This is deterministic logical serving evidence, not canonical wall-clock benchmark data.
Model prefill and decode calls both remain per request.

| Seq | Time | Event | Request | Status | Token | Active | Waiting | Cache | Reserved |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | 0.000000 | submitted | short | waiting |  | 0 | 1 | 0 | 0 |
| 1 | 0.000000 | submitted | medium | waiting |  | 0 | 2 | 0 | 0 |
| 2 | 0.000000 | submitted | cancelled | waiting |  | 0 | 3 | 0 | 0 |
| 3 | 0.000000 | admitted | short | prefilling |  | 1 | 2 | 0 | 6 |
| 4 | 0.000000 | admitted | medium | prefilling |  | 2 | 1 | 0 | 15 |
| 5 | 0.000000 | admitted | cancelled | prefilling |  | 3 | 0 | 0 | 26 |
| 6 | 1.000000 | submitted | long | waiting |  | 3 | 1 | 0 | 26 |
| 7 | 1.000000 | admitted | long | prefilling |  | 4 | 0 | 0 | 37 |
| 8 | 1.000000 | prefill_started | short | prefilling |  | 4 | 0 | 0 | 37 |
| 9 | 1.000000 | prefill_started | medium | prefilling |  | 4 | 0 | 0 | 37 |
| 10 | 1.000000 | prefill_started | cancelled | prefilling |  | 4 | 0 | 0 | 37 |
| 11 | 1.010000 | token | short | prefilling | 7 | 4 | 0 | 2 | 37 |
| 12 | 1.010000 | token | medium | prefilling | 13 | 4 | 0 | 7 | 37 |
| 13 | 1.010000 | token | cancelled | prefilling | 14 | 4 | 0 | 11 | 37 |
| 14 | 2.000000 | prefill_started | long | prefilling |  | 4 | 0 | 11 | 37 |
| 15 | 2.010000 | token | long | prefilling | 2 | 4 | 0 | 19 | 37 |
| 16 | 2.010000 | token | short | decoding | 8 | 4 | 0 | 20 | 37 |
| 17 | 2.010000 | token | medium | decoding | 10 | 4 | 0 | 21 | 37 |
| 18 | 2.010000 | token | cancelled | decoding | 4 | 4 | 0 | 22 | 37 |
| 19 | 3.000000 | cancelled | cancelled | cancelled |  | 3 | 0 | 17 | 26 |
| 20 | 3.010000 | token | short | decoding | 11 | 3 | 0 | 18 | 26 |
| 21 | 3.010000 | token | medium | decoding | 4 | 3 | 0 | 19 | 26 |
| 22 | 3.010000 | token | long | decoding | 16 | 3 | 0 | 20 | 26 |
| 23 | 4.010000 | token | short | decoding | 10 | 3 | 0 | 21 | 26 |
| 24 | 4.010000 | token | medium | decoding | 0 | 3 | 0 | 22 | 26 |
| 25 | 4.010000 | token | long | decoding | 15 | 3 | 0 | 23 | 26 |
| 26 | 5.010000 | token | short | decoding | 7 | 3 | 0 | 24 | 26 |
| 27 | 5.010000 | finished | short | finished |  | 2 | 0 | 18 | 20 |
| 28 | 5.010000 | token | medium | decoding | 2 | 2 | 0 | 19 | 20 |
| 29 | 5.010000 | finished | medium | finished |  | 1 | 0 | 10 | 11 |
| 30 | 5.010000 | token | long | decoding | 8 | 1 | 0 | 11 | 11 |
| 31 | 5.010000 | finished | long | finished |  | 0 | 0 | 0 | 0 |
