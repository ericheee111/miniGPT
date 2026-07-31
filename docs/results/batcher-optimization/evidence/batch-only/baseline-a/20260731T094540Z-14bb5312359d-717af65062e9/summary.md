# stage8-batcher-only-i7-14700

This is an isolated `TokenBatcher.next_batch()` microbenchmark, not end-to-end training.

| Case | Median ms/batch | Median tokens/s | CV % | MAD ms | Stability |
|---|---:|---:|---:|---:|---|
| char-gpt-b4-t128 | 0.0123785 | 41362039.019267276 | 6.518785370734396 | 0.00011872500000000182 | unstable |
| char-gpt-b16-t128 | 0.018667955000000003 | 109706713.99197179 | 2.2133424947496674 | 0.0002946100000000007 | stable |
| char-gpt-b32-t128 | 0.026737305 | 153194198.14375457 | 5.523400410484794 | 0.0011175400000000002 | unstable |
| char-gpt-b16-t256 | 0.01895573 | 216082419.40563616 | 12.560555402816778 | 0.0005779500000000007 | unstable |

Each replicate ran in a fresh process. No outliers were removed.
Peak RSS means worker lifetime peak RSS.
