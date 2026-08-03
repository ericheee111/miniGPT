# stage8-batcher-only-i7-14700

This is an isolated `TokenBatcher.next_batch()` microbenchmark, not end-to-end training.

| Case | Median ms/batch | Median tokens/s | CV % | MAD ms | Stability |
|---|---:|---:|---:|---:|---|
| char-gpt-b4-t128 | 0.01023239 | 50037185.83830366 | 5.80967192914903 | 0.0003168699999999986 | unstable |
| char-gpt-b16-t128 | 0.011197770000000001 | 182893558.27097714 | 2.3252938221154396 | 8.95049999999984e-05 | stable |
| char-gpt-b32-t128 | 0.01183003 | 346237498.975066 | 7.154256410091402 | 0.0003466400000000005 | unstable |
| char-gpt-b16-t256 | 0.01146681 | 357204837.2651156 | 5.8944376001778345 | 0.00023643499999999838 | unstable |

Each replicate ran in a fresh process. No outliers were removed.
Peak RSS means worker lifetime peak RSS.
