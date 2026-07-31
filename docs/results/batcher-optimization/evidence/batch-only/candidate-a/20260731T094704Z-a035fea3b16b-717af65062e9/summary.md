# stage8-batcher-only-i7-14700

This is an isolated `TokenBatcher.next_batch()` microbenchmark, not end-to-end training.

| Case | Median ms/batch | Median tokens/s | CV % | MAD ms | Stability |
|---|---:|---:|---:|---:|---|
| char-gpt-b4-t128 | 0.01000334 | 51182904.90976015 | 7.351484685312454 | 0.0003069049999999997 | unstable |
| char-gpt-b16-t128 | 0.01105458 | 185262578.94917762 | 4.623603174006116 | 0.0003273149999999999 | stable |
| char-gpt-b32-t128 | 0.01192916 | 343360303.65926856 | 3.2134564637473217 | 0.0003416749999999996 | stable |
| char-gpt-b16-t256 | 0.011923705 | 343517388.26145065 | 10.503280846034453 | 0.0002277999999999985 | unstable |

Each replicate ran in a fresh process. No outliers were removed.
Peak RSS means worker lifetime peak RSS.
