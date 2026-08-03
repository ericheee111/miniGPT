# stage8-batcher-only-i7-14700

This is an isolated `TokenBatcher.next_batch()` microbenchmark, not end-to-end training.

| Case | Median ms/batch | Median tokens/s | CV % | MAD ms | Stability |
|---|---:|---:|---:|---:|---|
| char-gpt-b4-t128 | 0.014058825 | 36418406.23238429 | 8.965172407339596 | 0.0010580599999999996 | unstable |
| char-gpt-b16-t128 | 0.020495580000000003 | 99923983.6101247 | 10.678826798913853 | 0.0021877200000000006 | unstable |
| char-gpt-b32-t128 | 0.027124395 | 151007976.39910492 | 5.64919631531206 | 0.0006575400000000016 | unstable |
| char-gpt-b16-t256 | 0.0214954 | 190552397.25708756 | 20.394480174658796 | 0.001587874999999999 | unstable |

Each replicate ran in a fresh process. No outliers were removed.
Peak RSS means worker lifetime peak RSS.
