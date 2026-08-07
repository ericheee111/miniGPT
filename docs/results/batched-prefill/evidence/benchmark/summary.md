# Fresh-process serving benchmark

Overall strict verdict: `not_comparable`.

Profiler timings are excluded from this canonical comparison.

| Scenario | Verdict | Conclusion | Baseline | Candidate | Speedup | Candidate CV | Prefill batch | Prompt waste |
|---|---|---|---|---|---:|---:|---:|---:|
| burst-equal-length | not_comparable | not_comparable | continuous_decode | continuous |  | 6.321500160964121 | 8.0 | 0.0 |
| burst-mixed-lengths | not_comparable | not_comparable | continuous_decode | continuous |  | 13.86338884785554 | 4.0 | 0.22807017543859648 |
| short-prompt-heavy | not_comparable | not_comparable | continuous_decode | continuous |  | 1.9438029746726144 | 4.0 | 0.125 |
| long-prompt-heavy | not_comparable | not_comparable | continuous_decode | continuous |  | 3.6083415848260763 | 2.4 | 0.125 |
| staggered-arrivals | not_comparable | not_comparable | continuous_decode | continuous |  | 13.10766801565876 | 1.6 | 0.0967741935483871 |
| high-padding-pressure | not_comparable | not_comparable | continuous_decode | continuous |  | 10.8796569502212 | 1.0 | 0.0 |
