# Stage 11A fresh-process serving benchmark

Overall strict verdict: `pass`.

Profiler timings are excluded from this canonical comparison.

| Scenario | Verdict | Conclusion | Speedup | Reference CV | Continuous CV | Avg batch | Waste |
|---|---|---|---:|---:|---:|---:|---:|
| burst-2 | pass | improved | 1.239x | 3.8710509364215615 | 7.325320412167548 | 2.0 | 0.0 |
| burst-4 | pass | improved | 1.696x | 6.064234828429397 | 4.112513073205086 | 4.0 | 0.0 |
| burst-8 | pass | improved | 2.220x | 4.280582865419838 | 2.2897191334858427 | 8.0 | 0.0 |
| staggered-arrival | pass | improved | 1.435x | 2.3044283594651134 | 7.222611000655162 | 2.8 | 0.09411764705882353 |
| mixed-cache-lengths | pass | improved | 1.255x | 5.628445538074065 | 2.562482158962507 | 2.3636363636363638 | 0.354251012145749 |
| cancellation | pass | improved | 1.268x | 9.847879120557174 | 3.436093395808997 | 2.3333333333333335 | 0.0 |
