# All-level CPU timing and accuracy

Each preset covers the same 100 checker plays, 100 cube actions and 100 post-move evaluations. The paired measurements use 16 logical CPUs on a Ryzen 7 9700X with GCC under local x86-64 WSL2 and MSVC on Windows. Both builds run their public Stage 11 analyzers with identical position seeds, bearoff database and weights. Timings cover each API call and exclude analyzer construction and serialization. The selected engine is `5de8023`. Changes are relative to fresh controls at `1ecca39`. The earlier published time for that commit is listed separately. Each paired total is one complete pass. The optimized 1P panel takes about 20 ms for all 300 calls, so even small absolute timing fluctuations are visible as percentages.

The runner calls `BgBotAnalyzer.checker_play`, `cube_action` and `post_move_analytics` directly. It reuses one analyzer per strength with ordinary engine caches, without a warmup, progress callback or optional cube-detail tree. Each truncated position gets a reproducible seed before timing. This clears the prior dice schedule, so its regeneration falls inside the timed call. A long-lived analyzer with a fixed seed can reuse that schedule. Application cache hits, queuing and request handling fall outside the measurement. [Benchmark runner](../../../../scripts/benchmark_multi_ply.py) · [Warm fixed-seed 2T calls with progress](app-style/README.md)

The pre-optimization engine is `139427d`. Historical runs used the same hardware and corpus in separate sessions on November 22, 2026 (UTC). All original Windows timings and Linux 1P–4P timings are from unmodified builds. The Linux machine uses a Windows-mounted filesystem under WSL2. Repeated weight-file loading inside the original 1P cube calls makes that timing especially filesystem-dependent. [Original Windows recordings](../search_pass2/windows/full_outputs.zip) · [Original Linux deterministic recordings](../search_pass2/linux/full_outputs.zip) · [Two-thread comparison with Linux-native and mounted weights](inline/README.md)

The original parallel truncated implementation could read an opening reply before another worker finished writing it. The archived Windows timings include this behavior. The original fourth-ranked move of `checker-money-priming-10` had a 2T win probability of 0.893650 against an independent full-game rollout of 0.819967. Some single-batch parallel full-game reference calls also used this historical code. [Cache-race investigation and serial replay](../truncated_16t/README.md#parallel-cache-correction)

The rollout reference supplies 4,724 probability and equity fields per preset. It covers 378 rolled-out checker alternatives, 100 cube decisions and 100 post-move positions. All 1,709 legal checker alternatives are compared directly between the two builds. Material means a change in absolute error greater than 0.0025 in probability or 0.005 in equity. [Reference provenance](../../rollout_positions/README.md) · [Independent verification script](audit.py)

The fixed references use one 1,296-game batch with variance reduction and 3P checker and cube settings. For 37 checker positions the existing archival references used the same strength in 1,296-game batches up to their target standard error. The archive omits their individual batch counts.

## Linux

| Preset | Before all optimizations (s) | Published checkpoint (s) | Current control (s) | Selected (s) | Saved this pass | Speedup since start |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1P | 11.145 | 0.020 | 0.023 | 0.022 | 5.0% | 504.79× |
| 2P | 12.727 | 0.220 | 0.226 | 0.232 | -2.9% | 54.86× |
| 3P | 18.291 | 1.868 | 2.097 | 1.952 | 6.9% | 9.37× |
| 4P | 153.584 | 32.401 | 36.409 | 34.789 | 4.5% | 4.41× |
| 1T | 39.936 † | 7.953 | 8.417 | 6.118 | 27.3% | 6.53× |
| 2T | 222.486 † | 53.410 | 61.510 | 45.869 | 25.4% | 4.85× |
| 3T | 610.438 † | 191.230 | 219.263 | 185.208 | 15.5% | 3.30× |

† The original Linux truncated timings use the original engine with the isolated parallel rollout cache publication fix. The original Linux deterministic timings use unmodified `139427d`. [Historical Linux truncated control](historical-linux/README.md)

An intermediate release at `82e63972379b77ab52ebafc716620e813ff7baa1` ran this corpus in an earlier session on the same machine. Timings below compare those independently recorded full panels with the selected build. [Intermediate release recordings](../search_pass3/release/linux/full_outputs.zip)

| Preset | Intermediate total (s) | Selected (s) | Speedup vs intermediate | Materially farther / closer |
| --- | ---: | ---: | ---: | ---: |
| 1P | 0.021 | 0.022 | 0.96× | 0 / 0 |
| 2P | 0.355 | 0.232 | 1.53× | 0 / 0 |
| 3P | 2.515 | 1.952 | 1.29× | 0 / 0 |
| 4P | 47.637 | 34.789 | 1.37× | 0 / 0 |
| 1T | 9.859 | 6.118 | 1.61× | 0 / 0 |
| 2T | 67.298 | 45.869 | 1.47× | 0 / 0 |
| 3T | 239.575 | 185.208 | 1.29× | 0 / 0 |

**3T repeat measurements**

| Measurement | First build | Checkpoint (s) | Selected (s) | Wall time saved | Speedup since start |
| --- | --- | ---: | ---: | ---: | ---: |
| Main measurement | Checkpoint | 219.263 | 185.208 | 15.5% | 3.30× |
| [linux-3t-candidate-first](qualified/linux-3t-candidate-first/verification.json) | Selected | 197.465 | 169.234 | 14.3% | 3.61× |
| Sum of all runs | — | 416.728 | 354.443 | 14.9% | 3.44× |

Every repeated full response matched the corresponding build on the main measurement. The pooled percentage uses the summed times for all recorded runs. The pooled speedup divides the original time for one pass by the mean selected time per pass.

| Preset | API (100 calls) | Original total (s) | Total seconds, checkpoint → selected | Speedup since start | Median ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1P | Checker | 0.410 | 0.016 → 0.016 | 25.63× | 0.112 → 0.117 | 0.283 → 0.311 |
| 1P | Cube | 10.733 | 0.004 → 0.004 | 3012.75× | 0.033 → 0.032 | 0.079 → 0.042 |
| 1P | Post-move | 0.003 | 0.003 → 0.003 | 1.17× | 0.027 → 0.025 | 0.064 → 0.032 |
| 2P | Checker | 1.372 | 0.151 → 0.157 | 8.76× | 1.023 → 1.014 | 3.449 → 3.562 |
| 2P | Cube | 11.085 | 0.046 → 0.048 | 231.35× | 0.480 → 0.489 | 0.740 → 0.812 |
| 2P | Post-move | 0.270 | 0.028 → 0.027 | 9.82× | 0.283 → 0.279 | 0.513 → 0.468 |
| 3P | Checker | 6.528 | 1.770 → 1.674 | 3.90× | 10.844 → 9.986 | 53.341 → 45.436 |
| 3P | Cube | 11.298 | 0.197 → 0.165 | 68.41× | 1.710 → 1.503 | 4.561 → 3.660 |
| 3P | Post-move | 0.465 | 0.130 → 0.112 | 4.14× | 1.080 → 0.976 | 2.853 → 2.391 |
| 4P | Checker | 123.716 | 30.953 → 28.471 | 4.35× | 144.006 → 130.240 | 922.284 → 819.468 |
| 4P | Cube | 21.916 | 3.040 → 3.314 | 6.61× | 29.985 → 34.857 | 63.439 → 68.963 |
| 4P | Post-move | 7.952 | 2.417 → 3.004 | 2.65× | 24.014 → 30.825 | 53.028 → 66.278 |
| 1T | Checker | 24.543 | 5.011 → 3.653 | 6.72× | 42.385 → 31.217 | 124.684 → 99.279 |
| 1T | Cube | 7.669 | 1.746 → 1.241 | 6.18× | 20.573 → 14.603 | 30.355 → 23.334 |
| 1T | Post-move | 7.724 | 1.660 → 1.225 | 6.30× | 19.742 → 15.213 | 28.602 → 22.856 |
| 2T | Checker | 122.711 | 34.327 → 26.057 | 4.71× | 308.626 → 233.627 | 968.082 → 732.914 |
| 2T | Cube | 49.606 | 14.025 → 9.753 | 5.09× | 159.282 → 114.637 | 260.608 → 176.973 |
| 2T | Post-move | 50.169 | 13.158 → 10.059 | 4.99× | 156.101 → 117.702 | 245.250 → 177.307 |
| 3T | Checker | 346.658 | 122.311 → 105.352 | 3.29× | 964.903 → 843.820 | 3313.862 → 2975.389 |
| 3T | Cube | 132.649 | 48.793 → 39.773 | 3.34× | 530.407 → 399.713 | 877.708 → 751.078 |
| 3T | Post-move | 131.131 | 48.159 → 40.083 | 3.27× | 519.161 → 431.332 | 912.159 → 776.579 |

| Game plan (60 calls per preset) | 1T wall time saved | 2T | 3T |
| --- | ---: | ---: | ---: |
| Attacking | 28.5% | 26.5% | 16.1% |
| Priming | 26.0% | 27.5% | 14.5% |
| Anchoring | 27.7% | 20.4% | 14.7% |
| Racing with contact | 24.1% | 27.6% | 17.9% |
| Pure race | 40.8% | 29.4% | 11.6% |
| Match (150 calls) | 26.1% | 28.6% | 13.5% |
| Unlimited (150 calls) | 28.6% | 21.6% | 17.6% |

Each probability MAE averages 2,890 equally weighted probability observations. Equity MAE averages 1,834 equity observations. The threshold applies separately to each observation. Direct shifts include all public checker alternatives, including those without a rollout.

| Preset | Probability MAE, checkpoint → selected | Equity MAE | Materially farther / closer | Largest direct probability / equity shift | Whole outputs identical (/300) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1P | 0.00698275 → 0.00698275 | 0.03526136 → 0.03526136 | 0 / 0 | 0.00000000 / 0.00000000 | 300 |
| 2P | 0.00544002 → 0.00544002 | 0.02736282 → 0.02736282 | 0 / 0 | 0.00000000 / 0.00000000 | 300 |
| 3P | 0.00514367 → 0.00514367 | 0.02431037 → 0.02431037 | 0 / 0 | 0.00000000 / 0.00000000 | 300 |
| 4P | 0.00445193 → 0.00445193 | 0.01940722 → 0.01940722 | 0 / 0 | 0.00000000 / 0.00000000 | 300 |
| 1T | 0.00464839 → 0.00464839 | 0.01707575 → 0.01707575 | 0 / 0 | 0.00000006 / 0.00000042 | 31 |
| 2T | 0.00415231 → 0.00415231 | 0.01591040 → 0.01591026 | 0 / 0 | 0.00004637 / 0.00008842 | 18 |
| 3T | 0.00369214 → 0.00369212 | 0.01160757 → 0.01160754 | 0 / 0 | 0.00001104 / 0.00000815 | 12 |

| Preset | Same best checker | Same full checker order | Same cube action | Checker agrees with rollout, checkpoint → selected | Cube agrees with rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1P | 100/100 | 100/100 | 100/100 | 83 → 83 | 86 → 86 |
| 2P | 100/100 | 100/100 | 100/100 | 81 → 81 | 83 → 83 |
| 3P | 100/100 | 100/100 | 100/100 | 87 → 87 | 95 → 95 |
| 4P | 100/100 | 100/100 | 100/100 | 87 → 87 | 98 → 98 |
| 1T | 100/100 | 100/100 | 100/100 | 92 → 92 | 98 → 98 |
| 2T | 100/100 | 100/100 | 100/100 | 90 → 90 | 96 → 96 |
| 3T | 100/100 | 100/100 | 100/100 | 90 → 90 | 98 → 98 |

Cumulative accuracy compares the selected build directly with the recorded original predictions using the same rollout references as the per-pass comparison.

| Preset | Probability MAE, original → selected | Equity MAE | Materially farther / closer | Best checker agrees with rollout, original → selected | Cube agrees with rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1P | 0.00698275 → 0.00698275 | 0.03526136 → 0.03526136 | 0 / 0 | 83 → 83 | 86 → 86 |
| 2P | 0.00544002 → 0.00544002 | 0.02736282 → 0.02736282 | 0 / 0 | 81 → 81 | 83 → 83 |
| 3P | 0.00514367 → 0.00514367 | 0.02431037 → 0.02431037 | 0 / 0 | 87 → 87 | 95 → 95 |
| 4P | 0.00445193 → 0.00445193 | 0.01940720 → 0.01940722 | 0 / 0 | 87 → 87 | 98 → 98 |
| 1T | 0.00464839 → 0.00464839 | 0.01707575 → 0.01707575 | 0 / 0 | 92 → 92 | 98 → 98 |
| 2T | 0.00415231 → 0.00415231 | 0.01591040 → 0.01591026 | 0 / 0 | 90 → 90 | 96 → 96 |
| 3T | 0.00369213 → 0.00369212 | 0.01160756 → 0.01160754 | 0 / 0 | 90 → 90 | 98 → 98 |

[Inputs, full analytics and per-call timings](qualified/linux/full_outputs.zip) · [Per-field errors and latency distributions](qualified/linux/comparison.json) · [Recomputed scores, hashes and subgroup errors](qualified/linux/verification.json)

## Windows

| Preset | Before all optimizations (s) | Published checkpoint (s) | Current control (s) | Selected (s) | Saved this pass | Speedup since start |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1P | 2.847 | 0.020 | 0.023 | 0.020 | 10.5% | 139.15× |
| 2P | 4.444 | 0.174 | 0.194 | 0.191 | 1.6% | 23.22× |
| 3P | 11.312 | 1.889 | 1.963 | 2.290 | -16.7% | 4.94× |
| 4P | 176.073 | 31.885 | 33.790 | 33.077 | 2.1% | 5.32× |
| 1T | 35.043 | 8.432 | 8.740 | 5.900 | 32.5% | 5.94× |
| 2T | 197.482 | 52.704 | 57.157 | 42.754 | 25.2% | 4.62× |
| 3T | 561.767 | 191.970 | 207.159 | 169.495 | 18.2% | 3.31× |

**3P repeat measurements**

| Measurement | First build | Checkpoint (s) | Selected (s) | Wall time saved | Speedup since start |
| --- | --- | ---: | ---: | ---: | ---: |
| Main measurement | Selected | 1.963 | 2.290 | -16.7% | 4.94× |
| [windows-3p-control-first](qualified/windows-3p-control-first/verification.json) | Checkpoint | 2.068 | 2.010 | 2.8% | 5.63× |
| [windows-3p-candidate-first](qualified/windows-3p-candidate-first/verification.json) | Selected | 2.053 | 2.052 | 0.0% | 5.51× |
| Sum of all runs | — | 6.084 | 6.353 | -4.4% | 5.34× |

Every repeated full response matched the corresponding build on the main measurement. The pooled percentage uses the summed times for all recorded runs. The pooled speedup divides the original time for one pass by the mean selected time per pass.

| Preset | API (100 calls) | Original total (s) | Total seconds, checkpoint → selected | Speedup since start | Median ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1P | Checker | 0.719 | 0.016 → 0.015 | 47.70× | 0.136 → 0.126 | 0.340 → 0.326 |
| 1P | Cube | 2.126 | 0.004 → 0.003 | 669.49× | 0.035 → 0.032 | 0.042 → 0.049 |
| 1P | Post-move | 0.003 | 0.003 → 0.002 | 1.48× | 0.030 → 0.022 | 0.041 → 0.030 |
| 2P | Checker | 1.624 | 0.141 → 0.138 | 11.72× | 0.985 → 0.964 | 3.563 → 3.540 |
| 2P | Cube | 2.559 | 0.032 → 0.032 | 80.63× | 0.288 → 0.279 | 0.617 → 0.683 |
| 2P | Post-move | 0.261 | 0.022 → 0.021 | 12.36× | 0.186 → 0.187 | 0.454 → 0.410 |
| 3P | Checker | 7.820 | 1.713 → 2.007 | 3.90× | 10.764 → 12.814 | 45.111 → 52.697 |
| 3P | Cube | 3.037 | 0.138 → 0.162 | 18.74× | 1.189 → 1.308 | 3.429 → 3.966 |
| 3P | Post-move | 0.455 | 0.112 → 0.120 | 3.77× | 0.877 → 0.953 | 2.941 → 2.923 |
| 4P | Checker | 151.879 | 28.542 → 27.801 | 5.46× | 131.057 → 135.646 | 860.040 → 807.441 |
| 4P | Cube | 15.226 | 2.886 → 2.906 | 5.24× | 29.313 → 29.544 | 61.121 → 61.641 |
| 4P | Post-move | 8.968 | 2.362 → 2.371 | 3.78× | 24.788 → 24.804 | 51.062 → 52.419 |
| 1T | Checker | 20.657 | 5.145 → 3.470 | 5.95× | 43.324 → 30.587 | 127.079 → 85.852 |
| 1T | Cube | 6.780 | 1.820 → 1.230 | 5.51× | 21.032 → 14.680 | 33.080 → 23.433 |
| 1T | Post-move | 7.606 | 1.776 → 1.200 | 6.34× | 21.191 → 14.282 | 30.868 → 20.950 |
| 2T | Checker | 112.280 | 32.060 → 24.044 | 4.67× | 308.421 → 229.836 | 855.128 → 687.416 |
| 2T | Cube | 42.508 | 12.574 → 9.404 | 4.52× | 147.820 → 114.401 | 215.748 → 165.854 |
| 2T | Post-move | 42.694 | 12.523 → 9.306 | 4.59× | 150.218 → 108.029 | 213.076 → 163.374 |
| 3T | Checker | 322.641 | 116.390 → 95.230 | 3.39× | 959.851 → 774.336 | 3132.717 → 2519.495 |
| 3T | Cube | 120.183 | 45.543 → 36.816 | 3.26× | 503.321 → 383.764 | 846.387 → 703.083 |
| 3T | Post-move | 118.943 | 45.225 → 37.449 | 3.18× | 488.776 → 394.886 | 830.642 → 726.477 |

| Game plan (60 calls per preset) | 1T wall time saved | 2T | 3T |
| --- | ---: | ---: | ---: |
| Attacking | 31.7% | 24.9% | 17.9% |
| Priming | 33.4% | 26.5% | 18.1% |
| Anchoring | 31.3% | 24.6% | 18.4% |
| Racing with contact | 32.4% | 25.4% | 18.4% |
| Pure race | 42.2% | 21.1% | 17.3% |
| Match (150 calls) | 32.7% | 25.9% | 20.0% |
| Unlimited (150 calls) | 32.3% | 24.4% | 16.3% |

Each probability MAE averages 2,890 equally weighted probability observations. Equity MAE averages 1,834 equity observations. The threshold applies separately to each observation. Direct shifts include all public checker alternatives, including those without a rollout.

| Preset | Probability MAE, checkpoint → selected | Equity MAE | Materially farther / closer | Largest direct probability / equity shift | Whole outputs identical (/300) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1P | 0.00698275 → 0.00698275 | 0.03526136 → 0.03526136 | 0 / 0 | 0.00000000 / 0.00000000 | 300 |
| 2P | 0.00544002 → 0.00544002 | 0.02736281 → 0.02736281 | 0 / 0 | 0.00000000 / 0.00000000 | 300 |
| 3P | 0.00514367 → 0.00514367 | 0.02431038 → 0.02431038 | 0 / 0 | 0.00000000 / 0.00000000 | 300 |
| 4P | 0.00445193 → 0.00445193 | 0.01940719 → 0.01940719 | 0 / 0 | 0.00000000 / 0.00000000 | 300 |
| 1T | 0.00464839 → 0.00464839 | 0.01707575 → 0.01707575 | 0 / 0 | 0.00000006 / 0.00000083 | 58 |
| 2T | 0.00415231 → 0.00415230 | 0.01591025 → 0.01591039 | 0 / 0 | 0.00004637 / 0.00008842 | 29 |
| 3T | 0.00369210 → 0.00369211 | 0.01160711 → 0.01160717 | 0 / 0 | 0.00007331 / 0.00018728 | 29 |

| Preset | Same best checker | Same full checker order | Same cube action | Checker agrees with rollout, checkpoint → selected | Cube agrees with rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1P | 100/100 | 100/100 | 100/100 | 83 → 83 | 86 → 86 |
| 2P | 100/100 | 100/100 | 100/100 | 81 → 81 | 83 → 83 |
| 3P | 100/100 | 100/100 | 100/100 | 87 → 87 | 95 → 95 |
| 4P | 100/100 | 100/100 | 100/100 | 87 → 87 | 98 → 98 |
| 1T | 100/100 | 100/100 | 100/100 | 92 → 92 | 98 → 98 |
| 2T | 100/100 | 100/100 | 100/100 | 90 → 90 | 96 → 96 |
| 3T | 100/100 | 100/100 | 100/100 | 90 → 90 | 98 → 98 |

Cumulative accuracy compares the selected build directly with the recorded original predictions using the same rollout references as the per-pass comparison.

| Preset | Probability MAE, original → selected | Equity MAE | Materially farther / closer | Best checker agrees with rollout, original → selected | Cube agrees with rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1P | 0.00698275 → 0.00698275 | 0.03526136 → 0.03526136 | 0 / 0 | 83 → 83 | 86 → 86 |
| 2P | 0.00544002 → 0.00544002 | 0.02736281 → 0.02736281 | 0 / 0 | 81 → 81 | 83 → 83 |
| 3P | 0.00514367 → 0.00514367 | 0.02431037 → 0.02431038 | 0 / 0 | 87 → 87 | 95 → 95 |
| 4P | 0.00445193 → 0.00445193 | 0.01940720 → 0.01940719 | 0 / 0 | 87 → 87 | 98 → 98 |
| 1T | 0.00464839 → 0.00464839 | 0.01707574 → 0.01707575 | 0 / 0 | 92 → 92 | 98 → 98 |
| 2T | 0.00418336 → 0.00415230 | 0.01599934 → 0.01591039 | 0 / 3 | 90 → 90 | 96 → 96 |
| 3T | 0.00369202 → 0.00369211 | 0.01160697 → 0.01160717 | 0 / 0 | 90 → 90 | 98 → 98 |

[Inputs, full analytics and per-call timings](qualified/windows/full_outputs.zip) · [Per-field errors and latency distributions](qualified/windows/comparison.json) · [Recomputed scores, hashes and subgroup errors](qualified/windows/verification.json)
