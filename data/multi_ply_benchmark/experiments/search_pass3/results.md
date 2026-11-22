# Full-corpus timings and accuracy

Each row covers 100 checker, 100 cube and 100 post-move calls on the same 16-logical-CPU Ryzen 7 9700X. Linux timings were measured locally under WSL2 using GCC on x86-64. Windows uses MSVC. Each arm and preset starts in a fresh process. Timing excludes analyzer construction, serialization and disk writes.

## Linux

| Preset | Original (s) | Previous (s) | This pass (s) | Less time this pass | Speedup since the original |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1P | 11.145 | 0.021 | 0.020 | 3.2% | 545.55× |
| 2P | 12.727 | 0.355 | 0.220 | 38.1% | 57.92× |
| 3P | 18.291 | 2.515 | 1.868 | 25.7% | 9.79× |
| 4P | 153.584 | 47.637 | 32.401 | 32.0% | 4.74× |
| 1T | — | 9.859 | 7.953 | 19.3% | — |
| 2T | — | 67.298 | 53.410 | 20.6% | — |
| 3T | — | 239.575 | 191.230 | 20.2% | — |

The original deterministic engine is `139427d`. The previous engine is `82e6397`. Original timings come from the published historical run with the same inputs. The deterministic timings are freshly paired. All final truncated calls were retimed; their control recordings come from the matched seven-preset run earlier in this study. Dashes indicate that this study has no original measurement on that platform.

| Preset | Checker play, previous → current (s) | Cube action (s) | Post-move (s) |
| --- | ---: | ---: | ---: |
| 1P | 0.015 → 0.014 | 0.004 → 0.004 | 0.003 → 0.002 |
| 2P | 0.188 → 0.148 | 0.098 → 0.045 | 0.069 → 0.027 |
| 3P | 2.122 → 1.604 | 0.237 → 0.149 | 0.156 → 0.115 |
| 4P | 40.715 → 27.197 | 4.103 → 2.738 | 2.819 → 2.466 |
| 1T | 5.832 → 4.731 | 2.008 → 1.619 | 2.020 → 1.603 |
| 2T | 37.105 → 30.057 | 15.290 → 11.662 | 14.904 → 11.692 |
| 3T | 134.565 → 107.683 | 51.777 → 41.292 | 53.234 → 42.255 |

| Deep search | API | Median, previous → current (ms) | p95, previous → current (ms) |
| --- | --- | ---: | ---: |
| 3P | checker_play | 14.2 → 10.7 | 63.2 → 46.0 |
| 3P | cube_action | 2.0 → 1.4 | 5.4 → 3.2 |
| 3P | post_move_analytics | 1.3 → 1.0 | 3.6 → 2.6 |
| 4P | checker_play | 250.2 → 132.7 | 1193.5 → 776.0 |
| 4P | cube_action | 42.9 → 28.1 | 87.5 → 56.7 |
| 4P | post_move_analytics | 28.8 → 24.7 | 60.4 → 56.5 |

Each game-plan cell includes 20 checker, 20 cube and 20 post-move calls.

| Game plan | 2P time saved | 3P time saved | 4P time saved |
| --- | ---: | ---: | ---: |
| attacking | 31.2% | 22.4% | 29.3% |
| priming | 46.6% | 27.9% | 31.6% |
| anchoring | 46.2% | 26.9% | 33.7% |
| racing with contact | 28.3% | 25.4% | 35.6% |
| pure race | 10.7% | 25.8% | 30.5% |

The following counts compare absolute error with the existing full-game rollout for the same position, checker endpoint and field. Material means a change exceeding 0.0025 for a probability or 0.005 for an equity. The direct shift includes returned legal alternatives with no rolled equity. Tiny changes in low-ranked alternatives can change their order.

| Preset | Rollout fields | Materially farther / closer | Max direct probability shift | Max direct equity shift | Same best checker / cube | Same full checker order |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1P | 4,724 | 0 / 0 | 0.00000000 | 0.00000000 | 100/100, 100/100 | 100/100 |
| 2P | 4,724 | 0 / 0 | 0.00000036 | 0.00000271 | 100/100, 100/100 | 100/100 |
| 3P | 4,724 | 0 / 0 | 0.00000172 | 0.00000231 | 100/100, 100/100 | 100/100 |
| 4P | 4,724 | 0 / 0 | 0.00000691 | 0.00000715 | 100/100, 100/100 | 100/100 |
| 1T | 4,724 | 0 / 0 | 0.00000000 | 0.00000000 | 100/100, 100/100 | 100/100 |
| 2T | 4,724 | 0 / 0 | 0.00000036 | 0.00000155 | 100/100, 100/100 | 100/100 |
| 3T | 4,724 | 0 / 0 | 0.00000036 | 0.00000154 | 100/100, 100/100 | 100/100 |

[Full release inputs, timings and responses](release/linux/full_outputs.zip) · [Accuracy and latency distributions by API](release/linux/comparison.json) · [Earlier paired run, before the Linux 2P race adjustment](linux/full_outputs.zip)

## Windows

| Preset | Original (s) | Previous (s) | This pass (s) | Less time this pass | Speedup since the original |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1P | 2.847 | 0.020 | 0.020 | -0.1% | 143.26× |
| 2P | 4.444 | 0.330 | 0.174 | 47.2% | 25.54× |
| 3P | 11.312 | 2.515 | 1.889 | 24.9% | 5.99× |
| 4P | 176.073 | 52.642 | 31.885 | 39.4% | 5.52× |
| 1T | 35.043 | 10.457 | 8.432 | 19.4% | 4.16× |
| 2T | 197.482 | 64.995 | 52.704 | 18.9% | 3.75× |
| 3T | 561.767 | 226.101 | 191.970 | 15.1% | 2.93× |

The original deterministic engine is `139427d`. The previous engine is `82e6397`. Original timings come from the published historical run with the same inputs. The deterministic timings are freshly paired. All final truncated calls were retimed; their control recordings come from the matched seven-preset run earlier in this study. Dashes indicate that this study has no original measurement on that platform.

| Preset | Checker play, previous → current (s) | Cube action (s) | Post-move (s) |
| --- | ---: | ---: | ---: |
| 1P | 0.015 → 0.015 | 0.003 → 0.003 | 0.002 → 0.002 |
| 2P | 0.157 → 0.127 | 0.101 → 0.028 | 0.071 → 0.019 |
| 3P | 2.130 → 1.653 | 0.235 → 0.135 | 0.151 → 0.100 |
| 4P | 44.918 → 26.629 | 4.640 → 2.932 | 3.085 → 2.325 |
| 1T | 6.093 → 4.974 | 2.184 → 1.735 | 2.179 → 1.723 |
| 2T | 37.543 → 29.836 | 13.812 → 11.546 | 13.640 → 11.321 |
| 3T | 126.882 → 107.962 | 49.585 → 41.314 | 49.635 → 42.694 |

| Deep search | API | Median, previous → current (ms) | p95, previous → current (ms) |
| --- | --- | ---: | ---: |
| 3P | checker_play | 14.4 → 10.9 | 57.5 → 51.2 |
| 3P | cube_action | 1.9 → 1.1 | 6.1 → 3.4 |
| 3P | post_move_analytics | 1.3 → 0.8 | 3.8 → 2.6 |
| 4P | checker_play | 326.2 → 137.4 | 1132.3 → 779.2 |
| 4P | cube_action | 48.8 → 30.3 | 95.6 → 60.5 |
| 4P | post_move_analytics | 31.6 → 23.3 | 67.0 → 50.2 |

The following counts compare absolute error with the existing full-game rollout for the same position, checker endpoint and field. Material means a change exceeding 0.0025 for a probability or 0.005 for an equity. The direct shift includes returned legal alternatives with no rolled equity. Tiny changes in low-ranked alternatives can change their order.

| Preset | Rollout fields | Materially farther / closer | Max direct probability shift | Max direct equity shift | Same best checker / cube | Same full checker order |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1P | 4,724 | 0 / 0 | 0.00000000 | 0.00000000 | 100/100, 100/100 | 100/100 |
| 2P | 4,724 | 0 / 0 | 0.00000024 | 0.00000143 | 100/100, 100/100 | 100/100 |
| 3P | 4,724 | 0 / 0 | 0.00000172 | 0.00000231 | 100/100, 100/100 | 100/100 |
| 4P | 4,724 | 0 / 0 | 0.00000703 | 0.00000742 | 100/100, 100/100 | 100/100 |
| 1T | 4,724 | 0 / 0 | 0.00000000 | 0.00000000 | 100/100, 100/100 | 100/100 |
| 2T | 4,724 | 0 / 0 | 0.00000024 | 0.00000083 | 100/100, 100/100 | 100/100 |
| 3T | 4,724 | 0 / 0 | 0.00000024 | 0.00000154 | 100/100, 100/100 | 100/100 |

[Full release inputs, timings and responses](release/windows/full_outputs.zip) · [Accuracy and latency distributions by API](release/windows/comparison.json) · [Earlier paired run, before the Linux 2P race adjustment](windows/full_outputs.zip)

## Independent Linux repeat

Fresh processes reversed the initial build order and reran the full corpus.

| Preset | Previous (s) | This pass (s) | Less time |
| --- | ---: | ---: | ---: |
| 1P | 0.022 | 0.021 | 1.6% |
| 2P | 0.352 | 0.215 | 38.9% |
| 3P | 2.453 | 1.889 | 23.0% |
| 4P | 47.054 | 31.780 | 32.5% |

[Inputs and complete repeat outputs](release/linux-repeat/full_outputs.zip)
