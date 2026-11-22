# Full-corpus CPU benchmark results

Times are wall seconds for 100 checker, 100 cube and 100 post-move calls. Each build uses 16 CPU workers, Stage 11, the same boards, match contexts and bearoff database. The original is `139427d` and the checkpoint is `33c4aa1`.

## Linux (GCC, x86-64)

| Preset | Original s | Checkpoint s | Final s | Time saved this pass | Time saved since start | Speedup since start |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1P | 11.145 | 10.157 | 0.021 | 99.8% | 99.8% | 529.81× |
| 2P | 12.727 | 0.899 | 0.351 | 61.0% | 97.2% | 36.28× |
| 3P | 18.291 | 3.074 | 2.492 | 18.9% | 86.4% | 7.34× |
| 4P | 153.584 | 55.844 | 49.436 | 11.5% | 67.8% | 3.11× |

Each entry below is checkpoint → final seconds per 100 calls.

| Preset | Checker play | Cube action | Post-move |
| --- | ---: | ---: | ---: |
| 1P | 0.306 → 0.015 | 9.848 → 0.004 | 0.003 → 0.002 |
| 2P | 0.725 → 0.183 | 0.103 → 0.096 | 0.071 → 0.071 |
| 3P | 2.653 → 2.137 | 0.256 → 0.213 | 0.165 → 0.142 |
| 4P | 46.695 → 42.068 | 5.472 → 4.307 | 3.677 → 3.062 |

Single-call latency on the deeper deterministic searches, in milliseconds.

| Preset | API | Median before → after | p95 before → after |
| --- | --- | ---: | ---: |
| 3P | checker_play | 19.2 → 13.9 | 67.6 → 57.7 |
| 3P | cube_action | 2.1 → 1.8 | 5.7 → 4.7 |
| 3P | post_move_analytics | 1.4 → 1.1 | 3.5 → 3.3 |
| 4P | checker_play | 293.7 → 263.5 | 1250.8 → 1093.1 |
| 4P | cube_action | 60.2 → 46.6 | 108.5 → 91.6 |
| 4P | post_move_analytics | 35.9 → 30.9 | 81.4 → 70.6 |

The accuracy columns count fields with material changes in absolute error against the frozen rollout. Complete-response equality includes every field and the full order of legal moves.

| Preset | Scored fields | Farther / closer vs checkpoint | Farther / closer vs original | Identical responses vs checkpoint | Same checker order / cube choice vs checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1P | 4,724 | 0 / 0 | 0 / 0 | 300/300 | 100/100, 100/100 |
| 2P | 4,724 | 0 / 0 | 0 / 0 | 300/300 | 100/100, 100/100 |
| 3P | 4,724 | 0 / 0 | 0 / 0 | 300/300 | 100/100, 100/100 |
| 4P | 4,724 | 0 / 0 | 0 / 0 | 300/300 | 100/100, 100/100 |

Agreement of the final engine with the rollout reference. The checker equity test includes only selected moves with a rolled equity.

| Preset | Matches best checker | Within 0.005 of best / scorable | Matches optimal cube action |
| --- | ---: | ---: | ---: |
| 1P | 83/100 | 90/99 | 86/100 |
| 2P | 81/100 | 92/100 | 83/100 |
| 3P | 87/100 | 97/100 | 95/100 |
| 4P | 87/100 | 96/100 | 98/100 |

[All timed analytics, inputs and group timings](linux/full_outputs.zip) · [Checkpoint accuracy by field and API](linux/checkpoint_comparison.json) · [Original accuracy by field and API](linux/original_comparison.json)

## Windows (MSVC, x86-64)

| Preset | Original s | Checkpoint s | Final s | Time saved this pass | Time saved since start | Speedup since start |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1P | 2.847 | 2.752 | 0.021 | 99.3% | 99.3% | 138.68× |
| 2P | 4.444 | 1.142 | 0.334 | 70.8% | 92.5% | 13.31× |
| 3P | 11.312 | 4.027 | 2.639 | 34.5% | 76.7% | 4.29× |
| 4P | 176.073 | 62.655 | 53.586 | 14.5% | 69.6% | 3.29× |
| 1T | 35.043 | 10.954 | 10.283 | 6.1% | 70.7% | 3.41× |
| 2T | 197.482 | 71.486 | 65.673 | 8.1% | 66.7% | 3.01× |
| 3T | 561.767 | 263.273 | 232.095 | 11.8% | 58.7% | 2.42× |

Each entry below is checkpoint → final seconds per 100 calls.

| Preset | Checker play | Cube action | Post-move |
| --- | ---: | ---: | ---: |
| 1P | 0.676 → 0.016 | 2.074 → 0.003 | 0.003 → 0.002 |
| 2P | 0.943 → 0.163 | 0.117 → 0.098 | 0.082 → 0.072 |
| 3P | 3.548 → 2.290 | 0.305 → 0.212 | 0.174 → 0.138 |
| 4P | 52.497 → 45.871 | 6.028 → 4.597 | 4.130 → 3.119 |
| 1T | 6.439 → 6.085 | 2.246 → 2.103 | 2.270 → 2.095 |
| 2T | 40.880 → 37.029 | 15.469 → 14.481 | 15.138 → 14.163 |
| 3T | 149.036 → 130.373 | 57.716 → 50.843 | 56.521 → 50.879 |

Single-call latency on the deeper deterministic searches, in milliseconds.

| Preset | API | Median before → after | p95 before → after |
| --- | --- | ---: | ---: |
| 3P | checker_play | 27.3 → 14.6 | 87.2 → 63.1 |
| 3P | cube_action | 2.4 → 1.7 | 8.2 → 5.5 |
| 3P | post_move_analytics | 1.4 → 1.1 | 4.4 → 3.3 |
| 4P | checker_play | 375.0 → 324.0 | 1321.6 → 1148.0 |
| 4P | cube_action | 65.2 → 48.0 | 121.7 → 94.7 |
| 4P | post_move_analytics | 40.9 → 31.4 | 92.0 → 67.1 |

The accuracy columns count fields with material changes in absolute error against the frozen rollout. Complete-response equality includes every field and the full order of legal moves.

| Preset | Scored fields | Farther / closer vs checkpoint | Farther / closer vs original | Identical responses vs checkpoint | Same checker order / cube choice vs checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1P | 4,724 | 0 / 0 | 0 / 0 | 300/300 | 100/100, 100/100 |
| 2P | 4,724 | 0 / 0 | 0 / 0 | 300/300 | 100/100, 100/100 |
| 3P | 4,724 | 0 / 0 | 0 / 0 | 300/300 | 100/100, 100/100 |
| 4P | 4,724 | 0 / 0 | 0 / 0 | 300/300 | 100/100, 100/100 |
| 1T | 4,724 | 0 / 0 | 0 / 0 | 300/300 | 100/100, 100/100 |
| 2T | 4,724 | 0 / 0 | 0 / 3 | 300/300 | 100/100, 100/100 |
| 3T | 4,724 | 0 / 0 | 0 / 0 | 300/300 | 100/100, 100/100 |

Agreement of the final engine with the rollout reference. The checker equity test includes only selected moves with a rolled equity.

| Preset | Matches best checker | Within 0.005 of best / scorable | Matches optimal cube action |
| --- | ---: | ---: | ---: |
| 1P | 83/100 | 90/99 | 86/100 |
| 2P | 81/100 | 92/100 | 83/100 |
| 3P | 87/100 | 97/100 | 95/100 |
| 4P | 87/100 | 96/100 | 98/100 |
| 1T | 92/100 | 97/100 | 98/100 |
| 2T | 90/100 | 97/100 | 96/100 |
| 3T | 90/100 | 98/100 | 98/100 |

[All timed analytics, inputs and group timings](windows/full_outputs.zip) · [Checkpoint accuracy by field and API](windows/checkpoint_comparison.json) · [Original accuracy by field and API](windows/original_comparison.json)

## Linux truncated evaluations

The 1T checkpoint was rerun and 2T/3T use the published 16-worker Linux runs. The Windows table above also covers the original engine for all three truncated presets.

| Preset | Checkpoint s | Final s | Time saved | Identical responses |
| --- | ---: | ---: | ---: | ---: |
| 1T | 10.125 | 10.111 | 0.1% | 300/300 |
| 2T | 77.580 | 67.918 | 12.5% | 300/300 |
| 3T | 259.308 | 237.109 | 8.6% | 300/300 |

[All Linux truncated outputs](linux-truncated/full_outputs.zip) · [Accuracy by field](linux-truncated/comparison.json)

## Repeatability on Linux

A second run started fresh processes and reversed the order of the two builds. Each row covers 300 calls per build per pass. The pooled figure divides total seconds saved by total checkpoint seconds across both passes.

| Preset | Time saved, first pass | Time saved, reverse order | Time saved, pooled |
| --- | ---: | ---: | ---: |
| 1P | 99.8% | 99.8% | 99.8% |
| 2P | 61.0% | 60.8% | 60.9% |
| 3P | 18.9% | 20.8% | 19.9% |
| 4P | 11.5% | 5.5% | 8.5% |

[All reverse-order outputs](linux-repeat/full_outputs.zip) · [Repeat scorecard](linux-repeat/comparison.json)
