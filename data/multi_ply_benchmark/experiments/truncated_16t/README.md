# Truncated rollout benchmark on 16 CPU threads

This study measures the public Stage 11 checker-play, cube-action and post-move
analytics at 1T, 2T and 3T. Each preset uses the [same 100 checker positions
and 100 cube positions](../../README.md). The post-move call uses the cube
position with the board, scores and cube ownership switched to the previous
mover's perspective.

## Results

The 900 calls took **794.29 seconds on the original build and 411.11 seconds
on the final build**. That saves 383.18 seconds, or 48.2% of the measured wall
time. Each API column below shows original → final seconds for its 100 calls.

| Preset | Checker play | Cube action | Post-move | All 300 calls | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1T | 20.66 → 10.82 | 6.78 → 3.45 | 7.61 → 3.43 | 35.04 → 17.70 | 1.98× |
| 2T | 112.28 → 58.39 | 42.51 → 23.02 | 42.69 → 22.60 | 197.48 → 104.02 | 1.90× |
| 3T | 322.64 → 165.21 | 120.18 → 62.52 | 118.94 → 61.66 | 561.77 → 289.39 | 1.94× |

At 3T, checker-play p95 fell from 9.07 to 4.14 seconds per position.
Cube p95 fell from 2.50 to 1.13 seconds, and post-move p95 from 2.36 to
1.18 seconds. The [complete analytics archive](full_outputs.zip) contains
all 1,800 timed responses, the exact inputs, settings and timing statistics
by API, game plan and play format.

**898 of the 900 paired responses matched exactly**, including standard
errors. All checker rankings, cube decisions and cubeful equities matched.
The two differing responses were a 2T checker position with three values
materially closer to its rollout, and a 3T checker position with differences
below the thresholds. Zero values moved materially farther from a rollout.

The following mean absolute errors give each position equal weight. Within
a checker position, average over its genuinely rolled endpoints first.
Checker equity is the move's cubeful equity, and cube equity is the optimal
action's equity. These equity errors are identical for the two builds. The
win-probability column is in percentage points and uses the same weighting.

| Preset | Checker equity MAE | Cube equity MAE | Post-move equity MAE | Checker win MAE, original → final | Best checker matches rollout | Cube action matches rollout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1T | 0.02009 | 0.02857 | 0.02857 | 0.477 → 0.477 pp | 92/100 | 98/100 |
| 2T | 0.01363 | 0.03091 | 0.03091 | 0.375 → 0.360 pp | 90/100 | 96/100 |
| 3T | 0.01242 | 0.01884 | 0.01884 | 0.335 → 0.335 pp | 90/100 | 98/100 |

Among the three presets, 3T has the smallest equity MAE for every API on
this corpus. The [per-field comparison](comparison.json) includes original,
final and reference means, timing distributions, per-endpoint errors,
decision regret, and the identity of every material change.

## Presets

| Preset | Trials | Half-moves before truncation | Checker selection during a trial | Cube decision depth | Leaf evaluation |
| --- | ---: | ---: | --- | --- | --- |
| 1T | 72 | 5 | 1-ply | 1-ply | 1-ply |
| 2T | 360 | 7 | 2-ply on the first half-move, then 1-ply | 2-ply | 2-ply |
| 3T | 360 | 7 | 3-ply first, cached 1-ply reply, then 2-ply | 3-ply | 3-ply |

All three use variance reduction and the public presets' checker-move filters.
The native trial loop keeps its initial cube evaluator throughout the trial
and uses a 1-ply screen before escalating a possible double. The [rollout
specification](../../../../ROLLOUT.md) describes the policies and their
bearoff and terminal shortcuts.

## Measurement

The original is commit `139427d826dd67798226e7d23c003c90315306ba`. Its 73
C++ and Python files were verified against their Git blobs. Both builds used
MSVC Release on the same Windows Ryzen 7 9700X, the Stage 11 weights, the
same bearoff database and 16 logical CPUs. The native SHA-256 digests are
`973178549025a0c8bc2460f51dd10b547a229e29bfb874f8d994bb873c447011`
for the original and
`b8dddb53f7064bee9b98367019e1568e235486389740ea0b4f4db44dcfa468de`
for the optimized build.

Each reported build ran one complete pass at each level in a fresh process.
One public call ran at a time with 16 trial threads. The original timings
were recorded first. The candidate timings cover the build with the corrected
rollout move cache. Both arms visited the positions in the same order and
used the normal analyzer caches. Timing
surrounds each public call, excluding analyzer construction, serialization
and disk writes. The single pass measures elapsed time on this host. AWS
instance performance depends on the CPU and operating system.

Before each call the runner sets the seed to the unsigned big-endian integer
in the first four bytes of SHA-256 of `truncated-benchmark-v1:<source key>`.
Both builds use identical dice for the same position and preset. The frozen
full-game references use a separate seed schedule.

The [reference corpus](../../rollout_positions/README.md) covers all 300
positions/API combinations. It contains 378 genuine full-game checker
rollouts across the 100 checker decisions. Other legal moves have shallow
reference evaluations. We compare the entire public response between builds
and use the rolled checker endpoints when measuring accuracy against
full-game play. A change in absolute error exceeding 0.0025 in a probability
or 0.005 in an equity counts as material.

The reference values retain the original engine and dice policy. A frozen
reference produced by the original single-batch parallel cubeful path
shares its historical susceptibility to the cache race. The published
reference provenance identifies the source of each call. The serial
reference for the reproducibility check below uses the same truncated dice
as the parallel call and directly tests the thread-scheduling correction.

## Parallel cache correction

An original 2T checker result gave the fourth-ranked move on
`checker-money-priming-10` a win probability of 0.893650 and a win
standard error of 0.073448. The independent archived full-game rollout
for this endpoint gave 0.819967. The cubeful equity and move order were
stable, making the discrepancy visible only in the full analytics.

In the single-batch cubeful rollout an early trial could claim a
second-move cache entry whose prefill had already started. Both workers
could write the same entry and one could publish it while the other
was still accumulating the variance-reduction mean. The corrected engine
reserves the opening cache slots before dispatch, and each prefill owner
publishes the completed entry with a release store. An early trial waits
only for the roll it needs. The [regression test](../../../../tests/test_rollout_prefill.py)
compares complete serial and repeated parallel results on the two affected
benchmark positions.

| Fourth-ranked 2T move | Original | Final | Full-game rollout |
| --- | ---: | ---: | ---: |
| Win probability | 89.365% | 82.062% | 81.997% |
| Gammon-win probability | 20.513% | 18.806% | 18.704% |
| Cubeless equity | 0.99152 | 0.82800 | 0.82683 |

The win standard error fell from 0.07345 to 0.00621. The other changed
response was the second-ranked move on `checker-money-priming-07` at 3T.
Its largest probability shift was 0.0304 percentage points, and its cubeless
equity shifted by 0.0002955.

On the three positions flagged during investigation, the final build's
single-threaded replay and three repeated 16-thread calls with the same seed each
reproduced its complete timed response. Replaying the original parallel
build also returned those values on these attempts. That variation across
runs is consistent with the ownership defect in the original code. At 1T
the final build reproduced the original timed result on every position.
[Replay records and direct 3P check](cache_verification.zip)

The final build also reproduced the original deterministic 3P digest for
all 300 benchmark calls, with 3.97 seconds of total call time. All 20
targeted tests of rollout prefill, cube detail, cube-batch reuse and
bearoff cache behavior passed. [Targeted test results](targeted-pytest.json)

## Reproduce

From the engine repository root, use the runner once for each build and
preset. Both source directories and binaries must be inside the engine
checkout. Supply the original Python package with the original binary.

```sh
python scripts/benchmark_multi_ply.py --levels 1T --threads 16 --model stage11 --python-dir build/original/python --native-source build/original --extension build/original-bin/bgbot_cpp.<platform-extension> --output logs/truncated_benchmark_16t/1T-baseline
python scripts/benchmark_multi_ply.py --levels 1T --threads 16 --model stage11 --python-dir python --native-source . --extension build/optimized-bin/bgbot_cpp.<platform-extension> --output logs/truncated_benchmark_16t/1T-candidate
```

Repeat for 2T and 3T with the level and output directory changed. To rescore
the saved measurements, extract [the full outputs](full_outputs.zip) under
`logs/truncated_benchmark_16t/` and run the following command. The comparison
checks seeds, call order, position coverage, endpoint identity and the public
arguments before reporting the per-field changes and rollout errors.

```sh
python scripts/compare_truncated_benchmark.py --baseline logs/truncated_benchmark_16t/1T-baseline/results.jsonl logs/truncated_benchmark_16t/2T-baseline/results.jsonl logs/truncated_benchmark_16t/3T-baseline/results.jsonl --candidate logs/truncated_benchmark_16t/1T-candidate/results.jsonl logs/truncated_benchmark_16t/2T-candidate/results.jsonl logs/truncated_benchmark_16t/3T-candidate/results.jsonl --output logs/truncated_benchmark_16t/comparison.json
```
