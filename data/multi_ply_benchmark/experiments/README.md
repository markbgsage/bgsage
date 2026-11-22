# 3-ply search experiments, November 21, 2026

These experiments use the fixed 100 checker positions and 100 cube/post-move
positions, production Stage 11 weights, the bearoff DB, and cubeful public API
calls. The checker comparison covers every legal candidate (1,709 in total).
The machine has a Ryzen 7 9700X with eight physical cores and 16 logical CPUs.

## Comparison method

The control is a fresh MSVC Release build of `139427d826dd` with its original
Python package. The candidate uses the same toolchain, weights, and build
options. We also check against the historical binary that produced the first
timing run. Rebuilding the original source can change a few floating-point
values at approximately the sixth decimal place. Use the matched builds to
attribute changes to the optimizations.

The paired runner verified 73 native and Python files against their Git
blobs before measuring the control. An earlier run used a preserved binary
whose recorded source directory had subsequently been reused for profiling.
The official comparison uses a fresh control built from a verified export.
Its SHA-256 is `973178549025a0c8bc2460f51dd10b547a229e29bfb874f8d994bb873c447011`.
The optimized binary is `7476f24d8d5dcf69eb62ca05bc714ac0c6872cfd3204ba8d8b06923982170633`.
Each timing archive includes the loaded binary hash and source fingerprints.

The whole-API timing study uses the default cache behavior, public calls,
fresh processes, the same order of distinct positions, and two pairs per
thread count. The second pair reverses the execution order. Timing excludes
process startup, model construction, and result serialization. All 16 logical
CPUs are available to both arms. The separate reference workers are suspended
for this measurement and resumed afterward.

The accuracy target is the public full-game rollout for the benchmark
position. The database already supplies 37 full-game checker decisions with
224 rolled moves. Those historical 3-ply rollouts used variance reduction
and 1,296-game batches, with an adaptive stopping rule. The other checker
positions use the normal public rollout selector and promotion rules, with
3-ply checker and cube decisions and one fixed 1,296-game batch. All 100 cube
and 100 post-move positions already had complete results from the frozen
reference run. Match checker moves by resulting board and score only moves
actually rolled. The public output preserves lower-ply values and their
evaluation labels. The scorecard reports the two reference sources separately.
For each scalar, compare `abs(candidate - rollout) - abs(control - rollout)`.
A rise over 0.0025 in probability (0.25 percentage points), or over 0.005 in
equity, is a material regression. Also compare absolute errors, best checker
moves, cube actions, and every returned API field. The rollout standard errors
are saved with the targets. For example, a contact cube decision in the first
pass had ND and DT standard errors of 0.0103 and 0.0124.

The [completed reference corpus](../rollout_positions/results.jsonl) contains
all 300 benchmark calls. For the 63 fresh checker decisions, the public API
selected 154 moves for rollout. We reused 149 matching checkpoints and ran
five missing 1,296-game moves across four positions. The 37 archived checker
decisions supplied the other 224 rolled moves. Existing frozen cube and
post-move calls supplied all 200 results for those APIs.

## End-to-end timing results

Each time is the mean wall time in seconds for a pass of 100 calls, averaged
across the two execution orders. Speedups are geometric means of the paired
baseline/candidate ratios.

| Threads | Checker play | Cube action | Post-move | All 300 calls | Wall time saved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 27.08 → 9.57 s (2.83×) | 4.38 → 0.95 s (4.62×) | 1.22 → 0.57 s (2.16×) | 32.68 → 11.08 s | 66.1% |
| 4 | 15.72 → 5.51 s (2.85×) | 3.41 → 0.63 s (5.40×) | 0.86 → 0.42 s (2.06×) | 19.99 → 6.56 s | 67.2% |
| 8 | 9.38 → 3.63 s (2.58×) | 2.97 → 0.46 s (6.51×) | 0.57 → 0.29 s (1.93×) | 12.92 → 4.38 s | 66.1% |
| 16 | 7.58 → 3.30 s (2.30×) | 2.89 → 0.42 s (6.87×) | 0.44 → 0.23 s (1.91×) | 10.92 → 3.96 s | 63.8% |

At sixteen threads the mean of the two per-run p95s fell from 226 to 78 ms
for checker play, 40 to 9 ms for cube action, and 11 to 5 ms for post-move.
Each game plan accounted for 20 positions per API, equally split across money
and match. The speedups below use the same paired runs at sixteen threads.

| Game plan | Checker play | Cube action | Post-move |
| --- | ---: | ---: | ---: |
| Attacking | 2.56× | 6.13× | 2.08× |
| Priming | 2.50× | 6.42× | 2.19× |
| Anchoring | 2.25× | 7.55× | 1.92× |
| Racing with contact | 1.95× | 5.87× | 1.74× |
| Pure race | 1.64× | 10.63× | 1.20× |

All sixteen runs produced the same 300 complete analytics objects, including
ordered checker moves, probabilities, equities, and cube actions. They had
identical SHA-256 digests over the canonical call records with timing removed.
The [paired statistics](paired_3ply.json) include the per-run means, medians,
p95s and digests. The [raw benchmark outputs](paired_3ply_outputs.zip) contain
all 4,800 timed calls plus each run's metadata and grouped statistics.

Against the earlier archived binary, the largest change among 9,545 reported
probabilities was 0.000001724, and the largest among 5,827 equity fields was
0.000000805. Those maxima also bound the change in absolute error against
any common rollout for the affected field. The 3-ply outputs of the
matched-toolchain control reproduced the optimized predictions exactly.

### Linux x86-64 CPU check

The same 300 public calls ran under local Ubuntu WSL2 on the Ryzen using
GCC 15.2, the engine's Linux Release flags and the same Stage 11 weights.
Each row is one complete pass per binary. The original went first at two
threads and the optimized engine went first at sixteen.

| Threads | Checker play | Cube action | Post-move |
| ---: | ---: | ---: | ---: |
| 2 | 20.008 → 7.760 s (2.58×) | 12.541 → 0.992 s (12.64×) | 1.051 → 0.559 s (1.88×) |
| 16 | 6.232 → 2.684 s (2.32×) | 11.683 → 0.500 s (23.37×) | 0.431 → 0.263 s (1.64×) |

The original cube call reopened model weights on every position. WSL2 read
them through the Windows-mounted filesystem. The cube speedup therefore
includes filesystem overhead particular to this setup. AWS instance latency
and throughput require measurements on the target machine.

All checker rankings and cube actions matched. Of 15,372 probabilities and
equity values, the largest probability change was 0.000000120 and the largest
equity change was 0.000000954. Both are below the materiality limits even
before accounting for the direction of the change against a rollout.
The completed Linux comparison recorded zero material regressions.
[Linux methods and timings](linux_x86_64/README.md) · [Linux rollout accuracy](linux_x86_64/rollout_accuracy.json)

## Effect on rollouts

The [1T, 2T and 3T benchmark](truncated_16t/README.md) measures the full
100-position sets for checker play, cube action and post-move analytics at
each preset and compares their predictions with the fixed full-game rollouts.

The [rollout guard](../optimization_rollout_guard/README.md) measures public
calls on a match attacking position and an unlimited pure race for each API
at four threads. The production 360-game truncated preset sped up by 1.80×
for checker play, 1.66× for cube, and 1.65× for post-move. Full-game 3-ply
checker and cube play with a 36-game timing batch sped up by 1.61×, 1.64×,
and 1.63× respectively. All 24 paired outputs matched, including standard
errors.

At [1,296 full games per evaluation](../optimization_rollout_guard/full1296/summary.json),
the four checker calls per engine totaled 543.46 → 326.44 s (1.66×), cube
action 147.37 → 93.48 s (1.58×), and post-move 145.24 → 91.44 s (1.59×).
All 12 paired analytics, including their standard errors, matched exactly.
These throughput comparisons used the same two positions for each API.
The separate [accuracy scorecard](full_depth_accuracy.json) compares
3-ply predictions with the stored position rollouts. It reports both
endpoint-weighted error and the mean across completed positions so that
a checker roll with many legal moves need not dominate the result.

Of 1,709 legal checker moves in the returned position analytics, 378 have
rollout values. The other 896 one-ply and 435 two-ply moves retain their
reported depths and are excluded from absolute-error scoring. The original
and optimized engines both chose the rollout's best checker move on 87 of
100 positions. On 97 positions the chosen move was within 0.005 equity of
the rollout winner. The mean equity lost was 0.000524 and the maximum was
0.019250. Across the 378 rolled moves, mean absolute errors were 0.019704
in cubeful equity and 0.005149 in win probability. Giving each checker
position equal weight gives 0.021175 and 0.005563 respectively.

On all 100 cube references, both builds chose the rollout's cube action in
95 cases. Their mean absolute error was 0.008884 for win probability and
0.044907 for optimal equity. On all 100 post-move references, the corresponding
win-probability and cubeful-equity errors were 0.008736 and 0.077995.

## Ten ideas tested

The [public-call profiles](python_path_profiles.zip) cover three positions
per source and game plan, giving 30 per API. On the unmodified checker path,
533 per-move cube-aware traversals consumed 12.15 worker-seconds, overlapping
on two threads. The separate cubeless pass cost 0.63 s of the 6.89 s wall time.
For cube action, 1.20 s of 1.22 s was inside the unified binding. For post-move,
0.338 s of 0.356 s was inside the cubeful probability search. Those measurements
motivated both the Python call-path changes and the native evaluator profile.

| Idea | Result |
| --- | --- |
| 1. Skip preliminary cubeless search in cubeful checker analytics | Retained. The final pass evaluates every legal endpoint at full depth and overwrites both the equity and probabilities from the preliminary pass. On a 30-position screen, removing 0.63 s of redundant search reduced 6.89 s to 6.28 s with all results identical. |
| 2. Reuse the analyzer's loaded strategy for cube decisions | Retained. The original binding read and initialized a model for every cube position. The analyzer now supplies its strategy, and the cache key distinguishes different move-filter contexts. Both the original cubeful and cubeless searches still run. |
| 3. Hash legal endpoint boards with a vectorizable dot product | Retained. Hash collisions still require full board equality. This and idea 4 preserve first-seen move order. |
| 4. Skip identical intermediate boards when playing doubles | Retained. Equal boards at the same depth have the same remaining dice and legal continuations. Together with idea 3, this reduced 25,200 move-generation calls from 391 ms to 106 ms with identical ordered endpoints. |
| 5. Reuse blocked-point masks across escape features | Retained. Each side builds its mask once, and containment, mobility, and rescue read windows from those masks. |
| 6. Precompute double-hit geometry | Retained. The relationships among die distances fit a small constant table. Ideas 5 and 6 together reproduced all 7,805,316 checked input floats and all 300 public analytics results. |
| 7. Screen checker moves at 2-ply and promote close moves | Rejected for the full analytics API. A preliminary 3.97× checker speedup preserved every first choice, but left 1,135 moves at 2-ply. Of these, 880 changed cubeful equity by over 0.005. All four screened moves with position-benchmark rollout values materially worsened in cubeful equity. |
| 8. Cache exact extended contact features by board | Discarded. The 512-entry cache preserved every result but slowed the three calls in aggregate after ideas 5 and 6. |
| 9. Change checker scheduling | Discarded. On the 30-position screen, reusing Python workers took 7.21 s versus 6.89 s for the control. Parallelizing rolls within each candidate took 8.92 s versus 6.28 s for the direct candidate-parallel traversal. |
| 10. Share identical neural batches across the two cube-search policies | Retained. The policies still choose their own moves. Against the already optimized engine this reduced cube time by another 9.8% at two threads and 6.9% at sixteen, with identical full outputs. |

Other trials included skipping contact pip counts, hoisting leaf cube-state
expansion, changing hidden-layer accumulation, and vectorizing detection
of changed neural inputs. Sharing a neural delta base across different
rolls took 7.005 s versus 7.067 s and changed some floating-point values;
that version was discarded. Per-experiment timings and limitations are in the
[native search study](native_search/README.md) and
[encoder study](encoder_trials.md). The instrumented native pass attributed
89% of its elapsed time to candidate encoding, routing, and neural evaluation.

Two further CPU experiments improved that part of the search. The contact
encoder represents occupied points, made points and enemy blots as 24-bit
masks. For each roll it finds hittable blots and the largest pip loss with
shifts and intersections. The bar-entry case retains its separate pip-loss
calculation. This cut median contact encoding from 3,964 to 1,563 ns in the
component trial. The integrated source reproduced all 259,508,480 feature
floats on 589,792 boards, including both the real-position corpus and an
independently seeded board generator. [CPU encoder study](cpu_feature_masks/README.md)

The existing x86 AVX2 neural path now detects changed input columns eight at
a time and applies four ordered updates to each group of hidden activations
before storing it. On top of the new encoder, this saved a further 6.3–6.7%
in whole-call time at two threads and 3.7–6.6% at sixteen. The engine also
supports its original scalar path. The benchmarks used the local Ryzen CPU.
Performance on AWS instance families, including Graviton, is unmeasured.
[CPU neural study](cpu_nn/README.md)

The [post-move equity study](post_move_tree/README.md) also tested returning
the cubeful equity already calculated inside the search. This reduced mean
absolute error from 0.077995 to 0.045403 on the 100 fixed post-move cases.
Seventy-two equities moved materially closer and six moved materially farther.
An independently sampled set of 84 historical rollout positions gave 69
material improvements and one material regression. All five probabilities
and cubeless equity stayed identical. The timing pairs did not establish
a speed difference between the two methods. The production path retains
the original cubeful equity because six primary cases exceeded the material
regression threshold.

## Sharing cube-search work

The cube API walks both a live-cube tree and a cubeless tree. Their policies
can choose different moves. A prototype returned the live-cube tree's
probabilities directly. It substantially accelerated the cube call, but 11
probability fields and two cubeless equities moved materially farther from
the fixed rollouts. One gammon-win error increased by 1.52 percentage points.
The [cube policy study](cube_policy_experiments.md) reports the paired
probability errors and the cases where the two move policies diverged.

A second implementation preserves both choices. When both trees encounter
the same board, dice, evaluator, pinned network and complete ordered
candidate list, it reuses the previously computed neural outputs. Distinct
positions and reordered or filtered lists are evaluated normally. Scratch is
private to one public 3-ply cube call and bounded at about 149 KiB.

We measured the reviewed engine, this implementation, and a version that also
cached legal-move lists. For each thread count the three arms ran in all six
orders on an otherwise available machine. Each arm produced the same complete
analytics digest over all 100 cube cases.

| Threads | Reviewed engine | Share neural outputs | Also share legal lists |
| --- | ---: | ---: | ---: |
| 2 | 1.2477 s | 1.1254 s | 1.1165 s |
| 16 | 0.5367 s | 0.4998 s | 0.5112 s |

Sharing neural outputs improved on the reviewed engine in every one of the 12
matched blocks. Adding move lists raised the scratch bound to 1.62 MiB and
gave mixed incremental timings. See the [paired measurements](cube_memo_three_way.json)
and [3,600 full timed outputs](cube_memo_three_way_outputs.zip).

## Checks outside the timing corpus

Reusing a loaded strategy makes successive calls with different low-level
options especially important. A seven-checker race exposed reuse of a neural
value after switching to a bearoff DB. A contact position exposed reuse of
automatic cube efficiency after switching to a small positive override.
The cache key now includes the bearoff DB generation and exact override.
The native bindings honor an explicit DB even for a preloaded raw model,
reuse an existing wrapper for the same DB, and keep nested wrappers alive.
The [cache regression tests](../../../tests/test_multi_ply_bearoff_cache.py)
exercise these cases. The integrated build passed 258 engine tests, including
2–4 ply details, checker/cube symmetry, concurrent requests, rollout callbacks
and variance reduction. Twenty tests skipped missing data fixtures. Nine of
these subsequently passed on the 80 fixed contact positions. Three further
tests require an external GNU Backgammon installation. The CPU harness ignored
an unavailable optional DLL search directory during bearoff-test setup.
The [regression report](cpu_regression.json) and [full logs](cpu_regression_outputs.zip)
record the fixtures, test outcomes and separate legacy diagnostic discrepancies.

## Reproduction

Use `scripts/build_multi_ply_engine.ps1` to build a separate Windows Release
binary for each source tree. Use `scripts/measure_multi_ply_pair.py` with
`--baseline-python`, `--baseline-extension`, `--baseline-native-source`
(the clean export), `--baseline-revision 139427d826dd`, `--candidate-extension`,
`--threads 2 4 8 16 --pairs 2 --cpu-mask 0xffff` and a new `--output` directory.
The runner writes full analytics for every arm alongside the paired timings.
Use `scripts/compare_multi_ply_accuracy.py` on the baseline and candidate
JSONL files and `scripts/compare_rollout_performance.py` for the separate
truncated/full-rollout throughput checks. The [position reference instructions](../rollout_positions/README.md)
describe how to reuse saved references and run missing public calls in parallel.

For example, after extracting `paired_3ply_outputs.zip` into
`logs/paired_3ply` in the engine checkout, compare the complete corpus with
the matched control:

```sh
python scripts/compare_multi_ply_accuracy.py --baseline logs/paired_3ply/16t-1-baseline/results.jsonl --candidate logs/paired_3ply/16t-1-candidate/results.jsonl --require-complete --output logs/paired_3ply/accuracy.json
```

During a run, `--follow` refreshes the scorecard as additional public calls
complete. A separate [supplemental checkpoint archive](../rollout_3p_vr1296/README.md)
retains fixed-strength evaluations that had already finished for individual
checker moves. The normal position benchmark requires only the moves selected
by the rollout API.
