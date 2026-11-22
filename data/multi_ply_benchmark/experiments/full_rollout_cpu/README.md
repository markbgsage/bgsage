# Full-game rollout CPU optimization

## Results at 1,296 paths and 16 workers

Both CPU kernels made **all 15 public calls faster**. They rolled the same 14 nonterminal checker alternatives, five cube positions and five post-move positions, for **31,104 paths per engine**. The widest kernel completed the panel in **220.92 s**. The variant that preserves the AVX2 sigmoid's rounding took **226.50 s** and reproduced all 15 complete responses bit for bit. [Paired timings and per-position results](qualified/primary/summary.json) · [Inputs, complete analytics, settings and accuracy scores](qualified/primary/full_outputs.zip)

| Public call | Calls | Before all optimization (s) | Starting engine, fresh (s) | Wide kernel (s) | Speedup this pass | Speedup since start |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Checker play | 5 | 369.36 | 158.13 | 135.13 | 1.17× | 2.73× |
| Cube action | 5 | 141.62 | 52.52 | 42.34 | 1.24× | 3.35× |
| Post-move analytics | 5 | 134.49 | 51.35 | 43.45 | 1.18× | 3.10× |
| **Total** | **15** | **645.47** | **262.00** | **220.92** | **1.19×** | **2.92×** |

Speedups divide the wall-time totals. The historical column is the archived `139427d` control with the opening-cache correctness fix. It used the same inputs, seeds and 24 batches. The starting engine for this pass is `5de8023`. It took 254.12 s in the earlier study and 262.00 s when interleaved with the two new kernels. All 15 of its fresh responses matched the archive. [Historical control and earlier measurements](../full_rollout_1296/README.md)

Across the three public calls per game plan, the wide kernel's speedups this pass were 1.16× for attacking, 1.25× for priming, 1.19× for anchoring, 1.17× for racing with contact and 1.08× for pure race. The six-alternative match anchoring checker request took 72.27 s, down from 84.51 s.

Every checker ranking, rollout shortlist and cube action agreed across the three engines. For the wide kernel, the largest direct differences from the starting engine were 0.0000008494 in probability and 0.0000007749 in equity. Across 213 reference-backed values, zero moved materially farther or closer. Fourteen calls use seeded public rollout references. The match-anchoring checker reference comes from the adaptive archive, which omits its dice seed and exact batch count.

The wide kernel was faster than the bit-preserving variant on 13 of the 15 primary calls. Reversing the order of the three builds on a predeclared checker, cube and post-move request gave 69.08 s for the starting engine, 62.11 s for the wide kernel and 60.28 s for the bit-preserving variant. The same three primary calls had also slightly favored the bit-preserving variant, 61.44 s versus 61.69 s. The full panel favors the wide kernel by 5.58 s, or 2.5%, with zero material increases in error against the position references. [Reverse-order timings](qualified/reverse/summary.json) · [Complete repeat outputs](qualified/reverse/full_outputs.zip)

Three held-out boards used the second corpus position in their strata and one call per public API. All three got faster, from **30.79 s to 27.18 s in total (1.13×)**. Both engines ran five batches, made the same decisions and had zero material regressions across the 41 reference-backed values. The largest direct difference in either probability or equity was 0.0000001193. [Held-out timings](qualified/holdout/summary.json) · [Requests and full responses](qualified/holdout/full_outputs.zip)

## Measurement

The primary comparison uses 15 public calls on ten boards, with one checker, cube and post-move call for each of the five game plans. The ten boards cover both unlimited and match play within each plan. Each engine uses 16 trial workers on an eight-core, 16-logical-CPU Ryzen 7 9700X running x86-64 Linux under WSL2. Only one call runs at a time. Each build has a resident analyzer. Elapsed time starts at entry to the public method and ends when it returns the complete analytics, including any promoted checker moves. Model loading, process startup and app queueing are outside the measurement. [Workload and historical measurements](../full_rollout_1296/README.md) · [Runner](run.py)

The builds use identical Stage 11 weights, bearoff database, dice streams and public checker filters. The budget is 1,296 stratified paths per rolled checker alternative or cube position. Each uses the full-game horizon (depth zero), variance reduction and 3P checker and cube decisions, including the engine's usual opening reply, cube screen and 200-half-move safety limit. The runner alternates which of two builds runs first and rotates build order for comparisons of three or more. The archives contain request arguments, seeds, configuration, build identities, wall and process CPU time, and complete public responses.

Accuracy uses the frozen position rollout references. For each comparable probability or equity the report measures the change in absolute distance to its reference. The material limits are 0.0025 for an absolute probability and 0.005 for equity. The report also compares direct output differences, checker rankings, rollout alternatives and cube recommendations. [Rollout references](../../rollout_positions/README.md) · [Scoring and archive verification](report.py)

## Where the CPU time goes

Four instrumented full-budget requests attributed **81.7%** of summed worker time to checker selection, **11.7%** to the variance-reduction evaluation of the possible dice rolls, and **6.1%** to cube decisions. The opening and reply tables together took **0.15%**. These are worker-phase shares. The public wall timings come from builds without the timers. [Phase totals and per-position measurements](profile/summary.json)

## Candidate measurements

The results below give aggregate elapsed time relative to a control run in the same screen. The initial screen used 324 paths and subsequent screens used 1,296. Differences below a few percent on these small screens need confirmation on the paired panel.

| Idea | Screen result |
| --- | --- |
| Change how workers claim games | One game per claim saved 4.6% at 1,296 paths. Grouping games by opening dice saved 4.0% on a separate screen. Individual claims balance long games against short ones. An independent callback interval preserves progress cadence. |
| Reuse the unchanged opponent's contact features and table the exact escape and entry results | 3.5% less time in a three-call screen. Both scalar and AVX2 encoders reproduced the prior dense and incremental features. |
| Vectorize detection of changed points and auxiliary inputs | 2.9% less than baseline with opponent-feature reuse. The simpler encoder and scheduling combination saved 7.4% on the same four positions. |
| Widen the CPU hidden-layer update | The independent AVX-512 candidate saved 7.7% over four calls. Its largest direct probability and equity differences were below 0.0000004. |
| Compute the saved hidden sums once and skip predictions that no caller consumes | Saved 2.8% in isolation on three full-budget calls. The native probe compares saved sums, saved inputs and predictions across network widths that exercise SIMD tails and heap-allocated scratch space. |
| Memoize repeated board encodings or neural outcomes across search contexts | The feature cache added 4.7% wall time. The cross-base output cache saved 2.2% and changed some low-order output bits. |
| Share deeper subtrees, opening pairs or duplicate dice-node results | Subtree and pair caches saved 1.0% and 1.7% respectively. Removing duplicate static evaluations inside a dice node cost 0.4% on the initial screen. |
| Precompute cube leaves, specialize the driving cube or cache bearoff lookups | Cube leaves added 3.6% on the initial screen and driving-cube specialization added 6.4% at full budget. Bearoff caching saved 0.1% on the initial screen. |
| Tighten the checker equity window or cap the deep shortlist at eight | Reduced time by 8.9–15.7%, with material regressions. The tighter of the two checker equity windows moved double/take equity by 0.012 on an unlimited anchoring board. |
| Evaluate seven dice rolls before deciding whether to search the remaining rolls | Saved 0.2%, with seven material regressions in four calls. |

[Initial 324-path screen](screens/initial/summary.json) · [Full-budget screen](screens/full-budget/summary.json) · [Combined and cache candidates](screens/combined/summary.json) · [Neural and checker-search candidates](screens/kernels/summary.json)

The three-call integration screen measured the effect of CPU instruction support on the same hardware. The control took 34.486 s. Individual trial claims, opponent feature reuse and single-pass hidden-base preparation took 32.334 s on the AVX2 path (**1.07×**). Enabling the wide hidden update and sigmoid brought that to 30.068 s (**1.15×**). These three calls give a local AVX2 fallback measurement. AWS instance types may have different cache sizes, memory bandwidth and instruction support. [Integration timings](screens/integration/summary.json)

The production kernel dispatches to AVX-512F at runtime on compatible x86 GCC and Clang builds. The bit-preserving alternative uses the same wide polynomial and splits the final sigmoid division into two AVX2 operations. All 1,048,576 tested activations matched the starting engine under the split division. The production version's largest difference across the same activations was 0.000000179. The wider division and the split division were also compared through complete public rollouts. [Native neural and sigmoid checks](dense/README.md) · [Head-to-head full-game screen](screens/reciprocal/summary.json)

The Windows Release extension passed 239 engine tests covering search, move generation, contact encoding, cube handling, rollout caches, progress and cancellation. Nine additional callback checks verified the complete analytics with reporting enabled and disabled, as well as the event counts for a single batch, an adaptive target-SE cube rollout and a 1,296-path full game. [Windows regression run](qualified/windows-regressions.json) · [Progress results and full responses](progress/README.md)

## Other evaluation levels

Each preset ran all 100 checker plays, 100 cube actions and 100 post-move calls. All 700 checker rankings and all 700 cube recommendations agreed with the starting engine. Across 33,068 reference-backed probabilities and equities, zero moved materially farther or closer. The largest increases in absolute error were 0.000004471 in probability and 0.000007272 in equity. [Per-field comparison to rollout truth](qualified/nonfull/truth_bounds.json) · [Complete preset outputs](qualified/nonfull/full_outputs.zip)

These timings compare the current run with the archived starting engine on the same machine, in separate sessions. The primary full-rollout result above interleaves both builds in one session. At 1P the entire panel takes about 20 ms.

| Preset, 300 calls each | Archived starting engine (s) | Selected (s) | Factor |
| --- | ---: | ---: | ---: |
| 1P | 0.022 | 0.020 | 1.12× |
| 2P | 0.232 | 0.191 | 1.22× |
| 3P | 1.952 | 1.508 | 1.29× |
| 4P | 34.789 | 24.454 | 1.42× |
| 1T | 6.118 | 4.593 | 1.33× |
| 2T | 45.869 | 34.516 | 1.33× |
| 3T | 185.208 | 136.863 | 1.35× |

The factors use the unrounded measurements. [Timings and decision comparisons for all seven levels](qualified/nonfull/summary.json)

## Encoder verification

The native probe checks the scalar escape and bar-entry definitions, every contact input, and both race and contact sparse deltas against dense subtraction, including the dense fallback. Its corpus covers all 24,576 back-point/blocker table entries, all six-point bar-entry masks, legal moves, sibling moves that hit different blots, changes to the front checker, and independent positions. The shipped and selected encoders returned the same feature fingerprint across 656,128 boards and identical deltas on 315,648 pairs. This held in both the AVX2 and scalar builds. [Native test](encoder/encoder_check.cpp) · [AVX2 results](encoder/avx2.json) · [Scalar results](encoder/scalar.json)

To repeat the native check on an x86 Linux host with a C++17 compiler, run from the engine root. Add `--portable` to exercise the scalar path.

```sh
study=data/multi_ply_benchmark/experiments/full_rollout_cpu
python3 "$study/encoder/check.py" --tree baseline /path/to/baseline --tree selected . --output /tmp/encoder.json
```

## Reproducing the public timings

Use two Linux source exports with their own compiled extension and Python package, built with the same compiler flags, weights and bearoff database. Run from the engine checkout with an unused output directory. Omit a job selection to run all 15 primary calls. For a three-build comparison, add a third `--arm` with its export.

```sh
study=data/multi_ply_benchmark/experiments/full_rollout_cpu
python3 -B "$study/run.py" --arm baseline build/truncated_pass2/shipped \
    --arm selected build/truncated_pass2/full-wide-release --output logs/full-game-replay
python3 -B "$study/report.py" --run logs/full-game-replay --output logs/full-game-report
python3 -B "$study/report.py" --output logs/full-game-report
```

The last command independently recomputes the timings, ranking comparisons and rollout errors from the saved responses. The report checks the control extension, weights, database, seeds and calls against the qualified corpus. The source archive contains the patches to `5de8023`, per-file fingerprints and exact extension identities. [Source patches and build manifests](variants.zip)
