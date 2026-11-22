# Optimizing 2T and 3T on a CPU

The comparison starts at `60f93556137a7a5c4be671502f4b3f521998ff94`, the engine published in the [preceding timing study](../truncated_16t/README.md). Each preset runs the same 100 checker decisions, 100 cube decisions and 100 post-move calls. Both use 360 stratified trials, variance reduction and a seven-half-move horizon. The 2T cube and truncation evaluations use 2P. The corresponding 3T evaluations use 3P. [Current rollout algorithm](../../../../ROLLOUT.md)

## Timings and accuracy

On a Ryzen 7 9700X under Windows, with one public call using 16 trial threads at a time, 2T took **34.8% less wall time** and 3T took **12.2% less**. The fresh control reproduced all 600 published responses exactly. Each entry below gives the control and optimized times in seconds for 100 calls.

| Level | Checker play | Cube action | Post-move | Total for 300 | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2T | 62.87 → 40.88 | 23.68 → 15.47 | 23.10 → 15.14 | 109.64 → 71.49 | 1.53× |
| 3T | 171.38 → 149.04 | 64.47 → 57.72 | 64.10 → 56.52 | 299.95 → 263.27 | 1.14× |

All 100 checker rankings and all 100 cube decisions matched at each level. Across **9,448** scored probabilities and equities, zero changes crossed either material threshold in either direction. **539 of the 600** complete responses were identical, including the standard errors. The largest public probability shift was **0.00693 percentage points** and the largest equity shift was **0.000161**. The candidate was faster on all 300 individual 2T calls and 283 of 300 at 3T. [All 1,200 timed outputs and settings](full_outputs.zip) · [Detailed scorecard](comparison.json) · [Regression tests](regression_tests.json)

Both executables used MSVC Release, the same production Stage 11 weights, bearoff database and per-position seeds. Each arm ran a single complete pass on all 16 logical CPUs in a fresh process, using normal analyzer caches. We measured 2T control then candidate, followed by 3T candidate then control. All other experiment workers were idle. The instrumented and prototype timings below used smaller affinity groups and do not form part of the full-corpus timings.

On the same CPU under Ubuntu 26.04 on WSL2, GCC 15.2 produced **34.2% less wall time at 2T** and **14.3% less at 3T**. Each build used the same 16 logical CPUs. All checker rankings and cube decisions matched, 545 of 600 complete responses matched exactly and zero of 9,448 scored fields changed materially. These figures measure the local Linux build. [Linux scorecard](evaluator/linux_final/comparison.json) · [All 1,200 Linux outputs](evaluator/linux_final/full_outputs.zip)

| Level | Linux checker play | Linux cube action | Linux post-move | Total for 300 | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2T | 67.26 → 42.37 | 26.08 → 18.08 | 24.49 → 17.13 | 117.83 → 77.58 | 1.52× |
| 3T | 175.63 → 143.34 | 63.54 → 58.20 | 63.29 → 57.76 | 302.45 → 259.31 | 1.17× |

As a separate full-game throughput check, both engines replayed the existing match attacking and unlimited pure-race cases twice per API with 36 games, 3P checker/cube decisions and 16 workers. All 12 full-game responses, including their standard errors, matched exactly. Checker time fell from 7.46 to 7.08 seconds, cube from 2.09 to 1.90 seconds, and post-move from 2.02 to 1.84 seconds. The accuracy benchmark uses the frozen 1,296-game references. [Full-game outputs and timing](fullgame_guard.zip)

## Where the time went

Across 30 public calls at each preset, 2T spent 78.1% of instrumented worker time computing the variance-reduction mean. After legal-move generation, this phase selects the 1P winner for each roll, weights the probabilities and performs the per-branch cubeful conversions. At 3T the largest costs were the final board evaluation (31.8%), the variance-reduction mean (29.3%), trial cube decisions (18.0%) and selection of the actual checker play (15.7%). These measurements sum elapsed time across workers. The final end-to-end timings measure elapsed time around the public API. [Phase profile, source instrumentation and full outputs](profile/README.md)

## Ten ideas tested

| Idea | Implementation and result |
| --- | --- |
| 1. Share neural evaluation across sibling moves | The Stage 11 cube-aware selector routes the pre-move board once and uses the network's sibling batch for candidate probabilities. In the isolated screen it cut 2T time by 21.9% and 3T by 6.6%. Two alternatives, routing once with dense forwards and retaining the dense winner, gave smaller gains. A full forward handles a saturated 64-feature delta buffer. [Evaluator variants and checks](evaluator/README.md) |
| 2. Share later VR move selections across trials | A 2,048-slot table holds all 21 winner indices and probabilities for one board and selection-cube context. An interleaved 60-call screen with exact outputs cut wall time by 10.4% at 2T and 4.0% at 3T. The table costs about 1.46 MiB shared by all trial workers. [Paired results](profile/README.md) |
| 3. Allocate the hybrid search cache on demand | Allocate the per-thread 262,144-entry cache at first insertion. Skip the memory wipe when empty. In a four-worker cube call, working-set growth fell from 298.4 to 162.6 MiB. The isolated 16-thread comparison saved 7.1% at 2T and 1.4% at 3T. [Memory measurements](screen_summary.json) · [Isolated timings](cache-isolated.json) |
| 4. Invalidate the cube search cache in constant time | Advance a thread-local generation. A lookup stops at the first slot outside the current generation. Only generation wraparound resets all 32,768 stamps. In the seven-position screen, adding generation stamps to the lazy hybrid cache was about 0.5% slower. A larger randomized interleaved comparison covering every format and plan took 10.88 → 10.65 s at 2T and 28.81 → 28.56 s at 3T. Process CPU time also fell at both levels. The change is small relative to run-to-run variation. [Interleaved cache measurements](cache-generation/measurements.json) |
| 5. Remove repeated opening work | Tested skipping a repeated opening-reply ranking and skipping redundant legal-move generation when reading the reply table. Each preserved the results. Neither produced a consistent win on both presets. [Native variants](trial/README.md) |
| 6. Reduce work on rolls or cube branches the trial does not play | Tested streaming the 21 candidate lists and specializing selection to the driving cube branch where safe. The streaming variant was 2.4% faster at 3T and slower at 2T. The four opening, branch and streaming changes combined measured 0.960× and 1.003×. Discarded. [Trial timings](trial/screen.json) |
| 7. Cache the cubeful VR mean of the second half-move | Memoize the 21 branch conversions after the opening cube action. All 42 paired comparisons matched exactly. The time ratios were 1.000× at 2T and 0.968× at 3T. Discarded. [Opening-mean experiment](trial/mean-screen.json) |
| 8. Reduce the trial count from 360 to 288 | Keeps complete 36-roll strata. Four values became materially farther from truth across the 12-call screen. Discarded. |
| 9. Shorten trials from seven half-moves to five | Kept the original final-board evaluation depth. Twenty-three values became materially farther from truth. Discarded. |
| 10. Reduce the final-board evaluation by one ply | Held the trial count, horizon and checker/cube policies fixed. Thirty values became materially farther from truth. Discarded. |

The three policy experiments used the same two cases per API and preset, covering both play formats and all five plans. A custom-policy control reproduced all 12 published preset outputs exactly. The material criterion is a deterioration in absolute error against the same frozen rollout, exceeding **0.0025** in probability or **0.005** in equity. Each count above represents one field on one position or checker endpoint. [Screen inputs, timings and complete outputs](screen_outputs.zip)

For example, the full-game no-double equity on `cube-money-racing-01` was 0.75849. Standard 3T returned 0.74095. At 288 trials the estimate fell to 0.73256, increasing the error by 0.00839. [Per-policy counts](screen_summary.json)

An additional experiment precomputed the pre- and post-Crawford equity tables, including terminal scores. All 1,922 score/phase lookups matched, as did the public analytics. Timings for its first pass were essentially tied with the matched cache variant at both presets. The later pass was faster in a changing load environment. The production match tables retain the original implementation. [Experimental patch](match_tables.zip) and [per-pass timing](screen_summary.json)

## Reproduce the public benchmark

To repeat the cache experiments, start from the control source. The `pos` arm replaces only `cpp/include/bgbot/multipy.h` with the accompanying version. The `both` arm also replaces `cpp/src/cube_eval.cpp`. The optional match-table arm applies the patch from `match_tables.zip` on top of `both`. Use a separate build directory for each arm. Run `screen.py --extension <binary> --policy preset --cpus 12 13 14 15 --output <new-directory>` for the original four-worker screen, or choose an affinity available on the machine. Use the same affinity with `check_cache.py` for the working-set and table checks. The policy switches and the fixed case IDs are in [the screen runner](screen.py).

Build the control from the above revision and the candidate from the accompanying source. Use the same compiler, CPU release flags, Stage 11 weights, bearoff database and 16 trial workers. Use a separate build directory for each extension. From the engine root, run the following command once for each level and arm, setting the paths for the chosen source and binary.

```sh
python scripts/benchmark_multi_ply.py --levels 2T --threads 16 --model stage11 --extension build/CONTROL_OR_CANDIDATE/bgbot_cpp.<extension> --native-source SOURCE --output logs/truncated_search/final/2T-ARM
python scripts/compare_truncated_benchmark.py --levels 2T 3T --baseline logs/truncated_search/final/2T-baseline/results.jsonl logs/truncated_search/final/3T-baseline/results.jsonl --candidate logs/truncated_search/final/2T-candidate/results.jsonl logs/truncated_search/final/3T-candidate/results.jsonl --output logs/truncated_search/final/comparison.json
```

The runner sets a fixed seed per source position and records every returned field. The comparator pairs all checker endpoints by resulting board and checks position identity, call order and inputs. The frozen reference supplies 378 full-game checker endpoints and all 200 cube/post-move calls. Other legal checker endpoints retain their evaluation levels and participate in the direct output comparison. [Reference corpus and provenance](../../rollout_positions/README.md)
