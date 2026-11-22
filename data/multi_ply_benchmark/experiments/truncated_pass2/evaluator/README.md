# Neural evaluation and Python progress during truncated rollouts

The production Stage 11 model ranks sibling checker moves with a shared hidden-layer base. All three of its batch methods derive changed inputs from the sibling boards, in input order, using the first nonterminal candidate as the base. Reaching the 64-input limit invokes the existing dense fallback. The checker analyzer passes a progress function into the native trial workers only when its caller requests progress. Both changes were built together for the public screen.

## Public calls

Ten seeded positions per level cover the five game plans and both play formats. Each level has four checker, three cube and three post-move calls. Both engines ran under Linux with four trial threads on CPUs 4–7 of a Ryzen 7 9700X. The runner alternated binary order by position. Every field in all 20 pairs matched, including every checker endpoint, decision and standard error. The sample contains 310 scored fields with 1,296-game references. [Per-case analytics](direct/outputs.zip) · [Per-API scores](direct/accuracy.json)

| Preset | Prior third-pass build | Sibling deltas and opt-in progress | Reduction |
| --- | ---: | ---: | ---: |
| 2T | 2.862 s | 2.807 s | 1.9% |
| 3T | 14.289 s | 14.038 s | 1.8% |

These small-sample totals span all three APIs. For example, the seven cube and checker calls together improved at each preset while the three post-move calls took 0.025 s longer at 2T and 0.056 s longer at 3T. [Binary hashes and aggregate times](direct/summary.json)

## Neural microbenchmarks

The component benchmark uses the existing corpus of 1,422 legal siblings and the 400-hidden-unit Stage 11 attacking/racing contact network. A timing pass performs 32 complete traversals including encoding and branching between first and subsequent siblings. Each arm ran three times on one Linux CPU, with four timing passes per implementation per launch. The table shows median ns per candidate and interquartile range for the direct-delta path. Each arm also recorded all five output fields for every candidate under both encoding paths, totaling 14,220 floats. [Distributions and output hashes](kernels.json) · [Native probe](kernel.cpp) · [Variant generator](kernels.py)

| Implementation | Direct-delta median | 25th–75th percentiles | Full-input median | Maximum change from incumbent |
| --- | ---: | ---: | ---: | ---: |
| Current activation and accumulation | 839 ns | 822–872 ns | 855 ns | 0 |
| Fuse sigmoid with five output sums | 851 ns | 831–886 ns | 880 ns | 0 |
| Hold 32 hidden units in registers across every changed column | 837 ns | 808–858 ns | 908 ns | 0 |
| Evaluate a degree-five Chebyshev exponential on a reduced interval | 823 ns | 802–832 ns | 828 ns | 2.38e-7 |

Fusing the hidden activation with the output sums and changing the accumulation tile gave no repeatable improvement. The shorter exponential polynomial reduced component time by about 2%. Its public screen gave mixed results and overlapped another study on the same computer. The measured 2T times were 2.817 s and 3.655 s, and the 3T times were 8.917 s and 8.456 s. All 14 checker/cube decisions agreed, and no field moved materially farther from truth. The largest shifts in the returned probabilities and equities were 1.06e-6 and 1.55e-6 respectively. The engine keeps its original activation, accumulation and exponential. [Public timing](activation/summary.json) · [Public outputs](activation/outputs.zip) · [Accuracy](activation/accuracy.json)

## Trial and integration checks

Native workers that report progress acquire Python's GIL. The default checker path now avoids that acquisition and the native progress counter. An integration test checks that workers receive a null callback in the default case, that supplying a callback produces cumulative reports through the last trial, and that returned analytics match. A second test cancels a native checker rollout with no callback and successfully reuses the analyzer after resetting cancellation. Both tests passed on Windows x86-64. [Regression tests](../../../../../tests/test_rollout_progress_callback.py)

Ten two-thread Linux configurations separately exercised closed-bar passes, terminal post-move analysis, moves entering the bearoff database, both players already in bearoff, disabled database, disabled and late-disabled cube-aware checker play, nested checker/cube evaluators, and target-standard-error batching. The stronger nested checker and nested cube policies each run two child trials. All ten control/candidate actions agreed, including every exact database and terminal case. The maximum public changes were 3.73e-9 probability and 1.20e-7 equity. [Custom configurations and complete outputs](custom.zip) · [Runner](special_cases.py)

## Reproduction

Build the third multi-ply release at `1ecca39` and a copy containing only the three Stage 11 sibling-delta substitutions and the checker progress change. Use the same GCC release flags, Python 3.14, model weights and bearoff database for both. The frozen build folders must include their `python/` package and native extension. Run this study from the engine root.

```sh
study=data/multi_ply_benchmark/experiments/truncated_pass2/evaluator
python3 -B "$study/screen.py" BASELINE_FOLDER CANDIDATE_FOLDER build/evaluator-screen
python3 -B "$study/special_cases.py" BASELINE_FOLDER INTEGRATED_FOLDER build/custom-cases.zip
```

For the component experiment, `kernels.py` uses the GCC commands for the repository's Linux release build and the direct-delta source export at `build/truncated_pass2/eval-direct/`. The corpus is the `micro/corpus.txt` entry in the [CPU neural study archive](../../cpu_nn/measurements.zip), extracted to `build/nn_trials/corpus.txt`.

```sh
python3 -B "$study/kernels.py" build direct fuse exp tile
python3 -B "$study/kernels.py" run direct fuse exp tile
```
