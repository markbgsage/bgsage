# Native phase profile of truncated rollouts

The 2T preset spent **78.1%** of measured worker time computing the variance-reduction (VR) mean. At 3T the largest costs were the truncation leaf (**31.8%**), the VR mean (**29.3%**), trial cube decisions (**18.0%**) and actual checker selection (**15.7%**). Optimizing only the VR mean leaves most of the 3T cost intact.

## Measured work

The source snapshot is `60f93556137a7a5c4be671502f4b3f521998ff94`. For each public API and each preset the study selected the first position in each of the five plan categories for both play formats. Each line in the table covers ten positions. The native phase columns are **worker-seconds summed across the four workers**. The public wall clock covers one complete call at a time, including work outside the timed native phases. Process CPU time measures the scheduled CPU consumption of the whole process. Worker elapsed time can include preemption and simultaneous work, so it need not match either CPU or wall time.

| Level and API | Wall s | CPU s | First roll setup | Second roll setup | Cube check | Legal moves | VR mean | Actual checker | VR luck | Truncation | Other trial work |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2T checker play | 18.60 | 71.14 | 0.53 | 0.94 | 3.41 | 2.52 | 56.99 | 2.16 | 0.01 | 4.90 | 0.02 |
| 2T cube action | 7.16 | 27.19 | 0.12 | 0.28 | 2.59 | 0.83 | 21.05 | 0.79 | <0.01 | 1.99 | <0.01 |
| 2T post-move | 6.96 | 26.53 | 0.12 | 0.29 | 2.52 | 0.82 | 20.40 | 0.77 | <0.01 | 1.97 | <0.01 |
| 3T checker play | 48.10 | 184.64 | 6.83 | 0.90 | 24.87 | 2.45 | 54.41 | 34.34 | 0.11 | 61.14 | 0.04 |
| 3T cube action | 16.89 | 65.14 | 1.82 | 0.27 | 16.15 | 0.78 | 19.11 | 7.71 | 0.04 | 19.77 | 0.01 |
| 3T post-move | 16.54 | 63.78 | 1.76 | 0.26 | 15.83 | 0.75 | 18.71 | 7.52 | 0.04 | 19.35 | 0.01 |

First-roll setup selects opening moves. Second-roll setup builds the 21 reply entries for a first roll. Cube check includes screening, possible deep cubeful evaluation and applying the chosen decisions. VR mean includes selecting a static winner for each of the 21 dice outcomes and weighting the five resulting probabilities and per-branch cube values. Actual checker time covers selecting the move played on the sampled dice. VR luck updates the trial's per-branch and cubeless accumulators. Truncation evaluates the final board at the preset leaf depth. Public Python filtering, thread dispatch and result assembly are outside the phase counters.

The native profile and the uninstrumented control reproduced the archived complete results for all 60 calls, including standard errors. Both used the public Stage 11 presets, the standard bearoff database, the archived per-position seeds and Windows logical CPUs 8 through 11 on a Ryzen 7 9700X. Other experiments used distinct CPU affinities on the same machine during profiling. Compare the four-worker full-call measurements within this study. The [16-worker qualification](../../truncated_16t/README.md) provides the historical production-width timings.

## Reproduction

From the engine checkout, set `study` to `data/multi_ply_benchmark/experiments/truncated_search/profile`. Run these commands with the Python interpreter used to build the native extension and the engine's normal models and database present. Each source export must use a new directory under `build`, and each invocation of the sampler needs a new output directory. Replace the extension suffix with the one for the active Python ABI.

```powershell
$study = 'data/multi_ply_benchmark/experiments/truncated_search/profile'
python "$study/instrument.py" build/phase-source
./scripts/build_multi_ply_engine.ps1 -Python (Get-Command python).Source -Source build/phase-source -BuildName multi_ply_phase_profile -Jobs 2
$binary = (Get-ChildItem build/multi_ply_phase_profile/bgbot_cpp*.pyd).FullName
python "$study/sample.py" --extension $binary --output build/phase-results --threads 4 --cpu-mask 0xf00 --profile
```

For the control, use an uninstrumented release build at the specified revision and omit `--profile`. The original control binary in this study has the same SHA-256 digest as the optimized arm in the 16-worker qualification. The [native profile measurements](native/measurements.json) and [raw control measurements](control/measurements.json) record the engine hash, elapsed time, process CPU time and exact-output check per position. Their sibling `calls.jsonl.gz` files contain all call inputs and complete public outputs.

## Cross-trial VR memo experiment

The experimental table retains the 21 selected static winner indices and probability vectors after a trial leaves the opening reply cache. A subsequent trial with the same board, ordered selection cube states and cube efficiency can reuse the entire fan. The table belongs to a single public rollout, has 2,048 direct-mapped slots, and uses acquire/release publication. Occupied slots for other keys fall back to fresh evaluation. The implementation continues to compute all means with the original arithmetic, dice order and current branch state.

In the independent consecutive passes, 2T took 32.04s in the control and 31.93s with the memo. 3T took 90.48s and 87.52s. Hit rates were 10,508/83,641 (12.6%) at 2T and 10,484/83,577 (12.5%) at 3T. In the [interleaved comparison](paired/measurements.json) all 60 paired responses reproduced the archive exactly. The memo reduced summed wall time by 10.4% at 2T and 4.0% at 3T. Across both levels, 53 of 60 calls were faster. The full API response from each build for each pair is in the [paired output archive](paired/calls.jsonl.gz).

| Preset and API | Control wall s | Memo wall s | Control CPU s | Memo CPU s |
| --- | ---: | ---: | ---: | ---: |
| 2T checker play | 18.76 | 16.93 | 70.98 | 64.00 |
| 2T cube action | 7.03 | 6.30 | 26.50 | 24.09 |
| 2T post-move | 7.05 | 6.21 | 26.66 | 24.06 |
| **2T total** | **32.84** | **29.44** | **124.14** | **112.16** |
| 3T checker play | 51.62 | 49.49 | 195.09 | 187.34 |
| 3T cube action | 17.66 | 16.95 | 67.62 | 64.78 |
| 3T post-move | 17.93 | 17.27 | 68.31 | 66.28 |
| **3T total** | **87.21** | **83.70** | **331.03** | **318.41** |

The production table allocates once per rollout position with VR enabled, depth greater than two half-moves and at least 288 trials. The screening implementation enabled it for 72 or more trials to count visits. Both sampled presets use 360. Full-game rollouts have depth zero, and 1T uses 72 trials. The table occupies approximately 1.46 MiB in the measured 64-bit build, shared by all workers within the rollout position. The production code omits the experimental process-wide visit counters.

`memo_experiment.py` prepares this variant from the same revision. Build and sample it with the procedure above, substituting `memo_experiment.py` for `instrument.py` and `--memo-counts` for `--profile`. The [memo measurements](memo/measurements.json) include eligible visits and hits. To time the arms side by side, run `paired.py --control <control-extension> --candidate <memo-extension> --output build/paired-results`. Its two processes retain separate loaded engines and execute one call at a time. It shuffles the same 60 cases with a recorded seed and alternates which arm runs first. Construction, serialization and disk IO occur outside the timed interval.
