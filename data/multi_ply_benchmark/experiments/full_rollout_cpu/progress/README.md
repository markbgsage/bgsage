# Full-game progress reporting

Workers claim individual games so another worker can pick up the next path as soon as it finishes. For parallel full games the engine reports progress about once per percent, with a minimum interval of eight completed games and an update at the end of each batch. This keeps callback traffic bounded for callers that publish each event. Serial runs and truncated configurations retain their own reporting intervals.

The check uses an unlimited pure-race position from the timing corpus. It records the complete cube analytics, including standard errors, and every callback. The custom 37-path cube comparisons run at fixed budget and with a target ND standard error of 10⁻¹², which forces two complete batches. The native cubeless post-move entry point is checked separately. The 1,296-path case uses 3P early and late checker and cube policies. The shorter cases use 1P. All full runs use variance reduction and a zero truncation depth.

Three builds run each request. The pre-optimization control here is the starting engine for this full-game pass. The single-claim experiment changes only the size of the work assignments. The selected build controls notifications independently of worker claims. The audit checks callback totals and batch-completion events, compares every returned analytic between builds, and replays each request with the same seed and callbacks disabled. Parallel workers can deliver intermediate events in a different order. [Callback counts and binary identities](summary.json) · [Full responses, progress events, settings and source identities](full_outputs.zip)

| Request | Workers | Before | Single-game claims alone | Selected |
| --- | ---: | ---: | ---: | ---: |
| Fixed 37-path cube or native cubeless post-move | 1 | 37 | 37 | 37 |
| Fixed 37-path cube or native cubeless post-move | 16 | 5 | 37 | 5 |
| Two 37-path cube batches with target SE | 1 | 74 | 74 | 74 |
| Two 37-path cube batches with target SE | 16 | 10 | 74 | 10 |
| Full 3P cube with 1,296 paths | 16 | 108 | 108 | 108 |
| Five-path full-game cube | 16 | 1 | 5 | 1 |
| Custom three-half-move cube with 37 paths | 16 | 37 | 37 | 37 |

All complete analytics, including standard errors, agreed across the three builds and across the runs with and without callbacks. The 37-path serial and parallel calls also agreed for fixed budgets and target-SE stopping.

From the engine root, audit the saved data with the following command.

```sh
study=data/multi_ply_benchmark/experiments/full_rollout_cpu/progress
python3 -B "$study/check.py"
```

To record a fresh run, supply three qualified build directories containing the native extension, matching Python package, Stage 11 weights, and bearoff data. The output directory must be new.

```sh
python3 -B "$study/check.py" --before ORIGINAL_DIR --single-claim SINGLE_CLAIM_DIR \
    --selected SELECTED_DIR --record logs/full-rollout-progress-local
```
