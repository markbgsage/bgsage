# Truncated rollout CPU profile

At checkpoint `1ecca3902fc7d04e35c43b9fcd3ec45920d0fcb5`, evaluating legal plays across
the 21 possible dice outcomes for variance reduction used **74.2% of measured
2T worker time**. At 3T the largest cost was the truncation tree (**35.6%**),
followed by the variance-reduction fan (**22.0%**), trial cube decisions
(**20.0%**) and choosing the sampled checker play (**17.7%**).

The sample contains one position from each game-plan and play-format
combination for each API. All 60 complete public results, including checker
alternatives and standard errors, exactly matched their archived outputs from
this checkpoint. The experiment reused the seeds in the archived calls.
[Full inputs and outputs](windows/calls.jsonl.gz) · [Per-call measurements and counts](windows/measurements.json)
· [Checkpoint reference archive](../../search_pass3/release/windows/full_outputs.zip)

| Preset and API | Public wall s | Process CPU s | First roll setup | Second roll setup | Cube | Legal moves | VR fan | VR mean and cache | Actual checker | VR luck | Truncation | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2T checker | 10.76 | 37.28 | 0.48 | 0.62 | 2.61 | 0.61 | 31.79 | 0.12 | 1.36 | 0.01 | 4.15 | 0.01 |
| 2T cube | 3.70 | 14.03 | 0.09 | 0.15 | 1.78 | 0.18 | 10.32 | 0.03 | 0.43 | <0.01 | 1.40 | <0.01 |
| 2T post-move | 3.71 | 13.94 | 0.10 | 0.18 | 1.78 | 0.18 | 10.30 | 0.03 | 0.43 | <0.01 | 1.43 | 0.01 |
| 3T checker | 41.01 | 133.00 | 6.46 | 0.67 | 25.34 | 0.67 | 34.53 | 0.14 | 32.28 | 0.15 | 58.23 | 0.04 |
| 3T cube | 11.30 | 43.80 | 1.35 | 0.15 | 12.10 | 0.17 | 9.81 | 0.03 | 5.64 | 0.03 | 14.72 | 0.01 |
| 3T post-move | 10.98 | 42.66 | 1.31 | 0.15 | 11.70 | 0.16 | 9.57 | 0.03 | 5.51 | 0.03 | 14.35 | <0.01 |

Phases are elapsed worker-seconds summed across four threads on Windows
logical CPUs 8–11 of a Ryzen 7 9700X. Worker clocks include time the OS
deschedules a worker. Public wall time includes the Python API, filtering,
thread coordination and result assembly. The VR fan timer covers evaluation
and best-play selection across the 21 rolls on cache misses. The VR mean timer
covers cache lookup, storing and retrieving the results, probability weighting
and cube conversion. Both exclude legal move generation. Other experiments
shared the machine during this profile. Use the full-corpus paired timing
study for end-to-end speed comparisons.

## Reproduce

From the engine checkout, use the Python interpreter and pybind11 installation
for the normal Windows CPU build. The regular engine build and models at
the checkpoint must be available. Supply its Python package directory using
`--python-dir` so the sampler uses the same API and weight registry.

```powershell
$study = 'data/multi_ply_benchmark/experiments/truncated_pass2/profile'
python -B "$study/prepare.py" build/replay-truncated-profile-source
./scripts/build_multi_ply_engine.ps1 -Python (Get-Command python).Source -Source build/replay-truncated-profile-source -BuildName multi_ply_replay_truncated_profile
$extension = (Get-ChildItem build/multi_ply_replay_truncated_profile/bgbot_cpp*.pyd).FullName
python -B "$study/measure.py" --extension $extension --python-dir python --output build/replay-truncated-profile-results
```

The preparation script exports the exact Git revision and adds native timers
only inside the new build directory. The sampler verifies its outputs against
the published checkpoint. Both commands require a fresh output directory.
[Prepare instrumented source](prepare.py) · [Run the measurements](measure.py)
