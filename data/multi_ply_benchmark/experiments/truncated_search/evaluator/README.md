# One-ply evaluation during rollout trials

Three CPU implementations were screened on the public 2T and 3T presets. Each level used the same ten cases, covering all five game plans in both money and match play. Four cases used checker play, three used cube action and three used post-move analytics. One Stage 11 analyzer per level and executable evaluated the same input and seed with four native trial threads. Each implementation occupied logical CPUs 4–7 by itself during its timed call. The order of the four implementations rotated by position.

| Selection implementation | 2T total | 3T total | Largest probability shift | Largest equity shift | Largest SE shift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original individual evaluations | 5.849 s | 19.635 s | — | — | — |
| Classify the pre-move board once, then evaluate each candidate in full | 5.512 s | 19.116 s | 0 | 2.39e-7 | 1.87e-9 |
| Classify once and reuse sibling hidden-layer sums | **4.568 s** | **18.334 s** | 4.48e-7 | 2.39e-7 | 1.04e-8 |
| Classify once, keep the dense winner’s probabilities and rank only the driving cube during base-level VR | 5.271 s | 18.659 s | 5.97e-8 | 2.39e-7 | 1.87e-9 |

Sibling batching cut measured wall time by 21.9% at 2T and 6.6% at 3T. All three variants preserved all 14 best checker and cube decisions and moved zero of 310 scored fields materially closer to or farther from the full-game reference. The probability and equity shifts in the table include every returned checker endpoint, including those without a full-game reference. The SE column covers all public probability and equity standard errors. Each original control call reproduced every field of the published control with 16 trial threads. [Field comparisons](score.json) · [Timing and binary hashes](summary.json) · [Full analytics for all 80 calls](full_outputs.zip)

The retained selector batches candidate ranking for the built-in backgame-aware pair models. VR evaluates the selected board with the active strategy, so a mixed bearoff decision can rank with the neural model and return the database result for its winner. A separate native fixture checks single- and multi-cube selection against dense ranking for 21 opening rolls at two cube states. It also exercises terminal outcomes, custom evaluators, database transitions and the dense fallback when the sparse input-delta array fills. [Native checks](native_checks.cpp)

## Linux portability check

The final production build ran all 300 public calls at each preset with 16
trial threads on Ubuntu 26.04 WSL2. Control and candidate ran separately at
each level, using identical seeded inputs. This comparison includes the VR
cache. [Build source identities](linux_final/inputs.json) · [Guest resources](linux_final/environment.json)

| Level | Checker play, 100 calls | Cube action, 100 calls | Post-move, 100 calls | Aggregate speedup | Wall time saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2T | 67.261 → 42.370 s | 26.077 → 18.076 s | 24.495 → 17.133 s | 1.519× | 34.2% |
| 3T | 175.625 → 143.342 s | 63.538 → 58.204 s | 63.289 → 57.762 s | 1.166× | 14.3% |

All 200 complete checker rankings and 200 cube decisions matched across
the two presets. The largest shifts were 0.0000190 in probability, 0.0000457
in equity and 0.00000136 in standard error. None of the 9,448 scored errors
moved materially in either direction against its frozen full-game reference.
The native release builds used the same WSL guest and CPU model as the panel
below. [Qualification](linux_final/qualification.json) · [Per-field scorecard](linux_final/comparison.json) · [All 1,200 complete public outputs](linux_final/full_outputs.zip)

### Interim four-thread panel

The same seeded panel ran under Ubuntu 26.04 WSL2, GCC 15.2.0 and Python 3.14.4 on the Ryzen 7 9700X. Both builds used the engine's Release CPU flags and four trial threads. This interim integration paired sibling batching with the hybrid position cache and generation-tagged cube cache. Both arms used the original rollout trial code from `60f9355`. The subsequently integrated VR cache was outside this Linux comparison. [Source identities](linux/source_identity.json)

| Level | Linux control | Linux candidate | Speedup | Entire calls identical | Decisions unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2T | 3.952 s | 2.655 s | 1.49× | 8/10 | 7/7 |
| 3T | 11.099 s | 9.810 s | 1.13× | 10/10 | 7/7 |

The maximum shift relative to the **Linux control** was 1.06e-6 for probability, 1.55e-6 for equity and 5.97e-8 for standard error. All 310 scored fields stayed within the material change thresholds against full-game truth. In the raw timing JSON, `matches_published_control` separately compares with the historical Windows outputs. These are local WSL timings on Windows-mounted model and database assets. [Field comparisons](linux/score.json) · [Timings and binaries](linux/summary.json) · [Full public outputs](linux/full_outputs.zip)

## Reproduction

The variant builder reads the reference engine from commit `60f93556137a7a5c4be671502f4b3f521998ff94`. The Windows script expects its Release CPU library and Python binding object in `build/multi_ply_cpu_fixed`. Run from the engine repository with a Python environment containing psutil.

```powershell
$study = 'data/multi_ply_benchmark/experiments/truncated_search/evaluator'
python "$study/variants.py"
& "$study/build.ps1"
python "$study/measure.py" --arms control route batch fuse --output build/trunc_eval_study/replication
python "$study/score.py" build/trunc_eval_study/replication
& "$study/build.ps1" -Checks
```

For Linux, the build script uses the GCC sysroot, compiler commands and object dependency lists from the [previous Linux build](../../linux_x86_64/README.md). It checks each engine source/header against the earlier compilation, compiles all affected objects in private snapshots, then links with two LTO jobs. From the engine repository in Ubuntu:

```bash
study=data/multi_ply_benchmark/experiments/truncated_search/evaluator
work=build/trunc_eval_study/linux
mkdir -p "$work"
git archive --format=tar --output="$work/control.tar" 60f93556137a7a5c4be671502f4b3f521998ff94 cpp
python3 -B "$study/linux_build.py"
python3 -B "$study/measure.py" --binary-root "$work" --arms control candidate --output build/trunc_eval_study/linux_replication
python3 -B "$study/score.py" build/trunc_eval_study/linux_replication
```

The complete final Linux comparison adds the VR cache and uses the guest's 16
logical CPUs. To reproduce it after the earlier Linux build, create the private
snapshots once and run the two presets in sequence:

```bash
python3 -B "$study/linux_build.py" --prepare-final
python3 -B "$study/linux_build.py" --build-final
python3 -B "$study/linux_qualify.py"
```
