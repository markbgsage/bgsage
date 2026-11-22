# CPU contact feature costs and exact dice geometry

The Stage 11 contact encoder calculates several of its 244 neural inputs by finding the dice rolls that can hit a blot. It repeats the calculation from both players' perspectives at each leaf. A native profile of 44,282 benchmark descendants measured a median 3,964 ns per board for the release encoder. One side's single-shot, double-shot and pip-loss functions took 107 ns, 428 ns and 490 ns respectively. The same profile measured 25 ns for back escape, 75 ns for the second moment and 28 ns for backbone. These are medians of three shuffled runs on one logical CPU, sharing the host with the reference rollouts. Every timed pass traversed the corpus four times. [Full profile](components.json)

## Tested calculations

Bit 0 represents point 1. Shifting a mask of occupied points right by a die value identifies destinations from every occupied source at once. Intersecting it with a mask of enemy blots finds direct hits. A double needs an open point at every intermediate stop. For unlike dice, either order may supply an open intermediate point. A double hit needs two distinct targets. The implementation accounts for stacking and for one checker stepping through two blots. The existing trained features have different rules for bar checkers in single hits, double hits and pip loss. The pip-loss computation retains the original bar-entry path. The no-bar case sums the highest reachable eligible blot for each roll.

| Change applied to the release encoder | All 244 inputs, ns/board | Feature bytes versus release |
| --- | ---: | --- |
| Release | 3,964 | Reference |
| Board masks for single and double hits | 2,317 | Identical |
| Board masks for no-bar pip loss | 2,871 | Identical |
| Both in one pass over the 21 rolls | 1,727 | Identical |
| Also reuse the masks for anchor, back-checker, bar-entry and backbone features | 1,563 | Identical |

The implementation uses unsigned bit operations and a highest-set-bit query. GCC/Clang and MSVC have native bit scans. A scalar loop supplies the portable fallback. It holds only data local to the current position. It can run on rollout threads without synchronization or cache management.

## Three-ply public calls

Each cell gives elapsed seconds for 100 fixed public calls per arm. The two-thread experiment used fresh processes and opposite arm order on each pass. The sixteen-thread experiment rotated the arm order per position with the reference rollouts paused. The reference for this experiment is the already optimized CPU release, SHA-256 `d983847b7194114e85be4edc0247f573c2d702257844c7907ed2b125e5d9878d`. The candidate changed only the contact encoder.

| API | 2 threads, release → candidate, pair 1 | 2 threads, release → candidate, pair 2 | Geometric speedup | 16 threads, release → candidate | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Checker play | 17.411 → 11.118 | 16.696 → 10.798 | 1.556× | 4.872 → 3.624 | 1.345× |
| Cube action | 1.480 → 1.182 | 1.451 → 1.058 | 1.310× | 0.573 → 0.450 | 1.274× |
| Post-move | 0.934 → 0.638 | 0.928 → 0.650 | 1.446× | 0.330 → 0.250 | 1.319× |

Every call in every arm matched in full. At two threads the hash of all 300 complete inputs, options and analytics, with elapsed time excluded, was `f0f30d701aa607eec84320b7861fb271cfc3331c28653e8d0ec00332882bcb98`. The 16-thread data is in the `enc16` directory of the combined CPU experiment archive under `base` and `encoder`. These are local Windows x86-64 measurements. [Two-thread summary](paired-2t.json) · [Two-thread full output and metadata](public-2t-full.zip) · [Sixteen-thread full outputs](../cpu_nn/measurements.zip)

## Differential checks

- All four compiled ablations and the release produced the same 115,248,320 bytes across the 196- and 244-input encoders on 65,482 boards. The corpus includes 44,282 real benchmark descendants, all relative placements of a checker behind two blots, sparse blockers, bar checkers, random legal boards and their flipped perspectives. [Per-arm hashes](components.json)
- The production-source candidate also passed the two existing extended-input test modules (95 tests) and reproduced the feature bytes on 65,504 boards via the linked Python extensions. The 22 named extra cases cover both perspectives on two on the bar, closed entry, high stacks, a backgame, bearing off, a terminal position, long doubles, and two hits from shared or separate checkers. [Release hashes](release_features.json) · [Production-source candidate hashes](final-features.json)
- A second generator independently constructed 524,288 valid boards. It assigned each point to at most one player and tracked at most 15 checkers per player across the board and bar. Each of eight strata contained 65,536 draws. They covered races, sparse contact, many blots, many blocks, mixed positions, one on the bar, multiple on the bar and both sides on the bar. Every one of 230,686,720 float32 outputs matched byte for byte against the release. The archived hashes include the raw board stream, each encoder output by stratum and both linked extensions. [Independent corpus results](random-parity.json)

## Reproduction

Run from the engine root with the project Python, NumPy, the Stage 11 weights and a release-mode CPU build. The historical MSVC recipe below expects a Ninja build at `build/multi_ply_release`. It links each private encoder against the same remaining release objects. The gzip archive contains the exact pre-experiment translation unit. The generator verifies its decompressed SHA-256 before preparing each ablation, so it can be rerun after the production source has changed.

```powershell
$study = "data/multi_ply_benchmark/experiments/cpu_feature_masks"
$work = "build/cpu_feature_masks"
$arms = @("baseline", "shots", "pips", "both", "reuse")
python "$study/variants.py" @arms
python "$study/boards.py" "$work"
foreach ($arm in $arms) { & "$study/build.ps1" -Name $arm }
python "$study/score.py" --arms @arms --output "$work/repeated-components.json"
& "$study/build.ps1" -Name baseline -Extension
& "$study/build.ps1" -Name reuse -Extension
$release = (Get-ChildItem "$work/baseline/bgbot_cpp*.pyd").FullName
$candidate = (Get-ChildItem "$work/reuse/bgbot_cpp*.pyd").FullName
python "$study/verify_extension.py" $release "$work/reference-features.json"
python "$study/verify_extension.py" $candidate "$work/candidate-features.json"
python "$study/random_parity.py" --release $release --candidate $candidate --output "$work/repeated-random.json"
python scripts/measure_multi_ply_pair.py --baseline-python python --baseline-extension $release --candidate-extension $candidate --threads 2 16 --pairs 2 --output logs/repeated_cpu_feature_pair
```

The timing runner also accepts `--baseline-native-source` and `--candidate-native-source` pointing to matching source trees for provenance. The linked binary and analytical output hashes identify the measured arms. On this host the bounded worker-window helper gave the timing processes a separate set of cores and restored the reference workers after each run.
