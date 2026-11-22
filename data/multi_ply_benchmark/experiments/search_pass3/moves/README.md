# Legal move generation and PubEval

Canonical generation reduced 3P wall time by 4.0% across the 300 public calls on four logical CPUs. Both full repeats improved. Every complete response matched the checkpoint, including the ranked alternatives. The native generator matched both public move orders across 571,608 ordered rolls. [Public timings](final/comparison.json) · [Complete analytics](final/calls.jsonl.gz) · [Native checks](native_check.json)

## Why the pruning is exact

For a double, a move origin has a single destination, including entry from the bar and removal by bearoff. Given the initial and final boards, work from the bar down toward point 1. At each origin, the starting number of friendly checkers plus arrivals from higher points minus the final number is precisely the number of departures. The endpoint therefore uniquely determines a multiset of origins.

A move from one origin cannot disable a currently legal move from another. It does not remove a friendly checker at that other origin or create an opposing made point. Entry from the bar takes precedence, in which case the bar is the sole legal origin. Moving forward can only make bearoff easier for another checker. Hits to a blot that is visited by both orderings end with the same opponent on the bar.

During the ascending depth-first search, mark a legal origin after exploring its child. Later siblings carry that bit through all their descendants. Any path that subsequently uses the marked origin has an equivalent ordering beginning with the earlier child. Conversely, the earliest legal ordering for the endpoint can contain no such skipped move. This produces each endpoint once in its original first-seen order. Doubles need neither intermediate board comparisons nor an endpoint hash table.

With mixed dice, the first order is searched in full. In the reverse order a second step that was legal at the start commutes with the first step, so its endpoint is already covered. If both steps start at the same point, the reverse path can take its second step there only when an additional checker was present. Hits and other coincident endpoints still receive a full board-equality check on a hash match. Both walkers preserve the rule requiring the maximum number of dice, the larger-die fallback, early bearoff and a single unchanged result when no move is possible. Sorted callers sort the unique endpoints lexicographically. [Production generator](../../../../../cpp/src/moves.cpp)

## Measurements

The two mixed/doubles profiles used the 413 registered and edge-case boards, all 21 rolls and 60 traversals per batch. The six doubles accounted for 74.8% of time in the checkpoint generator. For these rolls alone, the canonical generator took 0.195 seconds versus 1.168 seconds summed across the two batches, a 6.0× native speedup. Mixed-dice pruning with canonical doubles reduced mixed-roll time from 0.394 seconds to 0.272 seconds. These timers measure move generation only. [Profiler](profile.cpp) · [Native measurements](native_times.txt)

The Windows MSVC AVX2 study used four Stage 11 search threads on logical CPUs 8–11. Competing implementation processes each loaded their own extension, then ran one at a time with arm order rotated by position and reversed on alternate repeats. Every cell below compares complete responses from the same matched run. Times exclude serialization, model construction and loading the one-sided bearoff database. The 1P through 3P rows contain the 100 checker inputs, 100 cube decisions and the same 100 post-move inputs twice. The 4P screening panel uses ten positions covering both play formats and all five game plans, with three repeats.

| Level | Calls per build | Checkpoint seconds | Canonical seconds | Wall time saved | Exactly equal |
|---|---:|---:|---:|---:|---:|
| 1P | 600 | 0.0651 | 0.0637 | 2.1% | 600 |
| 2P | 600 | 1.4503 | 1.3904 | 4.1% | 600 |
| 3P | 600 | 14.8422 | 14.2439 | 4.0% | 600 |
| 4P | 30 | 42.1053 | 41.5286 | 1.4% | 30 |

The 4P panel had two slower individual repeats and 23 faster paired calls out of 30. Its aggregate timing is less precise than the repeated full corpus. [1P–3P measurements](final/comparison.json) · [All 1P–3P responses](final/calls.jsonl.gz) · [4P panel](deep/comparison.json) · [All 4P responses](deep/calls.jsonl.gz)

* Doubles pruning alone improved 3P by 3.6%. Reverse-order pruning alone increased 3P time by 3.5%. Combined, the two reduced 3P time by 3.8%. The shared specialized mutation implementation above retained the 3P gain. [Four-arm comparisons](ablation/comparison.json) · [Complete responses](ablation/calls.jsonl.gz)
* Routing sorted callers through the canonical generator removed the second recursive implementation. The separate three-arm run was effectively flat at 2P/3P relative to keeping the old sorted generator and 1.0% faster at 1P. [Sorted caller comparison](sorted/comparison.json) · [Complete responses](sorted/calls.jsonl.gz)
* A fixed table keyed by move hash deltas skips the initial hash and per-thread allocations. Mixed dice have at most 24 origins per step and two dice orders, so 2,048 slots cover even 1,152 distinct paths. The isolated improvement over the retained kernel was 0.4% at 2P and 0.7% at 3P. A separate PubEval prototype precomputed point scores by checker count, fused both borne-off counts into the score traversal and bypassed the 122-feature vector. Its complete corpus took 0.8% longer at 2P and 0.4% longer at 3P. Neither prototype entered the integrated build. All 3,000 candidate responses across the two experiments matched. [Incremental comparisons](extras/comparison.json) · [Complete responses](extras/calls.jsonl.gz)

## Native checks and reproduction

The verifier links the checkpoint move generator under a separate symbol. For every roll it compares every integer of every returned board in sequence. Inputs cover both sides of the benchmark positions, blocked and forced bar entry, an endpoint set above 2,048, 3,465 near-bearoff positions and 12,000 seeded boards. It tests the sorted and first-seen interfaces independently. [Verifier](verify.cpp) · [Corpus fixture builder](../../search_pass2/moves/fixtures.py) · [Native result](native_check.json)

The Windows replay uses the C++ headers and source from checkpoint `82e6397` and the other compiled objects from its CPU-only Release build. It applies one independent patch per arm. The default checkpoint build is `build/multi_ply_second_final`. The public timing script takes a separate extension for each arm and writes inputs, outputs, times and binary hashes. Use the checkpoint's Python package with Stage 11 models. [Replay script](build.ps1) · [Variant patches](variants.zip) · [Public timing runner](../../search_pass2/moves/measure.py)

```powershell
$study = 'data/multi_ply_benchmark/experiments/search_pass3/moves'
$runner = 'data/multi_ply_benchmark/experiments/search_pass2/moves'
$work = 'build/search_pass3_moves_replay'
& "$study/build.ps1" -BuildDirectory $work -Arms base,fast
python "$runner/fixtures.py" "$work/fixtures.txt"
& "$work/fast/verify.exe" "$work/fixtures.txt"
& "$work/base/profile.exe" "$work/fixtures.txt" 60
& "$work/fast/profile.exe" "$work/fixtures.txt" 60
$base = (Get-ChildItem "$work/base/bgbot_cpp.*.pyd").FullName
$fast = (Get-ChildItem "$work/fast/bgbot_cpp.*.pyd").FullName
python "$runner/measure.py" --python-dir python --mask 3840 --all-positions --level 1ply=2 --level 2ply=2 --level 3ply=2 --build "fast=$fast" --build "base=$base" --output "$work/full"
python "$runner/score.py" "$work/full" --output "$work/report"
```
