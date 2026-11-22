# CPU checker move generation and PubEval

The retained generator moves and restores checkers on a single 26-entry board during the depth-first search. It visits legal origins in ascending order using an occupancy bit mask. It updates the deduplication hash as each checker moves, including hits to the opposing bar, and verifies equality against the entire board on a hash match. Only new endpoints and the doubles-prefix cache require copies. The rule requiring as many dice as possible, the higher-die fallback, early bearoff and first-seen ordering all remain part of the generator. [Production implementation](../../../../../cpp/src/moves.cpp)

## Public analytics

With a matched control and two passes over all 300 public calls, the in-place implementation took 2.622 s versus 2.643 s at 2P and 16.540 s versus 16.853 s at 3P. All 1,200 candidate responses were exactly equal, including every checker endpoint and rank. The 2P difference was small and the second pass was slightly slower. Both 3P passes were faster. On the ten-case 4P panel, two repeats took 24.581 s versus 25.074 s with exact outputs. [Full-corpus timing](inplace_full_corpus/comparison.json) · [Complete full-corpus analytics](inplace_full_corpus/calls.jsonl.gz) · [Ten-case results](comparison.json) · [Complete ten-case analytics](calls.jsonl.gz)

The table gives total control time divided by candidate time in the same interleaved run. Values above 1 indicate a faster candidate. All four variants produced exactly the same public analytics as their control on every measured call.

| Implementation | 2P ten cases | 3P ten cases | 4P ten cases | 2P all 300 | 3P all 300 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Directly add nonzero PubEval feature contributions in point order | 0.987× | 0.990× | 0.974× | — | — | Discarded |
| Cache legal first steps for each die at eight recent board-hash slots per thread | 1.038× | 1.052× | 1.022× | 0.987× | 0.987× | Discarded |
| In-place search with incremental hashing | 1.027× | 1.018× | 1.020× | 1.008× | 1.019× | Retained |
| Cache non-double first steps and use in-place search on doubles | — | — | 1.074× | 0.985× | 0.997× | Discarded |

First-step caching and the combined variant lost their screening gains on the wider corpus. The combined variant's 4P result comes from a separate three-arm run with one control and both cache variants. First-step caching by itself measured 1.064× there. The larger tests covered all 100 checker positions and all 100 cube positions through each of the cube-action and post-move interfaces, twice per build and depth. [Cache variants, full corpus](full_corpus/comparison.json) · [Complete responses](full_corpus/calls.jsonl.gz) · [Separate 4P panel](hybrid_4p/comparison.json) · [4P responses](hybrid_4p/calls.jsonl.gz)

Each worker process loaded a separate native binary and the Python analyzer from commit `33c4aa1f6c28d12b846e66e745eef6c2692c5052`. The machine used MSVC Release with AVX2 and fast floating point. Stage 11, the bearoff database, four search threads and logical CPUs 0–3 were common to all arms. One request ran at a time. Arm order rotated by position and reversed on alternate repeats. The timer covers the public API call and excludes model construction and serialization. Normal cache behavior is included. The small panel covers the five game plans in both play formats across all three APIs. It used six repeats at 2P, four at 3P and two at 4P. The published binary and the matched relink produced identical full results in all 120 control comparisons. Separate runs have their own matched controls.

## Native identity

The native fixture evaluated all 36 ordered dice pairs on 413 boards, including both perspectives of the corpus, forced entry, blocked entry, hits, terminal positions, oversize bearoffs and a 2,054-endpoint double. Every implementation emitted the same 321,292 endpoints in the same order across 14,868 dice cases. The verifier hashes each result count followed by every 26-integer board in emission order.

On those endpoints it also compared 1,285,168 raw `double` PubEval scores covering both Tesauro and ListNet weights in both contact and race modes. The in-place and first-step-cache implementations reproduced every bit. Sparse PubEval produced different floating-point bits in 745,884 scores with a largest absolute difference of 1.42e−14. Algebraic equivalence of the sparse scoring loop did not yield binary equality with the optimized dense dot product. [Native results and hashes](native_check.json) · [Fixture generator](fixtures.py) · [Native checker](check.cpp)

## Reproduction

The [variant archive](variants.zip) contains an independent patch against `33c4aa1` for each implementation. The Windows builder takes the Python binding object, compiled library, CUDA stub and CPython import library from a CPU-only Release build of that commit. It exports the pinned native headers and affected sources into a new build-only directory. Its default baseline build directory is `build/multi_ply_truncated_final`. [Variant builder](build.ps1)

From the engine directory, set `$study` to `data/multi_ply_benchmark/experiments/search_pass2/moves` and run the following commands in PowerShell with the matching Python environment. For public timing, `build/pass2_checkpoint` contains the pinned Python source and its Stage 11 models. Pass a different frozen Python directory with `--python-dir` if needed.

```powershell
$work = 'build/search_pass2_moves_replay'
& "$study/build.ps1" -BuildDirectory $work
python "$study/fixtures.py" "$work/fixtures.txt"
foreach ($arm in 'base','sparse','first','inplace','hybrid') {
    & "$work/$arm/check.exe" "$work/fixtures.txt" "$work/$arm/boards.bin" "$work/$arm/scores.bin"
}
python "$study/check_outputs.py" $work "$work/native_check.json"
$base = (Get-ChildItem "$work/base/bgbot_cpp.*.pyd").FullName
$inplace = (Get-ChildItem "$work/inplace/bgbot_cpp.*.pyd").FullName
python "$study/measure.py" --all-positions --level 2ply=2 --level 3ply=2 --build "base=$base" --build "inplace=$inplace" --output "$work/full"
python "$study/score.py" "$work/full" --output "$work/report"
```

For the ten-case screen, omit `--all-positions` and supply the published binary, matched base, sparse, first and in-place binaries in that order. For the cache confirmatory runs use `base`, `first`, `hybrid` in that order. Every saved call contains its input, elapsed wall time, implementation, evaluation level, repeat and full analytics. [Timing runner](measure.py) · [Scoring and archiving](score.py)
