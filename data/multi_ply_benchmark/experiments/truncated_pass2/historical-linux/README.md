# Original Linux truncated rollout benchmark

The Linux baseline for 1T, 2T and 3T is **original + cache race fix**. The
analytics code and Python package come from `139427d826dd67798226e7d23c003c90315306ba`.
The original Windows measurements found that two workers could write the
same opening-cache entry during a cubeful rollout. A trial could then
consume an incomplete variance-reduction mean. The Linux baseline reserves
the first-move and reply entries before the pool starts. Each worker still
evaluates the same positions, and each trial still uses its indicated dice.
[Windows investigation and original measurements](../../truncated_16t/README.md)

The [source diff](prefill.patch) contains ten inserted lines in one function.
The [build manifest](provenance.json) verifies 73 original C++ and Python Git
blobs and records the original Linux CPU-only CMake wrapper. The build recompiles
`rollout.cpp` with the original GCC 15.2 release flags and replaces that one
object in the 15-member static archive. It uses the original Python binding
and CPU stub. The original and rebuilt extension SHA-256 digests, object-level
proof and identities of the 24 ordered Stage 11 weight slots are in the manifest.

Each preset runs sequentially in a fresh process with 16 rollout threads on
the Ryzen 7 9700X under WSL2. One public analytics call runs at a time.
Every preset covers 100 checker, 100 cube and 100 post-move requests, with
the same inputs, per-position seeds, variance-reduction settings and bearoff
database as the original Windows benchmark. Wall seconds cover each public
call, including use of the normal engine caches. Analyzer initialization and
writing the results occur outside the timed interval. The `git_commit` in
the benchmark metadata identifies the checkout running the measurement.
The native and Python source and extension identity identify the measured
historical engine. [Benchmark protocol](../../../README.md)

The [full outputs](full_outputs.zip) contain `results.jsonl`, `summary.json`
and `metadata.json` under `original/1T`, `original/2T` and `original/3T`.

| Preset | Checker play, s | Cube action, s | Post-move, s | All 300 calls, s |
| --- | ---: | ---: | ---: | ---: |
| 1T | 24.543 | 7.669 | 7.724 | 39.936 |
| 2T | 122.711 | 49.606 | 50.169 | 222.486 |
| 3T | 346.658 | 132.649 | 131.131 | 610.438 |

The [Windows comparison](windows_comparison.json) pairs every public call
with its original MSVC counterpart and scores both against the full-game
reference. Those pairs include a compiler and platform change in addition
to the cache correction. All 300 best checker choices and all 300 cube
decisions agreed between the two platforms. One checker position,
`checker-match-purerace-06` at 3T, reordered moves ranked 17–19. Six of
the 900 full responses matched exactly across platforms.

The three materially closer fields all came from the fourth-ranked move at
`checker-money-priming-10` at 2T. Its win probability changed from 0.893650
to 0.820617 against a full-game reference of 0.819967. Its gammon-win
probability changed from 0.205130 to 0.188065 against 0.187045, and its
cubeless equity changed from 0.991518 to 0.827998 against 0.826826. Zero
scored fields moved materially farther. The separately archived Windows
release with the ownership correction had zero material differences in
rollout error from this Linux baseline at all three levels.

The unmodified Linux binary on one worker reproduced all three complete
16-worker cache-regression outputs exactly, including the 2T values above.
[Serial controls and complete analytics](serial_check.json)

Run the following commands from the engine root in the historical Ubuntu
environment with its original source export, GCC build products and Stage 11 assets.
The first command verifies inputs and builds the comparator. The second runs
1T, 2T and 3T in order and compares the original Windows results. The three
serial controls run in a separate process in the gap between 2T and 3T.

```sh
python3 -B data/multi_ply_benchmark/experiments/truncated_pass2/historical-linux/replay.py build
python3 -B data/multi_ply_benchmark/experiments/truncated_pass2/historical-linux/replay.py run
```
