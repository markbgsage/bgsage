# Third CPU multi-ply search study

On 16 logical CPUs, the complete Linux panel took **1.868 s instead of
2.515 s at 3P** and **32.401 s instead of 47.637 s at 4P**. Reversing the
build order gave 1.889 vs 2.453 s and 31.780 vs 47.054 s, respectively.
All five game-plan groups improved at 2P, 3P and 4P. The Linux 1T, 2T and
3T panels each saved about 19–21% wall time. Windows saved 24.9% at 3P
and 39.4% at 4P. At all seven presets on both platforms,
every checker ranking and cube choice matched the preceding engine, and
zero rollout fields moved materially closer or farther.

This study starts at engine commit `82e63972379b77ab52ebafc716620e813ff7baa1`.
It measures all 100 checker calls, 100 cube calls and 100 post-move calls for
each preset. The paired builds use the same Stage 11 weights, bearoff database
and public inputs. The rollout reference is the existing 1,296-game corpus.
The timed interval covers the public analytics call, including the complete
list of legal checker choices. Each run retains every input and output.
All 1,709 legal checker endpoints are paired across builds. The 378 endpoints
with independent rolled values contribute to the truth comparison, along
with all 100 cube and all 100 post-move positions. This yields 4,724
rollout-scored probability and equity fields per preset.
[Full-corpus timings and accuracy](results.md) · [Reference coverage](../../rollout_positions/README.md)

The archive checks independently recompute the summaries and match the
measured binaries, source snapshots, ordered model weights and input files.
The final release was timed at every preset. At 1T–3T the comparisons use
the exact baseline recordings from the preceding paired run on each platform.
[Linux verification](release/linux/verification.json) · [Windows verification](release/windows/verification.json)
· [Reverse-order verification](release/linux-repeat/verification.json)
· [Full-game rollout timing check](fullgame_guard.md) · [Regression suite](release/windows-regressions.json)

## Experiments

1. **Compute bar-entry pip loss directly.** Compute which entries, intervening
   points and direct hits a roll can use. This removes an intermediate set of
   39 shot patterns. The first independent Linux screen reduced 3P time from
   5.398 to 5.121 seconds on 240 calls. [Encoder measurements](evaluator/README.md)
2. **Cull unreachable targets and evaluate rolls in SIMD lanes.** An off-bar
   checker can only hit a blot in front of it. Apply those masks to all 15
   mixed rolls and six doubles in three AVX2 groups. Together with direct
   bar-entry calculations, the final encoder saved 15.7% on all 300 3P calls
   and 18.5% on 60 stratified 4P calls. All responses matched. The independent
   portable build matched the same 63,910,432 reference feature bytes.
   [Ablations, scalar oracle and complete outputs](evaluator/README.md)
3. **Generate canonical doubles.** Once a legal origin has been explored at a
   decision point, later siblings can skip paths using that origin. The
   generator emits the first legal sequence for each endpoint directly. The
   six doubles took 0.195 s versus 1.168 s in the move-generation-only test.
   [Proof and measurements](moves/README.md)
4. **Eliminate commuting mixed-dice moves.** The reverse dice order explores
   newly available origins. Sorted and first-seen-order consumers use the
   same generator. Together with canonical doubles, full-corpus 3P took 4.0%
   less time. Each of the two move APIs matched its reference for 13,882,557
   endpoints. A fixed hash of movement deltas offered less than 1% on top.
   Table-based PubEval scoring increased wall time. [Independent alternatives](moves/README.md)
5. **Share first-layer neural work across dice.** Every roll at a node starts
   on the same board. Save that board's hidden sums once per node, worker and
   network and evaluate candidates by their input differences. Roughly 89%
   of dense bases could be reused in the 4P profile. Complete 2P, 3P and 4P
   calls took 5.5%, 7.2% and 6.3% less time in the independent screen.
   [Routing, arithmetic and results](tree/README.md)
6. **Cache individual endpoint network outputs.** Only about 7.5% of
   endpoint vectors repeated in the same 4P profile. Remembering whole
   endpoint vectors gave inconsistent savings over sharing the dense base.
   [Cache variants](tree/README.md)
7. **Share filtered cube candidates by move index.** The independent
   cubeless search reuses eligible neural results for the live search's
   selected moves, evaluates the rest and scores the full list under its
   own policy. All 200 3P/4P cube responses matched the node-base-only
   build, with a further 14.8% and 15.3% less time, respectively.
   [Complete cube panel and model-isolation tests](tree/README.md)
8. **Share completed subtrees between checker alternatives.** After each
   player has moved, paths from distinct legal root moves can meet. The
   four-core probe hit 12,501 of 127,612 probes at 4P and only 2 of 7,273 at
   3P. Parallel 4P checker calls share the four cube-state equities and five
   probabilities from matching completed 2P trees. The 20-position 4P
   checker screen took 27.337 s against 29.518 s. All 20 complete responses
   matched. [Cache counters](subtrees-cash/cache_counts.json) · [Paired outputs](subtrees-cash/outputs.zip)
9. **Parallelize the 21 dice at a 2P root.** On all 100 cube and post-move
   positions, twice, a four-thread Linux run took 0.153 s versus 0.293 s for
   cube decisions and 0.090 s versus 0.202 s for post-move calls. All 400
   responses matched. Roots invoked with one search worker stay serial.
   A final game-plan check found that 2P pure-race rolls on Linux are too
   cheap to justify waking helpers. Over four full-corpus runs, evaluating
   those race roots serially saved 59% on race cube calls and 68% on race
   post-move calls against the fully parallel implementation. All 1,200
   responses matched. Windows benefited from parallel evaluation on these
   same positions. [Initial screen](linux-shallow-cash/summary.json) ·
   [Width sweep, complete outputs and release policy](race2/README.md)
10. **Balance checker endpoints and dice rolls.** With 16 logical CPUs,
    57 of the 100 checker positions have fewer legal moves than workers.
    A nested dispatcher lets workers help with root rolls. Across two fresh
    20-position 4P screens, using at least two roll workers per endpoint took
    17.460 s against 20.211 s. The four-worker experiment showed a different
    tradeoff. Against the integrated engine with all the other changes,
    the selected scheduler took 27.838 s versus 30.893 s for all 100
    Linux 4P checker calls, with all complete responses identical.
    [Full integrated panel](integrated-schedule/summary.json) · [16-thread measurements](schedule16/summary.json) ·
    [Four-thread measurements](schedule4/summary.json)
11. **Precompute the whole match cash-point ladder.** The score determines
    the recube levels and MET payoffs. Preparing the entire ladder once per
    score left mostly interpolation work per candidate. The Linux 3P
    improvement was small and it did not improve the Windows 4P checker
    panel. The production match calculation stays as it was.
    [Linux panel](linux-shallow-cash/summary.json) · [Windows panel](subtrees-cash/summary.json) · [Prototype](subtrees-cash/cash.patch)

The four-core screens pinned workers to separate CPU groups and rotated arm
order within each position. Development elsewhere on the same machine can
affect these measurements. Absolute latency, CPU seconds and per-call parity
are in the saved archives. Individual percentage changes are relative to
each experiment's own control and cannot be added.

## Reproduce the comparison

Use the same compiler, Release flags, Python interpreter and pybind11
installation for both builds. The following Linux commands run from the
engine root. The source export stays inside the engine. It uses the same
model files and bearoff database as the candidate.

```sh
study=data/multi_ply_benchmark/experiments/search_pass3
base=build/replay-pass3-source
mkdir -p "$base"
git archive 82e63972379b77ab52ebafc716620e813ff7baa1 CMakeLists.txt cpp python | tar -x -C "$base"
ln -s "$PWD/models" "$base/models"
ln -s "$PWD/data" "$base/data"
for arm in base current; do
    src=.
    if [ "$arm" = base ]; then src="$base"; fi
    cmake -S "$src" -B "build/replay-pass3-$arm" -DCMAKE_BUILD_TYPE=Release \
        -DBGBOT_BUILD_APPS=OFF -DPYBIND11_FINDPYTHON=ON \
        -DPython_EXECUTABLE="$(command -v python3)" -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
    cmake --build "build/replay-pass3-$arm" --target bgbot_cpp --parallel
done
python3 -B "$study/paired.py" \
    --baseline build/replay-pass3-base "$base" --candidate build/replay-pass3-current . \
    --run-dir logs/replay-pass3 --output logs/replay-pass3-report \
    --levels 1P 2P 3P 4P 1T 2T 3T
```

Use new output directories for each run. Add `--candidate-first` to reverse
the arm order. The score compares the change in absolute error separately
for each field and rollout-scored checker endpoint. The candidate can also
be compared directly on all 1,709 legal checker endpoints, including those
without a full-game reference. [Timing runner](../../../../scripts/benchmark_multi_ply.py)
· [Rollout scoring](../../../../scripts/compare_truncated_benchmark.py)

For the final release builds, `python -B verify.py linux --release` or
`python -B verify.py windows --release` from this directory checks the stored archive
against the original source snapshots, binaries, corpus, model files and
rollout truth. To check the independent repeat, use
`python -B verify.py linux-repeat --release`. Omitting `--release` audits the
earlier paired builds from before the Linux 2P pure-race adjustment.
[Archive verifier](verify.py)
