# CPU sibling neural inference

Two changes reduce the cost of scoring legal checker moves from the same roll. The x86 implementation compares eight input differences at a time, visits changed columns in their original order, and applies four successive columns before storing a tile of hidden activations. The latter change saves three loads and stores of the same hidden values. Both paths use the same weights, sibling base, original limit of 64 changed inputs, and per-hidden-unit arithmetic order. They use the engine's existing AVX2 path. ARM and scalar builds continue through the original computation.

## Public 3-ply calls

The table compares the candidates against the improved CPU board encoder with the original neural kernel. Each column contains wall seconds for all 100 positions. Each build ran in its own preloaded process. For each position, the processes took turns, with their order balanced across positions. Each process saw the same shuffled positions and cache history.

| Threads | API | Encoder only | Four-column accumulation | Accumulation and SIMD scan | Time saved by both |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | Checker play | 10.967 | 10.858 | 10.234 | 6.7% |
| 2 | Cube action | 1.099 | 1.066 | 1.028 | 6.4% |
| 2 | Post-move | 0.771 | 0.740 | 0.723 | 6.3% |
| 16 | Checker play | 3.624 | 3.471 | 3.383 | 6.6% |
| 16 | Cube action | 0.450 | 0.436 | 0.433 | 3.7% |
| 16 | Post-move | 0.250 | 0.242 | 0.240 | 4.1% |

Every returned field matched for all 300 calls across the measured arms. On the preceding release with its older encoder, the scan alone changed 16-thread total time by less than 0.8% for any API. Both changes together reduced time by 2.1–3.1% on that release. A separate chronological 2-thread pair on the older encoder was mixed, ranging from a 5.0% speedup for cube action to a 2.7% slowdown for post-move. This sensitivity led to the per-position paired trials.

Measurements used a Windows Ryzen 7 9700X with 8 physical cores and 16 logical processors, MSVC 19.44 and the existing release compilation flags. VM performance depends on its CPU and available threads. The original native control was `d983847b7194114e85be4edc0247f573c2d702257844c7907ed2b125e5d9878d`. [Timings and bootstrap estimates](results.json)

## Component measurements and rollout guard

With the actual Stage 11 400-hidden-unit contact network, median inference time over a pass through 1,422 legal siblings was 541 ns for the original kernel, 496 ns for the scan, 495 ns for four-column accumulation and 446 ns for both. These 18 pass averages came from three launches per variant. This measurement includes the first full evaluation in every sibling group and excludes board encoding. The standalone probe compared 9,210 output floats bit-for-bit, including synthetic checks on partial vector widths and heap-backed networks. Its 35 shape combinations cover hidden widths of 4–513, input widths of 3–301, unchanged siblings and lists crossing the existing 64-delta limit.

The incremental rollout guard used the same assets, seeds and Python code, comparing the improved encoder against that encoder plus both NN changes. It covered the first match attacking and unlimited pure-race position for each API. Two pairs ran in AB and BA order, with four trial threads on four logical CPUs. The split scheduling window left the main reference workers running on the remaining twelve CPUs.

| Trial policy | Checker speedup | Cube speedup | Post-move speedup |
| --- | ---: | ---: | ---: |
| Production 3T, 360 games | 1.067× | 1.054× | 1.062× |
| Full game, 3-ply checker and cube, 36 games | 1.097× | 1.028× | 1.053× |

All 24 paired complete rollout outputs matched, including candidate boards, decisions, equities and standard errors. This guard measures speed and determinism. The dedicated 1,296-game reference corpus supplies the benchmark accuracy values. [Full analytics, per-case timings and probe inputs](measurements.zip)

## Reproduction

`prepare_variant.py` reads the original neural source directly from engine commit `139427d826dd67798226e7d23c003c90315306ba`. The `--source` option supplies all other engine sources, including the encoder. The resulting `both` source reproduces the measured Windows file SHA-256 `a49d23aefbdc63e704c7dae6c7e6144de2a10a8eb9b4caca29733d501023f21c`. The generator supports `base`, `scan`, `block` and `both`. The earlier `matched16/scan` arm used direct input comparisons before the final scan switched to comparing the actual vector differences against zero. Use `scan-compare` to reproduce that arm.

From the engine root with the local benchmark environment selected, the following PowerShell commands build a pair against the same encoder and benchmark all three public APIs. The build directories and the output directory must be new. Change the extension suffix to the one produced for your Python.

```powershell
$study = 'data/multi_ply_benchmark/experiments/cpu_nn'
$python = (Get-Command python -CommandType Application).Source
foreach ($v in 'base','both') {
    & $python "$study/prepare_variant.py" $v "build/nn_repro_$v"
    & scripts/build_multi_ply_engine.ps1 -Python $python -Source "build/nn_repro_$v" -BuildName "multi_ply_repro_$v"
}
& $python "$study/measure.py" --threads 2 --output build/nn_repro_2t --build "encoder=build/multi_ply_repro_base/bgbot_cpp.cp314-win_amd64.pyd" --build "encboth=build/multi_ply_repro_both/bgbot_cpp.cp314-win_amd64.pyd"
```

The runner writes each complete analytics call to its arm's JSONL file and checks per-case hashes. Repeat with `--threads 16` and a different output directory for the many-core comparison. For the rollout guard, use the engine's `compare_rollout_performance.py worker` on snapshots of the two builds with the same Python and pinned assets, four threads, and `--full-trials 36 --profiles 3T full`. The archive includes the elapsed seconds and random seeds for each individual call.
