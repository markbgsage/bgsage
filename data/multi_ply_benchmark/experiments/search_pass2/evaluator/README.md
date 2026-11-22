# CPU routing, contact features and sibling evaluation

The control is bgsage `33c4aa1f6c28d12b846e66e745eef6c2692c5052`. The comparisons below use the public Stage 11 analyzer on Windows x86-64 with the one-sided bearoff database. Each process had four workers and affinity to logical CPUs 4–7. The runner alternated binary order by position, used fresh processes on each repeat and timed only the public call. Startup, serialization and file I/O are excluded. Other development used separate CPUs. Every archive contains the inputs, elapsed time and complete analytics of each call, including every legal checker move.

## Independent implementations

| CPU change | 2P speedup | 3P speedup | 4P speedup |
| --- | ---: | ---: | ---: |
| Reuse flipped board and pip counts across specialist routing gates | 1.000× | 1.043× | 1.042× |
| Extract the two players' contact point masks in the same AVX2 traversal | 1.003× | 1.042× | 1.056× |
| Derive sibling network input differences from changed board points | 1.023× | 1.024× | 1.018× |

Each comparison above covers 120 calls per arm at 2P and 3P and 30 calls at 4P. The sampler selected uniformly across public API, benchmark source and game-plan strata. The routing and mask changes each matched all 270 complete analytics. Each saved sibling base still belongs to the same network as its candidates. Blended models and generic strategy evaluation keep their existing paths. A dense forward pass handles siblings with at least 64 changed inputs. [Routing and masks](routing_masks/score.json) · [Sparse differences](sparse_combined/score.json)

## Repeated all-position comparison

Each arm ran twice on all 100 checker positions, all 100 cube positions and the same 100 post-move positions at 2P and 3P. At 4P it ran twice on ten positions per API, one from each source and game-plan combination. Elapsed seconds are sums of the paired public calls.

| Level | Calls per arm | Control | Routing + masks | Also sparse | All changes vs control |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2P | 600 | 2.546 s | 2.361 s | 2.364 s | 1.077× |
| 3P | 600 | 17.914 s | 16.356 s | 15.963 s | 1.122× |
| 4P | 60 | 41.320 s | 37.660 s | 36.765 s | 1.124× |

The routing and mask pair matched every field on 1,260 calls. Adding the sparse implementation saved another 2.4% at both deeper levels. At 3P the matched speedups were 1.023× for checker play, 1.039× for cube action and 1.030× for post-move. A paired bootstrap across positions gave a 95% interval of 1.013–1.034× for the combined 3P timings and 0.978–1.018× at 2P. Repeats are paired before resampling, and strata retain their original sizes. The small 4P panel contains one position per stratum. [Per-API comparisons](additive/incremental.json) · [All timed outputs](additive/full_outputs.zip)

The sparse variant preserved every best move, complete move ordering and cube decision. Its largest movement from the control was 0.0000001193 in any probability and 0.000000775 in any equity. Zero endpoints moved materially closer to or farther from the recorded 1,296-game rollouts. Only checker candidates with a rollout reference enter the rollout-error comparison. All returned checker candidates enter the direct output comparison. Of 1,260 calls, 520 matched every field exactly. Within a single linked binary the direct sparse and existing full-input-scan kernels agreed on all 669,040 tested outputs. The tiny shifts between independently linked DLLs are consistent with differences in floating-point code generation. [Frozen-reference score](additive/score.json) · [Native comparison](check_deltas.cpp)

## Feature and routing checks

Each Windows variant regenerated the two complete feature vectors on 65,504 boards. All bytes agreed for all 196 race and 244 contact inputs. The six backgame-aware network configurations, from 19 through 24 networks, produced identical route indices across 393,024 classifications per arm. Directed tests supplement these samples for specialists occurring rarely in the underlying position data. The selected routing and mask pair passed 106 existing extended-input and specialist tests. [Control hashes and routing counts](parity/control.json) · [Probe and input details](verification.json)

The native differential test covers legal siblings and unrelated boards. For both input widths it constructs the dense inputs for each pair, checks the direct list in feature order and compares the network output to the existing dense-input-difference path. Of 133,808 tested encodings, 4,932 reached the dense fallback. The maximum output difference was zero.

## Reproduction

Run from the engine root. The Windows recipe expects the control's CPU-only MSVC Release objects under `build/multi_ply_truncated_final` and its Python package with model files available through `build/pass2_checkpoint`. The four patch files reconstruct the isolated implementations used for the screens. The `combined` arm contains routing and masks. The `all` arm adds sparse sibling differences.

```powershell
$study = 'data/multi_ply_benchmark/experiments/search_pass2/evaluator'
python "$study/prepare.py" route mask delta combined all
& "$study/build.ps1" -Arms @('route', 'mask', 'delta', 'combined', 'all')
python "$study/screen.py" --arms control route mask --per-group 4 --output build/det_eval2/routing-replay
python "$study/screen.py" --arms control delta combined --per-group 4 --output build/det_eval2/sparse-replay
python "$study/screen.py" --arms control combined all --per-group 10 --repeats 2 --output build/det_eval2/additive-replay
python "$study/score.py" build/det_eval2/additive-replay
python "$study/verify_selected.py" combined
```

The native probe expects the historical sibling corpus at `build/nn_trials/corpus.txt` (archive entry `micro/corpus.txt`) and the common feature corpus at `build/cpu_feature_masks/all.bin`. The feature corpus generator accepts the control extension with `--extension`. Run `build.ps1 -Arms all -DeltaCheck` to check both network kernels and `parity.py ARM OUTPUT.json` to reproduce the feature and routing digests. [Sibling corpus](../../cpu_nn/measurements.zip) · [Feature corpus generator](../../cpu_feature_masks/boards.py)

The final Linux verification uses the full integrated native sources, frozen separately from these Windows isolation screens. It checks SHA-256 for the control snapshot, recompiles affected translation units using the earlier CPU-only GCC Release settings and validates Stage 11 and the new binding. The optional `exact` arm changes only the implementations of encoding, network inference and batched cube evaluation, retaining all other integrated code. Its remaining objects, including the binding, are the same as the integrated build. [Linux build recipe](linux_release.py) · [Linux source and binary hashes](linux_identity.json)

On the 16-worker Linux comparison, adding the sparse implementation took 180 three-ply calls from 2.991 to 2.936 seconds (1.019×) and 60 four-ply calls from 20.310 to 20.274 seconds (1.002×). The 240 complete analytics matched exactly between these arms. The measured four-ply difference on Linux is too small to establish a performance gain. [Linux timing and output summary](../linux-sparse/summary.json)

The C++ constructor for a 17-network game-plan strategy also accepts prebuilt networks. A narrower network consumes the prefix of its slot's encoding. The compatibility check passes such siblings to the full-input difference kernel. For example, a 196-input network in a contact slot consumes the first 196 extended contact features. Comparing that case to independent dense evaluation exposed a 0.0226002 difference in the original sparse prototype. The correction reduced the maximum difference to 0.0000000596 over five public two-ply test cases, including arbitrary-width race and contact networks. Two networks wider than their slot now fail at construction before reading past the encoded input. The Stage 11 template eliminates the compatibility branch at compile time. [Native regression](check_slot_width.cpp) · [Results](prebuilt_slot_results.json) · [Guarded build recipe](slot_guard.py)
