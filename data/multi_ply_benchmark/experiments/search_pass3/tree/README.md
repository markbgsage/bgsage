# Sharing a neural base across the 21 dice rolls

Each pair network evaluates an endpoint by adding changed inputs to saved hidden-layer sums. The 21 rolls at an interior node all start from the same board. Using that board as the arithmetic origin lets each worker prepare the hidden sums once for each network it actually uses at that node. Network routing, legal moves, promotion filters, cube-aware ranking and the order of dice accumulation stay the same. Rollout-internal deep-prefilter trees use their first surviving sibling as the arithmetic origin.

## Wall time on the complete corpus

Each row covers 100 positions in both arms. The control is the published Windows Release extension from `82e6397`. The candidate includes only the network-base change. All calls use Stage 11, the one-sided bearoff database, four tree workers and logical CPUs 0–3. Each arm has its own process and receives the same shuffled cases. The order of the two arms rotates between cases. Wall time includes the public API call and excludes analyzer construction, serialization and process startup.

| Level | API | Control seconds | Per-node base seconds | Reduction |
| --- | --- | ---: | ---: | ---: |
| 2P | Checker | 0.518 | 0.485 | 6.4% |
| 2P | Cube | 0.169 | 0.164 | 3.0% |
| 2P | Post-move | 0.117 | 0.110 | 6.0% |
| 3P | Checker | 8.671 | 8.040 | 7.3% |
| 3P | Cube | 0.623 | 0.578 | 7.2% |
| 3P | Post-move | 0.458 | 0.431 | 5.9% |
| 4P | Checker | 130.209 | 122.518 | 5.9% |
| 4P | Cube | 12.579 | 11.612 | 7.7% |
| 4P | Post-move | 9.394 | 8.528 | 9.2% |

Changing the origin changes floating-point rounding. Across 900 public calls and all 1,709 checker endpoints at each level, all best decisions matched. None of the 28,635 probability or 17,481 equity predictions moved by a material amount. The largest changes were 0.00000704 probability and 0.00000743 equity. Against the frozen rollout references, the 8,670 comparable probability fields had mean absolute error 0.005011875 in both arms. The 5,502 comparable equity fields had mean absolute error 0.02369346 in both arms. These averages pool field values including the different equity fields. Each matched position and endpoint also passed the individual material-error test. [2P/3P scores](full_2p3p/scores.json), [complete 2P/3P responses](full_2p3p/calls.jsonl.gz), [4P scores](full_4p/scores.json), [complete 4P responses](full_4p/calls.jsonl.gz)

A separate 4P screen launched a fresh native process for each public call. Ten positions per API covered all unlimited/match and game-plan combinations. Checker took 12.739 versus 11.933 seconds, cube took 1.360 versus 1.233 seconds and post-move took 0.966 versus 0.939 seconds. A 3P model-switch guard exercised Stage 9 and Stage 11 at one, four and sixteen tree threads with the same four-core affinity. All 18 warm or parallel calls matched their corresponding cold single-thread outputs, including all 12 calls after switching models. [Fresh-process scores](cold/scores.json), [complete cold outputs](cold/calls.jsonl.gz), [isolation check](identity/summary.json), [model-switch outputs](identity/calls.jsonl.gz)

## Which work repeated

An instrumented prototype counted requests across ten positions per API at each depth. It used a 256-entry per-worker table for dense bases and an 8,192-entry per-worker table for endpoint vectors. Every lookup verified the full board and base identity. Across the three 3P APIs, 96,765 dense-base requests required only 13,096 preparations (86.5% reuse). The 1,153,524 endpoint-vector requests yielded 92,764 hits (8.0%). At 4P, 1,666,185 base requests required 181,090 preparations (89.1% reuse) and 30,372,854 endpoint requests yielded 2,302,919 hits (7.6%). [Per-case native counts](profile/measurements.json)

The exact original-sibling memoization design reproduced all 60 control outputs but slowed 3P checker calls from 0.612 to 0.749 seconds. Memoizing endpoint vectors with a common pre-roll base also failed to provide consistent gains over reusing only the base. The retained implementation needs neither a board hash nor a persistent result table. A node receives a unique token. Each worker stores its prepared inputs and hidden sums by recursion depth and network slot, replacing them the next time that slot sees a different node. [Three-arm screen](screen/scores.json), [complete prototype responses](screen/calls.jsonl.gz)

## Sharing live cube survivors with the cubeless headline

The live cube tree narrows lists with more than 16 candidates to the 15 highest PubEval scores. The independent headline tree evaluates the full list. With the same pre-roll arithmetic origin, a surviving endpoint has the same probabilities in both lists, even if it changes group order. The live traversal records the original generator index of each survivor. The headline traversal merges the stored probabilities into the complete legal list, evaluates the remainder and applies its own cube policy to all candidates.

Partial reuse requires the same dice path, full board, generator count, evaluator and pin, an unfiltered reader, and a known pair evaluator with a pre-roll neural base. Other evaluators retain full-list reuse only. Bearoff values, terminal probabilities and direct blended outputs are board-specific. The fixed x64 storage increases by 72 bytes per roll, from 324 to 396 bytes, for 181.2 KiB at 3P or 3.72 MiB at 4P. The cache is scoped to one public cube call.

With only the shared neural base as control, the complete 100-position cube benchmark took 0.4342 versus 0.3700 seconds at 3P, and 10.1491 versus 8.5964 seconds at 4P. This reduced time by 14.8% and 15.3%, respectively. All 200 full analytics responses matched exactly. [Full comparison](partial_full/scores.json), [complete outputs](partial_full/calls.jsonl.gz), [independent 20-position screen](partial/scores.json)

A separate routing check loaded Stage 10's blended backgame model, a 17-network standard pair and Stage 5's five-network fallback. Each ran under nested bearoff wrappers on two boards with one and four threads. The synthetic three-anchor position selected a blended Stage 10 index. All twelve candidate results matched the per-node control and their respective single-thread results exactly. [Routing check](routes/summary.json), [complete routing outputs](routes/calls.jsonl.gz)

## Reproduction

Export the control's `cpp/`, `python/` and `CMakeLists.txt` to an engine-local build directory, with the Stage 9 and Stage 11 weight files in its `models/` directory. Run `prepare.py <source> <arm>` and compile the result with `scripts/build_multi_ply_engine.ps1` into a private Release build. Arm `node` adds the per-node base and `partial` adds the filtered-cube sharing on top. Benchmark settings and native binary SHA-256 hashes are recorded in each comparison's `measurements.json`. The runner accepts `--cold` for per-call fresh processes and saves every call's complete inputs and outputs in compressed JSONL.

Run `measure.py --arm control=<control-extension> --arm node=<node-extension> --python-dir <source>/python --threads 4 --mask 15 --levels 2ply 3ply 4ply --per-stratum 10 --output <new-directory>`, followed by `score.py <new-directory>`. To repeat the filtered-cube comparison, supply the per-node extension as control, replace the candidate with the partial extension, and select `--apis cube_action --levels 3ply 4ply`.
