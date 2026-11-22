# Multi-ply timing positions

Two fixed sets let you time the public `BgBotAnalyzer` APIs and compare their
full outputs before and after an engine change.

The [3-ply experiments and timing results](experiments/README.md) use these
sets to compare the original and optimized builds. The
[1T, 2T and 3T comparison](experiments/truncated_16t/README.md) measures
the three public truncated-rollout presets on the same positions. The [position rollout
references](rollout_positions/README.md) reuse the archived full-game checker
decisions and complete the other positions with 1,296-game public analytics.

| File | Calls | Positions |
| --- | --- | ---: |
| `checker_play.json` | `checker_play(**position["inputs"])` | 100 |
| `cube_action.json` | `cube_action(**position["inputs"])` and `post_move_analytics(**position["post_move_inputs"])` | 100 |

Each file contains 100 distinct boards.

| Source | Attacking | Priming | Anchoring | Racing with contact | Pure race | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Money | 10 | 10 | 10 | 10 | 10 | 50 |
| Five-point match | 10 | 10 | 10 | 10 | 10 | 50 |

The engine labels for the last two categories are `racing` and `purerace`.
Classification is on the source board, from the player-on-roll perspective,
before any checker move. The quotas apply independently to both files.

## Sources and selection

The canonical inputs are `data/money_benchmark/benchmark.json.gz` (500 games)
and `data/match_benchmark/5pt/benchmark.json.gz` (130 five-point matches).
All three reference tiers (`3P`, `3T`, and `rollout`) are eligible on the same terms.

Within each source/kind/game-plan group, decisions are ordered by
`SHA-256("20261121:<mode>:<kind>:<source_key>")`, with source key as a
tie-breaker. The first 10 distinct boards are retained. Previously selected
boards in that same file are skipped, including across money/match groups.
Groups are visited in the file's source and plan order. This avoids repeated
starting positions or boards with different dice filling the set, and fixes
the selection independently of source row order. The original dice and game
state of the retained decision are preserved. The two files are sampled independently.

| Set | 3P references | 3T references | Rollout references |
| --- | ---: | ---: | ---: |
| Checker | 39 | 24 | 37 |
| Cube / post-move | 73 | 17 | 10 |

Each file embeds the source paths, compressed SHA-256 hashes, original dataset
metadata, and eligible pool sizes. Every position records the original source
key, zero-based decision index, reference tier, simulation seed, turn, player,
and match game number where applicable. Use the source key to recover the
original reference analytics. For accuracy comparisons, use each checker's
per-move `eval_level`. Even a `rollout` reference can contain moves the filter
evaluated at lower levels.

## API inputs and perspective

Pass `inputs` directly to `checker_play` or `cube_action`. It contains the
board, cube value and owner, away scores, Crawford flag, Jacoby and beaver
settings, cube cap (0 for unlimited), and dice for checker play. The money
simulations enabled Jacoby and beavers. Match rules disable both.

The cube source is a **pre-roll board for the player on roll**. To reuse the
same physical position for **post-move analytics of the previous mover**,
`post_move_inputs` flips the board, exchanges player/opponent cube ownership,
and swaps away scores. Cube value, Crawford and Jacoby remain unchanged.
It omits `beaver` and `max_cube_value`, which the public post-move method
does not accept.

The post-move call therefore reports the other player's probabilities and
equities. Its game-plan grouping describes the **source player on roll**.
Compare each API against its own saved outputs when optimizing. Post-move
cubeful equity uses the public method's own conversion.

The independent 1,296-game calls illustrate the distinction. In
`cube-money-purerace-08`, the cube API returned an on-roll win probability
of 0.744020 and the post-move API returned 0.256872 for the previous mover.
The difference from exact complements was 0.000892, the largest among the
100 paired positions.

## Rebuild and verify

Run from the `bgsage` repository root using a working Python environment:

```sh
python scripts/build_multi_ply_benchmark.py
python scripts/build_multi_ply_benchmark.py --check
python scripts/build_multi_ply_benchmark.py --check --verify-engine build/bgbot_cpp.<platform-extension>
```

The generator uses only the standard library. `--check` reconstructs both sets
and checks the saved files byte for byte, including source hashes and inputs.
`--verify-engine` additionally checks current game-plan classification, legal
checker moves, the board flip, and all 300 public API calls at 2-ply with finite
results. The extension must reside inside this repository; the checked-in
Python package and local model files are used.

## Run timings

```sh
python scripts/benchmark_multi_ply.py --levels 2ply 3ply --threads 2 --repeats 1
python scripts/benchmark_multi_ply.py --levels 4ply --threads 8 --repeats 3 --output logs/multi_ply_benchmark/4ply-8threads
python scripts/benchmark_multi_ply.py --levels 1T 2T 3T --threads 16 --model stage11 --output logs/multi_ply_benchmark/truncated
```

The runner discovers a compatible extension under `build/`. Use `--extension`
to select a specific in-repository build. Each selected API runs all 100
positions. For direct 2-, 3- and 4-ply evaluations, `--cube-details` requests
the optional per-roll cube breakdown and includes it in the timed call. New runs go under
`logs/multi_ply_benchmark/` by default. `--output` accepts a new directory
inside the repository.

The runner saves run metadata, one complete analytics result per timed call
in `results.jsonl`, and `summary.json` with overall and grouped timing/value
statistics. Timings cover the public API call. The model loads before timing,
and results are serialized afterward. Each level uses one analyzer with the
engine's ordinary caches across calls and repetitions. Keep the model, threads,
settings, position order, and cache policy fixed between versions. Match rows
by `(level, api, position_id, repeat)` to compare the complete `analytics`
objects, including checker ordering and per-move evaluation levels.

The truncated presets reset the dice seed before each timed call. The seed
comes from the source position and is shared across presets, builds and
repetitions. Cube and post-move calls on the same source use that same seed.
The stored full-game references use a separate seed schedule. Each truncated
record saves the seed, input context, wall time and all public fields,
including the available standard errors. Use
`scripts/compare_truncated_benchmark.py` to compare two complete runs with
the [position-level rollout references](rollout_positions/README.md).

The [2T and 3T optimization study](experiments/truncated_search/README.md) includes
native phase measurements, the tested alternatives and the complete paired analytics.

## Stage 11 baseline, 2026-11-21

The saved baseline used two configured threads, a fresh process, and one pass
over all positions per API at 2-ply and 3-ply. Both runs used the default cube
call without optional roll details. All 600 full analytics objects matched
exactly across the two runs. The table shows the saved run in milliseconds
per position on this Windows machine.

| API | 2-ply mean | 2-ply p95 | 3-ply mean | 3-ply p95 |
| --- | ---: | ---: | ---: | ---: |
| Checker play | 22.1 | 54.6 | 217.3 | 789.4 |
| Cube action | 26.7 | 31.1 | 39.7 | 64.7 |
| Post-move analytics | 2.5 | 6.1 | 10.9 | 30.3 |

See the [full per-call outputs](baseline_stage11/results.jsonl),
[group statistics](baseline_stage11/summary.json), and
[model, build and run metadata](baseline_stage11/metadata.json).
