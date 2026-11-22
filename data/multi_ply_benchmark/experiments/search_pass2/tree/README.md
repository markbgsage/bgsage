# Deterministic 3P and 4P tree experiments

The control is `33c4aa1f6c28d12b846e66e745eef6c2692c5052`. All arms used identical Stage 11 weights, the bearoff database, the Python wrapper exported from this revision, and MSVC Release. The Windows Ryzen 7 9700X measurements used four workers on logical CPUs 8–11. Other experiments ran on disjoint CPU affinities. Fresh processes visited the same shuffled positions, alternating build order within each case. Timing covers each public call, including all checker endpoints. Every actual input, output, elapsed time and process CPU time is in the sibling compressed JSON Lines files.

## Four independent changes

Each API/level cell used the first position from each play-format and game-plan combination, in two fresh-process repetitions. Positive numbers in the table are percent reductions in total wall time relative to the control. The six builds produced identical complete analytics objects on every call (120 paired inputs per build).

| Preset and API | Control seconds | 4P batch sharing | Dead-cube specialization | Skip terminal scans | Dynamic roll scheduling | All four |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3P checker play | 1.25 | 2.8% | 0.4% | 3.1% | 2.7% | 3.2% |
| 3P cube action | 0.10 | 0.1% | −1.3% | −1.8% | 3.4% | 5.8% |
| 3P post-move | 0.08 | 3.5% | 3.2% | 1.9% | 6.7% | 5.4% |
| 4P checker play | 31.61 | 1.3% | −2.1% | −1.5% | −0.2% | −2.7% |
| 4P cube action | 3.55 | 7.0% | 2.3% | 3.4% | 4.9% | 11.0% |
| 4P post-move | 2.49 | −0.8% | 0.2% | −0.4% | 3.1% | 3.5% |

The 4P sharing arm extended the 3P live/dead pair from a fixed root plus 21 children to cover all 463 possible interior roll paths. The original eligibility rules required matching board, strategy, root network and an unfiltered ordered batch of 2–16 endpoints. The specialization computed the synthetic centered 1/1 unlimited cube's probabilities while bypassing cube-state expansion and Janowski conversion. The terminal proof used the mover's remaining checker count: with more checkers than usable dice, this roll cannot finish the game and the opponent cannot bear off during it. Dynamic scheduling let a free worker claim the next roll, keeping the weighted reductions in the original order. The dynamic scheduler runs only when the search already dispatches dice rolls in parallel.

Both the broad dead-cube specialization and skipped terminal scans slowed 4P checker play. They were discarded. Some additional time differences occur on APIs a change does not exercise. For instance, 4P batch sharing only changes cube searches, and the single-position-per-cell sample is sensitive to host load, compilation and cache state. [Paired measurements](screen/measurements.json) and [all full outputs](screen/calls.jsonl.gz)

## Native work counts

A separate count-only build reproduced the control outputs on the same 60 cases. Each cell below covers ten cases per API. The batch counters count all candidate probability vectors requested in batches of at least two, including terminal and bearoff evaluations. The reuse counts come from the two-stage cube call. All other columns span all three APIs. Doubles account for six of the 21 distinct dice rolls.

| Level | Cube candidate requests | Cube vectors reused | Cube batches reused | Dead candidate scores simplified | Terminal scans avoided | Doubles endpoints | Other endpoints |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3P | 98,994 | 13,612 | 1,755 | 65,225 | 1,316,961 | 669,693 | 578,490 |
| 4P | 2,825,127 | 225,500 | 25,094 | 2,033,782 | 33,938,059 | 18,417,221 | 14,128,425 |

The 4P dead headline recursion evaluated 64,932 leaf boards across ten cube calls. The terminal proof alone avoided over 27 million board scans in the ten 4P checker calls. It still took longer in the paired end-to-end screen. [Per-position counters](counts/measurements.json) and [public results](counts/calls.jsonl.gz)

## Shared probabilities or a precomputed dead-cube winner

The compact variant precomputed the exact move that the second traversal would select under its centered 1/1 unlimited cube, using the normal cubeless tiebreak. It stored the chosen board and five probabilities. At eligible hits it could skip both generation and ranking. With 50 distinct cube positions and two fresh-process repetitions, the 4P control took 16.314s, shared probability batches took 15.265s and the compact winner table took 15.302s. The respective 3P totals were 0.737s, 0.758s and 0.744s. All 600 candidate responses matched. [Cube comparison](cube/measurements.json) and [complete outputs](cube/calls.jsonl.gz)

The final screen paired both designs with dynamic dispatch and scored twenty positions per API at each level. Across all 120 cases both designs matched the complete control output. The compact table did not yield an additional wall-time gain. Production retains the existing probability-batch representation and extends it to 4P. It allocates on a live root miss, with room for 22 nodes at 3P or 463 at 4P (149 KiB or 3.06 MiB on x64). During a cold search each roll writes to its own path. The second traversal starts after the writer has joined, compares the model, board and pin, and can fall back without recomputing other paths. A live root already cached needs no allocated storage.

| Level and API | Control seconds | Shared probabilities + dynamic | Shared winner + dynamic |
| --- | ---: | ---: | ---: |
| 3P checker play | 1.9901 | 1.9995 | 2.0037 |
| 3P cube action | 0.1028 | 0.0973 | 0.0981 |
| 3P post-move | 0.0736 | 0.0679 | 0.0696 |
| 4P checker play | 28.5336 | 28.2719 | 28.6592 |
| 4P cube action | 3.0090 | 2.7706 | 2.7691 |
| 4P post-move | 2.0720 | 1.9831 | 1.9972 |

Production uses the dynamic queue for parallel roots with 4–21 configured tree workers. This covers the tested width and the 21 distinct dice outcomes. Higher widths retain the original scheduling for nested jobs, where pool workers can themselves submit work. The weighted dice contributions always occupy their original array slots. [Final paired screen](final_screen/measurements.json) and [full responses](final_screen/calls.jsonl.gz)

## Reproduction

Export `cpp/` and `CMakeLists.txt` from the control revision to a private source tree inside `build/`. Point the source tree at the engine model files required during CMake configuration. Run `prepare.py <source> share4`, `terminals`, `dead` or `dynamic`; multiple arms can be specified for the combined build. The `winner` arm tests storing the dead policy's chosen board in the extended shared table. Compile each arm with `scripts/build_multi_ply_engine.ps1`, keeping its extension in its own directory. The `--profile` option on the four-arm combination enables work counters. Profiling has a separate untimed-value comparison from the uninstrumented timing study.

Run `measure.py --arm control=<original extension> --arm <name>=<variant extension> --python-dir <exported /python directory> --output <new engine directory> --rounds 2`; supply a separate `--arm` for each binary. The wrapper export's normal weights location must contain the real Stage 11 models. Timing settings and binary hashes are in each measurements file. `score.py <measurements.json>` produces sums and exact-output counts by level, API and variant. The Windows process CPU timer has coarse resolution for millisecond-scale calls. Use summed wall time to compare those calls.
