# Reusing work across the two cube-action search policies

The production 3-ply cube API evaluates the same pre-roll board twice. The
ND/DT search ranks checker moves under the primary cube state and uses a PubEval
prefilter when a roll has more than 16 legal endpoints. The search for the five
reported probabilities ranks all legal endpoints under dead-cube unlimited play.
The two searches can pick different moves and can use different root-pinned
networks.

## Reporting probabilities directly from the ND/DT search

The isolated prototype returned the probability vector already accumulated by
the ND/DT recursion and retained exact root bearoff probabilities. It preserved
the two perspectives' existing invert and clamp operations and the original
optional details traversal. It used the same production Stage 11 weights as the
reviewed 3-ply build. The 100 cube-action calls used two native threads pinned
to logical CPUs 8 and 9. Four fresh processes ran in ABBA order, with other
rollout work active. The reference for each position was its fixed 1,296-game
3-ply variance-reduced rollout.

| Output | Reviewed MAE | Single-search MAE | Materially closer | Materially farther |
| --- | ---: | ---: | ---: | ---: |
| Win | .008884 | .008508 | 9 | 0 |
| Gammon win | .008058 | .008152 | 6 | 9 |
| Backgammon win | .005328 | .005298 | 1 | 1 |
| Gammon loss | .006069 | .005769 | 6 | 1 |
| Backgammon loss | .005047 | .005005 | 0 | 0 |
| Cubeless equity | .028270 | .028183 | 4 | 2 |

The single-search prototype took 1.619 and 1.701 seconds for the 100 calls. The
reviewed build took 4.198 and 4.645 seconds. Every branch equity, optimal
action and optimal equity remained identical. Optional per-roll details also
matched exactly on one match contact position and one unlimited race.

The largest increases in absolute error were 1.52 percentage points for gammon
win on `cube-match-racing-08` and 0.02178 cubeless equity on
`cube-match-priming-01`. Priming and racing-with-contact positions accounted for
the two material cubeless-equity regressions. Pure races had two materially
closer probability fields and no material regressions in this sample. Eight
of the 20 pure races were exact root database lookups under both policies.
Skipping the old lookup wrapper on those eight saved only 0.004–0.005 seconds
per 100 calls. The unrestricted probability substitution was rejected.

[Field-level results by game plan, mode and database coverage](cube_tree_probabilities.json)

## Sharing identical candidate evaluations

The second prototype kept both search policies. At 3-ply the first two rolls
identify a maximum of 462 candidate batches per public cube call. Each first-pass
node records its pre-roll board, model and pinned network. The second pass uses a
saved batch only if all three match and the same roll produces the complete
originally ordered candidate list. A filtered or reordered list is excluded.
This preserves the neural network's original delta-evaluation base. The two
policies choose moves independently and can continue down different branches.

The fixed cache occupies 152,424 bytes per eligible call on x64 MSVC. Root
bearoffs, terminals, other depths and requests for per-roll details use their
original paths. All 100 complete API returns were identical across both paired
runs at both thread settings. At two threads the NN-only prototype took
3.617 and 3.827 seconds, versus 4.188 and 4.413 seconds for the reviewed build.
At sixteen requested threads confined to four logical CPUs, it took 1.633 and
1.479 seconds, versus 1.771 and 1.838 seconds for the reviewed build.

A separate prototype also saved the generated legal boards for rolls with at
most 32 endpoints. On the 92 eligible positions it reused 32,050 of 34,339
saved legal lists. It reused 15,692 NN batches. Its per-call allocation was
597 kB at the median and 904 kB at the maximum observed position. The array
sizes and list cap bound the x64 allocation for cached data to 1,702,808 bytes
per call, before allocator bookkeeping. Each independently tested variant has
its own paired control.

The subsequent three-way test gave every variant the same machine window.
Each entry below is the mean wall time for all 100 public 3-ply cube calls.

| Native threads | Reviewed | NN batches | NN batches and legal lists |
| ---: | ---: | ---: | ---: |
| 2 | 1.2477 s | 1.1254 s | 1.1165 s |
| 16 | 0.5367 s | 0.4998 s | 0.5112 s |

All 36 arm digests matched. NN-only sharing beat the reviewed build in each of
the 12 matched blocks. It was selected for the engine. Legal-list storage
saved another 0.009 seconds at two threads and cost 0.011 seconds at sixteen.
The production implementation supplies the cache only to the two passes of a
single eligible cube call. Rollouts and other standalone searches compile the
ordinary candidate batch path.

[Three-way timings](cube_memo_three_way.json) · [Complete outputs](cube_memo_three_way_outputs.zip) · [NN-only timings](multi_ply_cube_shared_experiment.json) · [Legal-list timings and memory](multi_ply_cube_moves_experiment.json)
