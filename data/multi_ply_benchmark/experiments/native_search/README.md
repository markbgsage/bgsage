# Native search and checker screening, 2026-11-21

These experiments used the Stage 11 CPU build, the fixed timing sets, and MSVC
19.44 on Windows. The C++ baseline source was `139427d826dd`. The native probe
called the serial 3-ply combined cubeful probability/equity tree on all 100
cube-set boards. This isolates tree cost from model construction, Python, and
the additional states carried by the public cube-decision API.

## Where the native tree spent its time

An instrumented pass took 2.522 seconds. The five timed regions were disjoint.
Elapsed time includes instrumentation and uncategorized work.

| Region | Seconds | Share |
| --- | ---: | ---: |
| Classify, encode and evaluate candidate networks | 2.246 | 89.0% |
| Generate moves for doubles | 0.144 | 5.7% |
| Generate moves for non-doubles | 0.025 | 1.0% |
| Rank candidate equities and break ties | 0.042 | 1.7% |
| Convert chosen leaf probabilities to equity | 0.017 | 0.7% |

The generator produced 431,473 doubles and 358,735 non-doubles endpoints.
The batch evaluator processed 781,186 candidates. The leaf timer covered
38,472 conversions.

## Five independent changes

Each arm ran against all 100 boards three times in a single instrumented
process pinned to logical CPU 12. Each board was evaluated under all variants
in seeded random order, resetting the thread-local transposition table before
each call. Times below sum the 300 calls. The median pairs each call with its
control on the same board in the same pass. Shared machine load makes small
wall-clock differences difficult to distinguish.

| Variant | Aggregate seconds | Median time / control | Decision |
| --- | ---: | ---: | --- |
| Control | 7.067 | 1.000 | Baseline |
| Vectorizable board hash, equality check on every occupied slot | 6.870 | 0.964 | Keep |
| Skip previously visited intermediate boards on doubles | 6.899 | 0.965 | Keep |
| Both move-generation changes | 6.846 | 0.964 | Keep together |
| Avoid counting pips when contact fixes cube efficiency | 6.983 | 0.998 | Discard |
| Expand the common leaf cube states once per 2-ply node | 7.127 | 1.000 | Discard |
| Share the first neural-network reference base across all serial rolls of a node | 7.005 | 0.983 | Discard |

The first four changes individually preserved all six output floats for every
board. Sharing the neural-network base changed the maximum probability by
0.000000119 and the maximum equity by 0.000000805. Its isolated elapsed-time
gain was below 1% in aggregate. Keeping per-node bases also adds per-thread
scratch storage and changes the order of floating-point accumulation.

For move generation alone, the four-way test covered the union of the two
timing files and 500 seeded samples from each source dataset, counting
repeated positions. It compared the entire sequence of endpoints across all
21 rolls on 1,200 positions, with four timed repetitions per arm. All 630,312
endpoint entries matched in order. Total times were 391.3 ms original,
217.1 ms hash only, 218.9 ms intermediate-prefix pruning only, and 106.0 ms
with both. The changes made generation 3.69 times faster on this workload.

The production hash table also grows before it fills. A 1–1 with 2,054 distinct
legal endpoints exposed an infinite probe loop in the original fixed table.
`move_capacity.cpp` passed all 21 rolls against the independent sorted
generator. Compile it with `cpp/src/moves.cpp` and `-Icpp/include`.

The cache fingerprint now includes the identity of the candidate filter and
whether deep pruning treats the current node as an entry or a descendant.
Switching the filter changed 72 of 100 cold 2-ply values. Switching the deep
pruning context changed 69. With the corrected key, going A→B and B→A recovered
the independent cold values for all 100 boards in each experiment.

`native_probe.zip` retains the exact instrumented C++ files, standalone move
benchmark and its positions, Python drivers, raw timings, neural outputs, and
per-region counters. Mode bits 1, 2, 4, 8 and 16 correspond to the five changes
in table order. The probe DLL is for measurement only. To rebuild it, export
the baseline's `cpp/` into a private subdirectory of `build/`, replace
`cpp/src/{moves,cube_eval}.cpp` with the archived files, and use the CPU CMake
build. The Python probes expect `build/multi_ply_engine` or `build/multi_ply_anchor`
for their respective private DLLs. Region counters require a single native
thread and the pinned driver selects logical CPU 12.

## Repeated bearoff database wrapper

The original public checker and post-move calls could pass both a strategy
already wrapped in `BearoffStrategy` and the same database as a binding
argument. The original binding wrapped the strategy a second time. A clean native probe
compared the two-layer call with omitting the redundant database argument.
Both arms used the same already-wrapped strategy, all 100 cube-set boards,
one CPU thread and three passes. Arm order was randomized for each board
and the transposition table cleared before every call. All 300 pairs had
exactly equal probabilities and equity. Two layers took 5.870 s and one
took 5.879 s. The median paired ratio was 0.99876. The trial did not
justify a production speed change. See `bearoff_elision.json.gz`.

The original `eval_each_impl` implementation holds one thread-local
`std::vector<int> rest`. When the first layer recurses into a second layer,
its `idxs` argument points into that same vector. The second layer calls
`rest.clear()` before reading `idxs`, invalidating the references to its
former elements. This alias exists with nested wrappers in the baseline.
With a single layer `idxs` initially points into a different vector and
the recursive call bypasses the wrapper branch. The timing experiment
found no value difference on the fixed boards and did not reproduce any
wrong output or crash. The release binding reuses a wrapper for the same
database. For distinct databases the evaluator walks the wrapper chain and
compacts surviving indices in place. The [wrapper and cache regression tests](../../../../tests/test_multi_ply_bearoff_cache.py)
cover the explicit-database and nested-wrapper cases.

## Two-ply checker shortlist

The screening arm first evaluated every legal endpoint at 2-ply. It evaluated
the top two and every move within 0.08 of the best at 3-ply. It re-ranked the
mixed-depth list and promoted any remaining 2-ply move that entered the top
two, until both leading moves had full-depth values. The experiment used the
normal public checker implementation, restricting root enumeration in the
private Python process to score shortlisted endpoints. The public API and
production filter settings were unchanged.

Both arms used two configured threads on logical CPUs 12–15. They used separate
processes, identical position order, and a clean native baseline rebuilt from
the baseline commit. Its SHA-256 is
`23396e8a5ed986e2560d859d47f9eac5a44b2ff96b34b73e1a496c3281bf36c1`.
The Python analysis path already skipped the redundant cubeless traversal.
The exact pass took 48.48 s under concurrent system load and the screened pass
12.20 s. These sequential runs give a provisional 3.97× speedup. This result
does not establish the speedup for the final combined engine.

Across 100 decisions, all first choices survived. Three top-two sets changed.
The screen kept 574 of 1,709 moves at 3-ply, including three promotions.
Among the 1,135 moves left at 2-ply, 880 changed cubeful equity and 872 changed
cubeless equity by more than 0.005 versus the full-depth control. Win
probability changed by more than 0.0025 on 731 moves.

The historical benchmark supplies rollout values for 224 comparable finalists.
The 221 retained finalists had identical results. On all three finalists
left shallow, cubeful equity moved materially farther from rollout
(sum of absolute-error increases 0.08138). Cubeless equity improved materially
on two. The fresh 1,296-game references also exposed material deterioration.
For one screened move in `checker-money-attacking-02`, the rollout gave
−0.107065, full 3-ply gave −0.072205, and the screen gave −0.313217.
Screening increased absolute error by 0.171291. In `checker-match-anchoring-04`
one move's error rose by 0.038059. Among the 176 shallow moves that had a
supplemental fixed-strength checkpoint, 84 cubeful equities moved materially
farther and 59 materially closer. The completed position benchmark includes
four rollout-scored moves that the shortlist would have left shallow. All
four moved materially farther in cubeful equity. The full-output API keeps
the exact evaluations. Reproduce the comparison with `score_shortlist.py`.

`exact.jsonl.gz` and `shortlist.jsonl.gz` contain timings and complete ranked
analytics per decision. `shortlist_summary.json` records the independent
reference comparison for each probability and equity, split by reported
depth. A material deterioration increases absolute error against the same
reference by more than 0.0025 (probability) or 0.005 (equity).
