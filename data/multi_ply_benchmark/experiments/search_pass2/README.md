# Second CPU multi-ply search study

On the 300-call deterministic benchmark, this pass reduced 3P wall time from
3.074 to 2.492 seconds on Linux and from 4.027 to 2.639 seconds on Windows.
At 4P, Linux went from 55.844 to 49.436 seconds and Windows from 62.655 to
53.586 seconds. All 1,200 complete deterministic responses on each platform
matched the checkpoint. On Windows all 900 truncated responses also matched,
with wall-time reductions of 6.1% at 1T, 8.1% at 2T and 11.8% at 3T.
On Linux, 1T was effectively flat at 10.111 seconds against 10.125, 2T
took 67.918 seconds against 77.580, and 3T took 237.109 seconds against
259.308. All 900 responses matched. Reversing build order in a separate
full-corpus Linux run gave 20.8% less time at 3P and 5.5% less at 4P.
[All levels, both historical comparisons, and times by API](results.md)

The starting point for this pass is bgsage `33c4aa1`, including the published
truncated-rollout optimizations. The original engine is `139427d`. All three
public calls use production Stage 11 and the one-sided bearoff database.
The [fixed corpus](../../README.md) covers 100 checker decisions and 100 cube
positions, each used once for cube action and once for post-move analytics.
Each preset therefore has 300 timed public calls. Measurements exclude
construction, result serialization and disk writes. They include all work
inside a call, including model loading when the implementation requests it.
Both control packages were checked against their Git revisions on both
platforms. The 24 ordered Stage 11 network slots, fixed positions, and bearoff
database matched the recorded identities. [Verified source and binary fingerprints](control_inputs.json)
The benchmark metadata contains the loaded extension's SHA-256 and the
fingerprint of each native and Python source file. Candidate runs were
measured from the working tree on the checkpoint commit.
[Archive integrity, counts and independently recomputed accuracy](final_checks.json)

The Linux measurements use GCC in a local WSL2 x86-64 guest on the same
Ryzen CPU. The engine and assets are on a Windows-mounted filesystem. This
matters especially for 1P cube action, because the original and checkpoint
implementations open the model files inside every timed call. An AWS VM's
filesystem and CPU will affect its absolute latency.

## Experiments and attribution

The initial screens pinned four workers to four logical CPUs on a Ryzen 7
9700X. Variants ran in separate processes and alternated order across matched
positions. Other development work used disjoint affinities. These measurements
identify which ideas deserve full-width testing. Each archive retains the
complete outputs as well as elapsed time.

1. **Remove repeated work at the 1P public boundary.** Cube action loaded the
   24-network model on every call. Reusing the analyzer's evaluator took its
   100 cube calls from 2.213 to 0.00675 seconds. Performing the relatively
   small checker equity conversions on the calling thread took 100 checker
   calls from 0.774 to 0.0224 seconds. All 300 mixed outputs matched,
   including the post-move control. [Paired results](static-1p/outputs.zip)
2. **Evaluate all checker endpoints in a single native call.** Share the
   model and database, release the GIL once and use the native worker pool.
   For 100 checker positions this took 2P from 1.071 to 0.517 seconds and 3P
   from 7.844 to 7.564 seconds. All outputs matched at 2P, 3P and in the
   20-position 4P screen. Claiming the next endpoint as a worker finishes
   saved a further 2.4%, 3.2% and 2.5% in a separate comparison. A forced
   endpoint can use the roll workers itself. [Native batch](checker-screen/summary.json)
   · [Scheduling](scheduling/summary.json)
3. **Reuse board information when choosing a specialist.** Compute race
   status, the opposing board and pip counts once, then evaluate the
   specialist gates in priority order. [Evaluator experiments](evaluator/)
4. **Extract both sides' contact masks together.** One traversal supplies
   the points needed by both sides' feature calculations. Routing and masks
   together reproduced the control's complete outputs across two full 2P and
   3P passes and 60 stratified 4P calls. [Repeated evaluator screen](evaluator/additive/summary.json)
5. **Generate sibling network deltas from changed board points.** This
   prototype carries sparse changes through encoding and network evaluation.
   On top of the routing and mask changes it saved roughly 2.4% at 3P and
   4P under MSVC. On the integrated GCC build, 180 3P cases took 2.991 vs
   2.936 seconds and 60 4P cases took 20.310 vs 20.274 seconds. All 240
   Linux outputs matched the equivalent build with dense input comparisons.
   Retained. [Windows ablation](evaluator/additive/summary.json)
   · [Linux ablation](linux-sparse/summary.json)
6. **Reduce legal move generation and deduplication costs.** Compare caching
   the six first-die move lists, changing the board in place with an incremental
   hash, and combining the two. The in-place walk preserved all 321,292
   ordered native endpoints and improved both complete 2P and 3P runs.
   First-die caching looked faster on a narrow panel but lost its advantage
   on the complete corpus. [Native checks](moves/native_check.json)
   · [Full-corpus experiments](moves/full_corpus/)
7. **Share work between the live cube policy and the cubeless headline
   policy.** Compare sharing raw 4P candidate batches with retaining the
   actual choice of the synthetic dead-cube policy. Ten 4P cube calls
   presented 2,825,127 candidate vectors, with 225,500 reusable across the
   two traversals. The raw batch cache took 2.771 seconds for 20 4P cube
   calls with dynamic dispatch, down from 3.009 seconds. The more compact
   winner-only cache offered no incremental time saving and was discarded.
   [Tree experiments and profile](tree/README.md)
8. **Balance the 21 dice rolls by cost.** Doubles contributed 56.6% of
   generated 4P endpoints on the native profile. Start expensive rolls early,
   let workers claim more work and reduce results in the original dice order.
   [Tree scheduling comparison](tree/README.md)
9. **Simplify the interior scoring and terminal tests.** Independently try a
   dedicated dead-unlimited tree, a parent proof that a position cannot yet
   be terminal and sparse arithmetic for PubEval. The terminal proof avoided
   33.94 million scans across the 30 profiled 4P calls, yet worsened 4P
   checker time. The dedicated dead tree had the same drawback. Sparse
   PubEval lost time at every screened depth. All three were dropped.
   [Native tree counters](tree/README.md) · [PubEval study](moves/comparison.json)
10. **Prepare score-dependent match equity payoffs.** A thread-local table
    reused the six possible match outcomes across cube conversions and cash
    point recursion. On full 2P and 3P runs, 300 calls took 0.7844 vs 0.7850
    seconds and 9.478 vs 9.496 seconds. Dropped.
    [Patch](match-screen/payoffs.patch) · [Comparison](match-screen/summary.json)

## Accuracy

The reference reuses the [frozen full-game rollout analytics](../../rollout_positions/README.md).
For every probability or equity with a rolled value, compare the change in
absolute error. A rise greater than 0.0025 in probability or 0.005 in equity
is a material regression. There are 378 rolled checker alternatives and
1,709 legal checker endpoints. All 1,709 enter the public-output and ranking
comparisons. Only the rolled alternatives enter the checker error calculation.
Cube and post-move each have 100 reference responses. Historical database
rollouts can comprise several adaptive 1,296-game batches.

The full-game throughput guard replayed an existing match attacking position
and an unlimited pure race twice through each API. It used 36 games, 3P
checker and cube decisions, variance reduction and four parallel trial
workers. All 12 paired responses matched, including the standard errors.
The four checker calls fell from 11.33 to 10.02 seconds, cube calls from
3.22 to 2.78 seconds and post-move calls from 3.23 to 2.80 seconds.
These short replays measure embedded search throughput. The accuracy
comparison uses the independent, frozen rollout corpus. [Full-game replays](fullgame_guard.zip)

## Repeat the qualification

The engine's [benchmark runner](../../../../scripts/benchmark_multi_ply.py)
loads an explicitly selected Python package and native extension. Its output
directory must be new. Use a Release build, the registered Stage 11 weights
and the benchmark's one-sided bearoff database. From the engine checkout,
the study wrapper selects the source export, package and binary corresponding
to each named version. The [control-input verifier](verify_inputs.py) lists
the required archives and build locations. The [Linux build procedure](evaluator/README.md)
records the GCC configuration and the deployment extension's source hashes.

```powershell
$study = 'data/multi_ply_benchmark/experiments/search_pass2'
python "$study/run.py" original 1P 2P 3P 4P
python "$study/run.py" checkpoint 1P 2P 3P 4P 1T
python "$study/run.py" candidate 1P 2P 3P 4P 1T 2T 3T
python "$study/qualify.py" windows --runs logs/search_pass2/windows --levels 1P 2P 3P 4P 1T 2T 3T
```

Run the deterministic commands with the Linux interpreter and pass
`linux --runs logs/search_pass2/linux --levels 1P 2P 3P 4P` to the scorer.
All timing runs use one process and all 16 logical CPUs. Separate runs for
each level start with a fresh analyzer. The qualification regenerates the
scorecards and archives only after checking the inputs, complete position
sets and thread settings. Windows truncated historical data come from the
preceding published experiments. Passing a fresh full run takes precedence.
