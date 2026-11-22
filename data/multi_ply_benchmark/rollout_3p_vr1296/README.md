# Supplemental fixed-strength rollout checkpoints

The [position benchmark](../rollout_positions/README.md) reuses the archived
rollout decisions and runs the normal public checker selector for the
remaining positions. This directory preserves an earlier run that also
evaluated lower-ranked legal moves. Its exhaustive queue was stopped when
the position-level scope was clarified. Its completed cube, post-move and
checker checkpoints are reusable. The instructions below document how
those checkpoints were produced.

The reference run uses Stage 11 from the saved timing baseline. Each simulated
position uses 1,296 stratified trials, variance reduction, cubeful trial
play, and full-game rollout mode. The early and late checker and cube
evaluators are all configured at 3-ply. The standard reply cache selects
the second checker play at 1-ply. The in-trial cube screen, terminal and
bearoff rules, and full-game safety limit also apply.
One fixed batch is used with no standard-error stopping. Each source key
determines its own 32-bit seed. All moves for a checker decision use the
same seed for lower-variance move comparisons.

There are 100 cube actions, 100 post-move evaluations of the corresponding
positions, and 1,709 legal candidate moves across 100 checker decisions.
Every candidate gets its own rollout, including moves that a normal
checker-play call would prune. The runner invokes the same cubeful rollout
used for each surviving candidate by the public checker API. It applies that
API's bearoff probability override and ranks the complete move list by
cubeful equity with the engine's usual tiebreakers. Terminal endpoints use the
engine's exact settlement. Post-move and cube actions use their respective
public APIs and retain their distinct cube-rule inputs.

From the engine repository root, using a Python environment compatible with
the baseline's compiled extension, run:

```sh
python scripts/multi_ply_rollout_truth.py prepare
python scripts/multi_ply_rollout_truth.py run --shards 2 --shard 0 --threads 4
python scripts/multi_ply_rollout_truth.py run --shards 2 --shard 1 --threads 4
python scripts/multi_ply_rollout_truth.py assemble
```

Start the two `run` commands in separate terminals. Each worker runs four
simultaneous trials. Resume a stopped worker using its original shard index.
Keep each shard assigned to at most one worker. `prepare` copies the native
extension, model weights, bearoff database, inputs and Python modules into
an isolated folder inside `logs/multi_ply_rollout_truth/frozen/`. The
manifest pins input and engine hashes from the timing baseline. A candidate
build can run alongside the frozen engine without changing the truth policy.
On a fresh or edited checkout, missing or changed Python sources are recovered
from the local baseline Git commit. LF and CRLF are tried to reproduce the
original bytes and only the recorded SHA-256 is accepted. The original native
binary is required by hash. If stored elsewhere under the engine checkout,
pass its path to `prepare --extension <path>`. A newly compiled extension
requires a separate reference generation identity.

For an unattended run, the optional `supervise` command uses `psutil` to
launch the same two workers and update assembled results as they finish.
Use this instead of launching the workers separately:

```sh
python scripts/multi_ply_rollout_truth.py supervise --shards 2 --threads 4 --cpu-mask 0xff
```

With workers already running, add `--adopt <shard-0-pid> <shard-1-pid>`
using their Python PIDs from the progress files. The supervisor validates
their command lines before adoption and can restart an exited shard up to
three times. Its lock is released automatically when it exits. The optional
CPU bit mask limits new workers to specific logical CPUs for concurrent
timing studies. Create `logs/multi_ply_rollout_truth/STOP` to stop new work
and let workers exit after their current units; remove it to resume.

To expand a live run from two four-thread workers to four after an isolated
timing study, use `repartition --shards 4 --threads 4 --adopt <pid0> <pid1>`.
The coordinator first drains the listed workers' active units, waits for the
old supervisor's lease, then supervises a new disjoint four-way split. Omit
`--cpu-mask` to use the machine's available CPUs. The same completed unit files
are used under either partition.

For a long run, `supervise --pooled --shards 4 --threads 4` lets every worker
claim the next unfinished unit. Use the same `--pooled` option with
`repartition` to switch a live static run after its outstanding units finish.
Every claim holds a kernel-owned byte-range lock on Windows or a file lock on
POSIX. The worker rechecks the canonical checkpoint after acquiring the lock
and publishes the completed result before releasing it. If a worker exits
mid-unit, the operating system releases its lock and another worker can resume
the unit. The lock files stay under the run's `leases/` directory. Run
`status --shards 4 --pooled` to see the pooled time estimate and live workers.

Each finished rollout is written atomically under
`logs/multi_ply_rollout_truth/units/`. Each worker records progress for
its current 1,296-trial unit in `progress-<shard>.json`. The first pass
covers the incumbent best checker move and cube/post-move positions across
the source game plans. Later passes cover the remaining candidates in
baseline ranking order. The first-pass ordering affects when a reference
becomes available. It has no effect on which moves qualify.

Run `python scripts/multi_ply_rollout_truth.py status --shards 4` for the
current four-worker split. It reports each worker's last trial count and
heartbeat age, completed and remaining units, and projected remaining hours
per shard. The longest shard determines projected wall time. Each remaining
unit uses finished endpoints from its own source position first. For a new
source, each observed position contributes one mean duration to its API, mode
and game plan, regardless of how many checker endpoints have already finished.
The report also shows the projection using median times for comparison.
Historical call times used for forecasting exclude complete, recorded
benchmark suspension windows spanned by the call. Saved checkpoint timings
retain elapsed wall time.
For a stratum with no finished measurements, the estimate pools modes, then
game plans, then APIs until it finds observations. Sparse contact samples and
changes in available CPU capacity can materially alter the projection.

`assemble` can run while workers are active. It exports `results.jsonl`
with one full public-shaped analytics object for each completed decision and
updates `coverage.json`. For checker moves the public `std_error` is the
standard error of cubeless equity, and `prob_std_errors` contains the five
probability standard errors. A second export, `checker_cubeful_se.jsonl`,
records the native cubeful-equity standard error for every completed checker
endpoint, including endpoints of incomplete positions. Join by
`position_id` and the mover-perspective result `board`. A checker position
enters the complete analytics file only when all
legal moves have finished. Partial checker candidates can be identified by
their exact endpoint board and inspected in their individual unit files. For
these partials, position-wide ranking and `equity_diff` are meaningful only
after assembly of the complete move list. The run is complete at 1,909 units
and 300 decision records.

### Rollout worker parallelism

On a Ryzen 7 9700X, four dedicated logical CPUs evaluated the same four
previously uncomputed contact endpoints under three process/thread splits.
The other twelve logical CPUs continued the reference run. Each topology
returned identical full public and native outputs, including every standard
error. The four baseline checkpoints became part of the reference corpus.

| Processes × trial threads | Four-unit wall time | Sum of unit seconds / workers | Peak total resident memory |
| --- | ---: | ---: | ---: |
| 1 × 4 | 224.12 s | 224.06 s | 0.96 GiB |
| 2 × 2 | 224.12 s | 220.19 s | 1.25 GiB |
| 4 × 1 | 257.46 s | 208.07 s | 0.55 GiB |

Each layout started in 0.31–0.35 s before its timing barrier. The serial
layout's longest endpoint determined its four-unit wall time. See the
[paired timings and memory measurements](parallelism/4slots/summary.json)
and the full outputs for [1 × 4](parallelism/4slots/1x4.jsonl),
[2 × 2](parallelism/4slots/2x2.jsonl) and [4 × 1](parallelism/4slots/4x1.jsonl).

The full-machine comparison used all 16 logical CPUs and 40 additional
endpoints from 24 checker positions. It covered four endpoints per combination
of mode and game plan. Each layout pulled from the same queue, ordered by
descending estimated expense using the source position's measured incumbent.
Forty paired public and native outputs matched exactly, including all SEs.

| Processes × trial threads | Startup | Batch wall time | Middle half of work | Sum of unit seconds / workers | Peak RSS | Min available machine RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 × 4 | 0.34 s | 489.59 s | 245.42 s | 489.04 s | 4.39 GiB | 9.57 GiB |
| 16 × 1 | 0.55 s | 614.67 s | 239.51 s | 452.64 s | 2.22 GiB | 11.93 GiB |

The four-worker layout finished the batch 125.08 s sooner. Sixteen serial
workers completed the middle half about 2.4% sooner, then waited on the five
longest remaining endpoints. The reference run uses four four-thread workers.
The initial two endpoints of `checker-money-purerace-02` used two trial
threads. A four-thread rerun matched both raw native results, including SEs,
and the public checker call matched the full assembled decision exactly.
Middle-half times measure elapsed time between 25% and 75% of the same
control-weighted workload. Each unit's fraction of completed trials is
weighted by its complete control duration, with linear interpolation between
five-second samples. Actual batch wall time uses a common start barrier and
the final worker's monotonic completion timestamp.

The [full timings](parallelism/16slots/summary.json) include per-unit
durations, completion timestamps, fixed-prefix timings, peak resident and
private memory, and the frozen reference manifest hash. The
[middle-half calculation](parallelism/16slots/sustained.json) uses the saved
[trial counts](parallelism/16slots/trial-progress.jsonl). Full unit outputs
are available for [4 × 4](parallelism/16slots/4x4.jsonl) and
[16 × 1](parallelism/16slots/16x1.jsonl).

Reproduce on an idle 16-logical-CPU machine following `prepare` above.

```sh
python scripts/benchmark_rollout_parallelism.py paired --cpu-mask 0xffff --threads 4 1 --pull --min-available-gib 3 --output logs/rollout_layout_repeat --units-file data/multi_ply_benchmark/rollout_3p_vr1296/parallelism/16slots/units.json
```
