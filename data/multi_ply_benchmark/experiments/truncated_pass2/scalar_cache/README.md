# First-use allocation of the shared N-ply table

The engine's cross-thread position table holds 2,097,152 board entries. Waiting
for the first actual lookup, reservation or insertion saved **about 288 MiB per
worker process** across the sampled 1T, 2T and 3T calls. On those cases the process
completed three calls on the same analyzer without needing the table.

The table is shared by parallel rollout trials for cubeless N-ply evaluation
and for eligible all-dead cubeful subtrees. When a search first needs the
table, one thread initializes its storage. Concurrent first callers wait for
publication, then use the same slots and probe sequence. A quiescent clear
before first use leaves the table empty. A clear after first use retains its
storage and resets its entries and counters.

## Measurements

These are means of two fresh Windows processes per arm and case. Each process
constructed one Stage 11 analyzer, ran one first call and two warm calls using
the same seed. Wall timers exclude output serialization. Peak RSS covers all
three calls. The balanced order alternated which arm ran first. All 60 complete
responses matched the published checkpoint, including errors, move order
and decisions. [Per-process timings and RSS](on_demand/measurements.json) ·
[All inputs and full responses](on_demand/calls.jsonl.gz)

| Call | Constructor ms, before → after | First call ms | Warm call ms | RSS after construction, MiB | Peak RSS across three calls, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1T checker | 81.4 → 45.3 | 120.1 → 120.6 | 115.4 → 114.3 | 372.5 → 84.4 | 537.3 → 249.5 |
| 1T cube | 79.8 → 44.3 | 49.6 → 50.4 | 44.3 → 44.9 | 372.2 → 84.2 | 534.3 → 246.3 |
| 1T post-move | 80.1 → 43.8 | 52.9 → 53.1 | 46.4 → 46.5 | 372.2 → 84.3 | 534.4 → 246.6 |
| 2T cube | 101.5 → 63.2 | 374.5 → 372.1 | 370.0 → 362.2 | 390.2 → 102.1 | 555.4 → 267.1 |
| 3T cube | 100.3 → 65.7 | 1192.1 → 1141.1 | 1135.7 → 1133.7 | 390.0 → 102.2 | 555.3 → 267.4 |

All workers used four Windows logical CPUs, 8–11, on a Ryzen 7 9700X.
This experiment measures initialization separately because the full benchmark
starts its call timer after constructing the analyzer. Memory savings depend
on whether a board and cube state use the shared table. A process that
begins a cacheable deep search allocates the full table.

An alternative implementation gated allocation on the effective search
policies. It saved the same memory at 1T. It still allocated the table for
both deeper cases because their policies allowed N-ply subtrees. All 30
responses from this trial matched the checkpoint. [Alternative patch](allocation.patch)
· [Measurements](gated/measurements.json) · [Full responses](gated/calls.jsonl.gz)

## Reproduction

Build the Windows CPU extension at `1ecca3902fc7d04e35c43b9fcd3ec45920d0fcb5`,
and a second copy with [the first-use allocation patch](demand.patch). From
the engine checkout, run the sampler under the Python interpreter that built
both extensions. Use the checkpoint's Python package directory and models.

```powershell
$study = 'data/multi_ply_benchmark/experiments/truncated_pass2/scalar_cache'
python -B "$study/measure.py" --control $controlExtension --candidate $candidateExtension --python-dir python --output build/replay-cache-memory
```

The [native concurrency test](first_use.cpp) releases 16 threads together on
each of eight fresh caches. Threads perform disjoint reservations, reads and
inserts, verify the published values and join. The main thread clears the
populated table, verifies that the prior lookup now misses and stores a new
value in the same storage. All eight rounds passed on Linux x86-64 with GCC
15. The guard can run on any CPU build with C++17 and the standard threading
library.

```sh
c++ -std=c++17 -O2 -pthread -Icpp/include data/multi_ply_benchmark/experiments/truncated_pass2/scalar_cache/first_use.cpp -o build/cache-first-use
build/cache-first-use
```
