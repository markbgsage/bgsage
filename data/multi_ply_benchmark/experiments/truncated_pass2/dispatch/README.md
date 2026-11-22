# Trial scheduling and cross-worker tree reuse

Giving a worker one truncated trial at a time reduced wall time at all three presets. With 72 trials and 16 workers, an assignment of eight games yields only nine assignments. At 360 games it yields 45. Individual trials take different amounts of time because of cube passes, checker choices, and the cost of evaluating their final boards. Smaller assignments let the available workers divide the remaining work more evenly.

Each cell gives wall seconds for 30 public calls on the same 16-thread Linux CPU. A cell covers 10 checker decisions, 10 cube decisions, and 10 post-move evaluations, with every game plan and both play formats in each API. A warm-up call precedes measurement in each process. The runner shuffles the case order, rotates the implementation measured first, and uses the corpus seed for each call.

| Sample | Eight trials per assignment | Four | Two | One | Wall time saved with one |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1T | 1.008 | 0.906 | 0.839 | **0.820** | 18.7% |
| 1T, second set of positions | 1.061 | 0.943 | 0.849 | **0.805** | 24.1% |
| 2T | 4.436 | 4.288 | 4.215 | **4.213** | 5.0% |
| 3T | 17.665 | 17.496 | 17.426 | **16.930** | 4.2% |

The initial 1T panel overlapped the end of a separate single-CPU probe. The second 1T panel used another 30 positions with the machine idle. Across both panels, **all 390 alternate results matched their respective eight-trial controls exactly**, including standard errors and rankings. The release selects one trial per claim for a positive truncation depth and fewer than 32 trials per worker. Full-game and larger workloads use eight. The worker checks cancellation between claims. Dice and reduction order remain indexed by the original trial. [All 510 timed calls and inputs](timed-outputs.zip) · [Results by API](summary.json)

The separate [nested-rollout case](nested/README.md) exercises a custom checker or cube evaluator that starts child rollouts on the same worker. It records complete serial and parallel results and the legal-board lifetime required for agreement.

## Shared two-ply tree experiment

The separate 3T experiment allowed a worker that missed its own two-ply cache to read a completed search from another worker on the same native rollout. The immutable direct-mapped entries compare the board, search key, evaluation and cube fingerprint, number of branches, and top-node flag. They store the five probabilities and up to eight branch equities. The table used 3,031,064 bytes per native endpoint rollout.

An eight-worker count on ten cube positions found **12,978 hits in 118,153 probes**, or 11.0%. With counters disabled, the 16-worker timing panel at eight trials per assignment took 17.665 s without the table and 17.526 s with it. Checker time increased from 10.212 s to 10.271 s while cube and post-move time decreased from 3.608 s to 3.565 s and from 3.845 s to 3.689 s respectively. The 0.8% aggregate difference and uneven API results did not justify keeping the table. [Per-position hit counts](counts.json) · [Full count-panel analytics](count-outputs.zip)

## Reproduction

The timing and count arms used the same GCC CPU release binary. The [experiment source archive](experiment-source.zip) contains all changed C++ and Python sources relative to engine commit `1ecca3902fc7d04e35c43b9fcd3ec45920d0fcb5`, the measured scheduler and cache switches, and the runners. Build a CPU-only engine with that source and the checkpoint weights and bearoff database. Place the extension and its Python source in `build/truncated_pass2/dispatch_tree/` under the engine checkout. Unpack the two runner files together. From the engine root, use the Python version matching the extension.

```sh
taskset -c 0-15 python /path/to/tools/sweep.py --output logs/dispatch-first
taskset -c 0-15 python /path/to/tools/sweep.py --levels truncated1 --offset 1 --output logs/dispatch-repeat
taskset -c 8-15 python /path/to/tools/counts.py logs/dispatch-counts
```

The runner sets `BGSAGE_BENCH_TRIAL_CHUNK` and `BGSAGE_BENCH_TRIAL_TREE` separately for each process. A label such as `8-1` selects eight trials and enables the experiment table. These switches belong to the research binary. The [timing metadata and binary hash](summary.json) identify the build.
