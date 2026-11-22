# Full-game rollout timing at 1,296 paths

The optimized engine completed 15 public full-rollout calls in **254.12 seconds**, down from **645.47 seconds** on the pre-optimization Linux control. The same positions and path budget took **60.6% less wall time**, a **2.54× speedup**. Every call got faster. [All 15 paired timings](summary.json) · [Inputs, settings, complete analytics and rollout-reference scores](full_outputs.zip)

| Public call | Matched calls | Before (s) | Optimized (s) | Speedup | Time saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Checker play | 5 | 369.36 | 154.38 | 2.39× | 58.2% |
| Cube action | 5 | 141.62 | 49.99 | 2.83× | 64.7% |
| Post-move analytics | 5 | 134.49 | 49.74 | 2.70× | 63.0% |
| **Total** | **15** | **645.47** | **254.12** | **2.54×** | **60.6%** |

Speedup divides total original wall time by total optimized wall time. Across individual calls the median factor was 2.67× and the range was 1.85× to 2.97×. Across the three requests for each plan the factors were 2.57× for attacking, 2.71× for priming, 2.47× for anchoring, 2.58× for racing with contact and 1.95× for pure race. The longest call, a six-alternative match-anchoring checker decision, went from 193.53 s to 84.01 s. These are sample measurements from ten distinct positions.

The two builds rolled the same 14 nonterminal checker alternatives plus the five cube and five post-move positions. That is **24 batches, or 31,104 paths per engine**. All five checker rankings and all five cube recommendations agreed. Among 213 probability and equity observations with rollout references, zero moved materially closer or farther. Across the complete public results the largest direct changes were 0.00000793 in probability and 0.0000209 in equity. The material thresholds were 0.0025 and 0.005 respectively. Fourteen calls have same-seed frozen references. The match-anchoring checker reference comes from the original adaptive rollout archive, which recorded eight rolled alternatives and did not record its exact batch count or dice seed. [Rollout references and provenance](../../rollout_positions/README.md)

## Cross-check against the unmodified original

Three cases ran a second time, reversing which build ran first and using the original binary with its historical cache behavior. They took **166.05 s vs 68.11 s**, a **2.44× speedup**. Those same three calls took 174.61 s vs 69.69 s in the main run (2.51×). For each case both historical builds returned identical complete analytics, and the repeated optimized call reproduced its first result exactly. [Three paired timings](sensitivity/summary.json) · [Full outputs and build identities](sensitivity/full_outputs.zip)

## Workload

The sample contains five public checker decisions and five pre-roll cube positions from the [timing corpus](../../README.md). Each cube position also supplies the post-move request from the previous player's perspective. Each API covers attacking, priming, anchoring, racing with contact, and pure race. The sample takes the first seeded corpus position in each selected stratum. For each plan the checker position uses one play format and the cube position uses the other, so the ten distinct positions cover all five plans in both match and unlimited play. The runner selects positions before either build reports a timing.

Each engine uses the same Stage 11 weights, bearoff database, position inputs and dice seed. Each full rollout uses a single batch of 1,296 stratified paths with variance reduction and a zero truncation depth. Early and late checker and cube evaluators are set to 3P, with no scheduled late-game reduction. Both builds use the engine's standard 1P opening reply and static cube screen. The public checker filter and promotions determine which legal alternatives receive their own 1,296-path batch. For a cube decision, no-double and double/take share the same paths. [Rollout algorithm](../../../../ROLLOUT.md)

The historical build comes from `139427d` with the ten-line opening-cache correction used by the other Linux timing studies. It prevents a trial from reading a partially written entry. The optimized engine uses `5de8023`. A separate three-call comparison runs the unmodified `139427d` binary and reverses the order of the same pairs. It covers the match-attacking checker decision, unlimited anchoring cube decision and match racing-with-contact post-move position. The harness checks both compiled extensions, Python analyzers, model files and database against the previously qualified binaries and assets. [Historical control and source patch](../truncated_pass2/historical-linux/README.md) · [Engine identities](../truncated_pass2/app-style/settings.json)

Runs use all 16 logical CPUs on one Ryzen 7 9700X under x86-64 Linux (WSL2). Each build has a long-lived analyzer in its own process. One public call runs at a time, using all 16 trial workers. The runner switches which build goes first between successive pairs. It times from the public call to its return, including checker filtering, promotions and native simulation. Analyzer construction, writing the response and application queue or network traffic are outside the interval. [Timing runner](measure.py)

## Reproduction

With the two qualified Linux builds and model assets available at the locations in the build manifests, run from the engine checkout using a new output directory.

```sh
study=data/multi_ply_benchmark/experiments/full_rollout_1296
python3 -B "$study/measure.py" --output logs/full-rollout-1296-local
python3 -B "$study/measure.py" --sensitivity --output logs/full-rollout-1296-original-local
```

To verify the saved timings, complete analytics, and accuracy scores without running the engine again, use the following commands. To package fresh logs, add `--run` with the corresponding output directory to each command.

```sh
python3 -B "$study/verify.py"
python3 -B "$study/verify.py" --sensitivity
```
