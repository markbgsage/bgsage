# Engine project notes

- **2026-11-22, full-game CPU scheduling.** Give workers individual paths
  at full depth. Keep callback spacing independent of the claim size so
  short fixed and target-SE jobs retain their reporting cadence. A 37-path
  parallel batch produced five notifications and a 1,296-path batch produced
  108. When optimizing move encoding, check every opponent point as well
  as the bar before reusing that side's features. Two sibling moves can hit
  different blots. The independent scalar and AVX2 encoder probes matched
  on 656,128 boards. On the local 16-thread Ryzen, all 15 full-budget public
  calls got faster (262.00 s to 220.92 s). A Linux build using only the AVX2
  path saved 6.2% on a separate three-call sample. Preserve CPU capability
  and sample size with speed claims. [Full-game study and checks](../../data/multi_ply_benchmark/experiments/full_rollout_cpu/README.md)

- **2026-11-22, fixed-budget full rollouts.** For full-game speed, time the
  whole public call with one batch of 1,296 paths per selected checker
  alternative or cube position. The shortlist and promotions determine how
  many batches a checker request runs. Compare the actual endpoint sets,
  including terminal shortcuts. The pre-optimization control with the opening-cache
  correction and `5de8023` used the same 24 batches across 15 calls. Direct paired Linux timings gave 2.54×
  overall and 2.39× for checker play. The standard opening reply and cube
  screen still apply at configured 3P strength. [Fixed-budget timings and verification](../../data/multi_ply_benchmark/experiments/full_rollout_1296/README.md)

- **2026-11-22, truncated-rollout CPU study.** At a single cube-aware VR
  node, different dice can reach the same endpoint. Share its static
  probabilities only within the same board, model and driving cube, checking
  full board equality. All 98,973 native winner indices matched the per-roll
  selector on 2,008 boards. Custom nested rollouts require each active
  generator to retain its own legal boards until it consumes the chosen index.
  A one-worker/two-worker replay caught a different cube decision with the
  same seed. The corrected full responses matched. When timing a caller that
  requests progress, measure inside its callback as well. On ten 2T checker
  positions, smaller work assignments raised the callback count from 1,352
  to 3,602. [Measurements and reproducer](../../data/multi_ply_benchmark/experiments/truncated_pass2/README.md)

- **2026-11-22, third multi-ply CPU study.** Within a standalone dice node,
  the pre-roll board is a stable neural base across rolls. A partial
  live-cube shortlist can serve the independent cubeless traversal only
  when both use compatible neural arithmetic and the original move indices
  are preserved. Parallel 4P checker alternatives can also share complete
  2P subtrees within the same decision. The integrated 16-thread scheduler
  cut the 100-position Linux 4P checker panel from 30.893 to 27.838 seconds
  with every full response equal. The new move generator reproduced both
  endpoint orders across 571,608 rolls. Check timings by game plan as well
  as by preset. On the same CPU, Linux 2P pure-race roots were fastest
  serially and Windows benefited from roll parallelism. [Study and reproduction](../../data/multi_ply_benchmark/experiments/search_pass3/README.md)

- **2026-11-22, deterministic CPU search.** Keep the analyzer's loaded
  evaluator for 1P cube calls and batch all legal checker endpoints in one
  native call at 2P–4P. All 2,100 checkpoint responses matched on each of
  Windows and Linux, covering 1P–4P and 1T–3T. The reverse-order Linux
  3P rerun confirmed the wall-time reduction.
  Grouped sibling evaluation must use the encoding of the selected network
  slot even when a prebuilt native network consumes only a prefix. A
  196-input net in a contact slot exercised this case, with a 0.0226 error
  before the fallback and at most 0.00000006 afterward. Measure complete
  public calls when screening move caches. A first-die cache won on the small
  panel and lost on the full corpus.
  [CPU study and reproducible checks](../../data/multi_ply_benchmark/experiments/search_pass2/README.md)

- **2026-11-21, truncated-rollout optimization.** At 2T the 21-roll
  variance-reduction mean took 78% of instrumented worker time. At 3T
  the largest costs were the truncation evaluation (32%) and the VR
  mean (29%). Share VR winner indices and probabilities only within the
  same rollout position, with the board, ordered selection cubes and
  cube efficiency matched. The four-worker replay matched all 60 public
  outputs. Both 16-worker 2T/3T comparisons preserved all checker
  rankings and cube decisions on Windows and Linux with zero material
  regressions against the frozen reference. The hybrid search table allocates on its first insert. In
  the four-worker memory check this avoided 136 MiB of resident memory.
  [Profile, public results and reproducible experiments](../../data/multi_ply_benchmark/experiments/truncated_search/README.md)

- **2026-11-21, 3-ply optimization.** Compare Release builds with the same
  compiler and flags, including the original Python package with the control
  binary. A rebuild of the original source shifted a few historical predictions
  by up to 0.000001724. Preserve every returned checker candidate when testing
  filters. A shortlist kept the best move on all 100 positions while changing
  hundreds of other reported equities materially. The [experiment report](../../data/multi_ply_benchmark/experiments/README.md)
  contains the measurements and reproduction commands. The [position references](../../data/multi_ply_benchmark/rollout_positions/README.md)
  reuse existing database rollout decisions. For missing decisions use the
  normal public checker rollout and its promotions. Only moves labeled
  `Rollout` in full-game decisions supply checker truth values.
- **2026-11-21, CPU deployment.** The target for these analytics is GPU-less
  AWS VMs. The portable bit-mask encoder reproduced 259,508,480 feature
  values, and ordered AVX2 sibling updates preserved the complete public
  results in paired runs. Keep architecture-specific claims tied to the
  measured hardware. The [encoder study](../../data/multi_ply_benchmark/experiments/cpu_feature_masks/README.md)
  and [neural study](../../data/multi_ply_benchmark/experiments/cpu_nn/README.md)
  record the independent comparisons on the x86 Ryzen machine. The final
  Git-verified original and candidate matched all 300 public analytics on
  Windows. Linux GCC had zero material regressions against those same 300
  position references. [Linux measurements](../../data/multi_ply_benchmark/experiments/linux_x86_64/README.md)
- **2026-11-21, parallel rollouts.** A prefill worker must own its first-roll
  and second-move cache slots before an early trial can request them. In
  `cpp/src/rollout.cpp` the single-batch cubeful path now reserves every
  scheduled slot before dispatch, and the owner publishes a complete value
  with a release store. The fixed-seed comparison exposed an inflated win
  estimate and standard error on a non-leading 2T checker move. Compare
  complete public results, including per-move probabilities and SEs, when
  verifying scheduling changes. Fixed serial and repeated 16-thread replays
  matched for all three flagged cases. Across 900 before/after pairs, 898
  complete responses matched, all actions matched and the three material
  changes improved on the frozen reference. [Truncated benchmark](../../data/multi_ply_benchmark/experiments/truncated_16t/README.md)
- The protected engine guide quotes older speed estimates and describes an
  ultra-late drop in both checker and cube strength, and on-demand claiming
  during single-batch cubeful prefill. Current cube decisions
  use the initial cube evaluator throughout a trial. The cached second
  checker move uses 1P at every preset. See [current search semantics](../../MULTI-PLY.md#9-multi-ply-search-inside-rollouts)
  and the [rollout configuration](../../ROLLOUT.md#17-configuration-reference).

The imported and original read-only engine memory indexes were unavailable
on this checkout. The engine guides, source and saved experiments supplied
the project context for these measurements.
