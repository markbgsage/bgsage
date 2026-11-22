# Engine project notes

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
