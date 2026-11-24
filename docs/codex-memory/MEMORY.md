# Engine project notes

- **2026-11-24, head-to-head experiment retention.** Keep generated game
  records, full reports and checkpoints under the Git-ignored `logs/`
  directory. Preserve run sources, configuration and aggregate estimates in
  Git. Verified the three previously tracked reports against their copies in
  `logs/` when removing the generated data from the tracked tree.

- **2026-11-24, Paskogammon Sage 1T vs XG Roller.** The head-to-head variant
  always starts the scattered side on roll. Sage 1T's pre-roll no-double
  values were −0.03618 for that orientation and +0.41667 for the other on
  build `309a5011`. Assign the engines to the scattered seat by game index,
  and use all 36 dice outcomes and the checker-time cube for first-roll luck.
  With seed `2026112344`, all 5,000 games verified as 2,500 starts per bot
  and 876 opening doubles. Within-opener SEs gave Sage a raw edge of −0.12080
  points per game (SE 0.04440) and adjusted edge of −0.01714 (SE 0.00723,
  95% CI [−0.03132, −0.00297]). The run took 3h 13m 10s with 256 game streams
  and 20 Sage compute workers. Three failed XG batches were replayed before
  their answers could affect play. Verified the original XG settings after
  exit. The benchmark archive contains the run driver and independent
  verification script. [Roller benchmark results](../../H2H_LUCK_ADJUST.md#sage-1t-vs-xg-roller)

- **2026-11-23, head-to-head luck adjustment.** Use the sampled entry of the
  same reference dice table that supplies the weighted mean. Substituting
  the played move only for the sampled entry can bias the edge. Score the
  standard opening across both potential openers and subsequent rolls at the
  checker-time cube after any take, always in one bot's points. The simulation
  transcript supplies those fields. `python -m bgsage.h2h` uses a complete
  2P ND table and reports SE and scoring CPU cost. On native build `309a5011`,
  512 Stage 11 1P-vs-2P games (run seed 20261123) lowered SE from 0.15053 to
  0.02426 points per game. Another 512 1P self-play games (seed 20261124)
  lowered it from 0.13914 to 0.01911. All 29 focused tests passed, and 24
  game records matched between one and six workers apart from timing fields.
  [Math and CLI](../../H2H_LUCK_ADJUST.md).

- **2026-11-23, parallel streams and repeated self-play.** The H2H CLI defaults
  to one game process per available logical CPU, caps by game count and the
  Windows pool limit, and keeps at most twice the worker count in flight. Sage now
  honors the one-thread limit for multi-ply search and rollout prefilters. With
  32 workers and 64 disjoint seed batches of 256 1P self-play games, the raw
  and adjusted batch-mean SDs were 0.17566 and 0.03170 points per game. The
  observed reduction was 82.0% (paired batch bootstrap 76.9%–86.0%). Check
  for duplicate derived seeds between batches even when the run seeds differ.
  [Repeated-batch result](../../H2H_LUCK_ADJUST.md#repeated-sage-1p-self-play-batches)

- **2026-11-23, paired 1P-vs-3P mean-shift test.** On native build `309a5011`,
  8,192 distinct games after a separate 256-game pilot gave adjusted-minus-raw
  −0.04422 points per game, paired SE 0.03339 and 95% CI [−0.10965, +0.02122].
  The run had roughly 80% power at a 0.094-point shift. Compare paired
  differences even when adjusted and raw have very different SEs. Exhaustive
  enumeration across 593 visited ordinary roll states and the joint opening
  found centering residuals below $4\times10^{-16}$ points. Sage 3P workers
  used about 708 MiB of private memory each on this Windows run. Native
  `multipy_parallel_run` obtains the hardware-sized pool before the serial
  fast path, giving each observed worker 33 threads. The 32-worker startup
  hit a memory limit. The measured run used 16 workers, plus 10 on the final
  two disjoint waves after memory freed up. The evaluator build stayed fixed.
  [Paired test results](../../H2H_LUCK_ADJUST.md#sage-1p-vs-3p-paired-mean-shift-check)

- **2026-11-23, external GNUbg H2H.** GNU Backgammon 1.08.003 is separately
  installed at `C:\Program Files (x86)\gnubg\gnubg-cli.exe`, the Windows path
  used by the legacy `sageapi` adapter. The H2H policy in `python/bgsage/gnubg.py`
  sends temporary command files to the executable and uses GNUbg's own ranked
  checker and cube hints. Set GNUbg's turn before sending an asymmetric simple
  board and verify the raw board echo, including the sign of the opponent's
  bar. The harness labels `1P`–`4P` map to native `0`–`3` ply. At a reached
  cube cap, disabling the GNUbg cube resets its value and ownership, so disable
  Jacoby there to preserve gammons on the already-turned cube. GNUbg subprocess
  cost is represented by elapsed worker time. The focused suite passed 43
  tests, all four native depths were probed, and 32 complete games exercised
  both seats and a cube cap of 2. Four seeded replays matched across one and
  four workers. [GNUbg usage](../../H2H_LUCK_ADJUST.md#run-sage-vs-gnubg)

- **2026-11-23, Sage 1P vs GNUbg native 0-ply timing.** Use Sage for luck
  scoring for both seats regardless of which bots play. The same 100 game
  seeds (`2026112331`) produced identical payoffs and luck values in serial
  and in 20 concurrent five-game streams. Elapsed time was 706.69 s serial
  and 77.14 s parallel during the overlapping runs. Sage's adjusted edge was
  −0.05023 points per game with SE 0.04286, versus raw −0.37000 with SE 0.28768.
  Serial luck scoring took 3.14 s. Reports are under `logs/` and the replay
  contributes one 100-game sample to inference.

- **2026-11-23, 1,000-game Sage 1P vs GNUbg native 0-ply cohort.** The fresh
  seed (`2026112332`) gave Sage +0.02873 points per game after adjustment,
  SE 0.01409 and 95% CI [+0.00112, +0.05634]. Raw mean was +0.03600 with
  SE 0.09182. The run retained 481 completed games from 30 streams and
  redistributed the remaining 519 across two 40-worker pools because of the
  Windows per-pool limit. File checks verified all 1,000 seeds and no overlap
  with the 100-game pilot. Observed throughput rose from 1.41 to 1.62 games per
  second across the two phases. A checkpoint showed 80 workers and 79 GNUbg
  subprocesses active. The phase timing uses different games and includes
  startup. Details are under `logs/sage-1p-vs-gnubg-1p-1000-20261123/`.

- **2026-11-23, external XG H2H.** `python -m bgsage.h2h` uses Windows Batch
  Analyze to play one XG seat against Sage under the standard simulator. Select
  a custom profile by the requested checker and cube depth, keep the opening
  book enabled, and check a known full-depth checker decision, cube decision
  and book move in each batch. Eagerly launching the desktop app minimized
  swallowed the Batch Analyze command on this installation. The two-stream
  native 1P replay matched both payoffs and Sage 2P luck values and took 116.3 s
  across 44 batches including calibration. Native 3P, staged 4P and Roller+
  calibrations verified both decision depths and restoration of the prior
  profile and book settings. The XG executable and a local complete `.xg`
  template are runtime inputs. [XG setup and rules](../../H2H_LUCK_ADJUST.md#run-sage-vs-xg)

- **2026-11-23, 1,000-game Sage 3P vs XG 3-ply cohort.** Seed `2026112342`
  used standard unlimited play, Jacoby on, cube cap 64, XG opening book on,
  and Sage 2P luck for both seats. Sage's adjusted edge was −0.00105 points
  per game, SE 0.01060 and 95% CI [−0.02182, +0.01972]. Raw edge was −0.11000
  with SE 0.08415. The run took 26m 33s across 584 XG decision batches and
  four planned XG restarts. Use separate pools to keep 80 game streams in
  flight with 12 Sage compute threads, and commit each seed before counting
  it as complete. Rechecked all 1,000 seeds, the game records, reported
  statistics and restored XG settings.
  [Summary and replay](../../H2H_LUCK_ADJUST.md#sage-3p-vs-xg-3-ply)

- **2026-11-23, 1,000-game Sage 1T vs XG Roller cohort.** Seed `2026112343`
  used the same native build `309a5011`, cube cap 64, Jacoby and Sage 2P luck
  reference as the 3P cohort, with a disjoint seed list. Sage's adjusted edge
  was +0.00171 points per game, SE 0.01211 and 95% CI [−0.02203, +0.02545].
  Raw edge was +0.14600 with SE 0.09177. The 43m 24s run used 80 game streams
  and 16 Sage compute workers. Seed Sage's internal 1T rollout separately
  for each decision from the game's seed so worker scheduling does not
  alter that RNG stream. Native Roller uses code `1000` on checker and cube
  records. Snapshot the existing profile before borrowing an inactive slot
  and verify the profile and analysis settings are restored at exit. All
  1,000 records and statistics were verified. The source archive retains the
  [standard run driver](../../data/h2h_benchmark/sage_1t_vs_xg_roller_1000_20261123.zip).

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
