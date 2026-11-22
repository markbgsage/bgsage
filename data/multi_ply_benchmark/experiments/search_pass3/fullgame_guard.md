# Full-game rollout timing check

The final Windows release completed 12 paired full-game calls with 16 trial
workers in **8.704 s versus 10.420 s**, saving 16.5% of wall time. These calls
use 36 games per rolled alternative on two existing benchmark cases and serve
as a throughput check. The accuracy reference is the separate 1,296-game
corpus. All checker orders and cube choices matched. The largest direct
probability shift was 0.000000060 and the largest equity shift was
0.000000775. [Final inputs, analytics, timings and binary hashes](release/fullgame_guard_16.zip)

## Earlier integrated builds

The Windows build with the 4P scheduler completed the same 12 paired calls
with 16 workers in **8.424 s versus 10.275 s** for the checkpoint, an 18.0%
wall-time reduction. All checker orders and cube decisions matched. Across
the full responses the largest direct probability shift was 0.000000060
and the largest equity shift was 0.000000775. [Complete 16-worker release outputs](fullgame_guard_16.zip)

The integrated Windows candidate completed 12 paired public calls in **13.420 s**, down from **15.439 s** at checkpoint `82e6397` (13.1% less wall time). Each individual pair improved, by 4.6% to 23.8%. [Full inputs, analytics, timings, configuration and hashes](fullgame_guard.zip)

| Public call | Pairs | Checkpoint seconds | Candidate seconds | Wall time saved |
|---|---:|---:|---:|---:|
| Checker play | 4 | 9.871 | 8.444 | 14.5% |
| Cube action | 4 | 2.809 | 2.491 | 11.3% |
| Post-move analytics | 4 | 2.760 | 2.484 | 10.0% |

The sample includes four benchmark positions. Checker play used `checker-match-attacking-01` and `checker-money-purerace-01`. Both cube action and post-move analytics used `cube-match-attacking-01` and `cube-money-purerace-01` with their respective archived inputs. Each build evaluated all six calls twice using the same per-position seeds. The second repeat reversed build order. Each arm ran in a fresh process with four workers on logical CPUs 12–15. The timer covered each public call. [Benchmark runner](../../../../scripts/compare_rollout_performance.py)

Each rollout used 36 full games, Stage 11, variance reduction, the one-sided bearoff database, and 3P checker and cube settings for both early and late play. The public checker shortlist and promotion rules applied. All 14 rollout-labeled checker entries across both repeats matched exactly, including standard errors. All eight cube and post-move responses also matched exactly. Checker move ordering and decisions agreed throughout. Across all reported probabilities and equities, including lower-ply checker alternatives, the direct change (candidate minus checkpoint) ranged from −1.49×10⁻⁸ to +5.96×10⁻⁸ for probabilities and from −1.19×10⁻⁷ to +7.75×10⁻⁷ for equities. The study measures accuracy against the separate frozen rollout references. [Rollout references](../../rollout_positions/README.md)

This guard used the integrated candidate before the optional 4P scheduler experiment. The native extension SHA-256 was `cacde7060a4336528bb76bbde51f99caa869ee6cdfb3a03d284488b26b9e282a` for the candidate and `fd4630f514159877922dfc425de3503694385a82d55d523c6f9964e2bdc3d728` for the checkpoint. All other snapshot manifest entries matched.
