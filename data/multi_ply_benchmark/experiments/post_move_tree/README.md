# Post-move cubeful equity from the 3-ply tree

The public method computes a 3-ply cubeful tree to select moves and estimate
five probabilities. It then applies Janowski cube conversion to these averaged
probabilities at the previous mover's root board. The **direct** variant
returns the equity already computed inside that same tree. The **bounded**
variant moves the public estimate toward the direct estimate by up to 0.0049
equity units. That limit was set from the requested 0.005 tolerance before
scoring the bounded variant.

## Results

The full-game reference used 1,296 games per position, variance reduction
and 3-ply checker and cube decisions. All 100 public results matched the
previously archived two-thread API outputs field for field. Both variants
preserved all five reported probabilities and cubeless equity exactly. The
post-move API reports no action. Its rollout equity equals the negative
of the paired rollout's optimal cube equity on all 100 boards. The existing
cube-action method agreed with 95 of those 100 rollout actions. Both
experiments change only the private copy of the post-move method.

| Cubeful measure | Public | Direct tree | Bounded |
| --- | ---: | ---: | ---: |
| Mean absolute error | 0.077995 | 0.045403 | 0.074477 |
| Median absolute error | 0.070344 | 0.035031 | 0.065444 |
| 95th-percentile absolute error | 0.177473 | 0.127334 | 0.172573 |
| Maximum absolute error | 0.281119 | 0.179728 | 0.276219 |
| Positions closer / farther / tied vs public | — | 83 / 15 / 2 | 85 / 13 / 2 |
| Materially closer / farther vs public | — | 72 / 6 | 0 / 0 |
| Largest increase in absolute error | — | 0.037035 | 0.004900 |

The 0.005 threshold applies to the change in absolute error for each equity,
and 0.0025 applies analogously to each probability. By the triangle
inequality, the bounded rule also limits any worsening against an unknown
reference to 0.0049 plus floating-point roundoff. The paired median change
in absolute error was −0.015573 for the direct estimate and −0.004900 for
the bounded estimate. These are finite rollout estimates. The calculations
do not infer a population error or assign statistical significance to
individual differences.

| Opponent's rollout decision | Positions | Public cubeful MAE | Direct cubeful MAE | Materially closer / farther |
| --- | ---: | ---: | ---: | ---: |
| No Double | 82 | 0.067304 | 0.048882 | 55 / 6 |
| Double/Take | 13 | 0.130420 | 0.040784 | 13 / 0 |
| Double/Pass | 5 | 0.117029 | 0.000372 | 4 / 0 |

The direct estimate handles the opponent's immediate doubling decision and
weights the options created by individual rolls. The Janowski conversion is
nonlinear, so applying it once after averaging the probabilities can give a
different answer from converting at the leaves and propagating branch
values. For example, the opponent can cash the cube on
`cube-money-attacking-03`. The rollout value was −1.000000, the public
value was −0.718881, and the direct value was −0.998142. Finite search still
depends on model and cube-efficiency estimates at its leaves. On
`cube-match-purerace-10`, the largest regression, the rollout was
−0.805087 (SE 0.002912), the public estimate was −0.796419, and the tree
gave −0.850789. This accounts for 0.037035 additional absolute error.

Four fresh-process AB/BA pairs were run for each variant, on Windows CPUs
14 and 15 with two analyzer threads, concurrent with rollout reference
generation. Totals sum public-call wall times in corpus order and exclude
constructor, serialization and file I/O.

| Trial | Mean public seconds / 100 | Mean variant seconds / 100 | Paired ratios, public / variant |
| --- | ---: | ---: | --- |
| Direct | 3.174 | 3.223 | 0.976, 1.004, 1.062, 0.893 |
| Bounded | 3.048 | 2.954 | 1.195, 1.024, 0.797, 1.153 |

The timing ranges overlap unity. Both variants perform the same native search
as the public API. The direct method skips the final conversion and the
bounded method adds a scalar clamp.

## Historical rollout cross-check

An independent SHA-256 ordering selected up to ten unique historical
rollout-tier cube boards per mode and game plan. Both board orientations and
all game/match seeds used by either primary set were excluded. The result
contains 84 positions from 67 additional games/matches, with 41 money and 43
five-point match positions. The remaining anchoring pool provided one money
and three match positions. Every other cell contributed ten. The previously
declared algorithms and materiality thresholds were used without adjustment.

| Cubeful measure | Public | Direct tree | Bounded |
| --- | ---: | ---: | ---: |
| Mean absolute error | 0.080040 | 0.027757 | 0.075782 |
| Median absolute error | 0.062947 | 0.012569 | 0.058047 |
| Positions closer / farther / tied vs public | — | 77 / 4 / 3 | 78 / 3 / 3 |
| Materially closer / farther vs public | — | 69 / 1 | 0 / 0 |
| Largest increase in absolute error | — | 0.018162 | 0.003017 |

The direct estimate reduced mean absolute error by 65.3% here. It materially
improved 38 money and 31 match positions, with one material regression in
match play. The historical escalation rule prioritized close cube decisions
(best-available gap below 0.02). This sample has 53 doubles in 84 positions,
compared with 18 in the primary 100. Different decision mixes and reference
configurations affect the size of the reduction.

The sources were introduced in engine commits `1796482` (money) and
`4e8cd04` (match). Their documented rollout configuration used 3-ply checker
and cube decisions through completion, variance reduction and 1,296-game
batches. The ND equity standard error target was 0.00255, with a cap of 16
batches. The default production model in both source revisions was Stage 9.
The archived per-position rows record the rollout tier, action and branch
equities. They do not identify the exact binary, weights, actual batch count
or achieved SE. The primary reference instead uses a fixed, fingerprinted
Stage 11 configuration and one 1,296-game batch per position. This cross-check
uses the negative of the archived branch chosen by the recorded double and
take decisions, with board, ownership and away scores flipped to the previous
mover. The script checks the branch against the archived action label.

## Reproduction

Run from the engine checkout with the reviewed extension, Stage 11 weights,
bearoff database and compatible Python with `psutil` installed.

```sh
python data/multi_ply_benchmark/experiments/post_move_tree/experiment.py --extension build/multi_ply_reviewed/bgbot_cpp.cp314-win_amd64.pyd --output logs/post_move_tree_direct_repeat --cpus 14 15 --pairs 4 --candidate tree
python data/multi_ply_benchmark/experiments/post_move_tree/experiment.py --extension build/multi_ply_reviewed/bgbot_cpp.cp314-win_amd64.pyd --output logs/post_move_tree_bounded_repeat --cpus 14 15 --pairs 4 --candidate bounded
python data/multi_ply_benchmark/experiments/post_move_tree/heldout.py --extension build/multi_ply_reviewed/bgbot_cpp.cp314-win_amd64.pyd --output logs/post_move_tree_heldout_repeat --cpus 14 15
```

The script replaces three portions of a private copy of the public Python
method. Both native wrappers use `cubeful_equity_nply_multi` with identical
inputs, default filters, bearoff DB, root routing, cube value, flipped cube
ownership and swapped away scores. The native equity is in the opponent's
basis-cube units and is negated on return to the previous mover. The C++
wrapper already converts match-winning chance into normalized equity. The
summaries include every scalar, the reference and engine hashes, and
nearest-rank distributions. The gzip files contain one row per position
with full analytics, rollout estimates and per-pass call times.
