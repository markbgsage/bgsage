# Position-level rollout references

This corpus covers the 100 sampled checker decisions, 100 sampled cube
decisions and 100 corresponding post-move evaluations. Each checker decision
contains its ranked list with the actual evaluation level of each move.

| Cases | Reference |
| --- | --- |
| 37 checker | Existing Open Sage rollout-tier decisions from the original money and five-point match archives. The archived ranked lists contain 224 rollout, 278 two-ply and 287 one-ply move records. |
| 63 checker | The frozen Stage 11 public `checker_play` call, with normal move selection and promotions. Each requested rollout uses one batch of 1,296 stratified games, variance reduction, cubeful play and three-ply checker and cube evaluators in all phases. |
| 100 cube and 100 post-move | Complete, previously saved calls under the same frozen 1,296-game policy, including the correct perspective and rule context for each API. |

The 63 frozen public checker calls requested 154 genuine rollout endpoints.
149 matched completed checkpoints and five required new simulations. Across
all 100 checker decisions, 378 of the 1,709 legal moves have rollout values,
435 have two-ply values and 896 have one-ply values.

For the frozen public simulations, the configured early and late checker
and cube strengths are all 3-ply. The trial engine's standard reply cache
selects the second checker play at 1-ply. The normal 1-ply cube screen also
applies. The archived decisions record their rollout policy but omit the
individual simulated games. [Trial implementation and presets](../../../MULTI-PLY.md#9-multi-ply-search-inside-rollouts)

The original benchmarks used full-game, three-ply checker and cube rollouts
with variance reduction. Their target standard error was 0.00255, with up to
16 batches of 1,296 paths. The archived rows omit actual per-position batch
counts and omit some standard errors. These fields are left null. Row
provenance identifies the archive, its hash, the original scenario seed and
its decision index. Ten archival rollout-tier cube records are retained for
independent comparison alongside the 100 full frozen cube outputs.

For the 63 fresh checker decisions, the normal public selector screens at
one ply with a 0.15 equity window, rescores survivors at two ply and selects
up to five within 0.08. It ensures two rollout candidates when available
and promotes a shallow move that becomes the cubeful leader. Other moves
retain their public fallback level. Restrict comparisons against rollout
truth to move records labeled `Rollout`.

The runner reuses a native result only when the pinned engine, fixed
position, deterministic dice seed, endpoint, cube and match context all
match. The native binding uses the endpoint and cube context for this
calculation and ignores the second board argument. This permits reuse
through either the normal shortlist or a promotion. Every output of a fresh
case passes through the original public filtering, fallback, bearoff,
ranking and formatting code.

- [Complete public-shaped case outputs](results.jsonl) include `reference_origin` and `provenance`.
- [Per-position coverage](position_coverage.jsonl) and the [coverage summary](coverage.json) identify completed cases and actual per-move levels.
- [Native checker cubeful standard errors](checker_cubeful_se.jsonl) are keyed by position and endpoint board. The public move's `std_error` measures cubeless equity. Archived rows retain their available public SEs.
- [Archived cube supporting records](archived_cube_support.jsonl) preserve available source fields.
- The [manifest](manifest.json) pins engine, sources, policies and output hashes.

From the engine checkout, prepare the frozen assets as described in the
[supplemental reference README](../rollout_3p_vr1296/README.md), then run:

```sh
python scripts/multi_ply_position_rollouts.py dry
python scripts/multi_ply_position_rollouts.py run --processes 4 --threads 4
python scripts/verify_multi_ply_position_rollouts.py --local-raw
```

The dry pass invokes the public selectors, reconstructs complete cases
from genuine checkpoints, and estimates outstanding miss counts. For an
unknown endpoint it uses the stored 3-ply prediction only to estimate
potential promotions. Any case containing such a placeholder stays out of
the reference file. The real pass checkpoints each requested missing
rollout under `logs/multi_ply_position_rollouts/native/` and each completed
checker decision under `checker/` there. Repeating the command after
interruption consumes completed checkpoints. `assemble` rebuilds the
published outputs from these results and the original archives without
simulation. The [earlier supplemental run](../rollout_3p_vr1296/README.md)
contains additional fully evaluated endpoints for separate analyses.
Verification checks the 300 selected inputs, archived values, cube/post
outputs and per-move levels. The optional `--local-raw` also regenerates
all legal boards and verifies frozen checker means, bearoff overrides and
standard errors against native checkpoints. Verification performs no
rollouts.
