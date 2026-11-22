# Rollout regression guard, 2026-11-21

The Git-verified matched-toolchain control and CPU release each evaluated the first match
attacking and money pure-race case for all three APIs. Each arm ran in a fresh
process using frozen code, the same Stage 11 weights, the same seed per source
position, four trial threads and CPU mask `0xff00`. The two pairs ran in AB
then BA order. The timer surrounds each public call, including its normal
checker filtering and any promotions.

| Trial policy | Checker speedup | Cube speedup | Post-move speedup |
| --- | ---: | ---: | ---: |
| Production 3T, 360 trials | 1.80× | 1.66× | 1.65× |
| Full game, 3-ply checker and cube, 36 trials | 1.61× | 1.64× | 1.63× |
| Full game, 3-ply checker and cube, 1,296 trials | 1.66× | 1.58× | 1.59× |

Speedup is the ratio of summed call times for four calls per API and policy.
All 36 matched calls had exactly equal complete analytics, including
candidate ranking, five probabilities, equities and reported standard errors.
The 1,296-trial measurement used the reference-strength trial policy. Its
checker timer includes the production prefilter and promotions. The
[position reference corpus](../rollout_positions/README.md) preserves the
actual evaluation depth of every returned checker move.

See [summary.json](summary.json) for paired timings and differences and
[settings.json](settings.json) for the experiment settings. Each JSONL file
in this directory contains twelve complete call records. The separate
[1,296-trial results](full1296/summary.json) have six complete calls per arm
file, along with their own [settings](full1296/settings.json). Across the
four 1,296-trial processes, identical seeds produced exactly equal full
outputs for each of the six cases. The hash manifests identify the Python,
native extension, model and data files used by each arm. Reproduce
with [the paired runner](../../../scripts/compare_rollout_performance.py).
