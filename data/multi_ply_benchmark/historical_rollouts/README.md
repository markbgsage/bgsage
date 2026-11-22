# Historical rollout references

These are the saved native Sage rollout means for positions in the two fixed benchmark sets. The 37 checker and 10 cube records are keyed by the benchmark position ID and the original decision key. [Public-shaped records](rows.jsonl.gz) · [Source keys, fields and coverage](audit.json)

| Sample | Source | 3P roots | 3T roots | Full-rollout roots | Full-rollout checker endpoints |
| --- | --- | ---: | ---: | ---: | ---: |
| Checker | Money | 22 | 9 | 19 | 122 of 489 on the 19 roots |
| Checker | Five-point match | 17 | 15 | 18 | 102 of 300 on the 18 roots |
| Cube | Money | 36 | 8 | 6 | |
| Cube | Five-point match | 37 | 9 | 4 | |

The 37 checker lists contain every legal endpoint and preserve the source's ranking, selection and recorded evaluation tier for each move. The first-ranked move received a full rollout in all 37 positions. In 35 positions the lower-ranked alternatives include filtered, shallower evaluations. Every legal move received a full rollout on `checker-match-purerace-02` and `checker-match-purerace-08`. Across all 100 fixed checker inputs there are 1,709 legal endpoints, of which 224 have historical full-rollout values. The historical files also use the move label `Rollout` for 73 truncated endpoints on 3T roots. The parent position tier distinguishes these.

The checker archive carries each endpoint board, cubeful equity, cubeless equity, five mover probabilities, evaluation level and saved cubeless-equity standard error. The historical binding took `std_error` from its cubeless rollout result and kept cubeful SE in a separate field. The compiled archive omitted cubeful and per-probability standard errors. The converter calculates the difference from the first-ranked move directly from the original equities. A stored zero error on a filtered move does not establish a stochastic sample for that move.

The 10 cube entries carry on-roll probabilities, no-double, double/take and double/pass equities, decision flags, beaver flag and recommended action. The selected branch gives the optimal equity. The archive omitted the public cubeless-equity scalar and all three cube standard errors. Those fields are null. The provenance records the ordinary cubeless equity calculated from the stored probabilities separately.

Cube archives support a previous-mover comparison. Flip the board, exchange cube ownership, swap the away scores, reverse wins and losses, and negate the recommended branch equity. The historical dataset did not save an independent post-move call, and the post-move API has its own evaluation and cube-rule inputs.

The money and match archives entered Git on June 24 and June 27, 2026, in `1796482` and `4e8cd04`. Both revisions defaulted to Stage 9. Their generator specified full 3-ply checker and cube play, variance reduction, a base seed of 42, and 1,296 games per batch. A target equity SE of 0.00255 and a 16-batch cap governed adaptive batching. Each row's saved seed identifies the simulated source game or match. The final files omit per-row batch counts, native build and model hashes, and actual rollout seeds. The records preserve the archived means at their original precision.

Run from the engine root to extract and verify without reevaluating any position. Supplying a compatible native extension also checks every historical checker endpoint against legal move generation.

```sh
python scripts/multi_ply_rollout_sources.py --verify
python scripts/multi_ply_rollout_sources.py --verify --extension path/to/bgbot_cpp
```
