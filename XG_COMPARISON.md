# Comparing the Open Sage Bot Engine to XG's Bot Engine

XG (eXtreme Gammon) is a standard reference for backgammon analysis, and its bot engine is considered a very strong one.

XG does not include an API to run its bot engine programmatically, so we could not run head-to-head games between Open Sage and XG at the scale required to identify small differences between them.

However, we did settle on an approach to test Open Sage against XG at scale using XG's Batch Analysis function.

Our goal was to compare Open Sage evaluations against XG evaluations at a comparable level. We compared:
* Sage 3T vs XG Roller ++. Both are truncated rollouts that incorporate variance reduction, truncate after 7 turns, and use 3-ply (or better) for decisions along each simulation path.
* Sage 2T vs XG Roller +. Like 3T/++ except that they make 2-ply decisions internally.
* Sage 1T vs XG Roller. Truncated rollouts with 72 (Sage 1T)/42 (XG Roller) paths, use variance reduction, truncate after 5 turns, and use 1-ply evaluations internally.
* Sage 3P vs XG 3-ply. Both are algorithms that look forward three plies (turns) and average the results over those possible futures. At the end of each path both do a 1-ply calculation - that is, the raw neural network output.
* Sage 4P vs XG 4-ply. Four-ply lookahead.

We looked at four approaches:

* Rollout PR: we simulated money games and match play over many games, rolled out the closest decisions, and scored bot decisions against these rolled out results, and ended up with a Performance Rating (PR) against the rollout truth. We store these benchmark decision results. Then we run each decision by a candidate bot and ask it to give its decision, and score its result against the benchmark equities.
* Disputed Positions: within the money benchmark above, we take the subset of the hardest (rolled-out) positions where Sage 3T and XG Roller ++ chose differently and — having both a Sage and an XG full rollout of each — score each engine's pick against both rollouts, to see which was closer without depending on a single engine's truth.
* Position-family benchmarks: back games, containment games, massive back games and the snake are rare in self-play games and historically the weakest part of any engine, so thirteen benchmarks measure them directly, each a family of real decisions with a rollout-grade reference. A whole-game Paskogammon benchmark — a variant played from a scattered start that produces far more of these positions than standard backgammon — rounds them out.
* Real-Match PR Agreement: instead of measuring strength against a rollout truth, we ask a practical question — if you analyze a real match in XG and again in Sage, do the two engines report the same Performance Rating? We re-analyzed hundreds of real tournament matches that had already been analyzed in XG, and compared the per-player PRs the two engines produced.

## Rollout PR Analysis

This is similar in approach to the analysis done on XG (and a number of other bots) in 2012: https://www.extremegammon.com/studies.aspx.

### Money Games

#### Rollout PR Algorithm

We simulated 500 money games of Sage 3P vs Sage 3P. We ran through all the decisions, and did a second pass, re-evaluating any decisions at Sage 3T where the best decision was within 0.05 equity of the next best decision. We then did a third pass, rolling out any decisions which Sage 3T evaluated as within 0.02 equity of the next best decision. We saved out all those results and counted them as the "true" decision results, against which we can benchmark any bot's decisions.

For rollouts we used Open Sage rollouts with 3P decisions for checker play and cube actions. We ran batches of 1,296 paths until the 95% accuracy range on the equity was less than 0.005, or it did 20,736 (=16 times 1,296) paths.

For a given bot (and evaluation level), we had the bot evaluate its decision for each one of those benchmark decisions, and scored it against the benchmark truth. We calculated a Performance Rating (PR) as the average error (as measured against the benchmark equities) multiplied by 500. We also broke out the results into checker play and cube action PRs.

For XG results, we manually ran XG's Batch Analyze on the 500 individual game files, then automatically parsed the XG decisions from the .xg files it generates (one per game). The Batch Analyze settings were 3-ply decisions, moving to the listed eval level for disputes.

#### Rollout PR Results

There were 17,535 decisions across 16,889 positions. Of the 16,889 positions, 7,652 were settled at 3-ply; the other 9,237 were re-evaluated at 3T, of which 3,260 settled there and 5,977 were rolled out. Some rollouts were very quick, while the slowest took well over an hour to roll out on a machine with 16 cores.

| Bot | PR | Checker PR | Cube PR| Pure Race | Racing | Attacking | Priming | Anchoring |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sage 3T | 0.23 | 0.20 | 0.41 | 0.02 | 0.25 | 0.20 | 0.32 | 0.32 |
| XG Roller ++ | 0.34 | 0.33 | 0.38 | 0.04 | 0.44 | 0.25 | 0.41 | 0.48 |
| Sage 2T | 0.27 | 0.24 | 0.43 | 0.02 | 0.32 | 0.20 | 0.40 | 0.36 |
| XG Roller + | 0.41 | 0.41 | 0.39 | 0.05 | 0.60 | 0.31 | 0.47 | 0.54 |
| Sage 1T | 0.49 | 0.50 | 0.43 | 0.04 | 0.58 | 0.43 | 0.62 | 0.64 |
| XG Roller | 0.53 | 0.54 | 0.48 | 0.05 | 0.64 | 0.45 | 0.71 | 0.67 |
| Sage 4P | 0.43 | 0.42 | 0.52 | 0.08 | 0.54 | 0.39 | 0.45 | 0.60 |
| XG 4-ply | 0.47 | 0.46 | 0.52 | 0.06 | 0.59 | 0.40 | 0.57 | 0.59 |
| Sage 3P | 0.60 | 0.60 | 0.61 | 0.13 | 0.78 | 0.53 | 0.67 | 0.75 |
| XG 3-ply | 0.57 | 0.57 | 0.58 | 0.05 | 0.72 | 0.48 | 0.73 | 0.72 |
| Sage 2P | 1.68 | 1.44 | 2.91 | 0.40 | 1.84 | 1.86 | 1.87 | 1.77 |
| Sage 1P | 2.55 | 2.42 | 3.24 | 0.42 | 2.71 | 2.79 | 3.13 | 2.74 |

Sage evaluations are stronger than their equivalent XG evaluations in every case except 3-ply, where XG is slightly stronger, but the two are very close. The edge is clearest at the truncated-rollout levels: Sage 3T scores 0.23 against XG Roller ++'s 0.34, and Sage 2T 0.27 against XG Roller +'s 0.41.

#### Running the Pipeline

The rollout-PR data set is built entirely by `scripts/benchmark_money.py`, run
from the `bgsage/` repo root — it resolves its Python path and the compiled
`bgbot_cpp.pyd` from inside `bgsage/`, so the only prerequisite is a local Open
Sage build (no external services). The build is three adaptive-precision passes,
each an independently resumable stage of `benchmark_money.py build`, so you can
run them one at a time, all locally. The `--n-games 100` below is just an example
— scale it up for a larger set.

**1. Simulate the games and capture 3P (pass 1).**

```bash
python scripts/benchmark_money.py build --stages pass1 --n-games 100 --workers 6
```

Plays `--n-games` Sage-3P-vs-Sage-3P money games (Jacoby + beavers on) across
`--workers` parallel self-play processes, capturing 3-ply checker and cube
analytics for every real decision. Writes one `build/stage1/seed_<N>.json` per
game; with `--write-txt` (on by default) it also writes an XG-import
`xg/seed_<N>.txt` transcript per game — those are the files you later batch-
analyze in XG to score XG against the same positions.

**2. Re-evaluate close decisions at 3T (pass 2).**

```bash
python scripts/benchmark_money.py build --stages pass2 --n-threads 16
```

Re-evaluates in-process every decision whose 3-ply best-vs-second-best gap is
under 0.05, using Sage 3T (the Roller++-style truncated rollout). `--n-threads`
is the thread count per evaluation. Appends to `build/stage2_3t.jsonl` and is
resumable (a re-run skips positions already done).

**3. Roll out the closest decisions (pass 3).**

```bash
python scripts/benchmark_money.py build --stages pass3 --n-threads 16
```

Rolls out every decision still within 0.02 equity after the 3T pass: 1,296-path
batches with 3-ply checker and cube decisions and variance reduction, repeated
until the 95% equity band is under ~0.005 or 16 batches (20,736 paths) are
reached. Appends to `build/stage3_rollout.jsonl`. **This is by far the longest
stage** — the hardest back-game positions take well over an hour each and run one
after another locally — but it is fully resumable, so you can stop and restart at
will.

After pass 3 the assembled benchmark is written to
`data/money_benchmark/benchmark.json`. (Running `build` with no `--stages` runs
all three passes in order.)

**4. Score a bot against it.**

```bash
python scripts/benchmark_money.py score --level 3ply --n-threads 16    # Sage 3P
python scripts/benchmark_money.py score --level truncated3             # Sage 3T
```

`--level` takes `1ply`–`4ply`, `truncated1`/`2`/`3` (= 1T/2T/3T) or `rollout`;
`--n-threads` scores positions concurrently. Decisions whose stored reference is
too coarse for how close they are (e.g. a not-yet-rolled-out position) are skipped
and reported, so a partially built data set still scores cleanly. Scoring writes
a resume cache, `scores/sage_<level>.jsonl`, and beside it
`scores/sage_<level>.picks.jsonl` — the board or cube action the bot chose for
every decision — which the two comparisons below read.

To score **XG**, batch-analyze the pass-1 `xg/*.txt` transcripts (with **Save
Games after analyze** checked) so each gets a matching `.xg` per game and level
(`seed_<N>_3p.xg` for the 3-ply batch, `seed_<N>.xg` for 4-ply, `_roller`, `_p`
and `_pp` for the three Roller levels), then:

```bash
python scripts/benchmark_pr_xg_levels_all.py --benchmark money
```

which reads XG's #1 decision per position at every level and scores it against
the same saved reference equities, printing the same PR breakdown and writing
`scores/xg_<level>.jsonl` + `.picks.jsonl` in the same layout as the Sage caches.

### Match Play

We repeated the Rollout PR experiment in match play, where the score on the board changes the value of every decision. A 5-point match is a good test case: the match score materially affects checker and cube decisions through these relatively short matches, so it exercises the engines' match-equity handling, not just their raw position evaluation.

#### Rollout PR Algorithm

We simulated 130 5-point matches of Sage 3P vs Sage 3P (both sides played by Sage at 3-ply). The match state — each player's away-count and the Crawford flag — is threaded through every evaluation, so all decisions, and the rolled-out "truth", are computed in match-equity (MWC) space against the correct score; cube decisions use the Kazaross-XG2 match equity table. Otherwise the method is identical to the money-game build: a first pass capturing 3-ply analytics for every decision, a second pass re-evaluating at Sage 3T any decision within 0.05 equity of its next-best alternative, and a third pass rolling out (1,296-path batches, 3-ply checker and cube decisions, variance reduction, repeated until the 95% equity band is under 0.005 or 20,736 paths) any decision still within 0.02 equity. The strongest tier reached for each decision is its benchmark truth.

For XG, we manually batch-analyzed the 130 match transcripts (3-ply decisions, upgrading to the listed eval level for disputes) — one `.xg` per match, each containing every game of the match — and scored XG's chosen decision against the same saved reference equities.

#### Rollout PR Results

There were 18,292 decisions across 17,892 positions. Of the 17,892 positions, 7,522 were settled at 3-ply; the other 10,370 were re-evaluated at 3T, of which 3,460 settled there and 6,910 were rolled out.

| Bot | PR | Checker PR | Cube PR| Pure Race | Racing | Attacking | Priming | Anchoring |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sage 3T | 0.23 | 0.20 | 0.46 | 0.07 | 0.22 | 0.20 | 0.28 | 0.31 |
| XG Roller ++ | 0.36 | 0.35 | 0.44 | 0.08 | 0.44 | 0.32 | 0.33 | 0.49 |
| Sage 2T | 0.26 | 0.24 | 0.40 | 0.02 | 0.35 | 0.18 | 0.31 | 0.35 |
| XG Roller + | 0.44 | 0.44 | 0.44 | 0.09 | 0.54 | 0.44 | 0.39 | 0.55 |
| Sage 1T | 0.47 | 0.47 | 0.47 | 0.06 | 0.55 | 0.45 | 0.55 | 0.56 |
| XG Roller | 0.51 | 0.51 | 0.56 | 0.09 | 0.63 | 0.49 | 0.50 | 0.65 |
| Sage 4P | 0.41 | 0.41 | 0.47 | 0.05 | 0.50 | 0.37 | 0.44 | 0.53 |
| XG 4-ply | 0.46 | 0.45 | 0.58 | 0.09 | 0.59 | 0.44 | 0.42 | 0.60 |
| Sage 3P | 0.56 | 0.54 | 0.71 | 0.05 | 0.73 | 0.55 | 0.59 | 0.63 |
| XG 3-ply | 0.54 | 0.53 | 0.62 | 0.09 | 0.68 | 0.53 | 0.52 | 0.68 |
| Sage 2P | 1.27 | 1.23 | 1.58 | 0.39 | 1.46 | 1.24 | 1.49 | 1.39 |
| Sage 1P | 2.22 | 2.21 | 2.35 | 0.38 | 2.43 | 2.41 | 2.44 | 2.47 |

As in money play, Sage's evaluations are stronger than the equivalent XG evaluation at every matched level except 3-ply, where the two are within noise (XG 0.54, Sage 0.56). The truncated-rollout levels that most users rely on — 3T and 2T — show Sage's clearest edge.

#### Running the Pipeline

The match data set is built by `scripts/benchmark_match.py`, the match-play twin of `benchmark_money.py`; the match length and number of matches are arguments, and the match state is threaded through every decision. The three passes are the same independently-resumable stages, run locally from a fresh `bgsage` checkout:

```bash
python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass1 --workers 6   # simulate + capture 3P; one XG-import .txt per match
python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass2 --n-threads 16  # re-evaluate close decisions at 3T
python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass3 --n-threads 16  # roll out the closest decisions
```

The assembled benchmark is written to `data/match_benchmark/5pt/benchmark.json` (shipped as `benchmark.json.gz`). Score a bot, or XG, exactly as in the money case:

```bash
python scripts/benchmark_match.py score --match-length 5 --level truncated3        # Sage 3T
python scripts/benchmark_pr_xg_levels_all.py --benchmark match --match-length 5    # XG at every level, from the batch-analyzed .xg files
```

As with the money build, pass 3 is by far the longest stage and fully resumable; the hardest back-game and long-race positions take well over an hour each.

## Disputed Position Analysis

The Rollout PR study scores both engines against a shared reference. A sharper, more direct question is: in realistic play, where do the two engines actually disagree on the best decision — and when they do, which one is right?

We answer it on the money games, reusing the benchmark built above. Among the hardest decisions — the rolled-out positions — we have **both** a full Sage rollout and a full XG rollout of each. We take the subset where **Sage 3T** and **XG Roller ++** chose differently, and score each engine's pick against both rollouts. Having both rollouts is the point: every disagreement is judged against Sage's rollout *and* XG's own, so the verdict does not depend on trusting a single engine's truth.

This method currently covers money games only: it needs an XG full rollout of the disputed positions, which we have run for the money benchmark but not for match play. Match-play strength is covered by the Rollout PR study above and the Real-Match Agreement study below.

### Money Game Results

Across the 5,969 rolled-out money positions with both rollouts, the two engines disagree on 1,286 checker plays and 80 cube decisions. We score each disagreement on the common set — the ones where both engines' picks were rolled by both rollouts — so the Sage-rollout and XG-rollout comparisons cover the identical positions.

**Checker play** — 1,286 disagreements, 1,139 on the common set:

| Reference | Sage 3T closer | XG Roller ++ closer | Neither | Sage 3T avg error | XG Roller ++ avg error |
| --- | ---: | ---: | ---: | ---: | ---: |
| vs XG rollout | 37.7% | 52.2% | 10.1% | 0.0030 | 0.0030 |
| vs Sage rollout | 50.0% | 38.1% | 11.9% | 0.0023 | 0.0040 |

On checker play — the large majority of disagreements — each rollout sides with its own engine on which move is best: by XG's rollout XG Roller ++ matches the best move more often, by Sage's rollout Sage 3T does. Measured by how much equity the disputed pick gives away, the two are level by XG's rollout (0.0030 each, PR 1.5) and Sage 3T is closer by Sage's rollout (0.0023 vs 0.0040).

**Cube decisions** — 80 disagreements:

| Reference | Sage 3T closer | XG Roller ++ closer | Sage 3T avg error | XG Roller ++ avg error |
| --- | ---: | ---: | ---: | ---: |
| vs XG rollout | 57.5% | 42.5% | 0.0110 | 0.0074 |
| vs Sage rollout | 61.3% | 37.5% | 0.0058 | 0.0102 |

Cube disagreements are far rarer (80 in all) and mixed: Sage 3T matches the rolled-out cube action more often under both rollouts, but by XG's rollout its misses are the more expensive ones, so on average error XG Roller ++ is closer there and Sage 3T closer by Sage's rollout. Given the small sample, read the cube panel as suggestive rather than decisive.

Because the two rollouts come from independent engines, their disagreement is itself informative: on the positions where two strong engines differ, the two rollouts differ about who is right, each leaning toward the engine that ran it. The disputed positions do not separate Sage 3T and XG Roller ++.

### Running the Pipeline

The disagreement study draws from the same money benchmark built for the Rollout
PR study, cross-referenced against XG's own full rollout of the hardest positions.
The only manual dependency is XG's Batch Rollout; two scripts do the rest.

1. Build the money benchmark (the Rollout PR pipeline above) and score Sage 3T
   against it, which records its picks in `scores/sage_truncated3.picks.jsonl`.
   Pass 1 also writes the XG-import transcripts to `data/money_benchmark/xg/`.
2. In XG, **Batch-Rollout** those positions with **Save Games after analyze**
   checked, so each rolled-out decision carries XG's own rollout equities in the
   resulting `.xg` files, and batch-analyze the transcripts at Roller ++ for XG's
   picks (`benchmark_pr_xg_levels_all.py --level rollerpp` records them).
3. Harvest and report:

```bash
python scripts/xg_benchmark_report.py                                                     # parses XG's rollouts into data/money_benchmark/xg_results/rollout.jsonl
python scripts/xg_dispute_analysis.py --sage-picks data/money_benchmark/scores/sage_truncated3.picks.jsonl
```

The second command prints the disputed-position report — every disagreement
scored against the Sage and XG rollouts in turn.

## Back Games, Containment Games and the Snake

Ordinary self-play games rarely produce a deep back game, a containment game or a far-side prime holding a single straggler, so the money and match sets above say little about how an engine plays them — and they are the positions where engines have historically been weakest. Thirteen position-family benchmarks measure them directly. Each follows the recipe of the money benchmark — real decisions, each with a rollout-grade reference — but the decisions come from inside the family:

* **The ten classic back games**, named by the two points the back-game player holds in the opponent's home board: the 2-1 back game holds the opponent's 2- and 1-points, the 5-4 holds the 5- and 4-points, and so on (2-1, 3-1, 3-2, 4-1, 4-2, 5-1, 5-2, 4-3, 5-3 and 5-4). Each benchmark starts from a small set of hand-curated seed positions and holds about 1,000 decisions, recorded only while both named anchors are still held and the holder is behind in the race.
* **Containment games**, defined by the escaper rather than the container: one side has borne off some checkers and has one to three checkers that were hit and must run the whole board home, while the other side, with a lost race, arranges whatever it has left to keep hitting them. Decisions are recorded while the escaper still has trapped checkers.
* **Massive back games**: three or more anchors in the opponent's home board, or two anchors with seven or more checkers back — the deep, timing-driven positions where the back-game player has committed most of the army.
* **The snake**: a run of four or more consecutive made points entirely on the opponent's half of the board, trapping a checker on the bar or in the holder's home board while the opponent's other ten or more checkers are already crunched home. A priming game rather than a back game, and one of the hardest shapes in backgammon to play.

Each benchmark's seeds define its family. Sage plays unlimited games out of those seeds against itself (cubeful, Jacoby and beaver on, 3-ply), and every decision that arises while the position still belongs to the family is recorded: a checker play when there are at least two legal moves with a meaningful equity spread between them, a cube position when the doubler's decision is not trivially obvious or a double is actually offered. Every recorded decision is then rolled out to produce its reference: 5,184 paths per position played to completion, with 3-ply checker and cube decisions for the first three half-moves and 2-ply thereafter, variance reduction on. A checker decision's reference carries the rolled-out equity of every candidate the reference player's move filter kept, and candidate-completion passes rolled out every move an engine under test actually chose, so each score below is against a rollout-graded candidate rather than a filter-precision estimate. A cube position where the opponent owns the cube is not a decision for the player on roll and is not scored.

### Results

PR of Sage from 2-ply up, against each family's rollout reference. Lower is better; the decision count is the number scored.

| Benchmark | Decisions | 2P Sage | 2P XG | 3P Sage | 3P XG | 4P Sage | 4P XG | 1T Sage | 1T XG | 2T Sage | 2T XG | 3T Sage | 3T XG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2-1 backgame | 1,000 | 2.26 | 2.84 | 1.01 | 1.64 | 1.03 | 1.22 | 0.61 | 1.67 | 0.41 | 0.98 | 0.38 | 0.93 |
| 3-1 backgame | 1,001 | 2.22 | 2.22 | 0.88 | 0.95 | 0.67 | 0.72 | 0.66 | 0.91 | 0.34 | 0.59 | 0.32 | 0.48 |
| 3-2 backgame | 1,008 | 2.07 | 2.39 | 0.91 | 1.31 | 0.78 | 0.99 | 0.65 | 0.95 | 0.44 | 0.61 | 0.25 | 0.51 |
| 4-1 backgame | 1,001 | 1.77 | 1.83 | 0.78 | 0.82 | 0.60 | 0.53 | 0.55 | 0.69 | 0.24 | 0.47 | 0.19 | 0.45 |
| 4-2 backgame | 1,000 | 2.16 | 1.96 | 0.73 | 1.05 | 0.69 | 0.55 | 0.63 | 0.79 | 0.36 | 0.61 | 0.27 | 0.43 |
| 5-1 backgame | 1,003 | 1.32 | 1.77 | 0.66 | 0.76 | 0.46 | 0.64 | 0.49 | 0.60 | 0.26 | 0.41 | 0.23 | 0.32 |
| 5-2 backgame | 1,002 | 1.53 | 1.60 | 0.49 | 0.82 | 0.48 | 0.63 | 0.53 | 0.63 | 0.33 | 0.41 | 0.24 | 0.30 |
| 4-3 backgame | 1,002 | 1.69 | 2.13 | 0.74 | 0.80 | 0.55 | 0.58 | 0.52 | 0.76 | 0.24 | 0.55 | 0.19 | 0.44 |
| 5-3 backgame | 1,003 | 1.86 | 2.13 | 1.02 | 0.89 | 0.58 | 0.63 | 0.68 | 0.81 | 0.35 | 0.46 | 0.21 | 0.44 |
| 5-4 backgame | 1,001 | 2.45 | 2.74 | 0.93 | 1.15 | 0.73 | 0.78 | 0.76 | 0.97 | 0.47 | 0.55 | 0.30 | 0.49 |
| Containment | 3,270 | 4.73 | 14.62 | 2.54 | 11.25 | 2.05 | 10.37 | 2.29 | 7.76 | 1.30 | 7.85 | 1.05 | 6.16 |
| Snake | 978 | 21.00 | 39.92 | 21.42 | 36.64 | 23.53 | 36.17 | 11.19 | 32.99 | 8.42 | 32.38 | 5.85 | 32.24 |
| Massive backgame | 2,103 | 4.33 | 5.76 | 2.56 | 4.44 | 2.17 | 3.68 | 1.91 | 3.14 | 1.39 | 2.75 | 1.09 | 2.64 |

Pooled over the ten classic back games (10,021 decisions), Sage's PR runs 1.93 at 2-ply, 0.82 at 3-ply and 0.66 at 4-ply, and 0.61, 0.34 and 0.26 at 1T, 2T and 3T; blunders (errors of 0.08 or more) fall from 28 at 2-ply to none at all at 3T. Containment games and massive back games are three to four times harder at every level — 1.05 and 1.09 at 3T — and the snake is in a class of its own: 21.00 at 2-ply and still 5.85 at 3T. It is also the one family where added PLY does not reliably help (2-ply 21.00, 4-ply 23.53) while the truncated-rollout levels do, which is what a family of hold-or-release decisions looks like: the choice turns on how a long containment plays out, and only simulation gets at that. Read the snake column as a measure of how far the position family is from solved rather than as a ranking of Sage's levels.

Against XG at matched levels, Sage leads on all thirteen benchmarks and at every level pooled: on the ten back games 1.93 against 2.16 at 2-ply, 0.61 against 0.88 at 1T and 0.26 against 0.48 at 3T. The margin widens with the difficulty of the family — at 3T, containment 1.05 against 6.16, massive back games 1.09 against 2.64, and the snake 5.85 against 32.24, a factor of five and a half. Only four of the 78 individual cells go the other way, all of them single back games at 2-ply to 4-ply where both engines are already under 1.1. XG has no programmatic interface, so its decisions are replayed through Batch Analyze from native `.xg` exports — see "Running the Pipeline" below.

### Running the Pipeline

The folder benchmarks live under `backgame_ref_positions/benchmark/` — one `<family> starting.txt` of seed positions and one `<family> rollout.jsonl` reference per family. Score any Sage level against a family with:

```bash
python scripts/score_backgame_pr.py --category "21 backgame" --level 3ply     # one family
python scripts/score_backgame_pr.py --category containment --level truncated3
```

`--category` takes any of the thirteen family names (`21 backgame` … `54 backgame`, `containment`, `snake`, `massive backgame`); `--level` the same levels as the money scorer. The report gives the family's PR, its checker and cube parts, the blunder count, and — because a pick outside the reference's rolled candidates is only ever valued at filter precision — the share of the error that sits on such picks, which should be near zero before a headline number is trusted.

To score XG, export the decisions as native `.xg` games, Batch-Analyze one copy of them per XG level, and harvest XG's chosen decision from the analysed files. With XG running on a Windows desktop the whole pass is scripted — `xg_batch_win.py` drives the Batch Analyze dialog through its Win32 controls (choose every file of a level's folder, select the level's analysis profile for each player, **Save Games after analyze** on, Start):

```bash
python scripts/export_folder_benchmark_xg.py generate      # data/backgame_xg/<family>/bench_shard_NNN.xg (+ sidecars)
python scripts/xg_folder_batch.py stamp                    # one flat folder per XG level: data/backgame_xg_flat/<tag>/
python scripts/xg_folder_batch.py analyze --level xg3ply   # XG: Batch Analyze all 81 shards of that level
python scripts/xg_folder_batch.py wait --level xg3ply      # until every shard has been rewritten
python scripts/xg_folder_batch.py score --level xg3ply     # -> data/backgame_xg_scores/xg_xg3ply_<family>.json
```

Level tags are `xg2ply`, `xg3ply`, `xg4ply`, `xgroller`, `xgrollerplus`, `xgrollerpp`; the profile each selects is the table at the top of `xg_folder_batch.py`. Each scoreable decision is written as a one-decision game — the position as the game's starting position, the mover on roll with the recorded dice — so XG's analysis of that record is its analysis of the benchmark decision. `score` reads XG's top-ranked move or recommended cube action from every analysed record, scores it with the same formulas, and reports any records XG left unanalysed.

## Paskogammon

Paskogammon is a backgammon variant played from a scattered opening position instead of the standard one. Games from that start produce far more back games and containment games than standard backgammon, which makes it a useful whole-game test of exactly the play the family benchmarks isolate. The benchmark is built like the money benchmark — 50 self-play games from the Paskogammon start (Jacoby and beaver on), every decision captured at 3-ply, close decisions re-evaluated at 3T, the closest rolled out — and scored the same way, over 2,556 decisions. XG was batch-analysed at every level it offers from 2-ply up; because the variant's starting position cannot be expressed in the text transcript format, the games are exported to XG as native `.xg` archives that carry the position explicitly.

| Bot | PR | Checker PR | Cube PR| Pure Race | Racing | Attacking | Priming | Anchoring |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sage 3T | 0.88 | 0.85 | 1.07 | 0.08 | 0.59 | 0.96 | 0.44 | 1.71 |
| XG Roller++ | 1.92 | 1.99 | 1.37 | 0.01 | 1.36 | 2.35 | 1.10 | 3.39 |
| Sage 2T | 1.46 | 1.51 | 1.08 | 0.03 | 1.36 | 1.79 | 0.51 | 2.35 |
| XG Roller+ | 2.44 | 2.59 | 1.45 | 0.02 | 1.97 | 2.48 | 1.54 | 4.19 |
| Sage 1T | 1.65 | 1.64 | 1.76 | 0.08 | 1.14 | 1.37 | 1.02 | 3.31 |
| XG Roller | 2.63 | 2.69 | 2.23 | 0.05 | 2.36 | 2.86 | 1.54 | 4.15 |
| Sage 4P | 1.90 | 1.92 | 1.72 | 0.14 | 1.64 | 2.43 | 1.04 | 2.88 |
| XG 4-ply | 2.62 | 2.66 | 2.32 | 0.05 | 2.22 | 2.76 | 1.58 | 4.32 |
| Sage 3P | 2.28 | 2.35 | 1.78 | 0.03 | 1.64 | 3.69 | 1.21 | 3.53 |
| XG 3-ply | 2.97 | 2.96 | 3.04 | 0.05 | 2.55 | 3.21 | 1.77 | 4.81 |
| Sage 2P | 4.57 | 4.37 | 5.93 | 0.16 | 3.97 | 5.16 | 3.95 | 6.30 |
| XG 2-ply | 4.38 | 4.50 | 3.58 | 0.23 | 3.52 | 5.83 | 2.98 | 6.50 |
| Sage 1P | 6.14 | 5.93 | 7.58 | 0.26 | 5.18 | 8.55 | 4.70 | 8.18 |

Every level of both engines finds Paskogammon several times harder than standard backgammon — the PRs are three to eight times the money-game figures — and Sage is ahead of XG at every matched level from 3-ply up: 3T 0.88 against Roller ++'s 1.92, 2T 1.46 against 2.44, 1T 1.65 against 2.63, 4-ply 1.90 against 2.62 and 3-ply 2.28 against 2.97. Only at 2-ply does XG come out ahead (4.38 against 4.57). Anchoring is the hardest game plan for both, and it is where Sage's margin is widest — 1.71 against Roller ++'s 3.39 at the strongest truncated level.

**A note on this benchmark's reference.** A rollout is itself played by an engine, and on a decision where two engines prefer different moves that choice can decide which move the rollout calls best. Those decisions — the ones where the engines being compared disagree — are therefore played out by the strongest engine available, and every engine's pick is forced into the set of moves the rollout evaluates, so no engine's answer is valued at coarser precision than another's. Each decision records the engine that played its rollout in `reference_player`.

### Running the Pipeline

```bash
python scripts/benchmark_pasko.py build --n-games 50 --stages pass1 --workers 6      # simulate from the Paskogammon start
python scripts/benchmark_pasko.py build --n-games 50 --stages pass2,pass3 --n-threads 16
python scripts/benchmark_pasko.py score --level truncated3                          # Sage 3T
python scripts/export_pasko_benchmark_xg.py                                         # native .xg games for XG (data/pasko_money_benchmark/xg_native/)
python scripts/benchmark_pr_xg_pasko.py --xg-dir data/pasko_money_benchmark/xg_native_xgrollerpp   # XG, one analysed folder copy per level
```

## Match PR Agreement on Real Matches

The analyses above measure *strength* — how close each engine's decisions are to a rolled-out truth. Another, equally practical question matters to anyone who uses an engine to study their own play: **if you analyze a real match in XG, note your Performance Rating, then analyze the same match in Sage, how close are the two PRs?** A player who has spent years building intuition for what a given PR means in XG should get essentially the same number from Sage.

To test this directly, we took a large set of real tournament matches that had already been analyzed in XG, re-analyzed every one from scratch in Sage, and compared the Performance Rating each engine assigned to each player.

### Evaluation Settings

Each match was re-analyzed in Sage at a **3-ply base**, with an **expert 3T pass** (a 360-path truncated rollout) applied to the decisions where the player's actual move disagreed with the 3-ply best. This mirrors how a strong XG analysis works — a base ply for the clear decisions, escalating to a truncated rollout for the close ones — and the two levels are matched in strength: **Sage 3P ≈ XG 3-ply** and **Sage 3T ≈ XG Roller ++** (the same level pairs compared in the studies above). Each engine then computes a PR per player from its own evaluations and its own decision counting — exactly what a user sees in each app.

### The Matches

The match files come from three 2026 tournaments — **UBC Texas**, **UBC Istanbul**, and **UBC Japan** — all 7-point matches, analyzed in XG and generously provided by **Máté Fehér**.

| Event | Matches |
|---|---:|
| UBC Texas 2026 | 100 |
| UBC Istanbul 2026 | 146 |
| UBC Japan 2026 | 44 |
| **Total** | **290** |

That is 290 matches and 580 individual player ratings. (One further match was set aside as a corrupted transcription.)

### Results

For each player in each match we have two Performance Ratings — XG's and Sage's — and their difference. Pooling all 580 player ratings:

| Per-player PR | XG | Sage | **Difference (Sage − XG)** |
|---|---:|---:|---:|
| Average | 4.36 | 4.36 | **+0.002** |
| Standard deviation | 2.08 | 2.10 | **0.37** |
| 95% range | 1.52 – 9.36 | 1.44 – 9.67 | **−0.76 – +0.74** |

The two engines agree almost exactly. The **average difference is +0.002 PR** — statistically indistinguishable from zero (95% confidence interval ±0.03; *p* = 0.90). The standard deviation of the difference (**0.37**) is small next to the spread in PR itself (**2.08**), so the disagreement on any single rating is minor relative to how much PR naturally varies from player to player and match to match. The two engines' per-player ratings correlate at **r = 0.98**.

In practical terms: a player who analyzes a match in Sage will, in the large majority of cases, see essentially the same Performance Rating that XG would give. As a measure of how well a match was played, the two engines are interchangeable.

## Conclusion

Open Sage and XG are close at every matched evaluation level. In the Rollout PR study — money play and 5-point match play alike — Sage's evaluations score better than the equivalent XG evaluation at every level except 3-ply, where the two are within noise, and the edge is clearest at the truncated-rollout levels most users rely on: 3T scores 0.23 against Roller ++'s 0.34 in money play and 0.23 against 0.36 in match play. The Disputed Positions study — which rolls out only the money positions where the two engines actually disagree and scores them against both engines' rollouts — is level on checker play, each rollout favouring the engine that ran it, with cube disagreements too rare and too split to call. The differences are small.

On the position families where engines have historically been weakest, Sage's play is measured to the same standard: pooled over the ten classic back games its 3T PR is under 0.7 and its 3-ply PR under 1.0, containment games and massive back games score between 2 and 3 at the truncated-rollout levels, and the snake remains the hardest family by a wide margin. In Paskogammon, the whole-game variant that produces these positions constantly, Sage is ahead of XG at every matched level from 3-ply up — 3T 0.88 against Roller ++'s 1.92 — and the margin is widest at 3-ply, where XG's cube PR is 3.04 against Sage's 1.78.

And on real matches, the two engines assign nearly identical Performance Ratings: across 290 tournament matches, the average difference between a player's Sage PR and XG PR is statistically indistinguishable from zero. Whether the test is strength against a rolled-out truth or simple agreement on how a real game was played, Open Sage and XG land in the same place.
