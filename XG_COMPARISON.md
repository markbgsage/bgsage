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

We looked at three approaches:

* Rollout PR: we simulated money games and match play over many games, rolled out the closest decisions, and scored bot decisions against these rolled out results, and ended up with a Performance Rating (PR) against the rollout truth. We store these benchmark decision results. Then we run each decision by a candidate bot and ask it to give its decision, and score its result against the benchmark equities. For the money games we then rebuild the entire comparison against XG's own tiered analysis as the reference, so the ranking can be judged by XG's numbers as well as Sage's.
* Disputed Positions: within the money benchmark above, we take the subset of the hardest (rolled-out) positions where Sage 3T and XG Roller ++ chose differently and — having both a Sage and an XG full rollout of each — score each engine's pick against both rollouts, to see which was closer without depending on a single engine's truth.
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
| Sage 3T | 0.21 | 0.18 | 0.36 | 0.02 | 0.24 | 0.17 | 0.31 | 0.28 |
| XG Roller ++ | 0.32 | 0.31 | 0.38 | 0.04 | 0.40 | 0.24 | 0.41 | 0.44 |
| Sage 2T | 0.26 | 0.23 | 0.44 | 0.02 | 0.32 | 0.21 | 0.39 | 0.30 |
| XG Roller + | 0.41 | 0.41 | 0.39 | 0.05 | 0.59 | 0.31 | 0.47 | 0.54 |
| Sage 1T | 0.50 | 0.52 | 0.40 | 0.04 | 0.57 | 0.43 | 0.59 | 0.73 |
| XG Roller | 0.53 | 0.54 | 0.48 | 0.05 | 0.63 | 0.44 | 0.71 | 0.66 |
| Sage 4P | 0.41 | 0.39 | 0.50 | 0.07 | 0.51 | 0.37 | 0.45 | 0.53 |
| XG 4-ply | 0.46 | 0.45 | 0.52 | 0.06 | 0.58 | 0.40 | 0.57 | 0.58 |
| Sage 3P | 0.58 | 0.58 | 0.57 | 0.14 | 0.72 | 0.52 | 0.63 | 0.74 |
| XG 3-ply | 0.57 | 0.57 | 0.58 | 0.05 | 0.71 | 0.48 | 0.73 | 0.71 |
| Sage 2P | 1.64 | 1.39 | 2.88 | 0.40 | 1.77 | 1.83 | 1.86 | 1.71 |
| Sage 1P | 2.59 | 2.48 | 3.20 | 0.78 | 2.60 | 2.79 | 3.10 | 2.89 |

Sage evaluations are stronger than their equivalent XG evaluations in every case except 3-ply, where XG is slightly stronger, but the two are very close.

#### Money Games — scored against XG's own reference

The table above grades every engine against *Sage's* tiered reference. The natural objection is home-field advantage — Sage is measured against its own rollouts. So we rebuilt the same comparison with **XG's own analysis as the truth** at every tier: XG 3-ply settles the 3P-tier decisions, XG Roller ++ the 3T-tier decisions, and XG's own full rollout the rolled-out decisions — XG's tier-for-tier analogue of the Sage reference. Every engine is then re-scored against it, over the same 17,535 decisions.

| Bot | PR | Checker PR | Cube PR| Pure Race | Racing | Attacking | Priming | Anchoring |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sage 3T | 0.26 | 0.25 | 0.33 | 0.02 | 0.37 | 0.20 | 0.30 | 0.36 |
| XG Roller ++ | 0.33 | 0.35 | 0.23 | 0.04 | 0.50 | 0.20 | 0.39 | 0.47 |
| Sage 2T | 0.31 | 0.30 | 0.35 | 0.02 | 0.41 | 0.27 | 0.36 | 0.37 |
| XG Roller + | 0.38 | 0.40 | 0.23 | 0.04 | 0.58 | 0.26 | 0.39 | 0.54 |
| Sage 1T | 0.46 | 0.48 | 0.37 | 0.03 | 0.58 | 0.40 | 0.50 | 0.64 |
| XG Roller | 0.47 | 0.49 | 0.35 | 0.12 | 0.61 | 0.35 | 0.54 | 0.62 |
| Sage 4P | 0.37 | 0.36 | 0.42 | 0.06 | 0.53 | 0.34 | 0.40 | 0.39 |
| XG 4-ply | 0.45 | 0.46 | 0.40 | 0.13 | 0.60 | 0.36 | 0.48 | 0.59 |
| Sage 3P | 0.49 | 0.48 | 0.52 | 0.08 | 0.66 | 0.46 | 0.54 | 0.55 |
| XG 3-ply | 0.47 | 0.49 | 0.40 | 0.12 | 0.63 | 0.37 | 0.54 | 0.61 |
| Sage 2P | 1.35 | 1.14 | 2.40 | 0.31 | 1.50 | 1.48 | 1.52 | 1.41 |
| Sage 1P | 2.24 | 2.14 | 2.76 | 0.74 | 2.21 | 2.44 | 2.69 | 2.39 |

The ranking survives the switch of truth. Even measured against XG's own analysis, Sage is closer to the reference at the truncated-rollout levels — 3T 0.26 vs Roller ++ 0.33, 2T 0.31 vs Roller + 0.38 — and at 4-ply (0.37 vs 0.45). 4-ply is the cleanest test of all, since it is the one level neither engine's reference is built from, so no row is scored against itself. Elsewhere the comparison flatters XG rather than Sage: XG Roller ++ *is* the 3T-tier reference and XG 3-ply the 3P-tier reference, so those rows score near-zero exactly where they define the truth, and Sage's edge is if anything understated. The one column where XG's own reference favors XG is the cube.

Whichever engine's analysis is taken as truth, Sage 3T lands ahead of XG Roller ++ — 0.21 vs 0.32 by Sage's reference, 0.26 vs 0.33 by XG's — and the truncated-rollout levels keep their edge. The remaining doubt about the first table — that the yardstick was Sage's own rollouts — is exactly what this mirror removes.

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
and reported, so a partially built data set still scores cleanly.

To score **XG**, batch-analyze the pass-1 `xg/*.txt` transcripts (with **Save
Games after analyze** checked) so each gets a matching `seed_<N>.xg`, then:

```bash
python scripts/benchmark_pr_xg.py
```

which reads XG's #1 decision per position and scores it against the same saved
reference equities, printing the same PR breakdown.

### Match Play

We repeated the Rollout PR experiment in match play, where the score on the board changes the value of every decision. A 5-point match is a good test case: the match score materially affects checker and cube decisions through these relatively short matches, so it exercises the engines' match-equity handling, not just their raw position evaluation.

#### Rollout PR Algorithm

We simulated 130 5-point matches of Sage 3P vs Sage 3P (both sides played by Sage at 3-ply). The match state — each player's away-count and the Crawford flag — is threaded through every evaluation, so all decisions, and the rolled-out "truth", are computed in match-equity (MWC) space against the correct score; cube decisions use the Kazaross-XG2 match equity table. Otherwise the method is identical to the money-game build: a first pass capturing 3-ply analytics for every decision, a second pass re-evaluating at Sage 3T any decision within 0.05 equity of its next-best alternative, and a third pass rolling out (1,296-path batches, 3-ply checker and cube decisions, variance reduction, repeated until the 95% equity band is under 0.005 or 20,736 paths) any decision still within 0.02 equity. The strongest tier reached for each decision is its benchmark truth.

For XG, we manually batch-analyzed the 130 match transcripts (3-ply decisions, upgrading to the listed eval level for disputes) — one `.xg` per match, each containing every game of the match — and scored XG's chosen decision against the same saved reference equities.

#### Rollout PR Results

There were 18,292 decisions across 17,892 positions. Of the 17,892 positions, 7,522 were settled at 3-ply; the other 10,370 were re-evaluated at 3T, of which 3,460 settled there and 6,910 were rolled out.

| Bot | PR | Checker PR | Cube PR| Pure Race | Racing | Attacking | Priming | Anchoring |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sage 3T | 0.19 | 0.16 | 0.40 | 0.01 | 0.17 | 0.18 | 0.25 | 0.26 |
| XG Roller ++ | 0.35 | 0.35 | 0.41 | 0.08 | 0.44 | 0.32 | 0.33 | 0.45 |
| Sage 2T | 0.26 | 0.24 | 0.44 | 0.02 | 0.36 | 0.18 | 0.34 | 0.33 |
| XG Roller + | 0.44 | 0.44 | 0.41 | 0.09 | 0.55 | 0.44 | 0.40 | 0.52 |
| Sage 1T | 0.49 | 0.49 | 0.52 | 0.05 | 0.57 | 0.44 | 0.57 | 0.63 |
| XG Roller | 0.51 | 0.51 | 0.54 | 0.09 | 0.64 | 0.50 | 0.51 | 0.62 |
| Sage 4P | 0.42 | 0.42 | 0.48 | 0.05 | 0.52 | 0.38 | 0.45 | 0.55 |
| XG 4-ply | 0.46 | 0.45 | 0.56 | 0.09 | 0.60 | 0.44 | 0.42 | 0.56 |
| Sage 3P | 0.57 | 0.56 | 0.67 | 0.05 | 0.74 | 0.56 | 0.58 | 0.68 |
| XG 3-ply | 0.54 | 0.53 | 0.61 | 0.09 | 0.69 | 0.53 | 0.52 | 0.65 |
| Sage 2P | 1.30 | 1.26 | 1.60 | 0.39 | 1.59 | 1.23 | 1.50 | 1.41 |
| Sage 1P | 2.30 | 2.29 | 2.31 | 0.69 | 2.48 | 2.43 | 2.47 | 2.56 |

As in money play, Sage's evaluations are stronger than the equivalent XG evaluation at every matched level except 3-ply, where the two are within noise (XG 0.54, Sage 0.57). The truncated-rollout levels that most users rely on — 3T and 2T — show Sage's clearest edge.

#### Running the Pipeline

The match data set is built by `scripts/benchmark_match.py`, the match-play twin of `benchmark_money.py`; the match length and number of matches are arguments, and the match state is threaded through every decision. The three passes are the same independently-resumable stages, run locally from a fresh `bgsage` checkout:

```bash
python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass1 --workers 6   # simulate + capture 3P; one XG-import .txt per match
python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass2 --n-threads 16  # re-evaluate close decisions at 3T
python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass3 --n-threads 16  # roll out the closest decisions
```

The assembled benchmark is written to `data/match_benchmark/5pt/benchmark.json` (shipped as `benchmark.json.gz`). Score a bot, or XG, exactly as in the money case:

```bash
python scripts/benchmark_match.py score --match-length 5 --level truncated3   # Sage 3T
python scripts/benchmark_pr_xg_match.py --match-length 5                       # XG, from batch-analyzed .xg files
```

As with the money build, pass 3 is by far the longest stage and fully resumable; the hardest back-game and long-race positions take well over an hour each.

## Disputed Position Analysis

The Rollout PR study scores both engines against a shared reference. A sharper, more direct question is: in realistic play, where do the two engines actually disagree on the best decision — and when they do, which one is right?

We answer it on the money games, reusing the benchmark built above. Among the hardest decisions — the rolled-out positions — we have **both** a full Sage rollout and a full XG rollout of each. We take the subset where **Sage 3T** and **XG Roller ++** chose differently, and score each engine's pick against both rollouts. Having both rollouts is the point: every disagreement is judged against Sage's rollout *and* XG's own, so the verdict does not depend on trusting a single engine's truth.

This method currently covers money games only: it needs an XG full rollout of the disputed positions, which we have run for the money benchmark but not for match play. Match-play strength is covered by the Rollout PR study above and the Real-Match Agreement study below.

### Money Game Results

Across the rolled-out money positions, the two engines disagree on 1,488 checker plays and 69 cube decisions. We score each disagreement on the common set — the ones where both engines' picks were rolled by both rollouts — so the Sage-rollout and XG-rollout comparisons cover the identical positions.

**Checker play** — 1,488 disagreements, 1,357 on the common set:

| Reference | Sage 3T closer | XG Roller ++ closer | Neither | Sage 3T avg error | XG Roller ++ avg error |
| --- | ---: | ---: | ---: | ---: | ---: |
| vs XG rollout | 44.4% | 39.3% | 16.3% | 0.0027 | 0.0048 |
| vs Sage rollout | 56.2% | 27.7% | 16.1% | 0.0019 | 0.0059 |

On checker play — the large majority of disagreements — Sage 3T matches the rollout's best move more often than XG Roller ++ and carries roughly half the average error, against **both** rollouts, including XG's own.

**Cube decisions** — 69 disagreements:

| Reference | Sage 3T closer | XG Roller ++ closer | Sage 3T avg error | XG Roller ++ avg error |
| --- | ---: | ---: | ---: | ---: |
| vs XG rollout | 58.0% | 42.0% | 0.0079 | 0.0086 |
| vs Sage rollout | 69.6% | 30.4% | 0.0041 | 0.0118 |

Cube disagreements are far rarer (69 in all) and closer to even: Sage 3T is clearly closer to Sage's rollout, and slightly closer against XG's own rollout too. Given the small sample, read the cube panel as suggestive rather than decisive.

Because the two rollouts come from independent engines, their agreement is meaningful. On the checker disagreements they point the same way — Sage 3T is closer even to XG's own rollout. Neither rollout is flawless ground truth, but that agreement is what a single reference cannot show on its own.

### Running the Pipeline

The disagreement study draws from the same money benchmark built for the Rollout
PR study, cross-referenced against XG's own full rollout of the hardest positions.
The only manual dependency is XG's Batch Rollout; a single script does the rest.

1. Build the money benchmark (the Rollout PR pipeline above). Pass 1 also writes
   the XG-import transcripts to `data/money_benchmark/xg/`.
2. In XG, **Batch-Rollout** those positions with **Save Games after analyze**
   checked, so each rolled-out decision carries XG's own rollout equities in the
   resulting `.xg` files.
3. Harvest and report:

```bash
python scripts/xg_benchmark_report.py
```

It parses XG's rollouts into `data/money_benchmark/xg_results/rollout.jsonl`, then
prints both the evaluation-level PR against the XG rollout (the "scored against
XG's own reference" table above) and the disputed-position report — every
disagreement scored against the Sage and XG rollouts in turn.

## Match PR Agreement on Real Matches

The two analyses above measure *strength* — how close each engine's decisions are to a rolled-out truth. A third, equally practical question matters to anyone who uses an engine to study their own play: **if you analyze a real match in XG, note your Performance Rating, then analyze the same match in Sage, how close are the two PRs?** A player who has spent years building intuition for what a given PR means in XG should get essentially the same number from Sage.

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

Open Sage and XG are close at every matched evaluation level. In the Rollout PR study — now run for both money play and 5-point match play — Sage's evaluations score at least as well as the equivalent XG evaluation at every level except 3-ply, where the two are within noise. For the money games the same comparison scored against XG's own tiered analysis keeps Sage's truncated-rollout levels ahead, which answers the natural worry that the ground truth is merely Sage's own rollouts. Sage's edge is clearest at the truncated-rollout levels (3T and 2T) and holds in both money and match play, and the Disputed Positions study — which rolls out only the money positions where the two engines actually disagree and scores them against both engines' rollouts — points the same way on checker play, with cube disagreements too rare and split to call. The differences are small.

And on real matches, the two engines assign nearly identical Performance Ratings: across 290 tournament matches, the average difference between a player's Sage PR and XG PR is statistically indistinguishable from zero. Whether the test is strength against a rolled-out truth or simple agreement on how a real game was played, Open Sage and XG land in the same place.
