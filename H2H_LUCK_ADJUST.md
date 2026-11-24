# Luck-adjusted head-to-head money games

## What the harness estimates

Fix two bot policies, A and B, including their checker play, cube decisions,
opening books and any policy randomness. Start each game from the standard
opening with a centered cube at 1. Fix the money-game rules for the run,
including Jacoby and the cube cap. A's expected net points per game against B
under those rules is $\mu$.

$$
\mu = \mathbb E_{A,B}[Y], \qquad
Y = \begin{cases}
+P & \text{if A wins},\\
-P & \text{if B wins}.
\end{cases}
$$

Here $P$ is the game's actual award in initial stake units. Bearing off awards
the current cube times the single, gammon or backgammon multiplier under the
configured rules. A cube pass awards the pre-offer cube to the doubler. Positive
$\mu$ means A has the edge. All games play to settlement under the two tested
policies. A separate reference evaluator supplies numbers for a control
variate. Its evaluations leave the moves, cube actions and settlement entirely
under the game policies and rules. Both the raw average and the adjusted average
estimate $\mu$. [Recorded game simulation](SIMULATION.md)

## Center each roll against its own alternatives

Let $H_t$ be the history immediately before chance event $t$. For an ordinary
roll this includes the board, who's on roll, the rules and the cube state
**after** resolving any offer and response. On double/take, use the accepted
cube value and the taker's ownership. On double/pass the game has already ended,
so there is no roll correction on that turn. The game retains all earlier roll
corrections. At the opening, $H_0$ precedes the contest that decides who starts.

For each possible roll $r$, define a score $q_t(H_t,r)$ in **points for A**.
The score approximates the expected final payoff after seeing that roll. The
same rule generates the full table of scores, including the entry for the
observed roll $R_t$. Let $p_t(r\mid H_t)$ be the actual conditional dice law.
Then the mean score, roll correction and game correction are

$$
\bar q_t(H_t) = \sum_r p_t(r\mid H_t)\,q_t(H_t,r), \qquad
D_t = q_t(H_t,R_t) - \bar q_t(H_t), \qquad
L = \sum_{t\ \text{reached}} D_t.
$$

**Ordinary rolls.** With independent fair dice and a score that is invariant to
die order, use the 21 unordered pairs. Each double has probability $1/36$ and
each non-double $2/36$. Let $c_t$ be the checker-time cube value and let
$e_t(r)$ be the reference continuation equity from the roller's perspective,
normalized to that cube. The score in the equation above is

$$
q_t(H_t,r) = s_t\,c_t\,e_t(r), \qquad
s_t = \begin{cases}+1 & \text{A rolls},\\ -1 & \text{B rolls}.\end{cases}
$$

Use the cube at that turn, even if the game later ends at a higher cube. For
example, a roll worth $+0.15$ above the mean for B at cube 4 contributes
$-0.60$ points to A's $L$. For an evaluator that already returns points, apply
only the perspective conversion. Average the cubeful scores for the separate
rolls because converting mean outcome probabilities to cubeful equity can give
a different result. [Engine rollout equity conventions](ROLLOUT.md#4-variance-reduction-vr)

**Opening contest.** With ties rerolled and automatic doubles disabled, the
first unequal ordered pair $(d_A,d_B)$ has 30 equally likely outcomes. The
winner plays both dice and there is no opening cube offer. Evaluate the opening
board from the winner's perspective and convert to A's perspective as above,
with $c_0=1$. The baseline averages both possible openers.

$$
\bar q_0 = \frac{1}{30}\sum_{d_A\ne d_B}q_0(d_A,d_B).
$$

Equivalently, average the 15 non-doubles for each opener and give each opener
weight $1/2$. Centering over only the observed winner's 15 rolls is also
mean-zero conditional on that winner. It leaves the random choice of opener
uncorrected. The proposed default is the joint opening correction so the
control can account for both parts of opening luck. If a future dice schedule
changes the sampling law, its centering and inference must follow that design.

For a fixed start with a designated first player, as in Paskogammon, the
first roll uses the ordinary 36-outcome baseline, including doubles. Resolve
any legal cube offer first and apply the same checker-time cube scaling as on
later rolls. When first-player assignments are fixed by design, estimate the
standard error from the separate opener groups and weight their means equally
for a balanced edge.

## Choose the score for a roll

For the first harness, use a fixed, inexpensive cubeful reference policy and
evaluator. For every possible roll, use the reference's preferred legal
checker move and value its continuation in the checker-time cube context. A
forced checker pass advances the turn with the same board. A terminal checker
move uses the exact award under the game rules, expressed in the evaluator's
perspective and units before conversion to A's points. Use a consistent turn
convention at nonterminal endpoints. If the evaluator scores
the next player before their roll, flip the board and cube ownership for that
call, then negate the value to express it for the mover. The resulting table
is usable even when the tested bot played a different move on the sampled roll.

An alternative is to ask the actual checker policy what it would play for
**every** roll, then score each continuation with the reference evaluator.
This can better match the game being measured, at the cost of extra policy
calls. For a stochastic checker policy, a roll score can average over its
action distribution or use a fixed reference rule. In all cases, use the same
table entry for the observed roll and for that roll's contribution to the mean.
Using the actual bot move only in the observed term adds
$\sum_r p_t(r\mid H_t)[q_{\mathrm{played}}(r)-q_{\mathrm{ref}}(r)]$ to the
conditional expected correction. That term can be nonzero even with the same
static evaluator on both sides. This risk applies to rollout paths that use a
deeper played move with a 1-ply reference baseline. [Trial move and baseline selection](ROLLOUT.md#8-checker-selection-during-trials)

`luck_from_equities` can do the weighted subtraction on a complete, validated
ordinary-roll table or one opener's non-double table. The joint opening table
needs both openers' scores. If using the analyzer's no-double per-roll details,
request them for the checker-time cube and verify that all expected rolls are
present with the right weights. Renormalizing an incomplete table can break
centering. A fixed skip rule based on the pre-roll history can safely assign
zero correction to costly or unsupported states. Skipping only when the
sampled roll fails can introduce bias. [Existing luck helper](python/bgsage/luck.py)

## Why the edge stays the same

Conditional on reaching a roll and on $H_t$, the reference table is fixed and
the observed dice have law $p_t$. Thus

$$
\mathbb E[D_t\mid H_t] = \bar q_t(H_t)-\bar q_t(H_t)=0.
$$

Set $D_t=0$ when the game has already ended. Reaching roll $t$ is decided
before its dice are seen, so these terms are martingale differences even though
earlier rolls change which boards, cube values and future rolls are reached.
With integrable accumulated corrections, for example with bounded scores and
a finite expected number of rolls, summing gives $\mathbb E[L]=0$. For any
fixed coefficient $\beta$, the adjusted outcome is

$$
Z = Y-\beta L, \qquad \mathbb E[Z]=\mathbb E[Y]=\mu.
$$

The same argument permits a reference or coefficient chosen from past data,
provided it is fixed before the dice it scores. Scoring a saved transcript
is valid if each table uses only the relevant pre-roll history and a fixed
mapping over candidate rolls. Selecting the table from the played move or a
later outcome can break the argument. Keep each $Z$ as computed, including
values outside the legal range of game awards. Clipping changes the
expectation. Report limits or failures as incomplete runs, since dropping games
based on their length or outcome can change the target. [Backgammon rollout derivation](https://www.bkgm.com/articles/GOL/Feb00/var.htm) · [General chance-node estimator](https://cdn.aaai.org/AAAI/2006/AAAI06-092.pdf)

## When the standard error falls

For finite second moments,

$$
\operatorname{Var}(Z)=\operatorname{Var}(Y)
+\beta^2\operatorname{Var}(L)-2\beta\operatorname{Cov}(Y,L).
$$

Start with $\beta=1$ because $L$ is already in points. This reduces variance
when $2\operatorname{Cov}(Y,L)>\operatorname{Var}(L)$. Unbiasedness holds even if
the reference is weak, whereas the gain depends on how well its corrections
track the fluctuations in actual payoffs. If $\operatorname{Var}(L)>0$, the
best fixed coefficient and its residual variance are

$$
\beta^*=\frac{\operatorname{Cov}(Y,L)}{\operatorname{Var}(L)}, \qquad
\operatorname{Var}(Y-\beta^*L)=\operatorname{Var}(Y)\bigl(1-\rho_{Y,L}^2\bigr),
$$

where $\rho_{Y,L}$ is the population correlation (assuming $Y$ also has
nonzero variance). Use $\beta=0$ if the correction is identically zero. To tune
$\beta$ or pick a reference, use a separate pilot and freeze the choice for the
measurement run. This keeps the coefficient independent of the evaluation
sample and avoids reporting a gain fitted to the same games.

In the ideal case $q_t(H_t,r)=\mathbb E[Y\mid H_t,R_t=r]$ under the actual
pairing. Then each $D_t$ is the change in conditional expected payoff caused
by the dice. With deterministic policies, a fixed initial setup and every dice
event included, the deltas telescope to $Y-\mu$. The corrected payoff with
$\beta=1$ is then constant. With approximate scores, $Z-\mu$ is the sum of the
errors in those deltas. The errors are themselves centered and orthogonal
across rolls, so their expected squared sizes add. This explains why the
reference's **differences across rolls** matter and why adding a constant to
every score at a state has no effect. Bot randomness introduces additional
chance events that a dice-only correction leaves in the residual. [Exact and approximate continuation values](https://cdn.aaai.org/AAAI/2006/AAAI06-092.pdf)

For a small exact check, consider a dead cube at 4 with A on roll and one
checker on the six-point. B has one checker on the one-point and wins on the
next turn if A fails to bear off. Both players have already borne off 14
checkers. A wins on 27 of 36 ordered rolls. The exact scores are $+4$ on those
rolls and $-4$ otherwise, so $\bar q=+2$. The correction is $+2$ on a win and
$-6$ on a loss, giving $Z=+2$ either way. The raw variance is 12 points squared.

## Estimate and report the gain

Use independently seeded games for the first comparison. The two means and
their sample standard errors use the same $n$ games.

$$
\hat\mu_{\mathrm{raw}}=\bar Y, \qquad
\hat\mu_{\mathrm{adj}}=\bar Z, \qquad
\widehat{\mathrm{SE}}_{\mathrm{raw}}=s_Y/\sqrt n, \qquad
\widehat{\mathrm{SE}}_{\mathrm{adj}}=s_Z/\sqrt n.
$$

Here $s_X^2=\sum_i(X_i-\bar X)^2/(n-1)$.

### Formula for the change in standard error

Let $s_L^2$ be the sample variance of total game luck and let
$s_{Y,L}=\sum_i(Y_i-\bar Y)(L_i-\bar L)/(n-1)$ be the sample covariance of
settlement with luck. For the fixed $\beta$ used in the run, the standard
error of the estimated average edge is

$$
\boxed{\displaystyle
\widehat{\mathrm{SE}}_{\mathrm{adj}}
=\sqrt{\frac{s_Y^2+\beta^2 s_L^2-2\beta s_{Y,L}}{n}}.}
$$

This equals $s_Z/\sqrt n$. For $s_Y^2>0$, the adjusted SE as a fraction of
the raw SE and the estimated percentage reduction are

$$
f_{\mathrm{SE}}
=\frac{\widehat{\mathrm{SE}}_{\mathrm{adj}}}
       {\widehat{\mathrm{SE}}_{\mathrm{raw}}}
=\sqrt{1+\frac{\beta^2 s_L^2-2\beta s_{Y,L}}{s_Y^2}},
\qquad
\text{SE reduction in percent}=100(1-f_{\mathrm{SE}}).
$$

The absolute gain is $\widehat{\mathrm{SE}}_{\mathrm{raw}}(1-f_{\mathrm{SE}})$
points per game. A negative gain means the adjustment increased the SE. For
example, with 10,000 games, $\beta=1$, $s_Y=3.0$, $s_L=2.5$ and $s_{Y,L}=6.5$,
the raw SE is $0.030$ points per game and the adjusted SE is $0.015$. That is
a 50% reduction in SE and a fourfold reduction in variance. These inputs
illustrate the calculation. Measurements from a separate pilot can supply the
variance and covariance terms to project the SE at a planned game count.

### Confidence intervals and compute cost

Report approximate 95% intervals as the mean plus or minus $1.96$ times the
sample SE. Use a fixed measurement sample size.
If future runs share dice across seat-swapped games, score both for A, average
within each pair, and apply these formulas to the independent pair means. Use
the same pair structure to compare raw and adjusted results. Games within a
pair can be correlated even when both marginal dice laws are fair.

Report the sample variance ratio $\widehat R=s_Y^2/s_Z^2$ alongside both SEs.
At the same game count the estimated SE is divided by $\sqrt{\widehat R}$.
For a fixed precision, a comparable adjusted run would need roughly
$1/\widehat R$ as many games. A variance ratio of 4 would halve the SE, or
quarter the required game count. If raw and adjusted runs take $t_{\mathrm{raw}}$
and $t_{\mathrm{adj}}$ seconds per game on the same compute setup, the estimated
speedup to a fixed precision is

$$
\widehat R\,\frac{t_{\mathrm{raw}}}{t_{\mathrm{adj}}}.
$$

Include the time to build the dice tables. A variance ratio above 1 can still
cost more compute if the per-game overhead exceeds the statistical gain.

## Run Sage vs Sage

From a standalone checkout, run `python -m pip install .` to build and install
the engine with its weights and bearoff database. A source checkout can also use
a current native build by putting its directory and `python/` on `PYTHONPATH`.
Run the harness in that environment:

```sh
python -m bgsage.h2h --a 1P --b 2P --games 100 --seed 42 > h2h.json
```

Both players accept `1P`–`4P`, `1T`–`3T` and `R` for full rollout. By default,
the runner uses one game process per available logical CPU, capped by the
game count and by Python's 61-worker limit on Windows. Each worker runs one
game at a time with single-threaded Sage analyzers, including rollout prefilters.
Workers take another game as soon as they finish, and the report preserves
game order by seed sequence. Set `--workers N` to select a worker count or
`--workers 1` for a serial run. When using `run_head_to_head` in a Python script,
call it under `if __name__ == "__main__":` for multiprocessing.

Results are always in A's points. The defaults use Jacoby, a cube cap of 1024
and a fixed $\beta=1$. Set `--no-jacoby`,
`--max-cube-value` or `--beta` to change those settings. The rules are those
of the standard-opening `bgsage.simulate_game` API. [Simulation rules](SIMULATION.md)

The scorer obtains a complete no-double dice table from a separate Sage
2-ply cube analysis, which returns a 1-ply continuation value for each dice
outcome. It reads the board, roller and checker-time cube from the simulator's
transcript. Keep the same run seed to replay the same per-game seed sequence.
The JSON report contains the game seeds, eval levels, production model,
native binary hash, rules, reference, $\beta$ and per-game raw and adjusted
points for verification. `summary` includes the two edge estimates, SEs,
approximate intervals, luck statistics, the SE reduction and the variance
ratio. A ratio with a zero denominator is reported as `null`.

The report's equal-SE speedup uses the summed CPU time for game play and
luck scoring. Overall wall time includes process startup and loading the
analyzers. The CPU estimate reflects the recurring cost on this build and
the observed sample variances. [Simulation API and transcript format](SIMULATION.md)

## Run Sage vs GNUbg

GNU Backgammon runs as a separately installed CLI. Select it for either seat
with `--bot-a gnubg` or `--bot-b gnubg`, and set that seat's level with `--a`
or `--b`. The levels use the same convention as Sage, so `1P`, `2P`, `3P` and
`4P` request GNUbg's native `0-ply`, `1-ply`, `2-ply` and `3-ply` respectively.
GNUbg applies its own move filters at the requested search depth.

```sh
python -m bgsage.h2h --a 2P --bot-b gnubg --b 2P --games 20 --seed 42 --workers 4
```

The runner looks for `gnubg-cli.exe` on Windows and `gnubg` on Unix, including
the standard installation locations. Set `GNUBG_CLI` or pass `--gnubg-cli`
with the executable path to choose a specific installation. Results include
GNUbg's version, executable hash and configured native depth.

GNUbg supplies its own checker and cube recommendations from temporary CLI
command files. The simulator supplies the dice, checks move legality and
settles the game. At the cube cap the adapter requests a dead-cube checker
evaluation with gammons counting. Before the cap, GNUbg uses its own cube
model for lookahead. The harness always scores luck for both players with the
same fixed Sage reference, regardless of the engines or playing levels selected.

For external-engine games, `estimated_worker_time_speedup_at_equal_se` includes
the time spent waiting on the GNUbg subprocess. The worker CPU counters cover
only the H2H worker processes, and the CPU speedup field is `null` for these
games. GNUbg itself is an optional system installation. [GNU Backgammon manual](https://www.gnu.org/software/gnubg/manual/gnubg.html)

## Run Sage vs XG

On Windows, the runner sends batches of checker and cube requests to a
dedicated eXtreme Gammon 2 instance. Select XG for either seat with
`--bot-a xg` or `--bot-b xg`. Its level uses the Sage convention. `1P` through
`4P` select the corresponding XG ply, and `1T` through `3T` select Roller,
Roller+ and Roller++. Set `--xg-profile` to the exact name of an existing
custom profile configured for the requested level on both checker and cube
decisions. If exactly one custom profile matches, its name is optional.

```powershell
python -m bgsage.h2h --a 1P --bot-b xg --b 1P --xg-profile "Custom Setting 1-ply" --games 20 --workers 8 --max-cube-value 64
```

The adapter starts XG from the standard installation path or `--xg-exe`.
Close any open XG session before running it. Use XG's English Batch Analyze
dialogs. The adapter temporarily selects the profile and enables the opening
book. It verifies live checker and cube depth, then restores the prior profile
selections and opening-book setting when the dedicated process closes. Opening
book decisions and XG's own early exits can report a different per-decision
depth. The report records the exact profile definition and executable hash.

The runner uses a local `.xg` game as a record template. Pass a completed
game containing checker and cube records with `--xg-template`. In a source
checkout the existing `data/money_benchmark/xg/seed_2.xg` is used when available.
The runner generates separate temporary batch files from that template.

The XG path uses standard unlimited play with Jacoby on, no beavers, a
standard opening contest and the selected cube cap in both engines. XG makes
its own checker and cube decisions. The shared simulator checks and settles
each game, and Sage scores luck for both players. `--workers` controls game
streams sharing one XG Batch Analyze instance. The JSON includes the same raw
and adjusted edge, SEs and confidence intervals, plus XG batch count and time.
The worker-time counters include waits for the shared XG instance.

## Validate the estimator

- Enumerate the 36 ordinary ordered outcomes and 30 decisive opening outcomes
  and verify probability mass one and a weighted correction of zero, including
  when the two openers have different value tables.
- Exercise A/B sign conversion, post-take cube value and ownership, double/pass,
  a forced checker pass and exact single, gammon and backgammon settlement
  with the applicable Jacoby state. Verify the dead-cube example above.
- Run identical-bot self-play, where symmetry gives an edge of zero, and a
  representative unequal pairing. Check that $\bar L$ is consistent with zero
  within its sampling uncertainty across independent runs. Compare the raw and
  adjusted SEs and report wall-clock efficiency on held-out games. The size of
  the gain is a property of the pairing and reference, to be measured by the
  harness.

### Repeated Sage 1P self-play batches

On 2026-11-23, 64 batches of 256 games used distinct dice seeds and the fixed
2P reference with $\beta=1$. Each batch supplied a raw and an adjusted edge
estimate. The sample standard deviation **across the 64 batch means** fell
from 0.17566 to 0.03170 points per game, an 82.0% reduction. Resampling whole
matched batches gave a 95% bootstrap interval of 76.9% to 86.0% for the
reduction. For comparison, the pooled game-level variances predict SEs of 0.20600
and 0.02902 for a 256-game batch. The adjusted grand mean was −0.00084 points
per game, with an across-batch SE of 0.00396. The run used 32 game processes and
completed 16,384 games in 66.6 seconds on that machine.

### Sage 1P vs 3P paired mean-shift check

On 2026-11-23, 8,192 games in 32 seed batches used Sage 1P for A, Sage 3P for
B and the fixed Sage 2P reference with $\beta=1$. A separate 256-game pilot
estimated the per-game SD of the shift for sample planning. The main
test used the adjusted-minus-raw payoff from each of the same 8,192 games.

| Estimate | Points per game for Sage 1P | SE | 95% interval |
| --- | ---: | ---: | ---: |
| Raw | −0.07349 | 0.03433 | [−0.14078, −0.00619] |
| Luck-adjusted | −0.11770 | 0.00498 | [−0.12746, −0.10795] |
| Paired shift | −0.04422 | 0.03339 | [−0.10965, +0.02122] |

The two-sided test of zero mean shift gave $p=0.185$. Using the observed
variance, the design had about 80% power for an absolute shift of 0.094 points
per game, and 85% power at the prespecified 0.10 margin. The game-level 90%
interval cleared the ±0.10 equivalence margin by less than 0.001. The separate
32-batch SE check was wider (0.03691) and did not clear that margin. The
adjustment reduced the estimated edge SE by 85.5%.

An exhaustive dice audit of 16 pilot games reproduced zero mean luck across
the 30 opening outcomes and all 36 outcomes at each of 593 later roll states.
The largest absolute residual was $4\times10^{-16}$ points. Luck scoring
used 1.1% of the total measured CPU time in the main experiment.

### Sage 1P vs GNUbg native 0-ply

On 2026-11-23, a fresh 1,000-game cohort (seed `2026112332`) gave Sage an
adjusted edge of +0.02873 points per game with SE 0.01409 and 95% interval
[+0.00112, +0.05634]. The raw edge was +0.03600 with SE 0.09182. The cohort
retained 481 games from the 30-stream phase and completed the remaining 519
with up to 80 streams across two process pools. All game seeds were distinct.

### Sage 3P vs XG 3-ply

On 2026-11-23, 1,000 games (seed `2026112342`) used the `Sage3P` XG checker
and cube profile, XG's opening book, Jacoby, a cube cap of 64 and the shared
Sage 2P luck reference. Results are in points per game for Sage.

| Estimate | Edge | SE | 95% interval |
| --- | ---: | ---: | ---: |
| Raw | −0.11000 | 0.08415 | [−0.27493, +0.05493] |
| Luck-adjusted | −0.00105 | 0.01060 | [−0.02182, +0.01972] |

The estimated SE fell by 87.4% on the same games. The run took 26m 33s with
80 game streams, 12 Sage compute workers and one XG session at a time. The
[full report](data/h2h_benchmark/sage_3p_vs_xg_3ply_1000_20261123.json)
records the native build hash, XG profile, seed and all 1,000 game outcomes.
With the Windows XG setup above and Sage available to Python, run
`python scripts/benchmark_h2h_sage3p_xg3p.py` from the repo root to replay the
cohort. It checkpoints completed seeds under `logs/sage-3p-vs-xg-3p-1000-replay/`.

### Sage 1T vs XG Roller

Both runs used Sage 2P to score luck, XG Roller for checker and cube play,
XG's opening book, Jacoby and a cube cap of 64. Results are in points per game
for Sage.

| Starting position | Estimate | Edge | SE | 95% interval |
| --- | --- | ---: | ---: | ---: |
| Standard | Raw | +0.14600 | 0.09177 | [−0.03386, +0.32586] |
| Standard | Luck-adjusted | +0.00171 | 0.01211 | [−0.02203, +0.02545] |
| Paskogammon | Raw | −0.12080 | 0.04440 | [−0.20782, −0.03378] |
| Paskogammon | Luck-adjusted | −0.01714 | 0.00723 | [−0.03132, −0.00297] |

The standard cohort completed 1,000 games in 43m 24s on 2026-11-23. It used
seed `2026112343` and reduced SE by 86.8%. The archive contains the game
records, engine identity and original Windows run driver.
[Standard cohort archive](data/h2h_benchmark/sage_1t_vs_xg_roller_1000_20261123.zip)

The Paskogammon cohort completed 5,000 games in 3h 13m 10s on 2026-11-24.
It used seed `2026112344` and reduced SE by 83.7%. The scattered side was on
roll from the same asymmetric start in every game. The engines alternated
that seat, with 2,500 starts each. There were 876 opening doubles. Compute
the overall SE as half the square root of the sum of the squared SEs from
the two opener groups. The archive includes the game records, starting
position, original Windows run driver and local verification script.
[Paskogammon cohort archive](data/h2h_benchmark/paskogammon_sage_1t_vs_xg_roller_5000_20261124.zip)
