# Rollout and Truncated Rollout Evaluation

Technical specification for the Monte Carlo rollout algorithms used for position
evaluation and doubling cube decisions in backgammon. This document describes the
mathematics, data structures, variance reduction, caching, parallelization, and
implementation details at a level sufficient for a complete reimplementation.

This document assumes familiarity with the concepts in `MULTI-PLY.md`, particularly
Janowski interpolation, cubeful equity, the evaluate-all-and-decide N-ply recursion,
and the 5-probability NN output format.

## Table of Contents

1. [Goal](#1-goal)
2. [Overview: Full vs Truncated Rollout](#2-overview-full-vs-truncated-rollout)
3. [Stratified Dice Generation](#3-stratified-dice-generation)
4. [Variance Reduction (VR)](#4-variance-reduction-vr)
5. [The Unified Trial Function](#5-the-unified-trial-function)
6. [Position Rollouts (Cubeless and Cubeful)](#6-position-rollouts-cubeless-and-cubeful)
7. [Cube Decision Rollout](#7-cube-decision-rollout)
8. [Move Selection Strategies During Trials](#8-move-selection-strategies-during-trials)
9. [Cube Decision Strategies During Trials](#9-cube-decision-strategies-during-trials)
10. [Move Caches (Move0 and Move1)](#10-move-caches-move0-and-move1)
11. [Truncation Evaluation](#11-truncation-evaluation)
12. [Parallelization](#12-parallelization)
13. [Statistical Aggregation](#13-statistical-aggregation)
14. [Performance Optimizations](#14-performance-optimizations)
15. [Match Play](#15-match-play)
16. [Best Move Selection via Rollout](#16-best-move-selection-via-rollout)
17. [Configuration Reference](#17-configuration-reference)

---

## 1. Goal

Rollout evaluation produces more accurate equity estimates than N-ply search alone
by simulating many trial games from a given position. Each trial plays the game
forward using quasi-random dice, with configurable strategy strength for move
selection and cube decisions. The results are aggregated with variance reduction
to produce:

- **Cubeless probabilities**: The 5 standard NN outputs (P(win), P(gw), P(bw),
  P(gl), P(bl)) estimated via Monte Carlo simulation.
- **Cubeless equity**: Derived from the mean probabilities.
- **Cubeful equity (single cube state)**: For checker play with a known cube
  state, a single-branch cubeful rollout produces a VR-adjusted cubeful
  equity from the same trials that produce the cubeless probs.
- **Cubeful ND / DT equities**: For cube decisions, two branches (No-Double and
  Double/Take) are simulated simultaneously from a pre-roll board, producing
  ND and DT cubeful equities with standard errors.

Rollouts are used for three purposes:

1. **Cubeless position evaluation**: Given a post-move board, estimate its
   cubeless probabilities and equity (used by the `Strategy` interface).
2. **Cubeful position evaluation**: Given a post-move board and a cube state,
   estimate the cubeful equity (with SE) along with the cubeless probabilities.
   Used by checker-play analytics so each candidate's cubeful equity is
   computed natively from rollout trial paths instead of post-hoc Janowski
   conversion of the rollout's cubeless probs.
3. **Cube decision evaluation**: Given a pre-roll board and cube state,
   estimate ND and DT cubeful equities for optimal cube action determination.

## 2. Overview: Full vs Truncated Rollout

### Full Rollout

Each trial plays the game to completion (until one side bears off all checkers).
The terminal outcome (single win, gammon, backgammon) is the raw result.

- `truncation_depth = 0` (play to completion)
- Typical configuration: 1,296 trials (= 36^2) for full stratification of the
  first two dice rolls
- With variance reduction, standard errors are typically 0.001-0.005 equity

### Truncated Rollout

Each trial plays forward a fixed number of half-moves, then evaluates the
resulting position with a neural network (at configurable ply depth). This is
faster than full rollout because games are cut short, but less accurate because
the truncation evaluation introduces NN bias.

- `truncation_depth > 0` (e.g., 5 or 7 half-moves)
- Typical configuration: 72-360 trials
- The truncation evaluation can use 1-ply, N-ply, or even a nested truncated
  rollout

### XG Roller Equivalences

| Level | n_trials | truncation_depth | decision_ply | late_ply | late_threshold |
|-------|----------|-------------------|-------------|----------|----------------|
| XG Roller (1T)    | 42  | 5 | 1 | -1 | 20 |
| XG Roller+ (2T)   | 360 | 7 | 2 | 1  | 2  |
| XG Roller++ (3T)  | 360 | 7 | 3 | 2  | 2  |
| Full Rollout (R)  | 1,296 | 0 | 1 | -1 | 20 |

These are the params to **replicate XG** exactly. Sage's own app levels diverge from
these — see Standard Configurations (App Levels) below. `truncated1` (1T) uses **72**
trials (2×36), not 42 (42 isn't a multiple of 36 and mis-weights the first roll).
`truncated2` (2T) uses `late_threshold=1`, `ultra_late_threshold=9999`, 2-ply cube
throughout, and a 2-ply truncation eval (it beats XG Roller+: benchmark PR 0.89 → 0.36).

> **3T note:** 3T also uses `ultra_late_threshold=9999` (3-ply early, then 2-ply for
> the rest of each trial — no 1-ply drop), which this 5-column table can't show; see
> the "Standard Configurations (App Levels)" table below for the full per-level row.

## 3. Stratified Dice Generation

Rollouts use quasi-random dice sequences rather than purely random dice. This
ensures that all 36 possible dice outcomes are represented equally at the first
roll, reducing variance without introducing bias.

### Hierarchical Permutation Array (GNUbg-style)

The dice generation uses a hierarchical permutation array with 6 levels, 128
turns, and 36 permutations per entry. This provides joint stratification across
multiple rolls.

**Structure:**
```
PerArray.perm[level][turn][index] -> dice_index (0..35)
```

Each entry is a Fisher-Yates shuffle of [0..35], generated from a seeded PRNG
(Mersenne Twister).

**Initialization:**
```
for each level i in 0..5:
    for each turn j in i..127:
        perm[i][j] = Fisher-Yates shuffle of [0..35] using MT19937(seed)
```

### Dice Sequence Generation

For each trial `t` and each half-move `m`:

**Quasi-random (m < 128):**
```
j = 0
k = 1  (= 36^0)
for i = 0 to min(5, m):
    j = perm[i][m][((t / k) + j) mod 36]
    k *= 36

die1 = j / 6 + 1
die2 = j % 6 + 1
```

This composition of hierarchical permutations ensures that:
- At level 0 (first roll): for every 36 consecutive trials, each of the 36 dice
  outcomes appears exactly once. With 1,296 trials (= 36^2), the first two rolls
  are jointly stratified — every pair of (roll0, roll1) appears exactly once.
- At deeper levels: the permutation composition provides quasi-random coverage
  while avoiding systematic correlations.

**Truly random (m >= 128):**

For moves beyond 128 half-moves, a per-trial PRNG generates uniform random dice:
```
trial_rng = MT19937(seed + t * 1000003 + 7)
die1 = uniform(1, 6)
die2 = uniform(1, 6)
```

This boundary is generous — real games rarely exceed 128 half-moves (64 full turns).

### Stratification and VR Interaction

When `n_trials % 36 == 0`, the first roll is perfectly stratified across all
trials. This means the VR luck at move 0 sums to exactly zero over all trials,
so VR computation is skipped at move 0 (a free optimization).

## 4. Variance Reduction (VR)

Variance reduction dramatically reduces the noise in rollout estimates by
tracking the "luck" component at each half-move and subtracting it from the
final result.

### Core Concept

At each half-move, the actual dice roll may be luckier or unluckier than average.
VR measures this luck and accumulates it. The final result is:

```
VR_result = raw_result - accumulated_luck
```

Since E[luck] = 0 over many trials, VR doesn't bias the estimate — it just
reduces variance by removing the known-random component.

### Per-Move VR Computation

At each half-move, VR computes:

**1. VR Mean (expected value over all 21 rolls):**

For each of the 21 possible dice outcomes, find the best move and evaluate the
resulting position at 1-ply. The weighted average is the expected value:

```
for each of 21 rolls (d1, d2, weight):
    best_probs[i] = evaluate_best_move_probs(board, d1, d2, base_strategy)

mean_probs[k] = sum(weight[i] * best_probs[i][k]) / 36.0   for k in 0..4
mean_equity = compute_equity(mean_probs)
```

**2. Actual value (the roll that was actually played):**

The position after the actual chosen move, evaluated at 1-ply:

```
actual_probs = base_strategy.evaluate_probs(chosen_board, board)
actual_equity = compute_equity(actual_probs)
```

When the decision strategy is 1-ply (base), the actual probs can be reused
directly from the VR mean computation (the roll's best-move probs). When the
decision strategy is N-ply, the chosen move may differ from the 1-ply best, so
a separate 1-ply evaluation of the chosen board is needed.

**3. Luck (actual - mean):**

```
luck_probs[k] = actual_probs[k] - mean_probs[k]    for k in 0..4
luck_equity = actual_equity - mean_equity
```

**4. Accumulation (perspective-aware):**

Luck is accumulated from the **starting player's (SP)** perspective. When it's
SP's turn, luck is added directly. When it's the opponent's turn, luck is
negated and probability components are cross-mapped:

```
if is_sp_turn:
    accumulated_luck[k] += luck_probs[k]
    scalar_luck += luck_equity
else:
    // Cross-map: opponent's P(win) becomes SP's P(lose), etc.
    accumulated_luck[0] -= luck_probs[0]       // P(win) -> -P(win)
    accumulated_luck[1] += luck_probs[3]       // P(gl) -> P(gw) for SP
    accumulated_luck[2] += luck_probs[4]       // P(bl) -> P(bw) for SP
    accumulated_luck[3] += luck_probs[1]       // P(gw) -> P(gl) for SP
    accumulated_luck[4] += luck_probs[2]       // P(bw) -> P(bl) for SP
    scalar_luck -= luck_equity
```

### VR Decoupling from Decision Strategy

VR always uses 1-ply for both the mean and actual evaluations, regardless of
the decision strategy's ply level. This is critical:

- Move selection during trials may use 2-ply, 3-ply, or truncated rollout
  evaluation to pick the best move.
- But VR measures luck = (actual_1ply - mean_1ply), where both sides use the
  same 1-ply evaluator. Biases cancel because both the mean and actual are
  evaluated at the same depth.
- This eliminates ~90% of the N-ply evaluations that would be needed if VR used
  the decision strategy's ply level.

When the bearoff database is loaded, the 1-ply evaluator used for VR is the
DB-aware variant: at any bearoff position encountered during VR mean or VR
actual evaluation, the exact DB probabilities are used in place of the NN
output. For the luck difference to remain unbiased, the VR mean's per-roll
baseline move must be the same move the trial actually plays — sharing the
evaluator for the probabilities is not sufficient. In the bearoff range the
decision strategy plays the DB-optimal move (an N-ply strategy short-circuits
its leaves to the DB), so the VR mean ranks its candidates by exact DB equity as
well (see §14), keeping both the selected move and its probabilities consistent
between mean and actual. Crucially, when the per-trial
trajectory enters the bearoff range and all reachable post-move states share
the same DB cubeless probabilities (e.g. a "saved-gammon" state where the
loser already has at least one checker borne off), VR luck at every move from
that point on is deterministically zero — not just zero in expectation. This
is what lets stratified-dice rollouts converge to the exact value at low trial
counts on positions whose outcome becomes deterministic after a few half-moves.

### VR Result Construction

At the end of a trial (terminal or truncation), the raw result is converted to
SP perspective and VR-corrected:

```
// Convert raw probs to SP perspective
if last_mover_is_sp:
    sp_probs = raw_probs
else:
    sp_probs = invert_probs(raw_probs)

raw_equity = compute_equity(sp_probs)

// VR correction
vr_probs[k] = sp_probs[k] - accumulated_luck[k]    for k in 0..4
vr_equity = raw_equity - scalar_luck
```

The per-trial VR-corrected probs and equity are returned as the trial result.

### Cubeful VR

When the trial carries any active cube branches (single-branch cubeful position
evaluation, §7, or two-branch cube decision evaluation, §8), each branch tracks
its own VR luck in cubeful basis-cube SP-perspective value space:

```
// For each active branch b:
if is_match:
    actual_val = cl2cf_match(actual_probs, branch.cube, cube_x)
else:
    actual_val = cl2cf_money(actual_probs, branch.cube.owner, cube_x, ...) *
                 branch.cube.cube_value / branch.basis_cube

mean_cf = weighted_average_over_21_rolls(cl2cf(roll_best_probs[i], branch.cube, ...))
luck_cf = actual_val - mean_cf

if is_sp_turn:  branch.vr_luck += luck_cf
else:           branch.vr_luck -= luck_cf
```

The cubeful VR always uses 1-ply probs (from the cubeless VR computation) with
Janowski interpolation for the cubeful conversion, regardless of the cube decision
strategy used during the trial. `basis_cube` is set to the input cube value at
trial setup so all per-trial cubeful equities are normalized to per-basis-cube
units (i.e. the value reported has the same scale as a cube=1 equity).

### VR Speed Optimizations

**Thinned VR:** At ultra-late moves (>= `ultra_late_threshold`), VR is computed
only at even half-moves. Odd ultra-late moves skip VR entirely. Since E[luck] = 0,
this doesn't bias the estimate — just increases variance slightly.

**Stratification skip:** When `n_trials % 36 == 0`, VR is skipped at move 0
because the stratified first roll ensures luck sums to exactly zero.

**1-ply reuse:** When the decision strategy is 1-ply (base), the VR computation's
best-move probs are reused directly for move selection — zero additional NN
evaluations for VR.

## 5. The Unified Trial Function

A single function, `run_trial_unified`, handles all three rollout modes
(cubeless position, cubeful position, cubeful cube decision). This eliminates
code duplication and ensures that cubeful overhead is zero when all branches
have dead cubes.

### Signature

```
TrialResult run_trial_unified(
    board,              // Starting position
    start_post_move,    // true = post-move (opponent first), false = pre-roll (SP first)
    branches[],         // Array of CubefulBranch (or null for cubeless)
    n_branches,         // 0 = cubeless, 1 = cubeful position, 2 = cubeful cube decision
    dice_seq,           // Pre-generated dice pairs for this trial
    max_moves,          // Maximum half-moves before forced stop
    move0_cache,        // Optional shared cache for first-move decisions
    move1_cache)        // Optional shared cache for second-move decisions
```

### Starting Convention

**Post-move start** (`start_post_move = true`):
- The input board is a post-move position (the mover just moved, opponent is
  about to roll). The board is flipped at the start so the opponent moves first.
- SP parity: `is_sp = (move_num % 2 == 1)` — the first mover (move 0) is the
  opponent, so move 1 is SP's turn.
- Used by both cubeless position rollout (`rollout_position`) and cubeful
  position rollout (`cubeful_rollout_position`).
- **Cube perspective flip at entry**: when `n_branches > 0`, each branch's
  cube state is flipped at trial start (owner-flip + match away-swap) to match
  the post-flip board's perspective. Inputs are conventionally in the
  post-move mover's (SP's) perspective; after the start-flip, the current
  mover is the opponent, so `branches[].cube` must be flipped to stay
  consistent with the loop's invariant that `branches[].cube` is always in
  the current mover's perspective. Phase 6's per-move flip then keeps it in
  sync for subsequent half-moves.

**Pre-roll start** (`start_post_move = false`):
- The input board is a pre-roll position (SP is about to roll). No flip at start.
- SP parity: `is_sp = (move_num % 2 == 0)` — move 0 is SP's turn.
- No entry cube flip is applied (the input cube perspective already matches
  the unflipped board).
- Used by cubeful cube decision rollout (`cubeful_cube_decision`).

### Per-Move Phases

Each half-move in a trial proceeds through 6 phases:

**Phase 1 — Cube Check (cubeful only, move > 0):**

If any branch has an active (non-dead) cube, evaluate whether the mover should
double. All active branches share the same board and strategy and differ only
in their `CubeInfo`, so they are evaluated together in a single batched call
whenever the evaluator supports it. Three cube evaluation modes are supported:

- **1-ply Janowski:** Get pre-roll probs once at 1-ply, then apply Janowski per
  branch via `cube_decision_1ply(probs, branches[b].cube, cube_x)`. Fastest.
- **N-ply cubeful recursion (screen + escalate):** A cheap 1-ply Janowski
  screen runs first on shared pre-roll probs; only branches the screen flags
  as doubles escalate to a single batched `cube_decision_nply_multi(board,
  cubes[], n, base, ply, …)` call over all flagged branches (one shared
  cubeful recursion with `cci = 2*n` and `fTop=true`, with the deep PubEval
  pre-filter enabled — see `MULTI-PLY.md` section 6). This keeps the deep
  recursion off the common no-double moves.

  **In the endgame the screen runs twice per branch** — once at the position's
  `cube_x` and, if that clears the branch, again at a dead cube (`x = 0`) —
  and the branch escalates if either wants to double. A screen false-negative
  is not a harmless under-count: it freezes that branch's cube for the rest of
  the trial, silently overriding the configured cube evaluator, and the missed
  double belongs to whichever side is on roll, so it can move the reported
  equity either way. Screening at `cube_x` alone does exactly that near the end
  of a game, where `cube_efficiency` floors a race at x = 0.6 but the real cube
  is dead, so Janowski's live-cube term overvalues holding it (a 7-vs-6-pip
  bearoff at 63.9% wins: 1-ply ND +0.47 / DT +0.34, no double; true cubeful
  ND +0.278 / DT +0.556, a clear double — a 0.13 gap, so no epsilon tolerance
  would catch it). The `x = 0` retry maximizes the doubling advantage
  `min(DT, DP) − ND` over x, since `DT = 2·cl2cf(OPPONENT, x)` falls as x rises
  while `ND = cl2cf(owner, x)` rises when the mover owns the cube; a branch
  both screens clear wants no double at **any** cube efficiency in [0, 1].

  The retry is closed-form on probs already in hand, but the escalations it
  admits are not free — a dead cube says "double whenever ahead", so running it
  unconditionally puts the deep recursion on roughly every move the mover
  leads (measured 1.56x / 1.47x / 1.48x wall-clock on the cube-action,
  checker-play and post-move rollout benchmarks). `is_endgame_cube_position()`
  therefore gates it to boards whose cube is genuinely near-dead — **at most
  two checkers a side (bar included) and at least one side within 24 pips** —
  which is where the mispricing lives. Gated, the same three benchmarks cost
  1.13x / 1.03x / 1.01x with values unchanged (worst equity delta 0.0000 /
  0.0043 / 0.0000). Mid-game positions keep the cheap `cube_x`-only screen,
  where the 0.68 live-cube model is calibrated and the cube really does stay
  live.

  At half-move 1 the board (one of 21, determined by the first roll) and the
  branch cube states are identical across every trial, so escalated move-1
  decisions are computed once per first roll and shared through the move-1
  cube-decision cache (see section 10).
- **Truncated rollout:** Per-branch, call inner
  `RolloutStrategy.cubeful_cube_decision(board, branch.cube)` (single-threaded).
  This mode is not batched across branches because each inner rollout has its
  own dice sequence and internal state; branches are processed sequentially.

Cube take/pass decisions are evaluated at the configured `cube` strategy
(defaulting to `decision_ply`) for the **entire trial** — unlike checker-play
move selection, they are NOT dropped to cheaper strategies at the late /
ultra-late thresholds. The trial's outcome is scored by the N-ply truncation
at `decision_ply`, so deciding take/pass at a shallower ply creates a
decision-vs-evaluation mismatch: the opponent takes doubles that a
consistent-depth evaluation would pass, and the deeper continuation then
over-credits the doubler (visible under Jacoby with a centered cube as a
No-Double equity above the +1.0 cash ceiling). The 1-ply screen keeps this
affordable. The `cube_late` config is accepted for compatibility but does not
affect take/pass decisions.

If the mover doubles:
- **Take:** `cube_value *= 2`, opponent now owns
- **Beaver:** `cube_value *= 4`, opponent owns (double + immediate redouble)
- **Pass:** Branch terminates with DP equity (the value of winning the current
  cube), VR-corrected: `final_equity = dp_value - accumulated_vr_luck`

Because each branch's cube decision depends only on its own cube state and the
shared board, the decisions produced by the batched call are applied
sequentially to `branches[]` after the call returns with no cross-branch
dependency.

If all branches terminate (all D/P'd), the trial ends early with a 1-ply
cubeless evaluation of the current position.

**Phase 2 — Move Generation:**

Generate legal moves for the actual dice roll. When VR is active, also generate
legal moves for all 21 possible rolls (needed for the VR mean computation).

When VR is skipped for this move, only the actual roll's candidates are generated.

Move generation throughout the trial loop (and inside the cubeful recursion)
uses the hash-dedup generator `possible_boards_unsorted`: duplicates are
rejected at insertion time via a generation-stamped hash table, and output is
in first-seen generation order. This avoids the O(n²) insertion-sorted dedup
of the board-sorted generator, which matters on doubles rolls with 30-90
candidates.

**Phase 3 — VR Mean Computation:**

When VR is active, evaluate the best move for all 21 rolls at 1-ply:
- For each roll, find the best candidate board via `best_move_probs_for_candidates`
  using the base (1-ply) strategy.
- Compute the weighted mean across all 21 rolls (cubeless and, if cubeful,
  per-branch cubeful means).
- Record the best candidate index for each roll (used for move reuse in Phase 4).

When using the move1 cache (move_num == 1), all of this is precomputed.

**Phase 4 — Move Selection:**

Pick the best move for the actual dice roll:

- **Move0 cache hit:** At move 0, if the cache has a precomputed result for this
  roll, use it directly.
- **Move0 cache miss:** Compute via CAS (compare-and-swap): first thread to claim
  the slot computes the result; others spin-wait.
- **1-ply (using base):** Reuse the VR computation's best candidate index.
- **N-ply cubeless (checker strategy is a `MultiPlyStrategy`):** call
  `best_move_index_cubeful_multi` with a single **dead cube**. With the cube
  dead the 1-ply filter scores by plain cubeless equity and the rescore runs
  the batched cubeful evaluation engine as a cubeless N-ply tree — so
  cubeless and cubeful trials select moves through the same engine. Other
  strategy types (child truncated-rollout evaluators) keep their own
  `best_move_index`.
- **Move1 cache hit:** At move 1, use the precomputed result.

**Phase 4b — VR Luck Computation:**

If VR is active:
1. Get the 1-ply probs of the chosen move. When using base for decisions, reuse
   the VR stored probs. When using N-ply, evaluate the chosen board at 1-ply.
2. Compute cubeful VR luck for each branch (if cubeful).
3. Compute cubeless VR luck and cross-map to SP perspective.

**Phase 5 — Terminal Check:**

If the chosen move ends the game (`check_game_over`):
- Compute terminal probs and equity.
- For cubeful branches: compute terminal value in the appropriate space (equity
  for money games accounting for gammons × cube value; MWC for match play).
  VR-correct and record `branch.final_equity = terminal_value - vr_luck`.
- For cubeless: convert to SP perspective, VR-correct, and return.

**Phase 6 — Board Flip:**

Flip the board to the next mover's perspective. For cubeful branches, flip the
cube ownership (PLAYER <-> OPPONENT) and, for match play, swap away scores.

### After the Loop: Truncation

If the trial reaches `truncation_depth` without terminating:
1. **Cubeful branches with N-ply truncation (`truncation_ply > 1`):** make a
   single `cubeful_equity_nply_multi` call over all unfinished branches (they
   share the truncation board and differ only in cube state) with the deep
   PubEval pre-filter enabled. The same tree walk also returns the node's
   **cubeless probabilities** (accumulated through the cubeful recursion —
   see `MULTI-PLY.md` section 4), which become the trial's cubeless
   truncation result after inverting to the last mover's perspective and
   clamping against `last_mover_board`. One tree walk serves both outputs;
   no separate cubeless evaluation runs.
2. **Otherwise** (cubeless trials, dead-cube branches, or 1-ply truncation):
   evaluate the last mover's post-move position cubelessly — exact DB probs
   for bearoff positions; the cubeful evaluation engine with a single dead
   cube when `truncation_ply > 1` (see §11, "Cubeless Truncation"); a 1-ply
   base evaluation otherwise — and clamp against `last_mover_board` so
   impossible outcomes (gammon/backgammon when bearoff has begun, backgammon
   when contact is broken and the danger zone is empty) are exactly zero.
   For cubeful branches with 1-ply truncation, apply Janowski to the clamped
   cubeless probs.
3. Convert to SP perspective, VR-correct, and return.

## 6. Position Rollouts (Cubeless and Cubeful)

Position rollouts evaluate a post-move position by running many trials forward
from that position. The same trial machinery is used in two flavors:

- **Cubeless** (`rollout_position`): no cube state is tracked; only cubeless
  probs/equity are produced.
- **Cubeful** (`cubeful_rollout_position`): a single cube branch is tracked
  through the trials, producing a VR-adjusted cubeful equity from the trial
  paths along with the cubeless probs/equity from the same trials.

Both share dice generation, move0/move1 caches, parallelization, and aggregation.

### Cubeless: `rollout_position(board) -> RolloutResult`

1. Pre-generate stratified dice sequences (cached at construction time).
2. Prefill move0 and move1 caches for the flipped starting board.
3. Run trials in parallel (or serial if single-threaded), each calling
   `run_trial_unified` with `start_post_move = true`, `branches = nullptr`,
   `n_branches = 0`.
4. Aggregate per-trial VR-corrected results into mean probs and standard errors.

`RolloutStrategy` implements the `Strategy` interface using this entry point:
- `evaluate_probs(board, …)` → `rollout_position(board).mean_probs`
- `evaluate(board, …)` → `rollout_position(board).equity`

This allows rollout to be used as a drop-in replacement for N-ply evaluation
anywhere a `Strategy` is expected.

### Cubeful: `cubeful_rollout_position(post_move_board, cube) -> CubefulPositionResult`

1. Pre-generate stratified dice sequences.
2. Create one branch template from the input cube:
   ```
   tmpl.cube = cube
   tmpl.basis_cube = cube.cube_value
   ```
   The cube is in the post-move mover's (SP's) perspective. The cube
   perspective flip described in §5 is applied inside `run_trial_unified` at
   trial start so that the branch's cube perspective matches the post-flip
   board (opponent's perspective at move 0).
3. Prefill move0 and move1 caches for the flipped starting board (same caches
   as the cubeless path). When `cubeful_trial_moves` is on, prefill receives
   the branch's cube state so `chosen[]` is cubeful-best; otherwise prefill
   uses cubeless selection.
4. Run trials in parallel, each calling `run_trial_unified` with
   `start_post_move = true`, `branches = [tmpl_copy]`, `n_branches = 1`.
5. Aggregate per-trial cubeful equities (basis-cube SP-perspective units) into
   a mean + SE, and aggregate the per-trial cubeless probs/equities the same
   way as the cubeless path.

The Python binding `RolloutStrategy.cubeful_evaluate_board(board, pre_move_board,
cube_value, owner, …)` exposes this entry point and returns a dict containing
both the cubeless results (`probs`, `equity`, `std_error`, `prob_std_errors`)
and the cubeful results (`cubeful_equity`, `cubeful_se`).

### Result Structure

```
struct RolloutResult {                      // Cubeless rollout result
    double equity;                           // Cubeless equity (SP perspective)
    double std_error;                        // SE of equity
    array<float, 5> mean_probs;              // VR-corrected per-prob means
    array<float, 5> prob_std_errors;         // SE per probability component
    double scalar_vr_equity;                 // Scalar-equity VR diagnostic
    double scalar_vr_se;
};

struct CubefulPositionResult {               // Cubeful position rollout result
    double cubeful_equity;                   // Mean cubeful equity (basis cube units)
    double cubeful_se;                       // SE of cubeful_equity
    RolloutResult cubeless;                  // Cubeless results from same trials
};
```

## 7. Cube Decision Rollout

The cube decision rollout evaluates a doubling decision by simulating two
branches — ND (No Double) and DT (Double/Take) — simultaneously with the same
dice sequences.

### Entry Point: `cubeful_cube_decision(pre_roll_board, cube)`

1. Pre-generate stratified dice sequences.
2. Create two branch templates from the cube state:
   - **ND branch:** Same cube state as input (player hasn't doubled).
   - **DT branch:** Cube value doubled, opponent owns.
   - Both branches share `basis_cube = cube.cube_value` for normalization.
3. Prefill move0 and move1 caches for the pre-roll board. When
   `cubeful_trial_moves` is on, prefill receives both branch cube states
   (ND + DT) and stamps `chosen[]` with the cubeful-best move for
   `branches[0]` (the ND branch, which both branches share).
4. Run trials in parallel, each calling `run_trial_unified` with
   `start_post_move = false`, `branches = [nd_copy, dt_copy]`, `n_branches = 2`.
5. Aggregate per-trial ND and DT equities into means and standard errors.

### Branch State: `CubefulBranch`

Used by both single-branch (§6 cubeful position) and two-branch (§7 cube
decision) modes:
```
struct CubefulBranch {
    CubeInfo cube;         // Current cube state (current mover's perspective)
    int basis_cube;        // For normalization
    double vr_luck;        // Accumulated VR luck (basis cube units, SP perspective)
    bool finished;         // Branch terminated (D/P, terminal, or truncation)
    double final_equity;   // Result (basis cube units, SP perspective)
};
```

`branches[].cube` is always in the current mover's perspective at any point
during the trial. The cube perspective flip at trial entry (§5) plus Phase 6's
per-move flip maintain this invariant.

### Cube Decision During Trials

At each half-move (except move 0), each active branch independently evaluates
whether the mover should double. The mover doubles if `cube_decision.should_double`
is true. The response determines the branch's fate:

- **Take:** Cube turns, branch continues with higher cube value.
- **Beaver:** Cube quadruples (double + immediate redouble), branch continues.
- **Pass:** Branch terminates immediately. The terminal value is:
  - Money game: `±cube_value / basis_cube` (sign depends on who passed).
  - Match play: MWC from `dp_mwc(away1, away2, cube_value, is_crawford)`.
  - VR-corrected: `final_equity = dp_value - vr_luck`.

### Dead Cube Optimization

When `cube_is_dead(branch.cube)` is true for all branches (e.g.,
`max_cube_value` is reached), the flag `cube_active` is set to false and all
cubeful overhead is skipped:
- No cube decisions evaluated.
- No cubeful VR luck tracked.
- Branch final equities computed from cubeless VR results with simple scaling.

This means cubeful rollouts with dead cubes have zero performance overhead
compared to cubeless rollouts.

### Result Structure

```
struct CubefulRolloutResult {
    double nd_equity;       // Mean ND equity (basis cube units)
    double nd_se;           // Standard error of ND
    double dt_equity;       // Mean DT equity (basis cube units)
    double dt_se;           // Standard error of DT
    RolloutResult cubeless; // Cubeless probs/equity from the same trials
};
```

The cubeless results are always computed alongside the cubeful branches (from the
same trial games), providing cubeless pre-roll probabilities for display.

## 8. Move Selection Strategies During Trials

Checker play (move selection) within trials uses configurable strategies at
different game phases.

### Strategy Selection Chain

For each half-move, the move selection strategy is chosen (first match wins):

1. **Ultra-late** (move_num >= `ultra_late_threshold`): `base_` (1-ply raw NN).
2. **Late** (move_num >= `late_threshold`): `checker_late_strat_`.
3. **Normal**: `checker_strat_`.

### Cube-Aware Selection (`cubeful_trial_moves`)

When `RolloutConfig.cubeful_trial_moves` is true AND the trial has at least
one active cube branch AND the half-move index is less than
`cubeful_late_threshold`, the strategy's `best_move_index_cubeful_multi` is
called instead of `best_move_index`. The chosen move is the candidate that
maximizes CUBEFUL equity (cl2cf) against the active branches' cube states.

- **Single branch** (`cubeful_rollout_position`, used by checker-play
  analytics): the chosen move is optimal for that branch's cube state.
- **Two branches** (`cubeful_cube_decision`: ND + DT): the multi-cube BMI
  returns per-branch best indices; the trial applies `branches[0]`'s pick
  to its shared board. Both branches evolve through the same trajectory,
  differing only in cube state.

The cubeful BMI call inside `MultiPlyStrategy::best_move_index_cubeful_multi`
runs a **per-cube 1-ply cubeful filter** (cl2cf per cube state, top
`max_moves` within `threshold` of each cube's cubeful-best) and unions the
survivors across cubes. This guarantees every cube's 1-ply cubeful favorite
reaches the N-ply rescore — a pure cubeless filter would drop candidates
whose cubeless equity sits well below the cubeless-best, including the
match-defensive plays preferred at extreme away scores (e.g. 1-away with
cube=2). The filter scores all candidates with the batched delta-evaluation
kernel (`batch_evaluate_candidates_equity_probs`), classified from the
pre-move board.

The union survivors are then evaluated at the configured N-ply via
`cubeful_equity_nply_multi` (serial, with the deep PubEval pre-filter
enabled — rescoring runs inside rollout trials, where per-call shifts
average out), **after flipping the candidate board and cube
states to the opponent's perspective** — `cubeful_equity_nply_multi`
expects a pre-roll position from the player-on-roll's POV and returns the
equity in that POV, so for a post-move candidate (mover already moved) we
flip first. The returned per-cube equities are then in the opponent's POV,
so the mover's best move is the one that **minimizes** the opponent's
equity (argmin per cube). This matches the `_cubeful_equity` helper in
the Python analyzer, which uses the same flip-and-negate pattern.

For 4-ply targets the filter chain has an intermediate 3-ply step that
still narrows survivors cubelessly — see the TODO note in `multipy.cpp`
about extending the per-cube cubeful treatment to that step for full
correctness in extreme match positions at 4-ply.

The Move0Cache and Move1Cache are populated **cube-aware** at prefill time:
`prefill_move0_cache` and `populate_move1_cache_entry` accept the active
branch cube states and call `best_move_index_cubeful_multi` instead of
`best_move_index`. Each trial then reuses the cached `chosen[]` at moves
0 and 1 — no per-trial cubeful BMI cost at those moves, and trial outputs
are deterministic across runs because every trial picks the same move at
each cache hit. Move1Cache's `mover_probs`, `roll_best_probs`, and the
`cl_mean` fields are cube-state-independent (1-ply cubeless) and shared
between cubeless and cube-aware modes — they feed the per-move cube
decisions and the cubeless VR mean.

The `cubeful_late_threshold` config caps cube-aware selection to early
half-moves: at moves >= this threshold the trial falls back to cubeless
BMI even when `cubeful_trial_moves` is on. Late-game cube state is usually
settled (the cube has either turned to a level both branches accepted, or
a D/P has terminated one branch), so cube-aware selection there adds
little signal but real cost. The default `cubeful_late_threshold = 0`
inherits from `ultra_late_threshold`; for full rollouts where
`ultra_late_threshold = 9999` keeps the cubeful path active for the entire
game, set `cubeful_late_threshold` lower (e.g. 12) to bound cube-aware
work to the early game.

### Strategy Construction

Strategies are built from `TrialEvalConfig` at `RolloutStrategy` construction:

- **1-ply** (`ply = 1`): Uses `base_` directly. No wrapping needed.
- **N-ply** (`ply > 1`): Wraps `base_` in `MultiPlyStrategy` with internal filter
  `{max_moves=2, threshold=0.03}`. Serial evaluation (`parallel_evaluate = false`)
  because parallelism operates across trials, not within them.
- **Truncated rollout** (`rollout_trials > 0`): Creates a child `RolloutStrategy`
  with `n_threads=1` (single-threaded inner rollout). This provides the deepest
  evaluation level available.

### Hybrid Mode

When a filter base strategy is provided, `MultiPlyStrategy` instances are created
in hybrid mode: the filter strategy handles 1-ply candidate scoring and opponent
move selection, while the base strategy handles leaf evaluations. See `MULTI-PLY.md`
section 9.

### VR Best-Candidate Reuse

When the decision strategy is 1-ply (base), the VR mean computation already
evaluated the best candidate for each roll. The trial reuses the best candidate
index directly — zero additional NN evaluations for move selection.

### Pre-Filter for N-ply VR

When the decision strategy is N-ply and VR is computing the mean across all 21
rolls, a generous 1-ply pre-filter (threshold=0.12, max=8 candidates) narrows
the candidate set before expensive N-ply evaluation. This avoids evaluating
clearly terrible candidates at N-ply depth while virtually never dropping a good
move (the threshold is 1.5x wider than the standard TINY filter of 0.08).

## 9. Cube Decision Strategies During Trials

Cube decisions during trials support three evaluation modes, configured
independently from checker play strategy.

### Evaluation Modes

**1-ply Janowski (default):**
- Get pre-roll probs once: `invert(base.evaluate_probs(flip(board), flip(board)))`.
- For each active branch, call `cube_decision_1ply(probs, branch.cube, cube_x)`.
  The NN evaluation that produced `probs` is shared across branches; only the
  per-branch Janowski conversion runs per branch.
- Fastest mode, using the standard Janowski interpolation.

**N-ply cubeful recursion (screen + escalate, batched across branches):**
- A 1-ply Janowski screen on shared pre-roll probs flags candidate doubles;
  branches the screen clears keep their cube unchanged (no double). The screen
  is evaluated at the position's `cube_x` and, on endgame boards
  (`is_endgame_cube_position()`: at most two checkers a side, one side within
  24 pips), again at a dead cube (`x = 0`), escalating if either wants to
  double — so where the cube is near-dead it cannot veto a double the
  configured evaluator would make at any cube efficiency in [0, 1]. See §5,
  Phase 1 for why a false-negative is not a safe miss.
- For the flagged branches, call `cube_decision_nply_multi(board, cubes[], n,
  base, ply, out[], …, deep_prefilter=true)`.
- Internally this runs a single cubeful recursion with `cci = 2*n` and
  `fTop=true`. The state layout is
  `[branch0_ND, branch0_DT, branch1_ND, branch1_DT, …]`; `fTop=true` suppresses
  `make_cube_pos`'s top-level DT expansion so the caller-constructed DT
  variants are used directly.
- Move selection (1-ply cubeful against the primary state, batched — see
  `MULTI-PLY.md` §6) and the per-roll NN evaluations are shared across all
  branches; only the per-state Janowski leaf conversions and `get_ecf3`
  cube-decision collapses differ per state.
- At half-move 1, escalated decisions are computed once per first roll and
  shared via the move-1 cube-decision cache (§10).
- Serial (1 thread) to avoid nested parallelism within trials.

**Truncated rollout (per-branch):**
- For each active branch, call
  `RolloutStrategy.cubeful_cube_decision(board, branch.cube)`.
- The inner strategy is a lightweight single-threaded rollout.
- Provides the deepest evaluation — a truncated rollout within a rollout.
- Not batched across branches: each inner rollout carries its own dice
  sequence, move0/move1 caches and per-trial state.

### Strategy Selection

Cube take/pass decisions use the configured `cube` strategy for the entire
trial — there is no late / ultra-late drop for cube decisions (see the
rationale in §5, Phase 1). The 1-ply screen keeps the configured-depth
evaluations off the common no-double moves.

### Configuration Resolution

The cube config defaults to inheriting `decision_ply` from the legacy fields.
When an explicit `TrialEvalConfig` is provided for cube, it overrides the
default.

```
cube_eval_config = resolve(config.cube, default=decision_ply)
```

The `cube_late` config is accepted for compatibility but does not affect
take/pass decisions.

## 10. Move Caches (Move0 and Move1)

All trials in a rollout share the same starting position, so the first two
half-moves have a finite, cacheable set of decisions.

### Move0Cache

At move 0, there are only 21 possible dice rolls. The first-move decision for
each roll is computed once and shared across all trials.

**Structure:**
```
struct Move0Cache {
    atomic<int> state[21];    // 0=empty, 1=computing, 2=ready
    Board chosen[21];         // Best post-move board for each roll
};
```

**Population:** The cache is prefilled before trial execution begins.
Prefilling uses the full checker strategy (e.g., 3-ply) for move selection,
not 1-ply. When multithreaded, `multipy_parallel_for` distributes the 21
entries across workers.

When `cubeful_trial_moves` is on, the prefill receives the active branch
cube states and calls `best_move_index_cubeful_multi` instead of
`best_move_index`. The cached `chosen[]` is then the cubeful-best move
under those cube states, matching what each trial would compute on its own.

**CAS protocol for on-demand population:** If a trial encounters an empty cache
entry (not prefilled), it atomically claims the slot via
`compare_exchange_strong(0 → 1)`, computes the result, then stores it with
`state = 2`. Other trials spin-yield until the entry is ready. The CAS
populate uses the same cubeless/cube-aware branch as prefill.

### Move1Cache

At move 1, the board depends on the move-0 roll, giving 21 possible boards
(one per first roll). For each, there are 21 second-roll decisions. The cache
precomputes all of this.

**Structure (per first-roll entry):**
```
struct Move1Cache::Entry {
    bool race;                         // Is the move-1 board a race?
    float cube_x;                      // Cube efficiency
    float mover_probs[5];              // Pre-roll probs for 1-ply cube decisions
    float roll_best_probs[21][5];      // VR mean: best move probs per second roll
    int best_candidate_idx[21];        // Index of best candidate per second roll
    double cl_mean_probs[5];           // Precomputed cubeless VR mean
    double cl_mean_eq;                 // Precomputed cubeless VR mean equity
    Board chosen[21];                  // Best post-move board per second roll
    float actual_probs[21][5];         // 1-ply probs of chosen move per second roll
};
```

**Population:** Move1 entries are populated after move0 entries. Each entry
requires ~21 NN evaluations (one per second roll). When multithreaded, entries
are distributed across workers. Like move0, a CAS protocol handles on-demand
population for entries not yet prefilled when a trial reaches move 1.

When `cubeful_trial_moves` is on, the prefill receives the active branch cube
states (flipped to move-1 mover's perspective inside `populate_move1_cache_entry`)
and `chosen[]` is the cubeful-best second move under those cube states. The
other entry fields (`mover_probs`, `roll_best_probs`, `actual_probs`,
`cl_mean_*`) are cube-state-independent and computed the same way in both
modes.

**Move1 uses 1-ply for move selection.** Unlike move0 which uses the full
checker strategy, move1 always uses the base (1-ply) strategy. The VR
averaging over many trials makes higher-ply move selection at move 1
unnecessary — the VR correction dominates the accuracy gain. In cube-aware
mode the 1-ply BMI is the cubeful variant against the active cube states;
in cubeless mode it is the standard `best_move_index`.

### Move-1 Cube-Decision Cache

At half-move 1 the board (one per first roll) and both branch cube states are
identical across all trials, so escalated N-ply cube decisions there are
deterministic per first roll. The first trial to need the decision for a
given first roll computes it (CAS-claimed) and stores the per-branch
`CubeDecision` results alongside each branch's cube-state fingerprint; every
later trial with the same first roll reads the cached decisions after
validating the fingerprints. This turns up to `n_trials` deep cube
evaluations at move 1 into at most 21.

### No-Barrier Design

Prefilling and trial execution are not separated by a barrier. Threads proceed
from prefilling to trial work as soon as their prefill work is done. Trials that
hit unpopulated cache entries compute them on demand via the CAS protocol. This
eliminates idle time from uneven prefill work distribution.

## 11. Truncation Evaluation

When a trial reaches `truncation_depth` half-moves without terminating, the game
is cut short and the position is evaluated with a neural network.

### Position-Based Probability Clamping

The cubeless probabilities returned by the truncation NN are run through
`clamp_probs_to_board(probs, last_mover_board)` before being used as either
the per-trial cubeless return value or the input to the 1-ply Janowski path
of cubeful truncation. The same clamp is applied at every other prob
production point in the trial: the VR mean (best-move probs over the 21
rolls), the VR actual (chosen-move probs), the cube-decision pre-roll probs,
the early-termination cubeless probs when all branches D/P, and the
move0/move1 cache prefill. The clamp enforces the same four position
invariants documented in `MULTI-PLY.md` §2 (player or opponent already has
bearoff progress; contact broken with the player or opponent absent from
the danger zone). Cubeful N-ply truncation goes through
`cubeful_equity_nply_multi`, where the leaf NN output is clamped inside the
cubeful recursion itself.

The result: at every truncation point and at every NN-driven step inside a
trial, gammon and backgammon outcomes that the position rules out are
exactly zero in the per-trial probability vector. For a position with
deterministic backgammon-loss probability (e.g. a pure race where the
player has cleared the opponent's home board), the per-trial backgammon
probabilities are deterministically zero too — so VR-corrected per-trial
probs are exact rather than "approximately exact."

### Cubeless Truncation

For trials with no active cube branches (cubeless rollouts, dead-cube
branches, branches that all finished before the truncation point) — and for
the cubeless probs when `truncation_ply == 1` — the truncation evaluation of
the last mover's post-move board (`last_mover_board = flip(current_board)`)
is, in priority order:

1. **Bearoff positions:** exact DB probs via
   `bearoff_db->lookup_probs(last_mover_board, post_move=true)`.
2. **`truncation_ply > 1`:** the cubeful evaluation engine with a single
   **dead cube** (`cube_value=1, max_cube_value=1`). With the cube dead,
   `cl2cf` bypasses Janowski everywhere, interior picks reduce to cubeless
   1-ply equity, and the tree is exactly a cubeless N-ply evaluation —
   the same batched walk (and the same `deep_prefilter=true` PubEval prune)
   the fused cubeful truncation uses. The tree's accumulated probs are
   inverted to the last mover's perspective.
3. **`truncation_ply == 1`:** `base.evaluate_probs(last_mover_board, …)`
   (the truncation strategy member is just the base strategy; N-ply
   truncation never goes through it).

Trials with active cube branches and `truncation_ply > 1` do not use this
path — their cubeless probs come from the cubeful tree walk (above).

### Cubeful Truncation

For cubeful branches at truncation, all active branches share the same
truncation board (every branch evolved through the same dice sequence and the
same cubeless move selections) and differ only in cube state. They are
therefore evaluated together in a single batched call.

**N-ply cubeful truncation** (`truncation_ply > 1`):
- Collect the cube state of every unfinished branch into an array `cubes[n]`.
- Call `cubeful_equity_nply_multi(board, cubes[], n, base, truncation_ply,
  out[], …, probs_out, deep_prefilter=true)`.
- Internally this runs a single cubeful recursion with `cci = n` and
  `fTop = false`. Move selection (1-ply cubeful against `cubes[0]`, via the
  batched candidate kernel) and the per-roll NN evaluations are shared across
  branches; only the per-state Janowski leaf conversions and `get_ecf3`
  cube-decision collapses differ per state.
- Returns a cubeful equity per branch that accounts for future cube actions,
  **and** the tree's cubeless probabilities from the same traversal — used as
  the trial's cubeless truncation result (see §5).
- The `board` is the **next mover's** pre-roll position (after Phase 6 flip).

**1-ply Janowski truncation** (`truncation_ply == 1`):
- Apply Janowski to the cubeless probs from the last mover's perspective.
- Requires flipping the cube ownership to match the last mover's view:
  `last_cube = flip_cube_perspective(branch.cube)`.

In both cases, each branch's result is VR-corrected:
`branch.final_equity = truncation_value[b] - branch.vr_luck`.

## 12. Parallelization

### Trial-Level Parallelism

Parallelism operates at the trial level: independent trials are distributed
across threads. N-ply strategies within trials use serial evaluation
(`parallel_evaluate = false`, `n_threads = 1`) to avoid nested parallelism.

### Thread Count

```
function rollout_thread_count(n_trials):
    if n_threads configured > 0:
        return min(n_threads, n_trials)

    n = hardware_concurrency

    // For truncated N-ply rollouts (truncation_depth > 0 && decision_ply > 1),
    // default to 1 thread to preserve cache locality.
    // Opt-in to parallelism via config.parallelize_trials = true.
    if !parallelize_trials && truncation_depth > 0 && decision_ply > 1:
        return 1

    return min(n, n_trials)
```

The conservative default of 1 thread for truncated N-ply rollouts exists because
`MultiPlyStrategy`'s thread-local `PosCache` (256K entries) provides significant
speedup when warm. Splitting trials across threads fragments the cache, often
making parallel execution slower than serial.

### Work-Stealing Pattern

Trials are distributed using an atomic counter with chunked dispatch:

```
const chunk_size = 8

atomic<int> next_trial = 0

thread_function:
    clear_thread_local_caches()          // N-ply PosCache + cubeful eval cache
    enable_shared_pos_cache()

    while (start = next_trial.fetch_add(chunk_size)) < n_trials:
        check_cancellation()
        for t in start..min(start + chunk_size, n_trials):
            trial_results[t] = run_trial_unified(...)

    disable_shared_pos_cache()
```

### SharedPosCache for Cross-Thread Sharing

When multiple threads run trials, a `SharedPosCache` (2M entries, lock-free CAS)
is activated to share N-ply position evaluations across threads. This prevents
redundant expensive evaluations when different trials reach the same positions.

Two producers feed it:
- `MultiPlyStrategy::evaluate_probs_nply_impl` nodes (the hybrid-evaluator
  cubeless N-ply recursion), keyed by (board, plies).
- **Dead-cube nodes of the cubeful evaluation engine** (the cubeless trial
  paths: trial move selection and cubeless truncation). When every valid cube
  state at a tree node is dead, the node's equities are fully determined by
  its cubeless probs (`equity == cubeless_equity(probs)`, which is linear in
  the probs), so only the probs need to be stored. Keys are the engine's
  (board, plies, cci, cube-fingerprint) cache key XOR a producer salt, so the
  two producers' entries can never be confused. Dead-node equities are always
  re-derived from the node probs — locally computed or shared-cache served —
  so results are bit-identical regardless of which thread computed an entry.

This cross-thread sharing is what makes 16-thread cubeless rollouts scale:
trial subtrees overlap heavily (especially the early-move N-ply selections,
whose position space is bounded by the stratified first rolls), and a per-
thread cache alone would re-evaluate each shared subtree once per thread.

The shared cache is cleared when load exceeds 75% capacity.

### Unified Prefill + Trial Threading

For cubeful rollouts, the same thread pool handles both prefilling (move0 + move1
cache population) and trial execution:

1. Threads claim prefill work via `atomic<int> next_roll`.
2. Each thread computes move0 + move1 for its claimed roll index.
3. Without waiting for all prefill to complete, threads proceed to trial
   work-stealing.
4. Trials handle missing cache entries on demand via CAS.

This unified approach keeps thread-local caches warm across prefill and trials.

### Persistent Thread Pool

The implementation uses `multipy_parallel_run` — a persistent thread pool shared
with `MultiPlyStrategy`. This avoids the overhead of creating and destroying
threads for each rollout call, which on Windows can exhaust TLS (Thread-Local
Storage) slots after thousands of cycles.

### Cancellation

A `cancel_flag` (atomic bool) can be set to abort an in-progress rollout.
Cancellation is checked between trial chunks (every 8 trials). When cancelled,
a `RolloutCancelled` exception is thrown after all running trials complete.

## 13. Statistical Aggregation

After all trials complete, per-trial results are aggregated into means and
standard errors.

### Mean and Standard Error

For each statistic `X` (probabilities, equity, per-branch equity):

```
mean_X = sum(X_t) / N
variance_X = sum(X_t^2) / N - mean_X^2
SE_X = sqrt(max(0, variance_X) / N)
```

The `max(0, ...)` guard handles floating-point underflow.

### Cubeless Aggregation

- `mean_probs[k]`: Mean of per-trial VR-corrected probability component k.
- `prob_std_errors[k]`: Standard error of each probability component.
- `equity`: Equity computed from mean probs (or equivalently, mean of per-trial
  equities since equity is linear in probs).
- `std_error`: Standard error of the equity.

### Cubeful Aggregation

- `nd_equity`, `nd_se`: Mean and SE of per-trial ND branch equities.
- `dt_equity`, `dt_se`: Mean and SE of per-trial DT branch equities.
- `cubeless.*`: Full cubeless results from the same trial games.

## 14. Performance Optimizations

### Move Candidate Pre-allocation

The trial function uses `thread_local` vectors for move candidates, pre-reserved
to capacity 24 (the typical maximum for non-doubles). This avoids repeated
allocation/deallocation within the hot loop.

### Batch Evaluation

When using the base (1-ply) strategy for VR mean computation at a position with
no bearoff candidate, `batch_evaluate_candidates_best_prob` evaluates all
candidates and returns both the best index and its probabilities in a single
pass — no redundant NN calls. (When a candidate is in the bearoff range, the VR
mean ranks candidates by exact DB equity instead; see "Bearoff Database
Integration".)

### Batched Interior Picks in the Cubeful Recursion

The in-trial N-ply cube decisions, the N-ply cubeful truncation, and the
cubeful BMI rescore all run on the cubeful evaluation engine described in
`MULTI-PLY.md` sections 4-6: batched delta-evaluation interior picks (grouped
by per-candidate NN classification), leaf reuse at 2-ply nodes, hash-dedup
move generation, the deep PubEval pre-filter (enabled for these
rollout-internal calls), and a per-thread cube-state-keyed memoization cache
that persists across calls within a rollout.

### Ultra-Late Threshold

Positions deep in a trial have diminishing impact on the final result. The
`ultra_late_threshold` (default 2 for truncated rollouts) drops checker play
to 1-ply at depth, eliminating expensive N-ply evaluations for moves that
barely affect the outcome. (Cube take/pass decisions do not drop — see §5,
Phase 1.)

For full rollouts with N-ply cube/checker strategies, set
`ultra_late_threshold = 9999` to disable ply reductions and use configured
strategies for the entire game.

### Bearoff Database Integration

When the bearoff database is loaded, every 1-ply evaluation that the rollout
performs goes through a DB-aware variant of the base strategy
(`base_bearoff_`): if the position being evaluated is in the bearoff range,
the DB returns exact cubeless probabilities; otherwise the call falls through
to the underlying NN. Specifically, the DB-aware base is used at:

- **VR mean** at every half-move (`best_move_probs_for_candidates` over all 21
  rolls). When no candidate is in the bearoff range, the underlying NN's
  optimized batch selects the best candidate and returns its probabilities in a
  single pass. When any candidate is in the bearoff range, candidates are ranked
  by exact DB equity instead (NN equity is used only for a non-bearoff candidate
  at a boundary position). Ranking by DB equity makes the per-roll baseline move
  identical to the move the decision strategy actually plays there — an N-ply
  checker strategy short-circuits its leaves to the DB, so it plays the
  DB-optimal move. This match is required for the luck term `actual − mean` to
  stay unbiased: a baseline that selected a different, weaker move than the
  trial plays would be systematically below the actual value and bias the
  VR-corrected result. The selected baseline move's probabilities are DB-exact.
- **VR actual** at every half-move when the chosen move's probabilities are
  evaluated for luck computation (the N-ply decision path).
- **Move-1 cache prefill** for both `mover_probs` and `roll_best_probs`.
- **Early-termination cubeless probabilities** when all branches D/P at the
  same half-move.
- **Cube decisions** during trials (1-ply Janowski path uses `mover_probs`
  populated from the DB directly when the current board is bearoff).

Other bearoff-DB use points:

- **Bearoff positions as input:** If the starting position is a bearoff
  position, every per-trial evaluation hits the DB directly, so the rollout
  returns exact results (the simulation still runs, but VR luck is
  deterministically zero so the trial-mean is the exact value).
- **At truncation:** bearoff truncation positions short-circuit to exact DB
  probs (`lookup_probs(last_mover_board, post_move=true)`); both the cubeful
  N-ply truncation and the dead-cube cubeless truncation
  (`cubeful_equity_nply_multi`) use `base_bearoff_` so their 1-ply leaves
  are DB-exact (see §11).

The bearoff database is also propagated to the internal N-ply strategies
(checker, late checker, inner rollouts for cube decisions) so that their
evaluations use exact bearoff probs at leaf nodes.

### Move Filter for N-ply Cube Decisions

A cheap pre-filter strategy (e.g., PubEval) can be set via `set_move_filter`.
This is propagated to all internal `MultiPlyStrategy` instances and used in the
N-ply cubeful recursion during trials to narrow candidate moves before expensive
full-model evaluation.

## 15. Match Play

Match play rollouts operate in MWC (Match Winning Chance) space for cubeful
branch tracking.

### Cubeful Branch Values

- **Terminal:** `cubeless_mwc(terminal_probs, away1, away2, cube_value, is_crawford)`
- **Double/Pass:** `dp_mwc(away1, away2, cube_value, is_crawford)`
- **Cube VR:** `cl2cf_match(probs, branch.cube, cube_x)` per-branch
- **Truncation (1-ply):** `cl2cf_match(probs, last_cube, trunc_x)`
- **Truncation (N-ply):** `cubeful_equity_nply_multi(...)` (batched over all
  active branches) → `eq2mwc(...)` back to MWC per branch

### Perspective Flips in Match Play

When the board flips at the end of each half-move, match-play branches also swap
their away scores:
```
swap(branch.cube.match.away1, branch.cube.match.away2)
```

SP-perspective conversion uses MWC complement:
- SP's MWC = `mwc` when it's SP's turn
- SP's MWC = `1 - mwc` when it's the opponent's turn

### Jacoby Disabled

Jacoby rule is automatically disabled for match play. The `jacoby_active()` check
on `CubeInfo` returns false when match state is present.

## 16. Best Move Selection via Rollout

`RolloutStrategy` exposes two entry points for ranking candidate moves with
rollout evaluation. Both use the same 1-ply pre-filter to narrow candidates
before launching rollouts.

### Cubeless ranking — `best_move_index(candidates, pre_move_board)`

Used by callers that don't carry cube state (e.g. anywhere a `Strategy`'s
`best_move_index` is invoked):

1. **1-ply filter:** Score all candidates at 1-ply. Sort by equity descending.
2. **Threshold filter:** Keep top `max_moves` within `threshold` of the best
   (using the rollout config's filter preset, typically TINY: 5 moves, 0.08).
3. **Cubeless rollout each survivor:** Call `rollout_position(candidate)` for
   each surviving candidate.
4. **Pick the best:** Return the candidate with the highest cubeless rollout
   equity.

### Cubeful ranking — checker-play analytics

The checker-play analytics path (the `BgBotAnalyzer.checker_play` interface)
applies a richer pipeline that uses the cubeful rollout for each survivor:

1. **1-ply scoring** (with bearoff DB awareness on bearoff candidates):
   compute cubeless probs and 1-ply Janowski cubeful equity for every
   candidate.
2. **Filter to rollout survivors:** narrow the candidate set down to the
   moves worth rolling out. The exact filter depends on
   `prefilter_threshold` — see "Two-stage prefilter" below. A min-2 rescue
   grabs the next-best non-survivor when only one survives, so the cubeful
   sort always has at least two rollout-quality entries.
3. **Cubeful rollout each survivor:** call `cubeful_rollout_position(board,
   cube)` for each candidate, producing both cubeless probs/equity and
   cubeful equity from the same trial paths (no post-hoc Janowski).
4. **Sort by cubeful equity:** rank survivors by the rollout-native cubeful
   equity. Non-survivors keep the equity from whichever evaluation level
   produced their result.

Thread-local caches are cleared before evaluation to prevent cross-strategy
contamination.

#### Two-stage prefilter (`prefilter_threshold`)

Step 2 has two modes, selected by the Python-side `prefilter_threshold`
parameter on `BgBotAnalyzer` / `_RolloutAnalyzer`. This is a Python-layer
option only — the C++ `RolloutConfig` is unaware of it.

**Single-stage (`prefilter_threshold = 0`)** — the legacy filter, used by
1T and by user-configured truncated rollouts (`eval_level="rollout"` with
`truncation_depth > 0`):

- Apply the standard **TINY filter** (top `filter.max_moves` = 5 within
  `filter.threshold` = 0.08 of best) directly on the 1-ply cubeful
  equities. Survivors are rolled out; non-survivors keep their 1-ply
  equity in the result list.

**Two-stage (`prefilter_threshold > 0`)** — the default (threshold 0.15)
for 2T, 3T, and any non-truncated rollout (`eval_level="rollout"` with
`truncation_depth = 0`):

- **Stage 1 (loose 1-ply cull):** keep every candidate whose 1-ply
  cubeful equity is within `prefilter_threshold` of the 1-ply best. No
  count cap — the only job is to drop obvious garbage that 2-ply won't
  rescue.
- **Stage 2 (2-ply TINY):** rescore stage-1 survivors at 2-ply (with
  bearoff DB awareness when applicable) and apply the standard TINY
  filter (`filter.max_moves` = 5 within `filter.threshold` = 0.08 of the
  2-ply best). These stage-2 survivors are rolled out.
- **Non-survivors:** moves that passed stage 1 but failed stage 2 keep
  their 2-ply equity in the result list; moves that failed stage 1 keep
  their 1-ply equity. The `eval_level` field on each result reflects
  this (`"Rollout"`, `"2-ply"`, or `"1-ply"`).

**Why two stages exist.** A single 1-ply TINY filter can drop a move
whose true equity is best but whose 1-ply estimate is poor — at 2T, 3T,
and the full Rollout this happens often enough to be visible. Doing the
cheap loose 1-ply cull first and then a stricter 2-ply filter recovers
those moves at low cost: the 2-ply rescore runs on a small surviving set
(typically ~3–8 candidates), and 2-ply is accurate enough that the TINY
gate at that depth rarely drops the rollout winner. 1T isn't worth the
extra 2-ply scoring because the rollout itself is only 1-ply at 72
trials — the filter's accuracy ceiling is already the bottleneck.

A 2-ply strategy instance is lazily created on `_RolloutAnalyzer` only
when `prefilter_threshold > 0`, and the bearoff DB (if loaded) is wired
into it.

## 17. Configuration Reference

### RolloutConfig Fields

| Field | Default | Description |
|-------|---------|-------------|
| `n_trials` | 36 | Number of trial games per evaluation |
| `truncation_depth` | 7 | Half-moves before truncating (0 = play to completion) |
| `decision_ply` | 1 | Default checker play ply depth (legacy) |
| `truncation_ply` | -1 | Ply for truncation evaluation (-1 = same as `decision_ply`) |
| `enable_vr` | true | Enable variance reduction |
| `parallelize_trials` | false | Allow parallel trial dispatch for truncated N-ply |
| `filter` | TINY | MoveFilter for top-level candidate selection |
| `n_threads` | 0 | Thread count (0 = auto-detect) |
| `seed` | 42 | Seed for stratified dice generation |
| `late_ply` | -1 | Default late-game ply (-1 = same as `decision_ply`) |
| `late_threshold` | 20 | Half-move index where late strategies activate |
| `ultra_late_threshold` | 2 | Half-move where checker/cube drop to 1-ply |
| `checker` | unset | TrialEvalConfig: checker play strategy override |
| `checker_late` | unset | TrialEvalConfig: late-game checker play override |
| `cube` | unset | TrialEvalConfig: cube decision strategy override |
| `cube_late` | unset | TrialEvalConfig: late-game cube decision override |
| `cubeful_trial_moves` | true | When true, trial-level checker moves are picked by cubeful equity (cl2cf) against the branch cube state. See §8 "Cube-Aware Selection". Set false for cubeless trial move selection. |
| `cubeful_late_threshold` | 0 | When `cubeful_trial_moves` is on, drop to cubeless BMI at half-moves >= this value. 0 = inherit from `ultra_late_threshold`. Set lower than `ultra_late_threshold` (e.g. 12) for full rollouts to bound cube-aware work to the early game. |
| `cancel_flag` | null | Atomic bool for rollout cancellation |

### TrialEvalConfig Fields

| Field | Default | Description |
|-------|---------|-------------|
| `ply` | 0 | 0 = inherit from legacy fields, 1 = raw NN, 2+ = N-ply |
| `rollout_trials` | 0 | 0 = N-ply mode, >0 = truncated rollout with this many trials |
| `rollout_depth` | 5 | Truncation depth for inner rollout |
| `rollout_ply` | 1 | Decision ply within inner rollout |

### Internal Constants

| Constant | Value | Description |
|----------|-------|-------------|
| Trial chunk size | 8 | Work-stealing granularity |
| Internal filter (N-ply trials) | {2, 0.03} | MoveFilter inside trial MultiPly strategies |
| Deep pre-filter (cubeful recursion) | 16 → 14 | Built-in PubEval prune at sub-entry-ply nodes of rollout-internal cubeful evaluations, including the dead-cube cubeless trees (see `MULTI-PLY.md` §6) |
| VR pre-filter threshold | 0.12 | 1-ply threshold for N-ply VR candidate narrowing |
| VR pre-filter max | 8 | Maximum candidates after VR pre-filter |
| PosCache capacity | 256K | Thread-local N-ply position cache entries |
| SharedPosCache capacity | 2M | Cross-thread position cache entries |
| SharedPosCache clear threshold | 75% | Clear when inserts exceed this fraction |
| Stratified dice levels | 6 | Hierarchical permutation depth |
| Max stratified turns | 128 | Half-moves with quasi-random dice |

### Standard Configurations (App Levels)

| Level | n_trials | trunc_depth | decision_ply | late_ply | late_threshold | ultra_late | prefilter |
|-------|----------|-------------|-------------|----------|----------------|------------|-----------|
| 1T (XG Roller) | 72 | 5 | 1 | -1 | 20 | 2 | 0 (off) |
| 2T (XG Roller+) | 360 | 7 | 2 | 1 | 1 | 9999 | 0.15 |
| 3T (XG Roller++) | 360 | 7 | 3 | 2 | 2 | 9999 | 0.15 |
| R (Full Rollout) | 1,296 | 0 | 1 | -1 | 20 | 9999 | 0.15 |

(2T also pins cube at 2-ply early **and** late and uses a 2-ply truncation eval
(`truncation_ply=2`) — neither is shown as a column above.)

These per-level defaults are defined in one place — the `BgBotAnalyzer.__init__`
dispatch in [python/bgsage/analyzer.py](python/bgsage/analyzer.py), in the
`elif eval_level == "truncated1"/"truncated2"/"truncated3"/"rollout":` blocks.
Change them there and all callers (including the host app) pick up the new
defaults.

For full rollouts (`truncation_depth = 0`), `ultra_late_threshold` is set high
(9999) to keep configured checker/cube strategies active for the entire game.
The C++ `RolloutConfig.ultra_late_threshold` default of 2 is appropriate for
truncated rollouts; full-rollout callers must override it.

`prefilter` is the Python-side `prefilter_threshold` parameter on
`BgBotAnalyzer` / `_RolloutAnalyzer`. When > 0 it enables the two-stage
checker-play candidate filter (1-ply loose cull → 2-ply TINY) described in
§16; when 0 the legacy single-stage 1-ply TINY runs. The default is
0.15 for 2T, 3T, and any non-truncated rollout (i.e. `eval_level="rollout"`
with `truncation_depth = 0`); 0 for 1T and for user-configured truncated
rollouts (`eval_level="rollout"` with `truncation_depth > 0`). Users can
override the per-level default by passing `prefilter_threshold=...` to
`BgBotAnalyzer`.
