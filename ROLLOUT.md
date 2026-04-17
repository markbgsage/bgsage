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
6. [Cubeless Rollout (Position Evaluation)](#6-cubeless-rollout-position-evaluation)
7. [Cubeful Rollout (Cube Decision Evaluation)](#7-cubeful-rollout-cube-decision-evaluation)
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
- **Cubeful equities**: For cube decisions, two branches (No-Double and
  Double/Take) are simulated simultaneously, producing ND and DT equities with
  standard errors.

Rollouts are used for two purposes:

1. **Position evaluation**: Given a post-move board, estimate its cubeless
   probabilities and equity (used by the `Strategy` interface for checker play).
2. **Cube decision evaluation**: Given a pre-roll board and cube state, estimate
   ND and DT cubeful equities for optimal cube action determination.

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
- Typical configuration: 42-360 trials
- The truncation evaluation can use 1-ply, N-ply, or even a nested truncated
  rollout

### XG Roller Equivalences

| Level | n_trials | truncation_depth | decision_ply | late_ply | late_threshold |
|-------|----------|-------------------|-------------|----------|----------------|
| XG Roller (1T)    | 42  | 5 | 1 | -1 | 20 |
| XG Roller+ (2T)   | 360 | 7 | 2 | 1  | 2  |
| XG Roller++ (3T)  | 360 | 5 | 3 | 2  | 2  |
| Full Rollout (R)  | 1,296 | 0 | 1 | -1 | 20 |

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

VR always uses 1-ply (the base NN strategy) for both the mean and actual
evaluations, regardless of the decision strategy's ply level. This is critical:

- Move selection during trials may use 2-ply, 3-ply, or truncated rollout
  evaluation to pick the best move.
- But VR measures luck = (actual_1ply - mean_1ply), where both sides use the
  same 1-ply evaluator. Biases cancel because both the mean and actual are
  evaluated at the same depth.
- This eliminates ~90% of the N-ply evaluations that would be needed if VR used
  the decision strategy's ply level.

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

When running cubeful rollouts (section 7), each branch (ND and DT) tracks its own
VR luck in cubeful value space:

```
// For each branch b:
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
strategy used during the trial.

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

A single function, `run_trial_unified`, handles both cubeless and cubeful rollout
modes. This eliminates code duplication and ensures that cubeful overhead is zero
when all branches have dead cubes.

### Signature

```
TrialResult run_trial_unified(
    board,              // Starting position
    start_post_move,    // true = post-move (opponent first), false = pre-roll (SP first)
    branches[],         // Array of CubefulBranch (or null for cubeless)
    n_branches,         // 0 for cubeless, 2 for cubeful
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
- Used by cubeless rollout (position evaluation via `rollout_position`).

**Pre-roll start** (`start_post_move = false`):
- The input board is a pre-roll position (SP is about to roll). No flip at start.
- SP parity: `is_sp = (move_num % 2 == 0)` — move 0 is SP's turn.
- Used by cubeful rollout (cube decision via `cubeful_cube_decision`).

### Per-Move Phases

Each half-move in a trial proceeds through 6 phases:

**Phase 1 — Cube Check (cubeful only, move > 0):**

If any branch has an active (non-dead) cube, evaluate whether the mover should
double. All active branches share the same board and strategy and differ only
in their `CubeInfo`, so they are evaluated together in a single batched call
whenever the evaluator supports it. Three cube evaluation modes are supported:

- **1-ply Janowski:** Get pre-roll probs once at 1-ply, then apply Janowski per
  branch via `cube_decision_1ply(probs, branches[b].cube, cube_x)`. Fastest.
- **N-ply cubeful recursion:** Call `cube_decision_nply_multi(board, cubes[],
  n, base, ply, …)` once over all active branches. One shared `cubeful_recursive_multi`
  call with `cci = 2*n` and `fTop=true` produces ND/DT equities for every
  branch in a single pass, sharing move selection and NN evaluations across
  branches.
- **Truncated rollout:** Per-branch, call inner
  `RolloutStrategy.cubeful_cube_decision(board, branch.cube)` (single-threaded).
  This mode is not batched across branches because each inner rollout has its
  own dice sequence and internal state; branches are processed sequentially.

The cube evaluation strategy depends on the move number:
- Ultra-late (>= `ultra_late_threshold`): always 1-ply Janowski
- Normal: use the configured `cube_eval_config`
- Late (>= `late_threshold`): use `cube_late_eval_config`

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
- **N-ply (using checker strategy):** Call the full checker strategy's
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
1. Evaluate the last mover's post-move position using the truncation strategy
   (a separate `MultiPlyStrategy` instance with aggressive PubEval prefiltering;
   see §11 for details).
2. For cubeful branches with N-ply truncation (`truncation_ply > 1`): make a
   single `cubeful_equity_nply_multi` call over all unfinished branches with
   a tight single-candidate move filter `{1, 0.0}` so the cubeful tree and
   its move selection are shared across branches (see §11).
3. For cubeful branches with 1-ply truncation: apply Janowski to the cubeless
   probs.
4. Convert to SP perspective, VR-correct, and return.

## 6. Cubeless Rollout (Position Evaluation)

The cubeless rollout evaluates a post-move position by running many trials with
no cube interaction.

### Entry Point: `rollout_position(board)`

1. Pre-generate stratified dice sequences (cached at construction time).
2. Prefill move0 and move1 caches for the flipped starting board.
3. Run trials in parallel (or serial if single-threaded).
4. Aggregate per-trial VR-corrected results into mean probs and standard errors.

### Trial Invocation

Each trial calls `run_trial_unified` with:
- `start_post_move = true` (board is post-move, opponent moves first)
- `branches = nullptr, n_branches = 0` (no cube tracking)

The trial returns VR-corrected cubeless probs and equity from SP's perspective.

### Public Strategy Interface

`RolloutStrategy` implements the `Strategy` interface:
- `evaluate_probs(board, ...)` → `rollout_position(board).mean_probs`
- `evaluate(board, ...)` → `rollout_position(board).equity`

This allows rollout to be used as a drop-in replacement for N-ply evaluation
anywhere a `Strategy` is expected.

## 7. Cubeful Rollout (Cube Decision Evaluation)

The cubeful rollout evaluates a cube decision by simulating two branches — ND
(No Double) and DT (Double/Take) — simultaneously with the same dice sequences.

### Entry Point: `cubeful_cube_decision(pre_roll_board, cube)`

1. Pre-generate stratified dice sequences.
2. Create two branch templates from the cube state:
   - **ND branch:** Same cube state as input (player hasn't doubled).
   - **DT branch:** Cube value doubled, opponent owns.
   - Both branches share `basis_cube = cube.cube_value` for normalization.
3. Prefill move0 and move1 caches for the pre-roll board.
4. Run trials in parallel, each with a fresh copy of the two branch templates.
5. Aggregate per-trial ND and DT equities into means and standard errors.

### Branch State: `CubefulBranch`

Each branch carries:
```
struct CubefulBranch {
    CubeInfo cube;         // Current cube state (mover's perspective)
    int basis_cube;        // For normalization (same for all branches)
    double vr_luck;        // Accumulated VR luck (basis cube units, SP perspective)
    bool finished;         // Branch terminated (D/P, terminal, or truncation)
    double final_equity;   // Result (basis cube units, SP perspective)
};
```

### Trial Invocation

Each trial calls `run_trial_unified` with:
- `start_post_move = false` (board is pre-roll, SP moves first)
- `branches = [nd_copy, dt_copy], n_branches = 2`

Both branches share the same dice sequence and board evolution. They diverge only
in cube state: the DT branch starts with a doubled cube, which affects subsequent
cube decisions, terminal payoffs, and Janowski evaluations within the trial.

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

**N-ply cubeful recursion (batched across active branches):**
- Collect the cube state of every active branch that `can_double()`.
- Call `cube_decision_nply_multi(board, cubes[], n, base, ply, out[], filter, 1, …)`.
- Internally this runs a single `cubeful_recursive_multi` with
  `cci = 2*n` and `fTop=true`. The state layout is
  `[branch0_ND, branch0_DT, branch1_ND, branch1_DT, …]`; `fTop=true` suppresses
  `make_cube_pos`'s top-level DT expansion so the caller-constructed DT
  variants are used directly.
- Move selection (cubeless 1-ply, see §8 and `MULTI-PLY.md` §6) and the
  recursive cubeless NN evaluations are shared across all branches; only the
  per-state Janowski leaf conversions and `get_ecf3` cube-decision collapses
  differ per state.
- Numerically equivalent to calling `cube_decision_nply` individually per
  branch, up to floating-point ordering in the cubeful cache.
- Uses internal filter `{max_moves=2, threshold=0.03}`.
- Serial (1 thread) to avoid nested parallelism within trials.

**Truncated rollout (per-branch):**
- For each active branch, call
  `RolloutStrategy.cubeful_cube_decision(board, branch.cube)`.
- The inner strategy is a lightweight single-threaded rollout.
- Provides the deepest evaluation — a truncated rollout within a rollout.
- Not batched across branches: each inner rollout carries its own dice
  sequence, move0/move1 caches and per-trial state.

### Strategy Selection

Same late/ultra-late chain as checker play:
1. Ultra-late: always 1-ply Janowski.
2. Late: `cube_late_eval_config_` (resolved from config).
3. Normal: `cube_eval_config_`.

### Configuration Resolution

Cube configs default to inheriting `decision_ply` from the legacy fields. When
explicit `TrialEvalConfig` is provided for cube, it overrides the default.

```
cube_eval_config = resolve(config.cube, default=decision_ply)
cube_late_eval_config = resolve(config.cube_late or config.cube, default=late_ply)
```

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

**Population:** The cache is prefilled before trial execution begins. Prefilling
uses the full checker strategy (e.g., 3-ply) for move selection, not 1-ply.
When multithreaded, `multipy_parallel_for` distributes the 21 entries across
workers.

**CAS protocol for on-demand population:** If a trial encounters an empty cache
entry (not prefilled), it atomically claims the slot via
`compare_exchange_strong(0 → 1)`, computes the result, then stores it with
`state = 2`. Other trials spin-yield until the entry is ready.

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

**Move1 uses 1-ply for move selection.** Unlike move0 which uses the full checker
strategy, move1 always uses the base (1-ply) strategy. The VR averaging over many
trials makes higher-ply move selection at move 1 unnecessary — the VR correction
dominates the accuracy gain.

### No-Barrier Design

Prefilling and trial execution are not separated by a barrier. Threads proceed
from prefilling to trial work as soon as their prefill work is done. Trials that
hit unpopulated cache entries compute them on demand via the CAS protocol. This
eliminates idle time from uneven prefill work distribution.

## 11. Truncation Evaluation

When a trial reaches `truncation_depth` half-moves without terminating, the game
is cut short and the position is evaluated with a neural network.

### Truncation Strategy (Separate Instance)

The truncation strategy is always a separate `MultiPlyStrategy` instance (never
shared with `checker_strat_`), even when `truncation_ply == decision_ply`. This
allows truncation-specific tuning: the truncation strategy uses more aggressive
PubEval prefilter parameters (threshold=8, keep=6 vs the default 20/15) to reduce
the number of NN encodings per opponent roll in the N-ply recursion. This is safe
because truncation evaluations are averaged over hundreds of trials, so slightly
noisier individual evaluations wash out in the mean.

### Cubeless Truncation

The truncation strategy evaluates the last mover's post-move board:
```
last_mover_board = flip(current_board)
probs = truncation_strat.evaluate_probs(last_mover_board, last_mover_board)
```

The truncation strategy can be:
- 1-ply (base) when `truncation_ply = 1`
- N-ply when `truncation_ply > 1` (wraps base in `MultiPlyStrategy` with
  aggressive PubEval prefiltering)

### Cubeful Truncation

For cubeful branches at truncation, all active branches share the same
truncation board (every branch evolved through the same dice sequence and the
same cubeless move selections) and differ only in cube state. They are
therefore evaluated together in a single batched call.

**N-ply cubeful truncation** (`truncation_ply > 1`):
- Collect the cube state of every unfinished branch into an array `cubes[n]`.
- Call `cubeful_equity_nply_multi(board, cubes[], n, base, truncation_ply,
  out[], filter, 1, …)`.
- Internally this runs a single `cubeful_recursive_multi` with `cci = n` and
  `fTop = false`. Move selection (cubeless 1-ply) and the recursive cubeless
  NN evaluations are shared across branches; only the per-state Janowski leaf
  conversions and `get_ecf3` cube-decision collapses differ per state.
- Uses a tight single-candidate move filter `{max_moves=1, threshold=0.0}` instead
  of the usual `{2, 0.03}`. Since move selection inside the cubeful recursion is
  always 1-ply cubeless, keeping only the 1-ply best move per roll at each node
  reduces the cubeful tree by up to 2^depth while producing the same result.
- Returns a cubeful equity per branch that accounts for future cube actions.
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
    clear_thread_local_caches()
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

When using the base (1-ply) strategy for VR mean computation,
`batch_evaluate_candidates_best_prob` evaluates all candidates and returns both
the best index and its probabilities in a single pass — no redundant NN calls.

### Ultra-Late Threshold

Positions deep in a trial have diminishing impact on the final result. The
`ultra_late_threshold` (default 2 for truncated rollouts) drops both checker play
and cube decisions to 1-ply at depth, eliminating expensive N-ply evaluations for
moves that barely affect the outcome.

For full rollouts with N-ply cube/checker strategies, set
`ultra_late_threshold = 9999` to disable ply reductions and use configured
strategies for the entire game.

### Bearoff Database Integration

When the bearoff database is loaded:
- **Bearoff positions as input:** If the starting position is a bearoff position,
  exact cubeless probabilities are returned immediately (no simulation needed).
- **During trials:** Bearoff positions that arise during play are evaluated
  exactly via the database, bypassing NN evaluation.
- **At truncation:** If the truncation position is a bearoff, the database
  provides exact evaluation.

The bearoff database is propagated to all internal strategies (checker, late
checker, truncation, inner rollouts) so that their N-ply evaluations also use
exact bearoff probs at leaf nodes.

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

The `best_move_index(candidates, pre_move_board)` method selects the best move
among candidates using rollout evaluation:

1. **1-ply filter:** Score all candidates at 1-ply. Sort by equity descending.
2. **Threshold filter:** Keep top `max_moves` within `threshold` of the best
   (using the rollout config's filter preset, typically TINY: 5 moves, 0.08).
3. **Rollout each survivor:** Call `rollout_position(candidate)` for each
   surviving candidate.
4. **Pick the best:** Return the candidate with the highest rollout equity.

Thread-local caches are cleared before evaluation to prevent cross-strategy
contamination.

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
| Cubeful truncation filter | {1, 0.0} | MoveFilter for `cubeful_equity_nply_multi` at truncation |
| Truncation PubEval threshold | 8 | PubEval activates at this many opponent candidates |
| Truncation PubEval keep | 6 | PubEval keeps this many after filtering |
| VR pre-filter threshold | 0.12 | 1-ply threshold for N-ply VR candidate narrowing |
| VR pre-filter max | 8 | Maximum candidates after VR pre-filter |
| PosCache capacity | 256K | Thread-local N-ply position cache entries |
| SharedPosCache capacity | 2M | Cross-thread position cache entries |
| SharedPosCache clear threshold | 75% | Clear when inserts exceed this fraction |
| Stratified dice levels | 6 | Hierarchical permutation depth |
| Max stratified turns | 128 | Half-moves with quasi-random dice |

### Standard Configurations (App Levels)

| Level | n_trials | trunc_depth | decision_ply | late_ply | late_threshold | ultra_late |
|-------|----------|-------------|-------------|----------|----------------|------------|
| 1T (XG Roller) | 42 | 5 | 1 | 1 | 20 | 2 |
| 2T (XG Roller+) | 360 | 7 | 2 | 1 | 2 | 2 |
| 3T (XG Roller++) | 360 | 5 | 3 | 2 | 2 | 2 |
| R (Full Rollout) | 1,296 | 0 | 1 | 1 | 20 | 2 |
