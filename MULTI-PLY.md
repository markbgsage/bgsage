# Multi-Ply Evaluation

Technical specification for the N-ply evaluation algorithms used for doubling cube
decisions and checker play analysis in backgammon. This document describes the
mathematics, data structures, recursion, optimizations, and implementation details
at a level sufficient for a complete reimplementation.

## Table of Contents

1. [Goal](#1-goal)
2. [Background: Cubeless vs Cubeful Equity](#2-background-cubeless-vs-cubeful-equity)
3. [1-Ply Cubeful Evaluation (Janowski Interpolation)](#3-1-ply-cubeful-evaluation-janowski-interpolation)
4. [N-Ply Cubeful Evaluation (Evaluate-All-and-Decide)](#4-n-ply-cubeful-evaluation-evaluate-all-and-decide)
5. [Cube Decision Extraction](#5-cube-decision-extraction)
6. [Move Selection Within the Cubeful Recursion](#6-move-selection-within-the-cubeful-recursion)
7. [N-Ply Checker Play Evaluation](#7-n-ply-checker-play-evaluation)
8. [Best Move Selection (Iterative Deepening)](#8-best-move-selection-iterative-deepening)
9. [Hybrid Evaluator Mode](#9-hybrid-evaluator-mode)
10. [Caching](#10-caching)
11. [Parallelization](#11-parallelization)
12. [Match Play](#12-match-play)
13. [Special Rules: Jacoby and Beaver](#13-special-rules-jacoby-and-beaver)
14. [2-Ply Detail Extraction](#14-2-ply-detail-extraction)
15. [Constants and Presets](#15-constants-and-presets)

---

## 1. Goal

Given a pre-roll board position and the current cube state, determine the optimal
cube action: should the player double, and if doubled, should the opponent take or
pass? At N-ply depth, the algorithm produces three key equity values:

- **E_ND** (No Double): expected equity if the player does not double
- **E_DT** (Double/Take): expected equity if the player doubles and the opponent takes
- **E_DP** (Double/Pass): expected equity if the player doubles and the opponent passes

The optimal decision follows from comparing these three values. Deeper ply levels
produce more accurate estimates by searching the game tree further.

## 2. Background: Cubeless vs Cubeful Equity

### Neural Network Outputs

The engine uses neural networks that output 5 cubeless probabilities from the
perspective of the player who just moved (post-move, pre-opponent-roll):

| Output | Meaning |
|--------|---------|
| P(win) | Probability of winning (includes gammons and backgammons) |
| P(gw) | Probability of winning a gammon (includes backgammons) |
| P(bw) | Probability of winning a backgammon |
| P(gl) | Probability of losing a gammon (includes backgammons) |
| P(bl) | Probability of losing a backgammon |

### Cubeless Equity

Cubeless equity assumes no further cube action is possible:

```
E_cubeless = 2*P(win) - 1 + P(gw) - P(gl) + P(bw) - P(bl)
```

This is normalized to a cube value of 1. Range: [-3, +3] (backgammon loss to
backgammon win).

### Pre-Roll vs Post-Move Probabilities

The NN evaluates a post-move board (the position after a move, before the opponent
rolls). To get pre-roll probabilities for the player about to roll:

1. Flip the board to the opponent's perspective
2. Evaluate the NN (returns post-move probs from the mover's perspective)
3. Invert probabilities: swap win/loss components

```
invert_probs(p) = [1-p[0], p[3], p[4], p[1], p[2]]
```

The difference between `evaluate(board)` and `invert(evaluate(flip(board)))` is one
tempo — being on roll is an advantage.

### Average Win/Loss Values (W and L)

W and L represent the average value of wins and losses respectively, accounting for
gammons and backgammons:

```
W = 1 + (P(gw) + P(bw)) / P(win)        if P(win) > 0, else 1
L = 1 + (P(gl) + P(bl)) / (1 - P(win))   if P(lose) > 0, else 1
```

W ranges from 1 (no gammons) to 3 (all backgammons). L similarly.

## 3. 1-Ply Cubeful Evaluation (Janowski Interpolation)

The 1-ply cubeful evaluation converts cubeless probabilities to cubeful equity using
Janowski's interpolation method. This is the leaf evaluation used at the bottom of the
N-ply search tree.

### Cube Efficiency (x)

The Janowski parameter `x` (cube efficiency) controls how much of the live-cube
premium is realized:

- **Contact/crashed positions:** x = 0.68
- **Race positions:** x = 0.55 + 0.00125 * avg_pips, clamped to [0.6, 0.7]

Where `avg_pips = (player_pips + opponent_pips) / 2`. Using the average (rather than
just the roller's pips) makes the evaluation perspective-independent, eliminating a
systematic bias at odd ply levels.

### Money Game: Janowski Formula

```
E_cubeful = E_dead * (1 - x) + E_live * x
```

Where:
- `E_dead` = cubeless equity (full gammon values)
- `E_live` = live-cube equity (piecewise linear function of P(win))
- `x` = cube efficiency

### Live-Cube Equity (Money Game)

The live-cube equity is a piecewise linear function defined by take point (TP) and
cash point (CP):

```
TP = (L - 0.5) / (W + L + 0.5)
CP = (L + 1.0) / (W + L + 0.5)
```

**Cube centered:**
```
p < TP:   E_live = -L + (-1 + L) * p / TP           [too weak to take]
TP ≤ p < CP:  E_live = -1 + 2 * (p - TP) / (CP - TP)  [doubling window]
p ≥ CP:   E_live = +1 + (W - 1) * (p - CP) / (1 - CP) [too good to double]
```

Under Jacoby rule (cube centered), the extreme segments are clamped:
- p < TP: E_live = -1.0 (pass, lose 1 point)
- p ≥ CP: E_live = +1.0 (double/pass, win 1 point)

The middle segment is unchanged because once the cube is turned, gammons count.

**Player owns cube:**
```
p < CP:   E_live = -L + (1 + L) * p / CP
p ≥ CP:   E_live = +1 + (W - 1) * (p - CP) / (1 - CP)
```

**Opponent owns cube:**
```
p < TP:   E_live = -L + (-1 + L) * p / TP
p ≥ TP:   E_live = -1 + (W + 1) * (p - TP) / (1 - TP)
```

### 1-Ply Cube Decision (Money Game)

Given pre-roll cubeless probabilities and cube state:

```
E_DP = +1.0  (opponent passes, player wins 1 cube unit)

E_ND = cl2cf_money(probs, current_owner, x, jacoby_active)

E_DT = 2 * cl2cf_money(probs, OPPONENT, x, false)
```

The DT equity is computed as if the opponent holds the doubled cube. The factor of 2
normalizes back to the original cube value (since `cl2cf_money` returns equity per cube
unit, and the doubled cube is worth 2x).

Decision:
- **Should double:** `min(E_DT, E_DP) > E_ND`
- **Should take:** `E_DT ≤ E_DP`
- **Optimal equity:** `should_double ? min(E_DT, E_DP) : E_ND`

## 4. N-Ply Cubeful Evaluation (Evaluate-All-and-Decide)

### Overview

The N-ply algorithm carries an array of cube states through the entire recursive
search tree. At each level, states are expanded into No-Double and Double/Take
branches, evaluated recursively, then collapsed by making optimal cube decisions
using the full recursive values. This approach (matching GNUbg's
`EvaluatePositionCubeful4`) means cube decisions emerge from actual tree values rather
than from heuristic predictions.

### Cube Count Index (cci)

The algorithm tracks `cci` cube states simultaneously. Two operations expand and
collapse this array:

1. **make_cube_pos** (expand: cci → 2·cci)
2. **get_ecf3** (collapse: 2·cci → cci)

At the top level of a cube decision, cci=2 (one ND state, one DT state). At each
recursion level, states double: cci → 2·cci → 4·cci → etc. The implementation
supports up to MAX_CCI=64 states (sufficient for 6+ ply depth).

### make_cube_pos (State Expansion)

For each input cube state `i`, produces two output states:

```
output[2i]     = ND branch: same cube state (no double)
output[2i + 1] = DT branch: cube_value doubled, opponent owns
```

Parameters:
- **fTop**: If true, skip DT branch creation (the top level already provides separate
  ND and DT states).
- **fInvert**: If true, flip all output states to the opponent's perspective (swap
  PLAYER ↔ OPPONENT ownership, swap match away scores). Used when preparing states
  for recursion into the opponent's half-move.

A DT branch is marked unavailable (sentinel `cube_value = -1`) when the player cannot
legally double (cube not owned, Crawford rule, max cube value reached, etc.).

### get_ecf3 (State Collapse)

For each ND/DT pair, makes the optimal cube decision using full recursive equities:

```
for each pair i:
    if DT branch available:
        rND = expanded_equities[2i]

        if money game:
            rDT = 2 * expanded_equities[2i + 1]   // normalize doubled cube to cube=1
        else (match):
            rDT = expanded_equities[2i + 1]        // MWC space, no scaling

        if beaver enabled and rDT < 0:
            rDT = 2 * rDT                          // Double/Beaver = 2 * Double/Take

        if money:
            rDP = 1.0
        else:
            rDP = dp_mwc(away1, away2, cube_value, is_crawford)

        if rDT >= rND and rDP >= rND:
            result[i] = min(rDT, rDP)    // Double; opponent picks best response
        else:
            result[i] = rND              // Don't double
    else:
        result[i] = expanded_equities[2i]   // No cube available
```

### Core Recursion: cubeful_recursive_multi

The main recursive function evaluates a pre-roll position for all cci cube states
simultaneously.

**Inputs:**
- `board`: pre-roll position from the roller's perspective
- `aciCubePos[]`: array of cci cube states (all from roller's perspective)
- `cci`: count of cube states
- `strategy`: neural network evaluation strategy
- `plies`: remaining search depth
- `fTop`: true at the top level (suppresses DT expansion in make_cube_pos)

**Output:** `arCubeful[]`: array of cci equity values (money) or MWC values (match).

**Algorithm:**

```
function cubeful_recursive_multi(board, cubeStates, cci, plies, fTop):

    // 1. Cache lookup
    if cached(board, cci, plies, fTop):
        return cached_values

    // 2. Terminal check
    if game_over(board):
        for each cube state:
            compute terminal equity/MWC
        return

    // 3. LEAF NODE (plies ≤ 1): NN evaluation + Janowski
    if plies ≤ 1:
        probs = invert(strategy.evaluate(flip(board)))    // pre-roll probs
        x = cube_efficiency(board)

        expanded[] = make_cube_pos(cubeStates, cci, fTop, fInvert=false)

        for each expanded state (2·cci total):
            if money:
                expanded_equity[i] = cl2cf_money(probs, state.owner, x, jacoby_active)
            else:
                expanded_equity[i] = cl2cf_match(probs, state, x)

        return get_ecf3(expanded_equities, cci, expanded_states)

    // 4. INTERNAL NODE (plies > 1): recurse over 21 dice rolls
    expanded[] = make_cube_pos(cubeStates, cci, fTop, fInvert=true)
    expanded_cci = 2 * cci
    accum[expanded_cci] = {0}

    for each of 21 dice rolls (d1, d2, weight):
        candidates = legal_moves(board, d1, d2)

        if no legal moves:
            // Standing pat: flip board, recurse
            recurse(flip(board), expanded[], expanded_cci, plies-1) into temp[]
        else:
            best = pick_best_move(candidates)     // by cubeless 1-ply equity
            opp_board = flip(candidates[best])
            recurse(opp_board, expanded[], expanded_cci, plies-1) into temp[]

        for i in 0..expanded_cci:
            accum[i] += weight * temp[i]

    // 5. Average and flip perspective
    for i in 0..expanded_cci:
        if money:
            accum[i] = -accum[i] / 36       // negate for opponent→player
        else:
            accum[i] = 1 - accum[i] / 36    // MWC complement

    // 6. Un-invert cube states back to current player's perspective
    for each expanded state:
        flip_cube_perspective(state)

    // 7. Collapse via optimal cube decisions
    return get_ecf3(accum, cci, expanded_states)
```

### Perspective Semantics

This is the most subtle aspect of the algorithm. At each recursion level:

1. **Board perspective**: The board is always from the roller's perspective. After
   the roller picks their best move, the result is flipped to the opponent's
   perspective before recursing.

2. **Cube state perspective**: When recurring into the opponent's half-move,
   `make_cube_pos` with `fInvert=true` flips all cube states to the opponent's view
   (PLAYER ↔ OPPONENT, match away scores swapped). After the recursive call returns
   values from the opponent's perspective, the perspective flip (negate for money,
   1-complement for MWC) and `flip_cube_perspective()` restore the current player's
   view.

3. **Leaf evaluation**: At the leaf, `make_cube_pos` uses `fInvert=false` because
   the Janowski evaluation is applied directly from the current player's perspective.

4. **Terminal positions**: When a post-move position is terminal (all checkers borne
   off), the terminal probabilities are from the mover's perspective. Since the
   accumulated values are from the opponent-of-the-mover's perspective (matching the
   recursive convention), terminal probabilities are inverted before accumulation.

### Dice Rolls

There are 21 unique dice combinations (6 doubles + 15 non-doubles), with total
probability weight of 36:

- 6 doubles: (1,1), (2,2), (3,3), (4,4), (5,5), (6,6) — weight 1 each
- 15 non-doubles: (1,2), (1,3), ..., (5,6) — weight 2 each

The weighted average over all 21 rolls divided by 36 gives the expected value.

## 5. Cube Decision Extraction

### Top-Level Entry: cube_decision_nply

The top-level function initializes two cube states and calls the recursion:

```
function cube_decision_nply(board, cube, strategy, n_plies):

    if n_plies ≤ 1:
        probs = invert(strategy.evaluate(flip(board)))
        return cube_decision_1ply(probs, cube)

    // Two initial states: ND and DT
    states[0] = cube                              // ND: current cube state
    states[1] = cube with (value=2*cv, owner=OPPONENT)  // DT: doubled cube

    cubeful_recursive_multi(board, states, cci=2, n_plies, fTop=true)
    // fTop=true prevents make_cube_pos from creating additional DT branches
    // at the top level (we already have separate ND and DT states)

    // results[0] = ND value, results[1] = DT value (at doubled cube scale)
```

### Money Game Decision

```
E_ND = results[0]
E_DT = 2 * results[1]     // scale from doubled-cube to original cube
E_DP = 1.0

if beaver enabled and E_DT < 0:
    E_DT = 2 * E_DT       // Double/Beaver equity
    is_beaver = true

should_double = min(E_DT, E_DP) > E_ND
should_take = E_DT ≤ E_DP
optimal_equity = should_double ? min(E_DT, E_DP) : E_ND
```

### Match Play Decision

```
nd_mwc = results[0]
dt_mwc = results[1]
dp_mwc = MET_after(away1, away2, cv, player_wins=true)

// Convert to equity for display (all at original cube value)
E_ND = mwc2eq(nd_mwc, away1, away2, cv, is_crawford)
E_DT = mwc2eq(dt_mwc, away1, away2, cv, is_crawford)
E_DP = mwc2eq(dp_mwc, away1, away2, cv, is_crawford)

// Decision in MWC space (canonical for match play)
should_double = min(dt_mwc, dp_mwc) > nd_mwc
should_take = dt_mwc ≤ dp_mwc
```

Special case: **post-Crawford automatic double** — when the trailing player (away > 1)
faces an opponent at 1-away, they should always double (losing costs the match
regardless of cube value, but winning scores more points).

### Cubeful Equity (Single Value)

The `cubeful_equity_nply` function returns a single cubeful equity value for a
position (not a full cube decision). It initializes cci=1 with the given cube state
and calls `cubeful_recursive_multi` with fTop=false. The internal expansion/collapse
handles all cube branching automatically.

## 6. Move Selection Within the Cubeful Recursion

At each internal node of the cubeful recursion, the algorithm must pick the best move
for each dice roll. Move selection uses **cubeless 1-ply equity** — the same move is
used for all cube states simultaneously. This is a key simplification: cubeful move
selection would require evaluating each candidate under each cube state independently,
which is prohibitively expensive and produces negligible accuracy improvement.

### Two-Stage Move Filtering

When a PubEval pre-filter strategy is provided and the number of legal moves exceeds
a threshold (16), a two-stage filter is applied:

**Stage 1: PubEval pre-filter:**
- Score all candidates with PubEval (fast linear evaluator)
- Keep the top 15 candidates (by evaluation score)
- Terminal positions (game over) always survive filtering

**Stage 2: Full model evaluation:**
- Score survivors with the main neural network strategy
- Pick the candidate with the highest cubeless equity
- If a `GamePlanStrategy` is available, use batch evaluation for efficiency

The PubEval pre-filter is provided by default in the `cube_decision_nply_unified`
binding. It reduces 1-ply NN evaluations on positions with many legal moves (doubles
can generate 30-90 candidates), with negligible impact on move selection accuracy
since PubEval's ranking correlates well with NN ranking for the top moves.

### Move Selection Constants

```
MOVE_FILTER_THRESHOLD = 16    // Apply pre-filter if > 16 candidates
MOVE_FILTER_KEEP = 15         // Keep top 15 after pre-filtering
```

## 7. N-Ply Checker Play Evaluation

The `MultiPlyStrategy` class implements N-ply lookahead for checker play (move
selection and position evaluation). This is a separate system from the cubeful
recursion in sections 4-6 — it evaluates **cubeless probabilities** at N-ply depth,
producing the 5 standard NN outputs. The cubeful recursion calls this system's base
strategy at leaf nodes, but this system is also used independently for checker play
analysis.

### Relationship to the Cubeful Recursion

The cubeful recursion (`cubeful_recursive_multi`) and the checker play evaluation
(`MultiPlyStrategy`) are related but independent systems:

- **Cubeful recursion** (section 4): Evaluates cube decisions. At leaf nodes it calls
  `strategy.evaluate_probs()` which goes to the base `GamePlanStrategy` (1-ply NN
  evaluation). At internal nodes it uses `strategy.evaluate_probs()` or
  `batch_evaluate_candidates_best_prob()` for move selection. It carries cube state
  arrays and operates in equity/MWC space.

- **Checker play evaluation** (this section): Evaluates cubeless probabilities at
  N-ply depth. Returns 5 probabilities. Used for checker play analysis ("which move
  is best?"), and also used within the cubeful recursion when the strategy passed to
  `cube_decision_nply` is a `MultiPlyStrategy` rather than a base `GamePlanStrategy`.

### Perspective Semantics

The most subtle aspect of the checker play evaluation is perspective management.

The neural network evaluates a **post-move board** (the position after a player has
moved, before the opponent rolls). It returns probabilities from the **mover's
perspective** — "how likely is it that the player who just moved to this position
will win?"

`evaluate_probs_nply(board, pre_move_board, plies)`:
- `board` is a post-move position from the current player's perspective
- `pre_move_board` is the board before the current player moved (used for NN plan
  classification only — which of the 5 NNs to select)
- Returns probabilities for the current player

When recursing after the opponent responds:
1. Flip `board` to get the opponent's pre-move board (`opp_board = flip(board)`)
2. Generate the opponent's legal moves from `opp_board`
3. The opponent's post-move positions are already from the opponent's perspective
4. Recurse on the opponent's best post-move position at `plies - 1`
5. **Invert** the returned probabilities to get current player's perspective

**Critical:** `evaluate(board)` ≠ `invert(evaluate(flip(board)))` — these differ by
one tempo (being on roll is an advantage).

### Core Algorithm: evaluate_probs_nply_impl

Evaluates a post-move position at N-ply depth, returning 5 cubeless probabilities.

```
function evaluate_probs_nply_impl(board, pre_move_board, plies, allow_parallel):

    // 1. Base case
    if plies ≤ 1:
        return base_strategy.evaluate_probs(board, pre_move_board)

    // 2. Terminal check
    if game_over(board):
        return terminal_probs(result)

    // 3. Cache lookup (thread-local)
    key = cache_key_for(board, plies)
    if cache_enabled:
        cached = thread_local_cache.lookup(key)
        if cached: return cached

    // 4. Shared cache lookup (active during parallel rollouts)
    shared_reservation = null
    if cache_enabled and shared_cache_active:
        result = shared_cache.lookup_or_reserve(key)
        if result.probs:
            thread_local_cache.insert(key, result.probs)  // promote
            return result.probs
        shared_reservation = result.reservation

    // 5. Flip to opponent's perspective
    opp_board = flip(board)

    // 6. Iterate over 21 dice rolls
    accum[5] = {0}
    for each of 21 dice rolls (d1, d2, weight):
        p1_probs = evaluate_single_roll(opp_board, d1, d2, plies)
        accum += weight * p1_probs

    // 7. Average
    avg = accum / 36.0

    // 8. Store in caches
    if cache_enabled:
        thread_local_cache.insert(key, avg)
    if cache_enabled and shared_cache_active:
        if shared_reservation:
            shared_cache.publish(shared_reservation, avg)
        else:
            shared_cache.insert(key, avg)

    return avg
```

### Opponent Move Selection (Per Roll)

For each of the 21 dice rolls, the algorithm generates the opponent's legal moves and
selects the best one. There are three paths depending on configuration:

**Pre-filtering (all paths):** If more than 20 legal moves exist and a cheap evaluator
(e.g., PubEval) is available, pre-filter to the top 15 candidates before full scoring.
Terminal positions always survive pre-filtering.

```
if move_prefilter and n_candidates > 20:
    score each candidate with cheap evaluator
    keep top 15 by score (terminals always survive)
```

**Path A — Forced move (1 legal move):**

If the opponent has exactly one legal move, evaluate it directly at `plies - 1`
and invert probabilities back to the current player's perspective. Terminal positions
are handled without recursion.

**Path B — Full-depth opponent mode (`full_depth_opponent = true`, plies > 2):**

The opponent evaluates ALL candidates at `(plies - 1)` depth and picks the move that
minimizes the current player's equity. This is the most accurate but slowest mode.

```
for each opponent candidate:
    opp_probs = evaluate_probs_nply_impl(candidate, opp_board, plies-1, false)
    p1_probs = invert(opp_probs)
    p1_equity = cubeless_equity(p1_probs)

// Opponent picks the move with lowest p1_equity (best for opponent)
return p1_probs with lowest p1_equity
```

This path can be parallelized internally: if `full_depth_threads > 1`, the candidates
are distributed across threads.

**Path C — Fast mode (default):**

The opponent picks their best move at 1-ply, then that single move is evaluated at
`(plies - 1)` depth. This is the standard mode used in practice.

```
// Score all candidates at 1-ply (using filter strategy if hybrid mode)
filter_s = filter_strategy ? filter_strategy : base_strategy
best_opp_idx = filter_s.batch_evaluate_candidates_equity(candidates, opp_board)

// Evaluate the best move at (plies - 1) depth
opp_probs = evaluate_probs_nply_impl(candidates[best_opp_idx], opp_board, plies-1, false)
return invert(opp_probs)
```

**2-ply optimization:** When `plies == 2` and no hybrid filter strategy is set, the
base case of the recursion (`plies - 1 = 1`) is a direct NN evaluation. Instead of
scoring candidates separately and then re-evaluating the best, the algorithm uses
`batch_evaluate_candidates_best_prob()` which returns both the best index and its
probabilities in a single pass — avoiding a redundant NN forward pass.

### Public Interface

```
// Evaluate a post-move position at N-ply depth (returns 5 cubeless probabilities)
evaluate_probs(board, pre_move_board) → probs[5]

// Evaluate a post-move position at N-ply depth (returns cubeless equity scalar)
evaluate(board, is_race) → equity

// Select the best move from a list of candidate post-move boards
best_move_index(candidates, pre_move_board) → index
```

All public methods delegate to `evaluate_probs_nply_impl` at the configured ply depth,
with `allow_parallel` set to true only at the outermost call level.

## 8. Best Move Selection (Iterative Deepening)

When selecting the best move among candidates at N-ply, the `best_move_index_impl`
function uses iterative deepening to progressively narrow the candidate set before
the expensive final evaluation.

### Algorithm

```
function best_move_index(candidates, pre_move_board, n_plies):

    if n_candidates ≤ 1: return 0
    if n_plies == 1: return base_strategy.best_move_index(candidates, pre_move_board)

    survivors = [0, 1, ..., n_candidates - 1]    // all indices

    // Phase 1: Iterative deepening filter chain
    for each step in filter_chain:
        if survivors.size() ≤ 1: break

        if step.ply == 1:
            // Batch 1-ply scoring (uses filter strategy if hybrid mode)
            filter_s = filter_strategy ? filter_strategy : base_strategy
            filter_s.batch_evaluate_candidates_equity(candidates, pre_move_board, equities)
        else:
            // N-ply evaluation for each survivor
            for idx in survivors:
                probs = evaluate_probs_nply_impl(candidates[idx], pre_move_board,
                                                  step.ply, allow_parallel=false)
                equities[idx] = cubeless_equity(probs)

        // Sort survivors by equity descending
        sort(survivors, by equities descending)

        // Filter: keep top max_moves within threshold of best
        best_eq = equities[survivors[0]]
        keep = count where (best_eq - equities[idx]) ≤ step.threshold
        keep = min(keep, step.max_moves)
        keep = max(keep, 1)
        survivors.resize(keep)

    // Phase 2: Final evaluation at full N-ply depth
    if survivors.size() == 1: return survivors[0]

    // Parallel evaluation of survivors (if enabled)
    for each survivor (parallel or serial):
        probs = evaluate_probs_nply_impl(candidates[idx], pre_move_board,
                                          n_plies, allow_parallel=false)
        equity = cubeless_equity(probs)

    return argmax(equity)
```

### Filter Chain Construction

The filter chain is built once at `MultiPlyStrategy` construction from the base
`MoveFilter` preset:

```
function build_filter_chain(n_plies, base_filter):
    chain = []
    if n_plies < 2: return chain

    // Step 1: Always filter at 1-ply with base preset
    chain.push({ply=1, max_moves=base.max_moves, threshold=base.threshold})

    // Step 2: At 4+ ply, add intermediate filter at (n_plies - 1)
    if n_plies >= 4:
        tight_threshold = max(0.01, base.threshold * 0.25)
        tight_max = max(2, base.max_moves * 2 / 5)
        chain.push({ply=n_plies-1, max_moves=tight_max, threshold=tight_threshold})

    return chain
```

**Default chains with TINY preset (5 moves, 0.08 threshold):**

| Target Ply | Step 1 | Step 2 | Final |
|------------|--------|--------|-------|
| 2-ply | 1-ply: keep 5 @ 0.08 | — | 2-ply |
| 3-ply | 1-ply: keep 5 @ 0.08 | — | 3-ply |
| 4-ply | 1-ply: keep 5 @ 0.08 | 3-ply: keep 2 @ 0.02 | 4-ply |

The intermediate step at 4-ply yields ~1.6x speedup by evaluating 2 instead of 5
candidates at full depth. The intermediate step is not used at 3-ply because 2-ply
rankings don't correlate well enough with 3-ply rankings on hard positions — the
intermediate filter prunes moves that turn out to be the 3-ply best.

### MoveFilter Presets

| Name | max_moves | threshold |
|------|-----------|-----------|
| TINY | 5 | 0.08 |
| NARROW | 8 | 0.12 |
| NORMAL | 8 | 0.16 |
| LARGE | 16 | 0.32 |
| HUGE | 20 | 0.44 |

### Parallelization of Final Evaluation

After the filter chain narrows candidates, the remaining survivors are evaluated at
full N-ply depth. If `parallel_evaluate` is enabled and `n_plies > 2`, survivors are
distributed across threads:

```
n_threads = min(parallel_thread_count(n_plies), n_survivors)
parallel_for(n_survivors, n_threads, evaluate each survivor)
```

Each thread calls `evaluate_probs_nply_impl` with `allow_parallel=false` to avoid
nested parallelism.

## 9. Hybrid Evaluator Mode

The `MultiPlyStrategy` supports an optional separate **filter strategy** for 1-ply
filtering and opponent move selection, while using the **base strategy** (the more
accurate model) for leaf evaluations at the deepest ply level.

### Motivation

If a smaller/faster neural network model is available, it can be used for the
"filtering" steps (1-ply candidate ranking, opponent move selection during recursion)
while the larger/more accurate model handles the final leaf evaluations. The idea is
that the fast model's move ordering is good enough for filtering, while accuracy
matters most at the leaves.

### How It Works

When `filter_strategy` is provided at construction:

- **1-ply candidate scoring** (in `best_move_index_impl` filter chain): Uses
  `filter_strategy.batch_evaluate_candidates_equity()` instead of `base_strategy`
- **Opponent move selection** (in `evaluate_probs_nply_impl` fast mode): Uses
  `filter_strategy.batch_evaluate_candidates_equity()` to pick the opponent's best
  move at 1-ply
- **Leaf evaluation** (at `plies ≤ 1`): Always uses
  `base_strategy.evaluate_probs(board, pre_move_board)`

### Construction

```
// Standard: single strategy for both filtering and evaluation
MultiPlyStrategy(base_strategy, n_plies, filter, ...)

// Hybrid: separate filter + leaf strategies
MultiPlyStrategy(base_strategy, filter_strategy, n_plies, filter, ...)
```

The `cache_salt` is derived from the base strategy pointer, ensuring different
strategy instances don't collide in the shared thread-local cache.

### Practical Note

Experiments with a half-size model (Stage 5 Small, 200h hidden) as filter strategy
and the full model (Stage 5, 400h hidden) as base showed marginal speedup at 4-ply
(~1.04x) because the NN forward pass is a minority of per-node cost — move
generation, input encoding, and cache management dominate. The hybrid mode is retained
for potential future use with faster filter strategies.

## 10. Caching

### Cubeful Recursion Cache

The cubeful recursion uses a thread-local open-addressing hash table with
epoch-based invalidation.

**Structure:**
- Fixed-size table: 8,192 entries (power of 2)
- Linear probing with max 4 probe steps
- Each entry stores: board, plies, cci, fTop, epoch, values[MAX_CCI]

**Hash function:**
```
h = plies * 0x9e3779b97f4a7c15
h ^= cci * 0x517cc1b727220a95
h ^= fTop * 0x6c62272e07bb0142
for each board[i]:
    h ^= board[i] * (0x9e3779b97f4a7c15 + i * 7)
    h = rotate_left(h, 7)
```

**Epoch-based invalidation:** A global atomic epoch counter is incremented before each
top-level cube decision call. Cache entries store the epoch they were written at.
Lookups reject entries from stale epochs. This solves cross-thread invalidation:
worker threads in the persistent thread pool retain their thread-local caches, but
stale entries are automatically rejected when the epoch changes.

```
// Before each top-level call:
global_epoch.fetch_add(1)

// On lookup: reject if entry.epoch != global_epoch
// On insert: stamp entry with current global_epoch
```

**Replacement policy:** On collision, replace the first empty, stale, or matching
slot (4 probe slots). If all 4 probe slots are occupied by current-epoch entries,
evict the first slot.

### Checker Play Position Cache (PosCache)

The `MultiPlyStrategy` uses a separate thread-local position cache for N-ply
checker play evaluation.

**Structure:**
- 256K entries (262,144), power of 2
- Open addressing with max 8 probe steps
- Auto-clear at 75% load factor
- Each entry stores: hash, plies, probs[5]

**Hash function:**
```
h = 0
for each board[i]:
    h ^= hash(board[i]) + 0x9e3779b9 + (h << 6) + (h >> 2)

// Incorporate configuration into hash:
h ^= cache_salt + 0x9e3779b97f4a7c15 + (h << 6) + (h >> 2)
h ^= plies + 0x9e3779b9 + (h << 6) + (h >> 2)
h ^= (full_depth_opponent ? 0x85ebca6b : 0x27d4eb2f)

key = h | 1    // ensure non-zero (0 is the empty marker)
```

### SharedPosCache (Cross-Thread)

For parallel rollouts, a lock-free shared position cache prevents redundant
evaluations across threads.

**Structure:**
- 2M entries, lock-free via CAS (compare-and-swap)
- State machine per entry: EMPTY(0) → CLAIMED(1) → COMPUTING(2) → READY(3)
- Max 8 linear probe steps

**Protocol:**
```
lookup_or_reserve(hash, plies):
    for probe in 0..8:
        if entry.state == READY and matches: return cached result
        if entry.state == EMPTY:
            CAS(EMPTY → CLAIMED): return reservation
    return miss (no empty slot)

publish(reservation, probs):
    write probs, set state = READY

abandon(reservation):
    set state = EMPTY
```

Threads that hit a COMPUTING entry spin briefly, then fall back to local computation.

## 11. Parallelization

### Thread Pool

A persistent thread pool (`MultiPlyThreadPool`) is created once and reused across
all evaluations.

- **Windows:** 8 MB stack per thread (required for deep recursion on crashed
  positions that generate hundreds of legal moves)
- **Thread count:** `min(hardware_concurrency, target)` where target depends on ply
  depth

**Thread count heuristic:**
```
if plies > 1:
    target = 4 * max(1, plies - 1)
    effective = (plies >= 4) ? min(8, target) : target
    n_threads = min(configured_threads, effective)
```

### Parallel Dice Roll Evaluation

In the cubeful recursion, the 21 dice rolls at each internal node can be evaluated
in parallel. This is enabled when:

```
allow_parallel = (n_threads > 1) and (plies > 2)
```

Parallelism is restricted to the top levels of the tree to avoid excessive thread
overhead. Child nodes use parallel evaluation only when `plies - 1 > 2`.

**Implementation:** `multipy_parallel_for(21, n_threads, fn)` distributes the 21 roll
evaluations across threads. The calling thread participates as worker 0. Each thread
writes its results into a separate slot of a results array (no synchronization needed
on the data). A completion counter + condition variable signal when all threads finish.

### Parallel Checker Play Evaluation

The checker play system has two parallelization points:

**1. evaluate_probs_nply_impl:** The 21 opponent-roll evaluations at the top level
of the recursion can be parallelized. Each thread evaluates a subset of the 21 rolls
and writes results into a shared array (no lock needed — each slot is owned by one
thread). Enabled only at the outermost recursion level:

```
allow_parallel = parallel_evaluate and (plies == n_plies) and (plies > 2)
n_threads = min(parallel_thread_count(plies), 21)
```

All deeper recursions force `allow_parallel = false` to avoid nested parallelism.

**2. best_move_index_impl:** After the filter chain narrows candidates, the surviving
moves are evaluated at full N-ply depth. If multiple survivors remain and parallelism
is enabled, they are distributed across threads:

```
use_parallel = parallel_evaluate and (n_plies > 2)
n_threads = min(parallel_thread_count(n_plies), n_survivors)
```

**3. full_depth_opponent mode:** When `full_depth_opponent = true` and `plies > 2`,
the evaluation of all opponent candidates at `(plies - 1)` depth can also be
parallelized across threads.

In all cases, each thread calls `evaluate_probs_nply_impl` with
`allow_parallel = false`, so only one level of parallelism is active at a time.

## 12. Match Play

Match play operates in **Match Winning Chance (MWC)** space internally, using the
Kazaross-XG2 Match Equity Table (MET).

### Key MET Functions

```
get_met(away1, away2, is_crawford) → MWC
    Lookup probability of winning the match from given score.

cubeless_mwc(probs, away1, away2, cv, is_crawford) → MWC
    Decompose NN probs into 6 exclusive outcomes (single win, gammon win,
    backgammon win, single loss, gammon loss, backgammon loss).
    Weight each by its MWC (e.g., gammon win = win 2*cv points).

eq2mwc(equity, away1, away2, cv, is_crawford) → MWC
    Linear: MWC = 0.5 * (equity * (MWC_win - MWC_lose) + (MWC_win + MWC_lose))

mwc2eq(mwc, away1, away2, cv, is_crawford) → equity
    Inverse: equity = (2*MWC - (MWC_win + MWC_lose)) / (MWC_win - MWC_lose)

dp_mwc(away1, away2, cv, is_crawford) → MWC
    MWC when opponent passes = MET_after(player wins cv points)
```

### Janowski in MWC Space (Match Play)

Match play uses Janowski interpolation in MWC space instead of equity space:

```
MWC_cubeful = MWC_dead * (1 - x) + MWC_live * x
```

Where `MWC_dead` is the cubeless MWC and `MWC_live` is computed via piecewise linear
interpolation using cash points derived recursively from the MET.

**Cash point computation** (`get_match_points`):
1. Find the dead-cube level: highest cube value where both players can survive a
   recube
2. At dead-cube levels: `CP = (DTL - DP) / (DTL - DTW)` using MET values
3. At live-cube levels: recursive formula using opponent's cash point from the level
   above
4. Auto-redouble detection: if a player would win the match at the new cube level,
   effective cube is 2×cv

Three `cl2cf_match` variants by ownership:
- **Centered:** 3-region piecewise linear (opponent too good | doubling window |
  player too good)
- **Player owns:** 2-region (below CP | above CP)
- **Opponent owns (unavailable):** 2-region (below opponent CP | above)

### Match Play in the N-Ply Recursion

The recursion operates entirely in MWC space for match play:

- Leaf evaluation: `cl2cf_match()` returns MWC
- Perspective flip: `1 - MWC/36` (complement) instead of `-E/36` (negate)
- Terminal positions: `cubeless_mwc(terminal_probs, ...)` instead of
  `cubeless_equity()`
- get_ecf3: DP value from `dp_mwc()`, DT not scaled (already in MWC space)
- Final result: converted to equity via `mwc2eq()` for display

### Crawford and Post-Crawford Rules

- **Crawford game** (is_crawford=true): No doubling allowed. `can_double()` returns
  false.
- **Post-Crawford** (one player at 1-away, not Crawford): Leader at 1-away cannot
  double (meaningless — they'd already win with a single win). Trailer should always
  double (automatic double logic).

## 13. Special Rules: Jacoby and Beaver

### Jacoby Rule

When Jacoby is active (money game, cube centered), gammons and backgammons count as
single wins/losses only. `jacoby_active()` returns true when:
`jacoby && is_money() && owner == CENTERED`.

**Effects on evaluation:**
- Terminal equity: `2 * P(win) - 1` (gammons zeroed)
- Live-cube extreme segments clamped to ±1.0
- Dead-cube equity still uses full gammon values (Janowski formulation)

Once the cube is turned (DT branch), `jacoby_active()` automatically becomes false
because `owner ≠ CENTERED`. No explicit deactivation needed.

### Beaver Rule

After being doubled, the opponent can immediately redouble (beaver) while retaining
cube ownership. Applied when DT equity < 0 from the doubler's perspective.

```
DB equity = 2 * DT equity
```

This is exact because `cl2cf_money()` returns equity normalized to cube=1,
independent of absolute cube value. A beaver doubles the cube value but keeps the
same ownership, so equity scales linearly.

Applied in both 1-ply and N-ply decisions, and within get_ecf3 at each recursion
level.

## 14. 2-Ply Detail Extraction

The `cube_decision_nply_with_details` function captures per-roll cubeful equities for
the first two turns of a cube decision analysis, providing visibility into the
internal structure of the N-ply evaluation.

### Output Structure

Two sections (ND and DT scenarios), each containing 21 player roll entries:

**PlayerRollDetail:**
- die1, die2: dice values
- post_move_board: board after player's optimal move (player's perspective)
- cubeful_equity: per-initial-cube, from player's perspective, incorporates
  opponent's optimal cube decision
- is_terminal: player's move ended the game
- opponent_dp: opponent has Double/Pass in this scenario (game over)
- opponent_rolls: 21 OpponentRollDetail entries (absent if terminal/DP)

**OpponentRollDetail:**
- die1, die2: dice values
- post_move_board: board after opponent's optimal move (player's perspective)
- cubeful_equity: per-initial-cube, from player's perspective, incorporates
  player's optimal cube decision

### Scaling Conventions

All equities are normalized to per-initial-cube units. The DT section equities
reflect the opponent's optimal cube action at the doubled cube level, scaled back to
per-initial-cube units.

Move selection at both levels uses 1-ply cubeless equity (matching the cubeful
recursion's internal behavior). Equities are evaluated at (N-1)-ply for player rolls
and (N-2)-ply for opponent rolls — so at 3-ply, player-roll equities are 2-ply
accurate and opponent-roll equities use 1-ply Janowski.

### Verification

The weighted average of `cubeful_equity` across all 21 ND player rolls (doubles
weight 1, non-doubles weight 2, divided by 36) equals `equity_nd`. Similarly for the
DT section.

## 15. Constants and Presets

### Cubeful Recursion

| Constant | Value | Description |
|----------|-------|-------------|
| MAX_CCI | 64 | Maximum simultaneous cube states |
| CUBEFUL_CACHE_SIZE | 8,192 | Cubeful cache entries per thread |
| Max probe distance | 4 | Linear probing for cubeful cache |

### Move Selection (Cubeful Recursion)

| Constant | Value | Description |
|----------|-------|-------------|
| MOVE_FILTER_THRESHOLD | 16 | Apply PubEval pre-filter if > 16 candidates |
| MOVE_FILTER_KEEP | 15 | Keep top 15 after pre-filtering |

### Move Selection (Checker Play)

| Constant | Value | Description |
|----------|-------|-------------|
| PREFILTER_THRESHOLD | 20 | Apply PubEval pre-filter if > 20 opponent moves |
| PREFILTER_KEEP | 15 | Keep top 15 after pre-filtering |

### Position Cache (Checker Play)

| Constant | Value | Description |
|----------|-------|-------------|
| CAPACITY | 262,144 (256K) | Entries per thread-local cache |
| MAX_PROBE | 8 | Linear probing distance |
| Load factor threshold | 75% | Auto-clear when reached |
| SharedPosCache CAPACITY | 2M | Cross-thread cache entries |

### Cube Efficiency

| Position Type | x Value |
|---------------|---------|
| Contact/crashed | 0.68 |
| Race (avg_pips ≤ 40) | 0.60 |
| Race (avg_pips = 80) | 0.65 |
| Race (avg_pips ≥ 120) | 0.70 |
| General race formula | 0.55 + 0.00125 * avg_pips, clamped [0.6, 0.7] |

### Thread Count

**Cubeful recursion** (`cube_decision_nply`, `cubeful_equity_nply`): The `n_threads`
parameter is passed through from the caller with no internal cap. Parallelism is
enabled when `n_threads > 1 && n_plies > 2`. The 21 dice rolls at each internal node
are distributed across the available threads. Child recursion restricts parallelism
to deeper nodes only (`child_allow_parallel = allow_parallel && (plies - 1 > 2)`).

**Checker play evaluation** (`MultiPlyStrategy::evaluate_probs_nply_impl`,
`best_move_index_impl`): Thread count is determined by `parallel_thread_count()`:

```
target = 4 * max(1, plies - 1)
n_threads = min(configured_threads, target)
```

| Ply Level | target |
|-----------|--------|
| 1-ply | 1 (no parallelism) |
| 2-ply | 4 |
| 3-ply | 8 |
| 4-ply | 12 |
| 5-ply | 16 |

The per-thread position cache (256K entries, ~7 MB) is small enough that 12-16
threads at 4-5 ply depth have acceptable total memory consumption (~100-110 MB)
while providing good parallelism across the 21 dice rolls.
