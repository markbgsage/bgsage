# bgsage — Open Sage Bot Engine Library

Neural-network backgammon engine with C++ core and Python interface. Licensed under MPL-2.0.

## Repo Boundary Rule

**This is a standalone library repo (`bgsage/`). When the working directory is
pointed at this folder, ALL new files, edits, and commits MUST stay within this
repo.** Never create or modify files outside this repo, even if parent CLAUDE.md
files (from a host project that vendors this repo as a submodule) are loaded as
context. New Python modules go in `python/bgsage/`, new scripts in `scripts/`,
new C++ in `cpp/`.

The same applies to *runtime* paths: every directory the code references —
compiled extension, weights, data, output dirs — must resolve inside `bgsage/`.
Scripts must not reach into a parent directory for `bgbot_cpp.pyd`, model
weights, or output logs. The compiled `bgbot_cpp.pyd` belongs in `bgsage/build/`,
default output directories resolve under `bgsage/logs/`, and scripts should set
`_PROJECT_ROOT = _SCRIPT_DIR.parent` (the bgsage repo root) — never
`.parent.parent` (which points at the host project). Several existing scripts
in `scripts/` still resolve paths via the host project root; treat that as
legacy and fix it when convenient.

## Git Worktree Rules

**CRITICAL: When working in a git worktree, ALL file operations (reads, edits,
writes, new files, builds, script execution) MUST use the worktree path — never
the main repo path.** The worktree path is shown in the environment as "Worktree
path" and is the primary working directory for the session.

- The worktree has its own branch. Commit and push from the worktree, then merge
  to main via PR or local merge — do NOT commit directly to main.
- Use relative paths or the worktree path for all tool calls. If you see
  yourself using the main repo path (e.g. `<repo-root>/` instead of
  `<repo-root>/.claude/worktrees/<name>/`), STOP and fix it.
- New files created in the main repo path will NOT be on the worktree branch.
- The MSVC build directory (`build_msvc/`) is shared across worktrees. After
  building, copy the `.pyd` to the worktree's `build/` directory.

## Project Structure

```
cpp/                         # C++ core (all performance-critical code)
  include/bgbot/            # Public headers
  src/                      # Implementation (.cpp/.cu)
  pybind/bindings.cpp       # pybind11 Python bindings
  CMakeLists.txt            # Full build (CUDA + CPU)
  CMakeLists_cpu.txt        # CPU-only build (Docker / macOS)
python/bgsage/               # Python package
  analyzer.py               # Public API: checker + cube analysis
  types.py                  # Dataclasses (Probabilities, MoveAnalysis, etc.)
  board.py                  # Board utilities (flip, possible_moves, etc.)
  weights.py                # Production model registry, WeightConfig, model discovery
  data.py                   # .bm file loading, training data parsing
  gnubg.py                  # GNUbg CLI wrapper for reference evaluation
  matchinfo.py              # Match play take points, dead cube take points, gammon prices
  text_export.py            # Galaxy/XG-compatible text export + move notation
  xg_compare.py             # Parse XG .xg files; compute per-game PR stats vs XG
scripts/                     # Training & benchmarking scripts
tests/                       # Python tests
models/                      # Production weights (5 files per model stage)
data/                        # GNUbg benchmark + training data + bearoff DB
```

## Architecture

**C++ core with Python presentation layer.** All compute (move generation, NN
evaluation, game simulation, training, benchmarking) runs in C++. Python handles
orchestration, configuration, data loading, and results display.

### 5-NN Game Plan Strategy

5 separate neural networks, selected by game plan classification:

| Network    | Hidden | Inputs | Description |
|------------|--------|--------|-------------|
| PureRace   | 200    | 196    | Contact broken, `is_race()` true |
| Racing     | 400    | 244    | Racing game plan, contact exists |
| Attacking  | 400    | 244    | Blitzing/hitting strategy |
| Priming    | 400    | 244    | Building a prime |
| Anchoring  | 400    | 244    | Defensive anchor strategy |

**Topology**: N_inputs -> N_hidden (sigmoid) -> 5 outputs (sigmoid)

**Outputs**: P(win), P(gammon_win), P(backgammon_win), P(gammon_loss), P(backgammon_loss)
- Equity = 2*P(win) - 1 + P(gw) - P(gl) + P(bw) - P(bl)

### Board Representation

- `std::array<int, 26>` — indices 1-24 are points (positive=P1, negative=P2)
- Index 0: P2 bar (always >= 0). Index 25: P1 bar (always >= 0)
- Board is ALWAYS from the perspective of the player on roll
- `flip()` reverses + negates + swaps bar after every move
- Starting: `[0,-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0]`

### Output Semantics (Critical)

The NN outputs probabilities from the perspective of the player who just moved,
representing the state **after the player's move, before the opponent rolls**.

To get pre-roll probabilities for the current player: (1) flip the board,
(2) evaluate the NN, (3) invert probs (P(win)->1-P(win), P(gw)<->P(gl), P(bw)<->P(bl)).

**Tempo effect**: `evaluate(board)` != `invert(evaluate(flip(board)))` — these differ
by one tempo. Being on roll is an advantage.

## Key C++ Components

| File | Purpose |
|------|---------|
| `board.h/cpp` | Board representation, flipping |
| `moves.h/cpp` | Legal move generation |
| `game.h/cpp` | Game simulation, self-play |
| `neural_net.h/cpp` | NN forward pass, backprop, weights |
| `encoding.h/cpp` | Input encoding, game plan classification |
| `training.h/cpp` | TD trainer, supervised trainer |
| `benchmark.h/cpp` | Benchmark scoring engine |
| `multipy.h/cpp` | N-ply search with caching |
| `rollout.h/cpp` | Monte Carlo rollouts with variance reduction |
| `cube.h/cpp` | Doubling cube (Janowski method) |
| `pubeval.h/cpp` | PubEval linear evaluator (reference opponent) |
| `bearoff.h/cpp` | One-sided bearoff database + BearoffStrategy wrapper |
| `cuda_nn.h/cu` | GPU training (CUDA/cuBLAS) |

## Production Model

The **production model** is the single source of truth for which trained weights
all scripts and the analyzer use by default. It is defined in one place:

```python
# python/bgsage/weights.py
PRODUCTION_MODEL: str = "stage5"   # ← change this line to promote a new model
```

The `MODELS` registry maps model names to their hidden sizes and weight file patterns:

```python
MODELS = {
    "stage6": {"hidden": (100, 300, 300, 300, 300), "pattern": "sl_s6_{plan}.weights.best"},
    "stage5": {"hidden": (200, 400, 400, 400, 400), "pattern": "sl_s5_{plan}.weights.best"},
    "stage4": {"hidden": (120, 250, 250, 250, 250), "pattern": "sl_s4_{plan}.weights.best"},
    "stage3": {"hidden": (120, 250, 250, 250, 250), "pattern": "sl_{plan}.weights.best"},
}
```

**To promote a new model:**
1. Add an entry to `MODELS` in `weights.py` with its hidden sizes and weight file pattern
2. Change `PRODUCTION_MODEL` to the new model name
3. That's it — all scripts and `BgBotAnalyzer` will use the new model automatically

**To benchmark an experimental model:**
All benchmark scripts accept `--model <name>` to override the production default:
```bash
python scripts/run_full_benchmark.py --model stage3
```

**Key API:**
- `WeightConfig.default()` → production model config
- `WeightConfig.from_model("stage3")` → specific model config
- `WeightConfig.add_model_arg(parser)` → adds `--model` to argparse
- `WeightConfig.from_args(args)` → resolves `--model` from parsed args
- `w.weight_args` → 10-tuple for C++ factory functions
- `w.hidden_sizes` → 5-tuple of hidden layer sizes
- `w.weight_paths` → dict of plan name → file path
- `w.validate()` → raises FileNotFoundError if any weight file missing

## Interfaces

**IMPORTANT: When writing scripts or code that needs checker play or cube action
analysis without targeting a specific model, always use `BgBotAnalyzer` from
`bgsage`.** This is the standard model-independent Python interface — it
abstracts away whether the production model is a 5-NN or 17-NN pair strategy.
Do NOT call low-level `bgbot_cpp` functions directly (e.g.
`cube_decision_nply`, `cube_decision_nply_pair`) unless you need to target a
specific model type or need parameters not exposed by the public API.

The library provides both **Python** and **C++** interfaces for five categories
of functionality. All Python functions default to the production model; pass
`weights=WeightConfig.from_model("stage3")` (or `model="stage3"` where
applicable) to use a different model.

### 1. Checker Play Analytics

Given a board position, two dice, and cube information, return all legal moves
ranked by equity, with cubeless post-move probabilities for each.

**Python** — `BgBotAnalyzer.checker_play()` (`python/bgsage/analyzer.py`):
```python
from bgsage import BgBotAnalyzer, STARTING_BOARD

analyzer = BgBotAnalyzer(eval_level="3ply", cubeful=True)
result = analyzer.checker_play(STARTING_BOARD, 3, 1, cube_value=1, cube_owner="centered")
# result: CheckerPlayResult with .moves (list[MoveAnalysis], best first)
for m in result.moves[:3]:
    print(f"{m.equity:+.3f}  {m.probs.win:.1%}  diff={m.equity_diff:+.4f}")

# Match play: add away1, away2, is_crawford keyword args
result = analyzer.checker_play(STARTING_BOARD, 3, 1, cube_value=1, cube_owner="centered",
                                away1=5, away2=3, is_crawford=False)
```

**C++** — Compose `possible_boards()` + `GamePlanStrategy::evaluate_probs()` + sort:
```cpp
#include "bgbot/moves.h"
#include "bgbot/strategy.h"
std::vector<Board> candidates;
possible_boards(board, die1, die2, candidates);
GamePlanStrategy strat(pr_w, rc_w, at_w, pm_w, an_w, ...);
for (auto& c : candidates) {
    auto probs = strat.evaluate_probs(c, board);
    double eq = NeuralNetwork::compute_equity(probs);
}
// Sort by equity descending.
```

### 2. Post-Move Position Analytics

Given a post-move board (right before the opponent's turn) and cube information,
return cubeful equity, cubeless equity, and cubeless probabilities.

**Python (single)** — `BgBotAnalyzer.post_move_analytics()` (`python/bgsage/analyzer.py`):
```python
from bgsage import BgBotAnalyzer

analyzer = BgBotAnalyzer(eval_level="2ply")
result = analyzer.post_move_analytics(post_move_board, cube_owner="centered")
# result: PostMoveAnalysis with .probs, .cubeless_equity, .cubeful_equity, .eval_level

# Match play:
result = analyzer.post_move_analytics(board, cube_owner="centered",
                                       away1=5, away2=3, is_crawford=False)
```

**Python (batch, parallelized)** — `batch_post_move_evaluate()` (`python/bgsage/batch.py`):
```python
from bgsage import batch_post_move_evaluate

positions = [
    {"board": board1, "cube_owner": "centered"},
    {"board": board2, "cube_owner": "player"},
]
results = batch_post_move_evaluate(positions, eval_level="1ply", n_threads=0)
# results: list[PostMoveAnalysis]
for r in results:
    print(f"CL={r.cubeless_equity:+.3f}  CF={r.cubeful_equity:+.3f}")
```

**C++** — `GamePlanStrategy::evaluate_probs()` + `cl2cf_money()`:
```cpp
GamePlanStrategy strat(...);
bool race = is_race(board);
auto probs = strat.evaluate_probs(board, race);
float cl_eq = NeuralNetwork::compute_equity(probs);
float x = cube_efficiency(board, race);
float cf_eq = cl2cf_money(probs, owner, x);
```

C++ batch: `bgbot_cpp.batch_evaluate_post_move(positions, strategy, n_threads)` via
pybind11; takes `list[(board, CubeOwner)]`, returns `list[dict]` with `probs`,
`cubeless_equity`, `cubeful_equity`.

### 3. Cube Action Analytics

Given a pre-roll board position and cube information, return cubeful equity
information for the three cube states (No Double, Double/Take, Double/Pass),
cubeless equity, and cubeless probabilities.

**Python** — `BgBotAnalyzer.cube_action()` (`python/bgsage/analyzer.py`):
```python
from bgsage import BgBotAnalyzer

analyzer = BgBotAnalyzer(eval_level="3ply", cubeful=True)
cube = analyzer.cube_action(board, cube_value=1, cube_owner="centered")
# cube: CubeActionResult with .equity_nd, .equity_dt, .equity_dp,
#   .should_double, .should_take, .optimal_action, .probs, .cubeless_equity
# jacoby=True by default for unlimited games; pass jacoby=False to disable

# Match play (Jacoby auto-disabled):
cube = analyzer.cube_action(board, cube_value=1, cube_owner="centered",
                             away1=5, away2=3, is_crawford=False)

# 2-ply details: per-roll breakdown for ND and DT scenarios (requires >= 2-ply)
cube = analyzer.cube_action(board, cube_value=1, cube_owner="centered",
                             incl_2ply_details=True)
# cube.details: dict with "nd" and "dt" keys, each a list of 21 player roll dicts
# See "2-Ply Detail Fields" below for structure
nd_rolls = cube.details["nd"]   # No Double scenario
dt_rolls = cube.details["dt"]   # Double/Take scenario
```

**2-Ply Detail Fields** (`incl_2ply_details=True`, requires >= 2-ply):

Returns per-roll details for the first two turns under both the **ND** (No Double)
and **DT** (Double/Take) scenarios. The ND section shows equities assuming the
player does not double; the DT section shows equities assuming the player doubles
and the opponent takes — even when those are not the optimal decisions. All boards
are from the original player's perspective. All equities are per-initial-cube, from
the player's perspective. No extra computational cost — data is captured from the
interior of the existing N-ply cubeful recursion.

`cube.details` — dict with two keys:
- `"nd"` — list of 21 player roll dicts (No Double scenario)
- `"dt"` — list of 21 player roll dicts (Double/Take scenario)

Boards (`checkers`) are identical in both sections (move selection is cubeless,
so cube state doesn't affect which move is chosen). Only the equities differ.

Each player roll dict (in both `"nd"` and `"dt"`):

| Field | Type | Description |
|-------|------|-------------|
| `die1` | int | First die (1-6) |
| `die2` | int | Second die (1-6) |
| `checkers` | list[int] | 26-element post-move board after optimal checker play |
| `cubeful_equity` | float | Cubeful equity incorporating opponent's optimal cube decision within this scenario. |
| `opponent_dp` | bool | Present and `True` when opponent has D/P in this scenario (game over) |
| `opponent_rolls` | list[dict] | 21 opponent roll details. **Absent** if the player's move is terminal, opponent has D/P, or the analysis is 2-ply. |

Each element of `opponent_rolls`:

| Field | Type | Description |
|-------|------|-------------|
| `die1` | int | First die (1-6) |
| `die2` | int | Second die (1-6) |
| `checkers` | list[int] | 26-element post-move board (player's perspective) |
| `cubeful_equity` | float | Cubeful equity incorporating player's optimal cube decision, scaled to initial-cube units. |

**ND section**: Cube remains at its current value. The opponent may double (if
they own or it's centered); per-roll equities reflect the opponent's optimal cube
action. The weighted average of `cubeful_equity` across all 21 ND player rolls
(doubles weight 1, non-doubles weight 2, divided by 36) equals `equity_nd`.

**DT section**: Cube is doubled (2× initial), opponent owns. The opponent may
redouble; per-roll equities reflect the opponent's optimal cube action at the
doubled cube level, scaled back to per-initial-cube units. The weighted average
of `cubeful_equity` across all 21 DT player rolls equals `equity_dt`.

Move selection at both captured levels uses 1-ply cubeless equity (this is what
keeps the ND and DT boards identical — cube state never affects the pick), with
the same PubEval keep-15 prefilter the recursion applies when a roll has >16
candidates. Note the recursion's own interior picks below these levels are 1-ply
CUBEFUL (see "N-Ply Cubeful Algorithm"), so per-roll detail equities can diverge
slightly from standalone (N-1)-ply evaluations of the same boards. Equities are
evaluated at (N-1)-ply for player rolls and (N-2)-ply for opponent rolls — so at
3-ply, player-roll equities are 2-ply accurate and opponent-roll equities use
1-ply Janowski.

**2-ply behavior**: at 2-ply only the player-roll level is captured — the
opponent-roll level would sit below the 1-ply leaf. Each post-move position is
evaluated directly at the 1-ply Janowski leaf (the same subtree the plain 2-ply
call evaluates), so per-roll equities are 1-ply accurate, `opponent_rolls` is
absent from every entry, headline equities match the plain (no-details) 2-ply
call, and the cost is that of a plain 2-ply cube evaluation. (Before 2026-06 the
details path always ran its two manual recursion levels, so a 2-ply details
request silently returned 3-ply-grade headline equities at ~21x the cost.)

**Python (batch, pre-roll)** — `batch_evaluate()` (`python/bgsage/batch.py`):
```python
from bgsage import batch_evaluate

positions = [{"board": b, "cube_value": 1, "cube_owner": "centered"} for b in boards]
results = batch_evaluate(positions, eval_level="3ply", n_threads=0)
# results: list[PositionEval] — includes probs, cubeless/cubeful equity, cube decision

# Match play: add optional away1, away2, is_crawford to position dicts
positions = [{"board": b, "cube_value": 1, "cube_owner": "centered",
              "away1": 5, "away2": 3} for b in boards]
```

**C++** — `evaluate_cube_decision()` (1-ply), `cube_decision_nply()` (N-ply),
`cube_decision_rollout()`:
```cpp
// 1-ply: evaluate_cube_decision(checkers, cube_value, owner, weight_args..., jacoby=false, beaver=false)
// N-ply: cube_decision_nply(checkers, cube_value, owner, n_plies, weight_args..., jacoby=false, beaver=false)
// Rollout: cube_decision_rollout(checkers, cube_value, owner, weight_args..., config..., jacoby=false, beaver=false)
// All accept optional jacoby/beaver to enable Jacoby/Beaver rules for money games
```

C++ batch pre-roll: `bgbot_cpp.batch_evaluate_positions(positions, strategy, n_threads)`
via pybind11; takes `list[(board, cube_value, CubeOwner[, away1, away2, is_crawford])]`,
returns `list[dict]`.

C++ batch checker play: `bgbot_cpp.batch_checker_play(inputs, strategy_1ply, [strategy_nply,]
filter_max_moves, filter_threshold, n_threads)` via pybind11; takes `list[dict]` with
`{board, die1, die2, cube_value, cube_owner}`, returns `list[dict]` each with `moves` list
sorted by cubeful equity desc. Two overloads: 1-ply (GamePlanStrategy only) and N-ply
(GamePlanStrategy + MultiPlyStrategy). Survivors evaluated at N-ply, rest at 1-ply.

**Python batch wrapper** — `batch_checker_play()` (`python/bgsage/batch.py`):
```python
from bgsage import batch_checker_play
positions = [
    {"board": b, "die1": 3, "die2": 1, "cube_value": 1, "cube_owner": "centered"}
    for b in boards
]
results = batch_checker_play(positions, eval_level="3ply", n_threads=0)
# results: list[CheckerPlayResult], each with .moves sorted best-first
```

### 4. Game Plan Classification

Given a board position, return the game plan for the player on roll and the
opponent. Plans: `"purerace"`, `"racing"`, `"attacking"`, `"priming"`, `"anchoring"`.

**Python (both plans)** — `classify_game_plans()` (`python/bgsage/board.py`):
```python
from bgsage import classify_game_plans

result = classify_game_plans(board)
# result: GamePlanResult with .player and .opponent
print(f"Player: {result.player}, Opponent: {result.opponent}")
```

**Python (player only)** — `classify_game_plan()` (`python/bgsage/board.py`):
```python
from bgsage import classify_game_plan
plan = classify_game_plan(board)  # -> str: "purerace", "racing", etc.
```

**C++** — `classify_game_plan()` (`encoding.h`):
```cpp
GamePlan gp = classify_game_plan(board);          // player on roll
GamePlan opp_gp = classify_game_plan(flip(board)); // opponent
```

C++ batch: `bgbot_cpp.classify_game_plans_batch(boards_np)` via pybind11;
takes `numpy array [N, 26]`, returns `int32 array` (0=purerace, 1=racing, etc.).

### 5. Game Utilities

Board manipulation, move generation, and game state queries.

**Python** — `python/bgsage/board.py` (all importable from `bgsage`):

| Function | Purpose |
|----------|---------|
| `flip_board(board)` | Flip to opponent's perspective |
| `possible_moves(board, die1, die2)` | All legal resulting positions |
| `possible_single_die_moves(board, die)` | Single-die moves with from/to (for UI) |
| `check_game_over(board)` | 0=not over, ±1/±2/±3 = single/gammon/backgammon |
| `is_race(board)` | True if contact is broken |
| `is_crashed(board)` | True if position is crashed |
| `invert_probs(probs)` | Swap player/opponent probabilities |
| `STARTING_BOARD` | Standard 26-element starting position |

**C++** — `board.h`, `moves.h`:

| Function | Purpose |
|----------|---------|
| `flip(board)` | Flip perspective |
| `possible_boards(board, d1, d2, results)` | Legal move generation |
| `check_game_over(board)` | Terminal detection |
| `is_race(board)` | Contact check |
| `is_crashed(board)` | Crashed position check |
| `invert_probs(probs)` | Invert 5-probability array |

### 6. Match Info

Match play gammonless take points, dead cube take points, and gammon prices.

**Python** — `python/bgsage/matchinfo.py`:

| Function | Description |
|----------|-------------|
| `take_points(away1, away2, cube_value)` | Gammonless live cube take points. Uses empirically-derived lookup tables (away-scores 2-5 for cv=1; select scores for cv=2) with Janowski approximation (x=0.68) as fallback. Handles automatic redouble cases (doubler wins match on single win at new cube level) with a special dead cube formula at 4C. Returns `(player1, player2)` tuple. |
| `take_points_dead_cube(away1, away2, cube_value)` | Gammonless dead cube take points, calculated exactly from the MET. Returns `(player1, player2)` tuple. |
| `take_points_janowski(away1, away2, cube_value, x)` | Gammonless Janowski live cube take points with explicit cube life index x. Returns `(player1, player2)` tuple. |
| `gammon_prices(away1, away2, cube_value)` | Gammon prices calculated exactly from the MET. Returns `(player1, player2)` tuple. |

All take points are gammonless (assume zero gammon/backgammon probability). For money
games: take point = 0.22, dead cube take point = 0.25, gammon price = 0.5 (both players).

### 7. Luck (Per-Roll Dice Fortune)

Given a position's cube analytics and the roll that actually happened, return how
lucky that roll was — a signed equity number.

**What the luck value means.** Luck is measured in **equity units, from the
perspective of the player who rolled**:

```
luck = actual_equity - average_equity
```

- `actual_equity` — cubeful equity after the *best play* with the roll that
  actually happened.
- `average_equity` — the weight-averaged equity over *all* possible rolls from
  the same position (doubles weight 1, non-doubles weight 2; ÷36).

So luck answers "how much did this roll help versus a typical roll here?"
**Positive = lucky** (the roll's best play beats an average roll), **negative =
unlucky**. Because it is a deviation from the mean, luck **averages to zero over
many rolls** — summing per-roll luck across a game isolates dice fortune from
decision quality (a player with high total luck won the dice, not necessarily the
game). This is the app-facing per-roll Luck shown in PR breakdowns; it is
distinct from **VR** (variance-reduction luck used internally to denoise
rollouts — see the Rollout section).

**This is a pure function over analytics, not a fresh evaluation.** The per-roll
equities are exactly the **ND (No Double) per-roll cubeful equities** that
`cube_action(incl_2ply_details=True)` already returns in `details["nd"]`. Luck
runs **no additional neural-net evaluation** — pass in analytics you already
have. Cube legality is irrelevant: the ND per-roll details exist whether or not
the player owns the cube.

**Python** — `python/bgsage/luck.py` (both importable from `bgsage`):

```python
from bgsage import BgBotAnalyzer, roll_luck

analyzer = BgBotAnalyzer(eval_level="3ply")

# Preferred: reuse analytics you already computed (no extra evaluation).
cube = analyzer.cube_action(board, cube_value=1, cube_owner="centered",
                            incl_2ply_details=True)          # the "bot analytics"
luck = roll_luck(cube, die1=3, die2=1)
# luck: LuckResult | None
#   .luck, .actual_equity, .average_equity, .ply, .level_label, .per_roll
print(f"{luck.luck:+.3f}  ({luck.level_label})")

# One-shot convenience (runs the cube analysis for you, then computes luck):
luck = analyzer.roll_luck(board, 3, 1)

# Opening roll excludes doubles (15 rolls, not 21):
luck = analyzer.roll_luck(STARTING_BOARD, 3, 1, is_opening_roll=True)
```

| Function | Description |
|----------|-------------|
| `roll_luck(cube, die1, die2, *, ply=None, level_label=None, is_opening_roll=False)` | Luck from a `CubeActionResult` produced with `incl_2ply_details=True`. `ply`/`level_label` default to being derived from `cube.eval_level` (an N-ply cube analysis gives (N-1)-ply luck). Returns `LuckResult` or `None`. |
| `luck_from_equities(per_roll, die1, die2, *, ply, level_label, is_opening_roll=False)` | Pure kernel over a sequence of `RollEquity`. Use when the per-roll equities come from somewhere other than `cube_action` (e.g. `batch_checker_play` best-move equities). |
| `BgBotAnalyzer.roll_luck(board, die1, die2, *, cube_value, cube_owner, away1, away2, is_crawford, jacoby, beaver, is_opening_roll=False)` | Convenience: runs `cube_action(incl_2ply_details=True)` at the analyzer's eval level, then computes luck. Prefer the free `roll_luck` when you already hold the cube analytics. |

**Match play** flows through the same `away1`/`away2`/`is_crawford` cube analysis,
so luck is expressed in match-equity-derived cubeful units automatically.

**Degenerate positions** (the actual roll missing from the details, or no details
present) return `None` — the caller decides how to record "no luck computed".

### Model Selection

All interfaces default to the production model. To use a specific model:

```python
from bgsage import BgBotAnalyzer, batch_evaluate, batch_post_move_evaluate
from bgsage.weights import WeightConfig

weights = WeightConfig.from_model("stage3")

# Single-position
analyzer = BgBotAnalyzer(weights=weights, eval_level="2ply")

# Batch pre-roll
results = batch_evaluate(positions, eval_level="1ply", weights=weights)

# Batch post-move
results = batch_post_move_evaluate(positions, eval_level="1ply", weights=weights)
```

### Return Types

| Type | Used by | Key fields |
|------|---------|------------|
| `CheckerPlayResult` | `checker_play()` | `.moves` (list[MoveAnalysis]), `.board`, `.die1`, `.die2` |
| `MoveAnalysis` | In CheckerPlayResult | `.board`, `.equity`, `.cubeless_equity`, `.probs`, `.equity_diff` |
| `PostMoveAnalysis` | `post_move_analytics()`, `batch_post_move_evaluate()` | `.probs`, `.cubeless_equity`, `.cubeful_equity` |
| `CubeActionResult` | `cube_action()` | `.equity_nd/dt/dp`, `.should_double`, `.should_take`, `.optimal_action`, `.probs`, `.details` (optional dict with `"nd"`/`"dt"` keys, with `incl_2ply_details`) |
| `PositionEval` | `batch_evaluate()` | `.probs`, `.cubeless_equity`, `.cubeful_equity`, `.equity_nd/dt/dp`, `.optimal_action` |
| `GamePlanResult` | `classify_game_plans()` | `.player`, `.opponent` |
| `LuckResult` | `roll_luck()`, `luck_from_equities()`, `BgBotAnalyzer.roll_luck()` | `.luck`, `.actual_equity`, `.average_equity`, `.ply`, `.level_label`, `.per_roll` (list[RollEquity]) |
| `RollEquity` | In LuckResult; input to `luck_from_equities()` | `.die1`, `.die2`, `.equity`, `.weight` (1 for doubles, 2 otherwise) |
| `Probabilities` | In all analysis types | `.win`, `.gammon_win`, `.backgammon_win`, `.gammon_loss`, `.backgammon_loss`, `.equity` |

## Bearoff Database

One-sided bearoff database for exact evaluation of endgame positions where all
checkers are in the home board (or borne off). Covers 54,264 positions
(C(21,6) = 15 checkers on 6 points). File: `data/bearoff_1sided.db` (~4.7 MB).

### What It Stores (Per Position)

- **Bearoff distribution** (32 × uint16): P(all checkers borne off in exactly k rolls)
- **Mean rolls** (float32): expected rolls to bear off all (for EPC)
- **Gammon distribution** (32 × uint16, only for all-15-on-board positions):
  P(0 checkers borne off after k rolls under optimal play)

### Position Indexing

Combinatorial number system (stars-and-bars). A 6-element checker count array
`[c1..c6]` maps to a unique index in [0, 54263]. O(1) arithmetic, no hash tables.

### Two-Sided Probability Computation

Combines two one-sided distributions for exact cubeless probs:
- P(win) = Σ P_player[i] × (1 - CDF_opponent[i-1]) (player on roll advantage)
- P(gammon_win) = Σ P_player[i] × ZeroOff_opponent[i-1]
- P(backgammon) = 0 (impossible in home-board bearoff)

### is_bearoff Check

Returns true when: both bars empty, player's checkers only on points 1-6,
opponent's checkers only on points 19-24. ~10 comparisons, negligible cost.

### Integration with Multi-Ply and Rollout

Both `MultiPlyStrategy` and `RolloutStrategy` accept an optional `BearoffDB*`
via `set_bearoff_db()`. When set:
- Multi-ply: bearoff positions short-circuit recursion (exact result returned)
- Rollout: bearoff input positions skip all trials (SE=0); truncation uses DB

The `BearoffStrategy` wrapper intercepts 1-ply leaf evaluations (VR, move selection).

### EPC (Effective Pip Count)

`BearoffDB::lookup_epc(board, player)` returns `mean_rolls × (49/6)`.
The mean_rolls includes the upcoming roll (1 checker on point 1 → mean=1.0 → EPC=8.167).

### Python API

```python
from bgsage import BgBotAnalyzer

# BearoffDB auto-loaded from data/ directory (bearoff_db=True by default)
analyzer = BgBotAnalyzer(eval_level="3ply")

# EPC for one side
epc = analyzer.epc(board, player=0)  # Returns float or None

# Disable bearoff DB
analyzer = BgBotAnalyzer(eval_level="1ply", bearoff_db=False)
```

### C++ API

```cpp
#include "bgbot/bearoff.h"
BearoffDB db;
db.load("data/bearoff_1sided.db");

if (db.is_bearoff(board)) {
    auto probs = db.lookup_probs(board);      // exact cubeless probs
    float epc = db.lookup_epc(board, 0);       // EPC for player on roll
}

// Wrap base strategy for automatic bearoff interception
auto base = std::make_shared<GamePlanStrategy>(...);
auto bearoff_strat = std::make_shared<BearoffStrategy>(base, &db);

// Set on multi-ply/rollout for deeper integration
multi_ply.set_bearoff_db(&db);
rollout.set_bearoff_db(&db);
```

### Generation

```bash
python scripts/generate_bearoff_db.py  # ~4 seconds, outputs data/bearoff_1sided.db
```

## Benchmark Scripts

All benchmark scripts default to the production model and accept `--model <name>`
to override. Scripts live in `scripts/`.

| Script | What it measures | Key args |
|--------|-----------------|----------|
| `run_full_benchmark.py` | Full suite: per-plan ER + contact/crashed/race ER + vs PubEval ppg + self-play distribution. Supports 1-ply through N-ply. | `--model`, `--ply N`, `--scenarios N`, `--threads N`, `--games N` |
| `run_rollout_benchmark.py` | Top-N worst 1-ply errors compared at 2-ply, 3-ply, 4-ply, rollout | `--model`, `--top N`, `--threads N` |
| `score_benchmark_pr.py` | Benchmark PR (equity error vs rollout reference, 103k decisions) | `--model`, `--plies N`, `--all-models`, `--all-plies` |
| `score_benchmark_pr_gnubg.py` | GNUbg's Benchmark PR (parallel GNUbg CLI subprocesses) | `--plies N` |
| `test_evaluate_probs.py` | Single position eval at 1-4 ply + GNUbg + rollouts | `--model`, `--checkers`, `--ply N` |
| `test_cube_decision.py` | Cube decisions vs 3 reference positions at 1-4 ply + rollout | `--model` |
| `test_unified_rollout.py` | Verify cubeful(max_cube=1) == cubeless at N-ply + rollout | `--model` |
| `eval_position.py` | Side-by-side Stage 5 vs GNUbg evaluation (cube action or checker play, 1-4 ply + rollout, money or match play) | `cube`/`checker` subcommand, `--checkers`, `--dice`, `--match`, `--score`, `--cube-value`, `--cube-owner` |
| `profile_cube_benchmark.py` | Profiling benchmark: 8 fixed cube positions (mix of money/match, bearoff/contact) evaluated serially with timing. For verifying optimisations don't change values. | level positional arg: `1ply`–`4ply`, `1T`, `2T`, `3T` |

```bash
# Full benchmark with production model (1-ply)
python scripts/run_full_benchmark.py

# Compare two models
python scripts/run_full_benchmark.py --model stage5
python scripts/run_full_benchmark.py --model stage3

# Multi-ply benchmark
python scripts/run_full_benchmark.py --ply 2
python scripts/run_full_benchmark.py --ply 3 --scenarios 500

# Score all registered models on Benchmark PR
python scripts/score_benchmark_pr.py --all-models

# Side-by-side Stage 5 vs GNUbg cube analysis (money game)
python scripts/eval_position.py cube --checkers "0,-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0"

# Side-by-side cube analysis (match play: 5-point match, player 3pts, opp 0pts)
python scripts/eval_position.py cube --checkers "..." --match 5 --score 3 0

# Side-by-side checker play analysis
python scripts/eval_position.py checker --checkers "..." --dice 3 1

# Side-by-side checker play analysis (match play)
python scripts/eval_position.py checker --checkers "..." --dice 3 1 --match 5 --score 3 0
```

## Comparing Sage to XG via .xg Files

The goal is to benchmark Sage's evaluations against XG (eXtreme Gammon),
historically considered the strongest backgammon engine. There are two
natural approaches:

1. **Head-to-head play** — Sage and XG play thousands of games against
   each other; tally points. XG has no API, so feeding moves between the
   two engines is a manual click-through ritual. Even at one game a minute,
   reaching a meaningful sample size takes far too long to be practical.

2. **Sage plays itself; XG scores it** — Sage plays both sides of many
   games, each game is exported as text, and XG's *Batch Analyze* feature
   scores the lot in a single pass. The Performance Rating (PR = equity
   error per decision × 500) measures how often Sage's chosen moves and
   cube actions deviate from XG's recommendations. Batch Analyze can chew
   through hundreds of files without intervention, so this is what we use.

### Workflow

Three steps; the middle one is manual because it goes through XG's GUI.

**1. Generate Sage-vs-Sage transcripts** with
[scripts/run_sage_vs_sage_games.py](scripts/run_sage_vs_sage_games.py):

```bash
# 200 games at 3-ply, 6 parallel worker processes
python scripts/run_sage_vs_sage_games.py 1 200 --level 3P --workers 6
```

```python
# Or in code:
from run_sage_vs_sage_games import run_sage_vs_sage_games

run_sage_vs_sage_games(
    initial_seed=1,    # game i uses seed (initial_seed + i)
    n_games=200,       # number of games to play
    level="3P",        # 1P/2P/3P/4P (N-ply) or 1T/2T/3T (truncated rollout)
    workers=6,         # parallel processes; pass 1 for serial
    out_dir=None,      # defaults to <project_root>/logs/sage_vs_sage
)
```

This writes `seed_<N>.txt` per game in the output directory. Each
transcript is a single money game (Jacoby + Beaver on, both sides
labelled "Sage") in Backgammon Galaxy / XG-import compatible text format.
With `workers > 1`, each worker process pre-loads its own analyzer at
`parallel_threads=1` so 6 workers × 1 thread don't oversubscribe the CPU.

**2. Run XG's Batch Analyze on the .txt folder** (manual):

Open XG → File → Batch Analyze → point it at the folder of .txt files.
**Critical:** check **"Save Games after analyze"** — without this, XG
prints summary stats but writes no per-game files, and step 3 has nothing
to read. When Batch Analyze finishes, the folder contains a matching
`seed_<N>.xg` next to each `seed_<N>.txt`.

**3. Aggregate PR stats** with
[scripts/aggregate_xg_pr.py](scripts/aggregate_xg_pr.py):

```bash
python scripts/aggregate_xg_pr.py [folder] [--pattern '*.xg']
```

For each .xg, the script parses turns via
`bgsage.xg_compare.parse_xg_game`, applies XG-style decision filters
(skip forced moves and trivial cube positions), and computes per-player
error totals + decision counts. It prints:

- Per-game total PR: `(P1_err + P2_err) / (P1_dec + P2_dec) * 500`.
- Across-games per-game PR mean / std dev / SEM.
- Aggregate PR computed from summed errors and summed decisions
  (weighted by decision count rather than equally per game).

### Public API (`bgsage.xg_compare`)

The .xg parser and PR aggregation are exposed for ad-hoc scripting:

```python
from bgsage.xg_compare import parse_xg_game, compute_game_pr_stats

with open("seed_8.xg", "rb") as f:
    turns = parse_xg_game(f.read())
stats = compute_game_pr_stats(turns)
# stats: {user_err, user_dec, bot_err, bot_dec, total_err, total_dec, pr}
```

`parse_xg_game` returns a list of turn dicts (`cube_action`,
`cube_analysis`, `checker_analysis`, `board_before`, `board_after`,
`dice`, ...). `compute_game_pr_stats` then calls `apply_decision_flags`
(also exposed) to populate `is_cube_decision` / `is_checker_decision`
plus the error fields, and aggregates them into the totals dict above.
The .xg file MUST contain exactly one game; `ValueError` otherwise.

## Building

### Windows (MSVC — required for Python 3.14)

**Python 3.14 is compiled with MSVC.** MinGW-compiled pybind11 modules crash due
to incompatible C runtime. Always use MSVC.

```powershell
# One-time CMake configure
cd build_msvc
cmake ..\cpp -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -Dpybind11_DIR=<path-to-pybind11>/share/cmake/pybind11

# Build
ninja bgbot_cpp
```

Need `#define NOMINMAX` before `#include <windows.h>` (std::min/max conflict).

### macOS (CPU-only)

```bash
mkdir build && cd build
cmake ../cpp -DCMAKE_BUILD_TYPE=Release -f ../cpp/CMakeLists_cpu.txt
make -j
```

On Apple Silicon: NEON intrinsics used instead of AVX2, no `-ffast-math`.

### Linux / Docker (CPU-only)

Use `CMakeLists_cpu.txt`. Flags: `-mavx2 -mfma -ffast-math -march=native`.

### Dependencies

- C++17, pybind11
- CUDA 13.1 toolkit (optional, for GPU training only)
- Python >= 3.10

## Input Encodings

**Tesauro (196 inputs)** — PureRace only:
- 4 thermometer inputs per point per player (24 points x 2 players x 4 = 192)
- Plus bar and borne-off per player (4 more)

**Extended (244 inputs)** — Racing/Attacking/Priming/Anchoring:
- 122 features per player
- [0-95]: Point encoding (same as Tesauro)
- [96]: Bar / 2.0
- [97-99]: Borne-off (3 buckets)
- [100-121]: 22 GNUbg-style features (escape, containment, timing, etc.)
- Requires `init_escape_tables()` once before use

## Training Pipeline

**CRITICAL: TD pre-training is required before SL.** SL from random init finds
unnatural minima. TD gives realistic probability distributions that SL refines.

1. **TD Self-Play (CPU)**: 25k-200k games @ alpha=0.1. Serial only (parallel TD deprecated).
2. **Supervised Learning (GPU)**: Backprop against GNUbg rolled-out probabilities.
   GPU via CUDA/cuBLAS, batch size 128.

**Key training rules:**
- All contact NNs train on ALL contact+crashed data, NOT game-plan subsets
- Narrow subsets cause catastrophic regression
- Game plan weight (`--gameplan-weight`) specializes each NN during SL

### Training a New Model from Scratch

To train a new 5-NN model (e.g., with different hidden sizes), follow these steps.
The process is long-running (~25-30 hours total for the current production schedule).

**Step 1: Create a training script** based on `scripts/run_stage5_training.py`.
Key parameters to customize:
- `N_HIDDEN` / `N_HIDDEN_PURERACE`: hidden layer sizes for contact / purerace NNs
- `MODEL_PREFIX` / `TD_MODEL_NAME`: file naming prefix (e.g., `sl_s5s` / `td_s5s`)
- `CONFIGS`: per-NN SL schedule (epochs, learning rates, game plan weights)

**Step 2: Register the model** in `python/bgsage/weights.py`:
```python
MODELS["stage5small"] = {
    "hidden": (100, 200, 200, 200, 200),
    "pattern": "sl_s5s_{plan}.weights.best",
}
```

**Step 3: Launch training as a detached process** (Windows).
Training is long-running and must survive past Claude Code's ~1h timeout.
Run from the bgsage repo root:
```bash
# IMPORTANT: Use python -u for unbuffered output. python must be on PATH.
powershell -Command "Start-Process -FilePath python -ArgumentList '-u','scripts\run_stage5small_training.py' -WindowStyle Hidden -RedirectStandardOutput 'logs\training.log' -RedirectStandardError 'logs\training_err.log'"
```

**Output buffering note:** Even with `-u`, C++ stdout from `bgbot_cpp` functions
(TD training, SL training) is internally buffered until the C++ function returns.
The TD benchmark_interval (default 10k games) triggers a benchmark + CSV write.
To monitor progress during TD training, check:
- `models/<td_model>_<plan>.weights` file timestamps (updated every benchmark_interval)
- `models/<td_model>.history.csv` (updated every benchmark_interval with game count +
  elapsed time + benchmark score). Note: CSV writes may also be delayed by OS buffering.

TD training prints benchmark scores every `benchmark_interval` games (default 10k).
SL training prints loss and benchmark scores every `print_interval` epochs (auto-set
to ~20 prints per phase, e.g., every 10 epochs for a 200-epoch phase).

**Step 4: Monitor training progress:**
```bash
# Check if process is alive
powershell -Command "Get-Process python* | Select-Object Id, CPU, StartTime"

# Check weight file timestamps (updated every benchmark_interval)
stat -c '%Y' models/td_s5s_racing.weights && date +%s

# Check history CSV
cat models/td_s5s.history.csv

# Check log output (may be delayed due to C++ internal buffering)
powershell -Command "Get-Content 'logs\training.log' -Tail 20"
```

**Step 5: After training completes**, run benchmarks:
```bash
python scripts/run_stage5small_benchmarks.py
```

**Estimated timing** (Stage 5 Small — 100h/200h hidden, Windows RTX 4070S):
- TD Phase 1 (200k games @ α=0.1): ~3.9 hours
- TD Phase 2 (1M games @ α=0.02): ~19 hours
- SL (5 NNs, ~2500 epochs each): ~2-4 hours (GPU)
- Total: ~25-27 hours

For reference, Stage 5 (200h/400h) TD training at the larger hidden size is ~2x
slower per game due to the larger matrix multiplies.

### Key Training Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_stage5_training.py` | Stage 5 (200h/400h) full training |
| `scripts/run_stage5small_training.py` | Stage 5 Small (100h/200h) full training |
| `scripts/run_td_gameplan_training.py` | TD self-play (standalone, low-level) |
| `scripts/run_gpu_sl_training.py` | GPU SL training (standalone, per-NN) |

### Standard TD + SL Schedule

The production training schedule (used for Stage 5):

**TD (CPU, serial):**
- Phase 1: 200k games @ α=0.1 (high learning rate for initial learning)
- Phase 2: 1M games @ α=0.02 (low learning rate for refinement)

**SL (GPU, per-NN):**

All contact NNs train on ALL contact+crashed data (not game-plan subsets). The
**game plan weight (gpw)** controls specialization: positions matching the NN's
game plan get `gpw` × the gradient weight, while all other positions get weight 1.0.
Higher gpw = stronger specialization toward that plan's positions. PureRace trains
on separate purerace-only data (gpw not applicable).

| NN | Schedule | gpw | Effective gradient % |
|----|----------|-----|---------------------|
| Racing | `100ep@α=20 → 200ep@α=10 → 200ep@α=3.1 → 500ep@α=1.0` | 2.0 | ~48% |
| Attacking | same | 5.0 | ~59% |
| Priming | same | 5.0 | ~56% |
| Anchoring | same | 1.5 | ~27% |
| PureRace | `200ep@α=20 → 500ep@α=6.3 → 500ep@α=2.0` | — | 100% (separate data) |

**gpw tuning:** Racing uses gpw=2.0 instead of 5.0 because Racing positions are
the most common contact plan (~37% of training data). At gpw=5.0, Racing positions
dominate 74% of the gradient, which destabilizes smaller networks (200h). The
threshold is between 63% (gpw=3.0, stable) and 74% (gpw=5.0, diverges). Larger
networks (400h, Stage 5) can handle gpw=5.0 for Racing.

Each SL phase resumes from the `.best` weights of the previous phase. Benchmark
scoring runs after each epoch; the best-scoring weights are saved as `.best`.

See "Benchmark Scripts" section above for all benchmarking commands.

### Hybrid Evaluator (Multi-Ply with Separate Filter Model)

The `MultiPlyStrategy` supports an optional separate filter strategy for 1-ply
filtering and opponent move selection, while using the main (leaf) strategy for
leaf evaluations. This allows using a fast/small model for filtering with an
accurate/large model for final evaluation.

**C++ API:**
```cpp
// Standard (single strategy for both):
auto strat = std::make_shared<MultiPlyStrategy>(base, n_plies, filter);

// Hybrid (separate filter + leaf):
auto strat = std::make_shared<MultiPlyStrategy>(base, filter_strat, n_plies, filter);
```

**Python API:**
```python
# Standard multi-ply
multipy = bgbot_cpp.create_multipy_5nn(*w.weight_args, n_plies=3)

# Hybrid multi-ply (fast filter + accurate leaf)
multipy = bgbot_cpp.create_multipy_hybrid_5nn(
    *w_leaf.weight_args,      # 10 args: 5 weight paths + 5 hidden sizes
    *w_filter.weight_args,    # 10 args: 5 weight paths + 5 hidden sizes
    n_plies=3)

# Hybrid rollout
rollout = bgbot_cpp.create_rollout_hybrid_5nn(
    *w_leaf.weight_args,
    *w_filter.weight_args,
    n_trials=360, truncation_depth=7, decision_ply=2)
```

The hybrid mode affects:
- `best_move_index_impl`: filter strategy scores candidates for ranking/pruning
- `evaluate_probs_nply_impl`: filter strategy selects opponent's best move at 1-ply
- Leaf evaluation (`plies=0`): always uses the base (accurate) strategy
- VR in rollouts: always uses base strategy (unaffected by hybrid mode)

## Multi-Ply Search

- 1-ply: Direct NN evaluation
- 2-ply: Average over 21 opponent rolls (~60x slower with TINY filter)
- 3-ply: Recursive (~800-1000x slower than 1-ply)

**Move filter**: After 1-ply scoring, keep top `max_moves` within `threshold` equity.
Default TINY: 5 moves, 0.08 threshold.
**The ranked list never carries a 1-ply value in its top two**: pruned
candidates keep their 1-ply equity, which is not on the survivors' N-ply
scale, so whichever of #1/#2 is stale is promoted (evaluated at N-ply) and
the list re-sorted until both are full-depth. See Stage 11s for the bug
this replaced.

### Iterative Deepening Filter Chain

When selecting the best move at N-ply (`best_move_index`), candidates are narrowed
through multiple filter passes at progressively deeper ply levels before the final
evaluation. This avoids evaluating all 1-ply survivors at the full (expensive) target
ply — intermediate passes at cheaper ply levels prune weak candidates early.

**Filter chain structure**: A sequence of `MoveFilterStep{ply, max_moves, threshold}`.
Each step scores all current survivors at the step's ply depth, then keeps the top
`max_moves` within `threshold` equity of the best. After all steps, the remaining
survivors are evaluated at the full target ply to determine the best move.

**Default chains** (auto-generated from the base `MoveFilter` preset via
`build_filter_chain()`):

| Target | Step 1 | Step 2 | Final |
|--------|--------|--------|-------|
| 2-ply | 1-ply: keep 5 @ 0.08 | — | 2-ply |
| 3-ply | 1-ply: keep 5 @ 0.08 | — | 3-ply |
| 4-ply | 1-ply: keep 5 @ 0.08 | 3-ply: keep 2 @ 0.02 | 4-ply |

The intermediate step is only added at 4-ply and above. At 3-ply, the chain is a
single 1-ply filter pass (same as the old behavior) because 2-ply rankings don't
correlate well enough with 3-ply rankings on hard positions — the intermediate filter
prunes moves that turn out to be the 3-ply best. At 4-ply, the intermediate 3-ply
evaluation is accurate enough for safe pruning, and the speedup is significant.

The second step uses a tighter filter derived from the base preset:
`max_moves = max(2, base.max_moves * 2/5)`, `threshold = max(0.01, base.threshold * 0.25)`.

**Example (4-ply with TINY filter)**:
1. Score all 16 legal moves at 1-ply → keep top 5 within 0.08 of best
2. Score 5 survivors at 3-ply → keep top 2 within 0.02 of best
3. Score 2 survivors at 4-ply → pick the best

Without iterative deepening, step 2 is skipped and all 5 survivors go directly to
4-ply evaluation. Since each 4-ply evaluation costs ~0.5s, evaluating 2 instead of 5
saves ~1.5s — roughly a **1.6x speedup** on 4-ply checker play.

**Implementation**: `MoveFilterStep` struct and `build_filter_chain()` in `multipy.h`.
The chain is built once in the `MultiPlyStrategy` constructor and stored as
`filter_chain_`. The `best_move_index_impl()` function in `multipy.cpp` loops through
the chain, evaluating survivors at each step's ply level through the cubeful
evaluation engine's dead-cube path (`cubeless_tree_probs`; the old recursion
only in hybrid / full-depth-opponent modes).

**Optimizations**: AVX2 FMA intrinsics, fast sigmoid LUT, open-addressing position
cache, incremental delta evaluation, transposed weight matrix.

## Rollout

Monte Carlo evaluation with variance reduction. Stratified first roll
(36 dice pairs). Parallelized trial execution via work-stealing
(`atomic<int> next_trial`).

**VR (variance reduction) decoupled from decision ply:** VR always uses 1-ply
(raw NN) for both mean and actual computations, regardless of the decision
strategy's ply level. Move selection still uses the full N-ply decision strategy.
Since VR tracks luck = (actual - mean) with both sides using the same ply, biases
cancel. This eliminates ~90% of N-ply evaluations. When `n_trials % 36 == 0`, VR
is skipped on move 0 (stratified dice makes luck sum to exactly zero). N-ply
strategies inside trials use serial evaluation (`parallel_evaluate=false`) — all
parallelism is across trial paths.

**Shared caches for trial acceleration:**
- **Move0Cache**: Pre-computes first-move decisions for all 21 stratified dice rolls.
  Shared across all trials (first roll is deterministic per trial index). Populated
  serially before trial threads start.
- **Move1Cache**: Pre-computes second-move decisions: for each of 21 first-roll
  outcomes, generates all 21 second-roll best moves at the configured decision ply.
  Avoids redundant N-ply best-move-index (BMI) calls across trials. Used by both
  cubeful (`cubeful_cube_decision`) and cubeless (`run_trials_parallel`) paths.
- **SharedPosCache**: Lock-free cross-thread position cache (2M entries, CAS-based
  state machine: EMPTY→COMPUTING→READY). Threads that hit a COMPUTING entry spin
  briefly then fall back to local computation. Eliminates redundant N-ply evaluations
  across threads when `n_threads > 1`.

**Unified trial function** (`run_trial_unified`): A single function handles both
cubeless (`n_branches=0`) and cubeful (`n_branches>0`) rollout modes. The
`start_post_move` flag controls starting convention: `true` for checker-play
evaluation (flip board, opponent first), `false` for cube decisions (no flip,
SP first). When all branches have dead cubes (`cube_is_dead()`), all cubeful
overhead is skipped — zero performance cost compared to dedicated cubeless code.

**Cubeful mode** (for cube decisions): Two-branch simulation — ND (no double)
and DT (double/take) branches share the same board evolution and dice. Cube
decisions during trials are configurable via `TrialEvalConfig`:
- 1-ply (default): `cube_decision_1ply()` — Janowski on 1-ply cubeless probs
- N-ply: `cube_decision_nply()` — full cubeful recursion (evaluate-all-and-decide)
- Truncated rollout: `cubeful_cube_decision()` on inner RolloutStrategy (n_threads=1)
Double/pass terminates the branch immediately. VR luck tracked in cubeful value
space per-branch (always 1-ply). Match play works entirely in MWC space
(`cl2cf_match`, `cubeless_mwc`, `dp_mwc`), with `away1/away2` swapped at each
perspective flip. Money game branches use equity-based logic unchanged. Jacoby
rule is propagated through `CubeInfo` on each branch; VR luck, terminal payoffs,
and truncation all respect `jacoby_active()`.

**Cubeful evaluation engine:** the N-ply cubeful evaluation engine in
[cpp/src/cube_eval.cpp](cpp/src/cube_eval.cpp) powers all cubeful analytics —
cube action, checker play, and post-move evaluation — both inside rollout
trials (escalated cube decisions, truncation eval, cubeful BMI) and at the
standalone 2-4 ply levels via the cube.h entry points (`cube_decision_nply*`,
`cubeful_equity_nply*`, `cubeful_probs_nply`,
`cubeful_probs_and_equity_nply`; the single-cube wrappers in cube.cpp compose
the multi entries). Key mechanisms: batched delta-eval interior picks with
1-ply cubeful move selection, leaf reuse at 2-ply nodes, hash-dedup move
generation (`possible_boards_unsorted`), a deep-node PubEval pre-filter
(16/14, enabled only for rollout-internal evaluations via the
`deep_prefilter` parameter), cubeless probs accumulated through the same tree
walk (so rollout truncation needs one walk, not two), a per-thread
cube-state-keyed memoization cache, the move-1 cube-decision cache, and a
1-ply screen that gates in-trial cube-decision escalations.

The same engine also runs the **cubeless** N-ply evaluation everywhere: the
rollout internals (in-trial N-ply move selection, cubeless truncation
evaluation) and the standalone paths (`MultiPlyStrategy::evaluate_probs_nply`
and the cubeless `best_move_index` rescore — i.e. post-move N-ply analytics,
benchmark ER scoring, self-play) all go through the engine with a single
**dead cube** (`cube_value=1, max_cube_value=1` — Janowski bypassed, interior
picks reduce to cubeless 1-ply). Dead-cube tree nodes are shared across trial
threads via the rollout's `SharedPosCache` (probs fully determine dead-node
values); bearoff positions short-circuit to exact DB probs. The old cubeless
recursion (`evaluate_probs_nply_impl`) survives only for the hybrid evaluator
(separate filter strategy). Note the engine classifies candidates by their
own boards (per-candidate NN selection) where the old recursion used the
pre-move board — values shift ~0.01-0.02 on some plan-boundary contact
positions, and the benchmark ER improved (stage9 contact 2-ply 7.99 → 7.76).
Full specification: **`MULTI-PLY.md`** (sections 4-7) and **`ROLLOUT.md`**.
Benchmarks against saved baselines: `scripts/bench_3t.py` (cube action),
`scripts/bench_3t_checker.py` (checker play, fixed dice in
`scripts/bench_checker_dice.json`), `scripts/bench_3t_postmove.py`
(post-move eval), `scripts/bench_3t_cubeless.py` (cubeless rollout of
post-move positions), `scripts/bench_nply_cubeless.py` (standalone N-ply
cubeless: post-move probs at 2-4 ply + contact/race benchmark ER) — each
supports `--save`/`--compare` with a material band of max(SE, 0.01) for
equities and 0.005 for probabilities (the cubeless rollout bench widens the
prob band to each prob's own SE).

### Truncated Rollouts (XG Roller-style)

Truncated rollouts are short Monte Carlo simulations truncated at a fixed depth
with N-ply evaluation at the truncation point. They are stronger than pure N-ply
search but faster than full rollouts, making them the best speed/accuracy tradeoff
for position evaluation.

**Key parameters:**
- `n_trials`: Number of trial games per candidate (72-360 typical for truncated rollouts)
- `truncation_depth`: Half-moves before truncating and evaluating with NN (0 = play to completion)
- `decision_ply`: Ply depth for move selection during early trial moves
- `truncation_ply`: Ply depth for evaluation at the truncation point (-1 = same as `decision_ply`).
  Using a lower ply here (e.g. 2-ply when `decision_ply=3`) gives a large speed improvement
  with small accuracy tradeoff, since truncation evaluation is the dominant cost.
- `late_ply`: Ply for move selection after `late_threshold` half-moves (-1 = same as `decision_ply`)
- `late_threshold`: Half-move index where decision ply switches from `decision_ply` to `late_ply`
- `ultra_late_threshold`: Half-move index where checker and cube evaluation drops to
  1-ply regardless of configured strategies (default 2). Set high (e.g. 9999) to
  disable ply reductions and use configured strategies for the full game — required
  for accurate full rollouts with N-ply strategies. At 1-ply, the VR best-candidate
  pick is reused directly — zero additional BMI cost. Also controls VR thinning:
  at ultra-late moves, VR is computed only at even half-moves (odd ones skipped).
- `enable_vr`: Variance reduction (always true for truncated rollouts, uses 1-ply)

**Checker play strategy selection during trials** (evaluated in order, first match wins):
- Race positions: always `base_` (1-ply, nearly perfect for pure races)
- At or after `ultra_late_threshold`: `base_` (1-ply)
- Before `late_threshold`: `checker_strat_` (configured checker evaluation)
- At or after `late_threshold`: `checker_late_strat_` (configured late checker evaluation)
- Truncation evaluation: the cubeful evaluation engine at `truncation_ply` (defaults to `decision_ply`); 1-ply truncation uses the base strategy directly

**Cube decision strategy selection during trials** (same fallback chain):
- Race positions: always `base_` (1-ply)
- At or after `ultra_late_threshold`: `base_` (1-ply)
- Before `late_threshold`: `cube_strat_` (configured cube evaluation)
- At or after `late_threshold`: `cube_late_strat_` (configured late cube evaluation)

Cube decisions get cubeless pre-roll probs via
`invert_probs(strat.evaluate_probs(flip(board), flip(board)))`, then apply Janowski.
This pattern works for any Strategy: 1-ply, N-ply (MultiPlyStrategy), or truncated
rollout (child RolloutStrategy).

**XG Roller equivalences** (XG uses XG ply convention = our convention):

| XG Level | n_trials | truncation_depth | decision_ply | late_ply | late_threshold |
|----------|----------|-------------------|-------------|----------|----------------|
| XGRoller           | 42  | 5 | 1 | -1 | 20 |
| XGRoller+          | 360 | 7 | 2 | 1  | 2  |
| XGRoller++ Checker | 360 | 5 | 3 | 2  | 2  |
| XGRoller++ Cube    | 360 | 7 | 3 | 2  | 2  |

**App level names**: `truncated1` is XG-Roller-style but uses **72** trials (2×36),
not XG's 42 — 42 isn't a multiple of 36, so it over-weights 6 ordered first rolls and
biases the rollout (benchmark PR 2.23 → 0.50 going 42 → 72). The `XGRoller` row above
(42) still shows how to replicate XG Roller exactly. `truncated2` is XG-Roller+-style
but diverges: cube 2-ply throughout, 2-ply truncation eval (`truncation_ply=2`), and
checker 1-ply only after the first ply (`late_threshold=1`, `ultra_late_threshold=9999`)
— beats XG Roller+ (benchmark PR 0.89 → 0.36) vs the old `ultra_late=2` 1-ply drop.
`rollout` = full rollout (1296 trials, play to completion). `truncated3` no longer
maps to a single XG level: it uses `truncation_depth=7`, `decision_ply=3`,
`late_ply=2`, `late_threshold=2`, **`ultra_late_threshold=9999`** (3-ply early, then
2-ply for the rest of each trial — no 1-ply drop), making it closer to XG Roller++
than the old `truncated3` = XG Roller++ Checker (trunc-5) mapping. See the
`truncated3` branch in `analyzer.py`.

```python
from bgsage import BgBotAnalyzer

# XGRoller equivalent
analyzer = BgBotAnalyzer(eval_level="rollout",
    n_trials=42, truncation_depth=5, decision_ply=1)

# XGRoller+ equivalent
analyzer = BgBotAnalyzer(eval_level="rollout",
    n_trials=360, truncation_depth=7, decision_ply=2,
    late_ply=1, late_threshold=2)

# XGRoller++ Checker equivalent
analyzer = BgBotAnalyzer(eval_level="rollout",
    n_trials=360, truncation_depth=5, decision_ply=3,
    late_ply=2, late_threshold=2)

# XGRoller++ Cube equivalent
analyzer = BgBotAnalyzer(eval_level="rollout",
    n_trials=360, truncation_depth=7, decision_ply=3,
    late_ply=2, late_threshold=2)
```

**VR speed optimizations:**
- **Thinned VR**: At ultra-late moves (>= `ultra_late_threshold`), VR is computed
  only at even moves. Odd ultra-late moves skip VR entirely. Since E[luck] = 0,
  this doesn't bias the estimate — just increases variance slightly. When
  `ultra_late_threshold` is set high (no ply reductions), thinning never activates
  and VR is computed at every move.
- **VR candidate prefilter**: When a roll generates >20 legal moves (common for
  doubles), candidates are pre-filtered to the top 20 by pip heuristic before
  1-ply evaluation. The actual roll's candidates are kept unfiltered for move
  selection. Reduces encoding cost for doubles with 50-96 candidates.
- **1-ply move1 selection**: Move1Cache uses 1-ply (base_) for move selection
  instead of late_decision_strat_. The VR averaging over many trials makes
  higher-ply move selection unnecessary in the move1 cache.
- **No prefill barrier**: Trials start immediately after each thread finishes its
  prefill work, without waiting for all 21 entries. run_trial_unified handles
  missing cache entries via CAS (compute on demand).

### Separate Checker/Cube Evaluation Strengths

Rollout trials support independent evaluation strengths for checker play (move
selection) and cube decisions. Each can be configured as N-ply or truncated rollout,
with separate late/ultra-late fallbacks. When no per-purpose configs are set, both
checker and cube default to `decision_ply` (`build_rollout_strategies` in
`rollout.cpp` resolves an unset `cube` config to `decision_ply` to match checker
play). This is why the named 1T/2T/3T levels do 1/2/3-ply cube decisions
respectively, not 1-ply.

**`TrialEvalConfig`** struct (C++: `rollout.h`, Python: `bgbot_cpp.TrialEvalConfig`):
- `ply`: N-ply depth (0 = unset/inherit, 1 = raw NN, 2+ = multi-ply)
- `rollout_trials`: When > 0, use truncated rollout instead of N-ply
- `rollout_depth`: Truncation depth for inner rollout (default 5)
- `rollout_ply`: Decision ply within inner rollout (default 1)

**`RolloutConfig`** fields for per-purpose evaluation:
- `checker`: Checker play evaluation config
- `checker_late`: Late-game checker play config
- `cube`: Cube decision evaluation config
- `cube_late`: Late-game cube decision config
- `ultra_late_threshold`: Half-move where checker/cube drop to 1-ply (default 2,
  set to 9999 to disable and use configured strategies for the full game)

**Checker play** uses the `Strategy` interface (`best_move_index`). When
`TrialEvalConfig.is_rollout()`, a child `RolloutStrategy` with `n_threads=1` is
created as the evaluation strategy.

**Cube decisions** use proper cubeful evaluation, NOT Janowski on cubeless probs:
- 1-ply (default): `cube_decision_1ply()` (Janowski on 1-ply cubeless probs)
- N-ply: `cube_decision_nply()` (full evaluate-all-and-decide cubeful recursion)
- Truncated rollout: `cubeful_cube_decision()` on an inner `RolloutStrategy`
  with `n_threads=1` (two-branch ND/DT cubeful rollout)

This means N-ply and rollout cube decisions during trials produce the same quality
of cube actions as the top-level cube analysis — cube decisions at each ply level
emerge naturally from recursion, not from heuristic Janowski conversion.

**Python API — low-level (`bgbot_cpp.cube_decision_rollout`):**

```python
import bgbot_cpp
from bgsage.weights import WeightConfig
w = WeightConfig.default()

# Full rollout, 3-ply for both checker and cube, no ply reductions
# IMPORTANT: set ultra_late_threshold=9999 for accurate full rollouts
# (default=2 drops to 1-ply at move 2+, biasing results)
result = bgbot_cpp.cube_decision_rollout(
    checkers=board,
    *w.weight_args[:5],   # 5 weight paths
    **dict(zip(['n_hidden_purerace','n_hidden_racing','n_hidden_attacking',
                'n_hidden_priming','n_hidden_anchoring'], w.hidden_sizes)),
    n_trials=1296, truncation_depth=0, decision_ply=1,
    n_threads=16, enable_vr=True,
    checker=bgbot_cpp.TrialEvalConfig(ply=3),
    cube=bgbot_cpp.TrialEvalConfig(ply=3),
    ultra_late_threshold=9999,
    progress=lambda done, total: print(f"{done}/{total}"),
)
# result: dict with equity_nd, equity_nd_se, equity_dt, equity_dt_se,
#   cubeless_equity, cubeless_se, probs, prob_std_errors, ...

# Full rollout, 3-ply checker, 1T cube (XG Roller-style cube decisions)
result = bgbot_cpp.cube_decision_rollout(
    checkers=board, *w.weight_args[:5],
    **dict(zip(['n_hidden_purerace','n_hidden_racing','n_hidden_attacking',
                'n_hidden_priming','n_hidden_anchoring'], w.hidden_sizes)),
    n_trials=1296, truncation_depth=0, decision_ply=1,
    n_threads=16, enable_vr=True,
    checker=bgbot_cpp.TrialEvalConfig(ply=3),
    cube=bgbot_cpp.TrialEvalConfig(rollout_trials=42, rollout_depth=5, rollout_ply=1),
    ultra_late_threshold=9999,
)

# Strategy object approach (reusable across positions)
rollout = bgbot_cpp.create_rollout_5nn(
    *w.weight_args[:5],
    **dict(zip(['n_hidden_purerace','n_hidden_racing','n_hidden_attacking',
                'n_hidden_priming','n_hidden_anchoring'], w.hidden_sizes)),
    n_trials=1296, truncation_depth=0, decision_ply=1,
    n_threads=16, enable_vr=True,
    checker=bgbot_cpp.TrialEvalConfig(ply=3),
    cube=bgbot_cpp.TrialEvalConfig(ply=3),
    ultra_late_threshold=9999,
)
result = rollout.cube_decision(checkers=board, cube_value=1,
    owner=bgbot_cpp.CubeOwner.CENTERED)
```

**`ultra_late_threshold` guidance:**
- Default `2`: fast, suitable for truncated rollouts (short games, ~5-7 moves)
  where most moves are within the threshold anyway
- `9999`: use configured strategies for the entire game — required for accurate
  full rollouts (`truncation_depth=0`) with N-ply or rollout cube evaluators
- Full rollout with 3-ply checker + 3-ply cube + `ultra_late_threshold=9999` gives
  results matching XG at ~100s/position (1296 trials, 16 threads)

**Progress callback**: Both `cube_decision_rollout()` and `rollout.cube_decision()`
accept `progress=callable` — called with `(completed, total)` periodically during
execution. Useful for progress bars in UI. The callback is called from worker threads
with the GIL automatically acquired.

**Not yet implemented:**
- Early stopping (XG Roller+ stops at 0.010 confidence, minimum 180 games)

## Doubling Cube

Janowski interpolation for both money games and match play. Optional Jacoby rule
for unlimited games (default on in Python API). Cube efficiency: 0.68 contact,
pip-dependent for race (unchanged for match play).

### Jacoby Rule

Optional rule for unlimited (money) games: while the cube remains centered (never
doubled), gammons and backgammons count as single wins/losses only. Once either
player doubles (cube is turned), gammon values are restored. Does not apply to
match play.

**Implementation:** `CubeInfo` carries a `bool jacoby` flag. `CubeInfo::jacoby_active()`
returns true only when: `jacoby && is_money() && owner == CENTERED`. When active:
W=1, L=1, dead-cube equity = `2*P(win) - 1`. The DT branch turns the cube →
`jacoby_active()` automatically becomes false (no explicit deactivation needed).

**Defaults:** Python public API defaults `jacoby=True`. C++ bindings default
`jacoby=false`. Auto-disabled when match play params are present.

### Beaver Rule

Optional rule for unlimited (money) games: after being doubled, the opponent can
immediately redouble (beaver) while retaining cube ownership. This punishes
incorrect doubles where DT equity < 0 from the doubler's perspective.

**Math:** DB (Double/Beaver) equity = 2 * DT equity. This is exact at all ply
levels because `cl2cf_money()` returns equity normalized to cube=1, independent
of absolute cube value. A beaver doubles the cube value but keeps the same
ownership (OPPONENT), so the equity scales linearly. No third recursion branch
is needed anywhere.

**When does beaver apply?** When DT < 0 from the doubler's perspective. Since
DB = 2*DT: when DT < 0, DB < DT < DP, so the opponent prefers beaver over
take. When DT >= 0, DB >= DT, so take is better for the opponent — standard
DT/DP logic applies.

**Output:** `CubeDecision` has `bool is_beaver`. When `is_beaver=true`, the
`equity_dt` field contains the DB equity (= 2*DT). `optimal_action` string:
`"Double/Beaver"` when `is_beaver && should_double`. `should_take = true` when
beaver applies (opponent IS accepting the double, plus beavering).

**Implementation:** `CubeInfo` carries a `bool beaver` flag. Beaver logic is
applied at the decision layer in `cube_decision_1ply_money()`, `get_ecf3()`
(N-ply), `cube_decision_nply()` (top-level), and `cube_decision_rollout()`
(binding). Rollout internal cube decisions via `cube_decision_1ply()` also
respect the beaver flag; a beaver results in cube_value *= 4 (double + beaver).

**Janowski is NOT affected** by beavers. The formulas (take point, cash point,
live cube equity, cube efficiency) are unchanged. Beavers are an additional
decision layer on top.

**Defaults:** Python public API defaults `beaver=True`. C++ bindings default
`beaver=false`. Auto-disabled when match play params are present.

### Max Cube Value (Cubeless Mode)

`CubeInfo.max_cube_value` caps the cube at a given value (0 = unlimited). When
`cube_is_dead(ci)` (max_cube_value > 0 && cube_value >= max_cube_value):
- `can_double()` returns false
- Janowski is bypassed (returns `cubeless_equity(probs)` directly)
- Rollout skips all cubeful overhead (zero performance cost vs cubeless)
- `should_double` is always false

Setting `max_cube_value=1, jacoby=False` produces cubeless-equivalent equity.
All cube decision bindings accept `max_cube_value` (default 0).

### Money Game

Three equities compared: ND (no double), DT (double/take), DP (double/pass = +1.0).
Double if `min(DT, DP) > ND`. Opponent takes if `DT <= DP`.
When Jacoby is active, ND uses W=1/L=1 (gammons zeroed); DT always has
`jacoby_active()=false` since the cube is turned.
When beaver is enabled and DT < 0, DB = 2*DT replaces DT in the decision:
opponent chooses min(DB, DP) vs ND.

### N-Ply Cubeful Algorithm (Evaluate-All-and-Decide)

The N-ply cubeful evaluation carries an **array of cube states** through the entire
recursion tree, rather than predicting cube actions at intermediate nodes. This
eliminates the need for heuristic cube-action predictions and produces accurate
cubeful equities at any depth.

**Core concept — cube count index (cci):** At each recursion level, the algorithm
tracks `cci` cube states simultaneously. Two helper operations expand and collapse
this array:

1. **make_cube_pos** (expand: cci -> 2*cci): For each input cube state, create two
   branches — a No-Double branch (same state) and a Double/Take branch (doubled cube,
   opponent owns). The DT branch is skipped when the player can't legally double.
   The `fInvert` flag flips cube perspective (PLAYER <-> OPPONENT) when entering the
   opponent's turn.

2. **get_ecf3** (collapse: 2*cci -> cci): For each ND/DT pair, compute the optimal
   cube decision using full recursive values: `rND` = recursive ND equity,
   `rDT` = 2 * recursive DT equity (money), `rDP` = +1.0 (money). If doubling
   improves equity (`min(rDT, rDP) > rND`), the result is `min(rDT, rDP)`;
   otherwise the result is `rND`.

**Recursion (`cubeful_recursive_multi`):**

- **Leaf (plies=0):** Single NN eval -> cubeless probs. Expand via make_cube_pos,
  apply Janowski (`cl2cf`) to each expanded state, collapse via get_ecf3.

- **Internal (plies>0):** Expand via make_cube_pos (with fInvert=true for opponent's
  perspective). For each of 21 dice rolls: generate moves, pick best by 1-ply CUBEFUL
  equity (`cl2cf` against `aciCubePos[0]`, the primary cube state — shared across
  all cube states; no per-cube subtree fork), flip to opponent perspective, recurse
  at plies-1. Average over 36 total weight, flip perspective back, collapse via
  get_ecf3.

**Top-level entry points:**

- `cube_decision_nply`: Starts with cci=2 (ND state + DT state), fTop=true. Returns
  both ND and DT equities from a single tree traversal.
- `cubeful_equity_nply`: Starts with cci=1, fTop=false. The internal expansion/collapse
  handles all cube branching automatically.

**Key properties:**
- Cube decisions at every level use full recursive values (not heuristic predictions)
- Move selection uses 1-ply cubeful equity (`cl2cf` against the primary cube state
  `aciCubePos[0]`) — captures match-awareness throughout the tree without per-cube
  forking. For multi-cube callers (e.g. `cube_decision_nply` carrying ND+DT) the pick
  is biased toward `cubes[0]`; this matches the rollout's shared-board MVP.
- Janowski `x` is applied at 1-ply leaf nodes for the final cubeful conversion AND
  at interior nodes for the cubeful move-pick (using each candidate's own cube_x)
- The cci array grows and shrinks at each level (1->2->4->...->collapse back)
- Both money game and match play use the same recursion; only the leaf conversion
  (`cl2cf_money` vs `cl2cf_match`) and get_ecf3 scaling differ

### Match Play

Match state: `MatchInfo{away1, away2, is_crawford}`. When `away1=0, away2=0`, falls
back to money game behavior (all existing callers unchanged).

**Key files:**
- `cpp/include/bgbot/match_equity.h` / `cpp/src/match_equity.cpp` — MET data + utilities
- `cube.h` / `cube.cpp` — `cl2cf_match()`, `cube_decision_1ply_match()`, `cubeful_mwc_recursive()`

**Hardcoded Kazaross-XG2 MET** (from GNUbg): 25x25 pre-Crawford + 25 post-Crawford values.
- `get_met(away1, away2, is_crawford)` → MWC for the player needing `away1` points
- `cubeless_mwc(probs, away1, away2, cv, is_crawford)` → weighted MWC from 6 outcomes
- `eq2mwc()` / `mwc2eq()` — linear conversion anchored at win/lose cv points
- `dp_mwc()` → MWC when opponent passes (player wins cv points)
- `can_double_match()` → Crawford/post-Crawford/dead cube rules

**Janowski in MWC space:** `MWC_cf = MWC_dead * (1-x) + MWC_live * x`
Three ownership variants: centered (3-region piecewise linear), owned (2-region),
unavailable (2-region). Unified dispatcher: `cl2cf()` → money or match.

**N-ply match recursion** works entirely in MWC space. Opponent decisions use MWC
maximization. Final results converted to equity via `mwc2eq()`.

**Crawford rule:** No doubling allowed. **Post-Crawford:** Leader at 1-away can't
double; trailer should double immediately.

**Equities are always normalized:** DP = +1.0 in equity space for both money and match
(by definition of the `mwc2eq` linear normalization). ND and DT vary by score.

## Current Best Scores (Production Model: stage5)

| Metric | 1-ply | Target |
|--------|-------|--------|
| Contact ER | 9.87 | < 10.5 |
| Race ER | 0.95 | < 0.643 |
| vs PubEval | +0.633 | > +0.63 |

Benchmark PR (103k decisions): 1-ply=2.47, 2-ply=1.85, 3-ply=1.53.

The production model is defined in `python/bgsage/weights.py` — see "Production Model"
section above. See `MODEL_BENCHMARKS.md` for full comparison of all trained models.

## Stage 6 (S6) — Mid-Size Model

**Purpose:** Mid-size model (100h PureRace, 300h contact NNs) between Stage 5 Small
(100h/200h) and Stage 5 (200h/400h). Tests whether 300h contact NNs close the gap
to 400h.

**Weights:** Registered as `"stage6"` in `python/bgsage/weights.py`. Weight files
are `sl_s6_{plan}.weights.best` in `models/`.

**Training:** Same TD + SL pipeline as Stage 5. TD: 200k games @ α=0.1 + 1M @ α=0.02.
SL: same schedule except Racing and Priming use gpw=2.0 (not 5.0 — gpw=5.0 causes
divergence at 300h for Priming, similar to Racing at 200h in S5S).

**Per-plan ER (1-ply):**

| Plan | Stage 5 (400h) | S6 (300h) |
|------|---------------|-----------|
| PureRace | 0.95 | 1.00 |
| Racing | 5.74 | 5.90 |
| Attacking | 8.74 | 8.73 |
| Priming | 8.59 | 9.58 |
| Anchoring | 11.06 | 11.34 |
| **Contact** | **9.87** | **10.09** |

**Summary:** Contact ER=10.09 meets the <10.5 target. Attacking is essentially
identical to Stage 5 (8.73 vs 8.74). Other plans show small regressions from the
reduced hidden size. Priming shows the largest gap (9.58 vs 8.59), likely due to
the lower gpw=2.0 needed to prevent divergence.

## Stage 7 (S7) — 17-NN Pair Strategy Model

**Purpose:** Uses 17 NNs (1 PureRace + 16 contact) selected by the (player, opponent)
game plan pair, instead of just the player's game plan. This allows each NN to
specialize for specific matchups (e.g., attacking vs anchoring). 14 distinct weight
files — 3 rare pairs (prim_prim, prim_anch, anch_prim, anch_anch) share one NN.

**Architecture:** `GamePlanPairStrategy` in `neural_net.h`. Array of 17 NNs indexed by
`pair_nn_index(player_gp, opponent_gp)`. Classification: `classify_game_plan(board)`
for player, `classify_game_plan(flip(board))` for opponent. PureRace selected when
`is_race(board)` is true.

**Hidden sizes:** 100h PureRace, 300h for all 16 contact pair NNs.

**Weights:** Registered as `"stage7"` in `python/bgsage/weights.py`. Weight files are
`sl_s7_{pair}.weights.best` in `models/`. Shared pairs use `sl_s7_prim_anch.weights.best`.

**Training:**
- TD: 300k games @ α=0.1 + 1.5M @ α=0.02 (50% more than S5/S6 due to 16 contact NNs)
- GPW scan: For each NN, test gpw values (1.5, 3, 5, 7, 10) with pair-filtered
  benchmarks. Pick the gpw that minimizes ER on the NN's specific (player, opponent)
  pair subset of the benchmark.
- SL: 4-phase schedule (100ep@α=20 → 200ep@α=10 → 200ep@α=3.1 → 500ep@α=1.0) using
  optimal gpw per NN. Pair-filtered benchmarks used for best-weight selection.

**Pair frequencies in training data and optimal GPW values:**

| NN | Freq | GPW | Pair ER (S7) | Pair ER (S6) | Δ |
|----|------|-----|-------------|-------------|---|
| race_race | 10.0% | 7.0 | 7.26 | 7.97 | -0.71 |
| race_att | 5.7% | 10.0 | 4.72 | 5.00 | -0.28 |
| race_prim | 9.5% | 5.0 | 4.38 | 4.51 | -0.13 |
| race_anch | 11.3% | 7.0 | 5.26 | 5.82 | -0.56 |
| att_race | 5.7% | 5.0 | 5.39 | 6.99 | -1.60 |
| att_att | 3.9% | 7.0 | 9.90 | 9.96 | -0.06 |
| att_prim | 6.3% | 5.0 | 8.31 | 8.33 | -0.02 |
| att_anch | 6.8% | 10.0 | 9.98 | 10.57 | -0.59 |
| prim_race | 9.5% | 7.0 | 8.50 | 9.40 | -0.90 |
| prim_att | 6.3% | 7.0 | 9.71 | 10.04 | -0.33 |
| prim_anch (shared) | 6.8% | 5.0 | 9.15 | 9.60 | -0.45 |
| anch_race | 11.3% | 7.0 | 11.79 | 11.84 | -0.05 |
| anch_att | 6.8% | 7.0 | 10.36 | 10.72 | -0.36 |

**Benchmark results (1-ply):**

| Metric | S7 (17-NN, 300h) | S6 (5-NN, 300h) | S5 (5-NN, 400h) |
|--------|-----------------|-----------------|-----------------|
| Contact ER | 9.76 | 10.09 | **9.66** |
| Race ER | 1.00 | 1.00 | **0.95** |

**Summary:** S7 reduces Contact ER from 10.09 (S6) to 9.76 — recovering 77% of
the gap between S6 and S5 (400h) through pair specialization alone, using the same
300h hidden size. All 13 canonical NNs beat their S6 baseline on pair-filtered
benchmarks. Biggest wins on cross-plan pairs (att_race: -1.60, prim_race: -0.90).

**C++ bindings:**
```python
# 1-ply scoring
bgbot_cpp.score_benchmarks_pair(scenarios, weight_paths, hidden_sizes)

# Multi-ply
multipy = bgbot_cpp.create_multipy_pair(weight_paths, hidden_sizes, n_plies=3)

# Rollout
rollout = bgbot_cpp.create_rollout_pair(weight_paths, hidden_sizes,
    n_trials=360, truncation_depth=7, decision_ply=2, n_threads=16)
```

**Training scripts:**
- `scripts/run_s7_training.py` — TD training
- `scripts/run_s7_sl_training.py` — GPW scan
- `scripts/run_s7_sl_phase34.py` — Full SL training (phases 3-4)

## Stage 8 (S8) — 17-NN Pair Strategy, 400h Contact, S5 Fallback

**Purpose:** Same 17-NN pair strategy architecture as Stage 7, but with 400h contact
NNs (matching Stage 5 hidden size) instead of S7's 300h. Tests whether pair
specialization + larger hidden layers can beat S5's single-plan 400h NNs. PureRace
weights copied from S7 (same 100h architecture).

**S5 fallback:** After training, each pair NN is scored on its pair-filtered benchmark
subset. Any NN with worse ER than S5's corresponding plan NN is replaced with the S5
weights, guaranteeing no regressions. 10 of 13 pair NNs beat S5; 3 were replaced
(race_prim, att_att, prim_att).

**Weights:** Registered as `"stage8"` in `python/bgsage/weights.py`. Weight files
are `sl_s8_{pair}.weights.best` in `models/`. 3 of the 14 canonical weight files
are copies of S5 plan weights (due to fallback).

**Hidden sizes:** 100h PureRace, 400h for all 16 contact pair NNs.

**Training:** Same pipeline as S7 with these differences:
- Contact NNs: 400h hidden (vs S7's 300h)
- GPW candidates: [2, 5, 7, 10, 12] (vs S7's [1.5, 3, 5, 7, 10])
- PureRace: copied from S7 (not retrained)
- S5 fallback applied after SL training

**Training schedule:**
- TD: 300k games @ α=0.1 + 1.5M @ α=0.02 (same as S7)
- GPW scan: For each of 13 canonical contact NNs, try gpw in [2, 5, 7, 10, 12]
  with phases 1-2 (100ep@α=20 + 200ep@α=10). Pick gpw minimizing pair-filtered ER.
- SL phases 3-4: 200ep@α=3.1 + 500ep@α=1.0 with optimal gpw, starting from
  scan's best weights.
- S5 fallback: replace any pair NN with worse pair-filtered ER than S5.

**Benchmark results (1-ply):**

| Metric | S8 (17-NN, 400h) | S7 (17-NN, 300h) | S5 (5-NN, 400h) |
|--------|-----------------|-----------------|-----------------|
| Contact ER | **9.49** | 9.76 | 9.87 |
| Race ER | 1.00 | 1.00 | 0.95 |
| vs PubEval | +0.633 | — | +0.633 |

**Multi-ply Contact ER:** 1-ply=9.49, 2-ply=8.44, 3-ply=7.76, 4-ply=7.66.

**C++ bindings:** Same as S7 — `score_benchmarks_pair`, `create_multipy_pair`,
`create_rollout_pair`.

**Training script:** `scripts/run_s8_training.py` — unified pipeline (TD + GPW scan +
SL phases 3-4 + S5 fallback + scoring + benchmarks).

## Training a New Pair Strategy Model (End-to-End Process)

This section documents the full algorithm for training a new 17-NN pair strategy
model (like S7 or S8), so future training runs can follow the same process.

### Prerequisites

- CUDA GPU available (for SL training)
- C++ build with pair strategy support (`td_train_gameplan_pair`, `cuda_supervised_train`,
  `score_benchmarks_pair`, `create_multipy_pair`, `create_rollout_pair`)
- Training data in `data/`: `contact-train-data`, `crashed-train-data`, `purerace-train-data`
- Benchmark files in `data/`: `purerace.bm`, `racing.bm`, `attacking.bm`, `priming.bm`,
  `anchoring.bm`, `contact.bm`, `crashed.bm`, `race.bm`

### Step 1: Register the Model

Add an entry to `MODELS` in `python/bgsage/weights.py`:
```python
MODELS["stage9"] = {
    "hidden": (100,) + (400,) * 16,        # (purerace_h,) + (contact_h,) * 16
    "pattern": "sl_s9_{plan}.weights.best",
    "plans": "pair",
    "canonical_map": [0,1,2,3,4,5,6,7,8,9,10,12,12,13,14,12,12],
}
```

### Step 2: Create the Training Script

Copy `scripts/run_s8_training.py` and update these constants:
- `N_HIDDEN`: contact NN hidden size (e.g., 400)
- `MODEL_PREFIX`: e.g., `'sl_s9'`
- `TD_MODEL_NAME`: e.g., `'td_s9'`
- `GPW_CANDIDATES`: list of gpw values to scan (e.g., `[2, 5, 7, 10, 12]`)
- PureRace source: which model to copy PureRace weights from

### Step 3: Launch Training

Training is long-running (~30-50 hours depending on hidden size). Launch as a
detached Windows process. Run from the bgsage repo root (python must be on PATH):

```bash
powershell -Command "Start-Process -FilePath python -ArgumentList '-u','scripts\run_s9_training.py' -WindowStyle Hidden -RedirectStandardOutput 'logs\s9_training.log' -RedirectStandardError 'logs\s9_training_err.log'"
```

### Step 4: Monitor Progress

Set up a cron job to show the user a training summary every 10 minutes. Use
`CronCreate` with `*/10 * * * *` and a prompt that runs these checks:

1. **Process alive?** `powershell -Command "Get-Process python* | Select-Object Id, CPU, WorkingSet64, StartTime | Format-Table -AutoSize"`
2. **Training log tail:** `powershell -Command "Get-Content 'logs\s9_training.log' -Tail 30"`
3. **Errors:** `powershell -Command "Get-Content 'logs\s9_training_err.log' -Tail 10"`
4. **Weight files:** `ls -lt models/td_s9* models/sl_s9* models/s9_gpw_scan/ 2>/dev/null | head -20`

The cron prompt should instruct Claude to summarize: what phase it's in (TD Phase 1,
TD Phase 2, GPW scan, SL phases 3-4, benchmarks), how far along, and any issues.

The training script also prints its own progress updates every 10 minutes to the log
(via the `ProgressTracker` class), but the cron job gives the user a live interactive
summary without needing to manually check logs.

You can also check manually:
```bash
powershell -Command "Get-Content 'logs\s9_training.log' -Tail 30"
```

Check weight file timestamps to verify training is progressing:
```bash
ls -la models/td_s9_*.weights
ls -la models/s9_gpw_scan/
ls -la models/sl_s9_*.weights.best
```

### Step 5: Training Pipeline Phases

The script runs these phases automatically:

1. **Copy PureRace** from previous model (no retraining needed if same 100h arch)
2. **TD Phase 1**: 300k self-play games @ α=0.1 (~5-11h depending on hidden size)
3. **TD Phase 2**: 1.5M self-play games @ α=0.02 (~25-55h)
4. **GPW Scan**: For each of 13 canonical contact NNs:
   - Try each GPW candidate with SL phases 1-2 (100ep@α=20 + 200ep@α=10)
   - Score using pair-filtered benchmarks (only positions matching the NN's
     player×opponent game plan pair)
   - Select GPW minimizing pair-filtered ER
   - Save results to `models/s*_gpw_scan/optimal_gpw.json`
5. **SL Phases 3-4**: For each canonical contact NN:
   - Resume from GPW scan's optimal `.best` weights
   - Train phases 3 (200ep@α=3.1) and 4 (500ep@α=1.0)
   - Use pair-filtered benchmarks for best-weight selection
   - Copy canonical weights to alias NNs (shared group)
6. **Scoring**: Pair-filtered comparison with reference model (e.g., S5)
7. **Benchmarks**: 1-ply contact/race, 2-4 ply contact, top-100 worst positions

### Step 6: Record Results

After training completes:
1. Update `MODEL_BENCHMARKS.md` with benchmark results
2. Update this `CLAUDE.md` with the new model section (hidden sizes, GPW values,
   pair-filtered ER results, training scripts)
3. Commit weight files and updated docs

### Recovery / Partial Reruns

The script supports resuming from any point:
- `--sl-only`: Skip TD, use existing TD weights
- `--phase34-only`: Skip TD and GPW scan, resume from saved optimal GPW
- `--score-only`: Skip all training, just score existing weights
- `--benchmark-only`: Run multi-ply and top-100 benchmarks only
- `--nn race_race att_att`: Train only specific NNs

## Stage 5 Small (S5S) — Fast Filter Model

**Purpose:** Half-size model (100h PureRace, 200h contact NNs) trained as a potential
fast filter for multi-ply search and truncated rollouts. The hypothesis was that using
a smaller model for 1-ply filtering and a full-size model for leaf evaluations could
speed up 4-ply and Roller++ calculations.

**Weights:** Registered as `"stage5small"` in `python/bgsage/weights.py`. Weight files
are `sl_s5s_{plan}.weights.best` in `models/`.

**Training:** Same TD + SL pipeline as Stage 5. TD: 200k games @ α=0.1 + 1M @ α=0.02.
SL: same schedule except Racing uses gpw=2.0 (not 5.0 — gpw=5.0 diverges at 200h due
to Racing's 37% share of training data dominating the gradient at 74%).

**Per-plan ER (1-ply):**

| Plan | Stage 5 (400h) | S5S (200h) |
|------|---------------|------------|
| PureRace | 0.82 | 1.23 |
| Racing | 5.74 | 6.40 |
| Attacking | 8.74 | 8.75 |
| Priming | 8.59 | 9.07 |
| Anchoring | 11.06 | 12.05 |
| **Contact** | **9.87** | **10.58** |

**Timing results — S5S is NOT significantly faster than S5 at high ply:**

The original hypothesis (2x faster NN → ~1.4-1.7x faster 4-ply) did not hold.
Profiling revealed that the NN forward pass (matrix multiply) is a minority of
total per-node cost at 4-ply depth. The dominant costs are:

1. **Move generation** (`possible_boards`): O(candidates) per dice roll, fixed cost
2. **Input encoding**: Computing 244 extended features (escape counts, containment, etc.)
   is a fixed cost that doesn't shrink with fewer hidden nodes
3. **Position cache divergence**: S5S's noisier 1-ply evaluations produce less consistent
   move ordering, creating more unique positions in the search tree (fewer cache hits)

| Level | S5 Time | S5S Time | S5S Speedup |
|-------|---------|----------|-------------|
| 1-ply | 10.8ms | 7.7ms | 1.40x |
| 2-ply | 88ms | 79ms | 1.11x |
| 3-ply (1T) | 146ms/pos | 96ms/pos | 1.52x |
| 4-ply (1T, cache on) | 1,598ms/pos | 1,533ms/pos | 1.04x |
| 4-ply (1T, cache off) | 4,401ms | 4,634ms | 0.95x (slower) |

With cache disabled, S5S does 3x fewer leaf evaluations but each takes 3x longer
in amortized terms — the fixed overhead (move gen + encoding + filtering) dominates.

**Hybrid evaluator** (S5S filter + S5 leaf) was also tested. It's slower than pure S5
at 4-ply due to overhead from managing two strategy sets. At Roller++ it showed a
modest 1.19x speedup but crashed on large benchmark runs due to memory accumulation
in thread-local caches.

**Conclusion:** Halving hidden nodes does not meaningfully speed up multi-ply search
because the NN matrix multiply is not the bottleneck. Future speedup efforts should
target move generation, encoding, or cache efficiency rather than smaller networks.

## Benchmark Data Format

GNUbg `.bm` files. Each "move" line:
```
m <position_string> <die1> <die2> <best_pos> <2nd_pos> <2nd_err> ...
```
Score = mean error * 1000 (millipips). Lower is better.

## GNUbg Training Data Format

Each line: `<20-char position string> <P_win> <P_gw> <P_bw> <P_gl> <P_bl>`

## Rules for Experiments

1. Every experiment lives in its own directory
2. **NEVER modify `cpp/src/`, `cpp/include/`, or evaluation code during an experiment**
   without explicit approval
3. Always compare against the current best model

## Ply Counting Convention

We use the XG convention where 1-ply = raw NN evaluation. GNUbg calls raw NN
evaluation "0-ply". So GNUbg's 0-ply = our 1-ply, GNUbg's 1-ply = our 2-ply, etc.
Keep this in mind when comparing results.

## C++ Gotchas

- `std::fixed` is sticky — always reset with `std::defaultfloat` at start of
  functions that need default formatting
- `MultiPlyStrategy::get_cache()` uses `thread_local static PosCache` — ALL instances
  share the same cache. Call `clear_cache()` between strategy comparisons.
- Rollout `NeuralNetwork` transposed-weight init uses `std::call_once` for thread safety

## Back Game Data Files

Back game positions and rollout probabilities for training specialized back game NNs.
All files in `data/`. Generated by `scripts/generate_backgame_data.py` (positions)
and `scripts/rollout_backgame_positions.py` (rollout probabilities via Parallelizor).

### Position files (input)

One position per line: 26 space-separated integers (board representation).

| File | Positions | Description |
|------|-----------|-------------|
| `player-backgame-train-data` | 93,770 | Player BG training positions |
| `player-backgame-benchmark-data` | 10,419 | Player BG benchmark positions |
| `opponent-backgame-train-data` | 93,770 | Opponent BG training positions |
| `opponent-backgame-benchmark-data` | 10,419 | Opponent BG benchmark positions |

### Rollout files (output)

One position per line: 26 space-separated integers (board) followed by 5 floats
to 4 decimal places (cubeless rolled-out probabilities: W, Gw, Bw, Gl, Bl).
Probabilities are post-move, from the perspective of the player whose checkers
are positive. Rolled out with Stage 8 pair model, 1296 trials, 3-ply cubeless
decisions throughout, PubEval 20/15 prefilter, VR enabled.

| File | Description |
|------|-------------|
| `player-backgame-train-rollout` | Rolled-out probs for player BG training |
| `player-backgame-benchmark-rollout` | Rolled-out probs for player BG benchmark |
| `opponent-backgame-train-rollout` | Rolled-out probs for opponent BG training |
| `opponent-backgame-benchmark-rollout` | Rolled-out probs for opponent BG benchmark |

### Back game criteria

A position qualifies as a **player back game** if: game plan pair is (anchoring,
racing), player is behind in the race (higher pip count), and player holds ≥2
anchors in the opponent's home board (points 19-24). **Opponent back game** is the
mirror: (racing, anchoring) pair with opponent behind and holding ≥2 anchors in
player's home board (points 1-6).

### Benchmark ER (1-ply Stage 8)

Player back game: **45.98** (vs overall contact ER of 9.49 — 4.8× worse).

### Stage 9 Back Game NN Training Process

The Stage 9 back game NNs (player_bg, opponent_bg) are trained iteratively,
re-rolling out training and benchmark data between rounds to improve target
quality as the model improves.

**Round 1 — Bootstrap from S8 rollouts:**
1. Roll out all player/opponent back game training and benchmark positions using
   Stage 8 pair model (1,296 trials, 3-ply cubeless decisions throughout, PubEval
   20/15 prefilter, VR enabled). Output: `*-backgame-*-rollout` files.
2. SL training for both player_bg and opponent_bg NNs:
   - 100k steps @ α=3.1
   - 250k steps @ α=1.0
   - Starting from S8 fallback weights (anchoring pair NN)

**Round 2 — Re-rollout with improved model:**
1. Re-roll out all training and benchmark positions using the Stage 9 model
   (same rollout settings: 1,296 trials, 3-ply, VR, PubEval prefilter). The
   improved back game NNs produce more accurate rollout targets.
2. Redo SL training with the same schedule:
   - 100k steps @ α=3.1
   - 250k steps @ α=1.0
   - Starting from round 1's best weights

The re-rollout step is necessary because the S8 rollout targets have ~22 ER
noise on back game positions (measured by comparing S8 vs S9 rollouts on 10
random positions). Once the model's back game ER drops below ~20, the training
data noise becomes the limiting factor for further improvement.

### Stage 10 (S10) — Gated Paskogammon Backgame Hybrid

Stage 10 is Stage 9 plus a **precision-gated committee of a second backgame NN
pair**. It is the full Stage 9 19-NN model **carried unchanged**, plus two extra
backgame NNs trained on Paskogammon-game backgame rollouts, used ONLY inside a
tight structural gate. Outside the gate — including every non-backgame position
and all ordinary (shallow) backgames — S10 is **bit-identical to Stage 9**.

**Architecture (21 NNs).** Indices 0-18 are exactly Stage 9's NNs (including
`sl_s9_player_bg`/`sl_s9_opponent_bg` at 17/18). Indices 19/20 are the extra
Paskogammon-trained backgame NNs (`sl_s10_player_bg`/`sl_s10_opponent_bg`, 400h).

**Gate + blended eval.** `BackgameAwarePairStrategy` accepts 19 **or** 21 NNs.
With 21, a *detected* backgame (same detection as S9) is routed to the committee
only when the backgame side has

```
anchors >= 3   OR   (anchors == 2  AND  back checkers >= 7  AND  pips >= 200)
```

Inside the gate both backgame NNs evaluate the same inputs and the probs are
mixed, `probs = (1-w)·base + w·pasko`, with an anchor-zone weight (**0.80** for
4+ anchors, **0.70** for 3, **0.60** for the massive-2-anchor arm) scaled by a
pip ramp (×0.5 at ≤170 pips → ×1.0 at ≥230). Constants: `BACKGAME_GATE_*` /
`BACKGAME_BLEND_*` in `neural_net.h`. `select_nn_idx` reports gated positions
via sentinels 21/22; the batch delta-eval kernels evaluate those per-candidate
(no shared delta base). Non-gated backgames return 17/18 and keep the fast
paths. With 19 NNs everything is off (pure S9).

**Why a gate.** The pasko nets are decisively better on *deep/massive*
backgames but indistinguishable-to-worse on ordinary ones, and no board feature
separates standard-game from pasko-game shallow backgames — an ungated blend
improved the neutral backgame benchmarks but leaked small divergences from S9
into full-game rollout-PR scores everywhere. The gate confines the pasko nets
to the region where they provably win (verified on 60.7k rolled-out backgame
positions): **1-ply ER, Paskogammon backgame benchmark: S9 43.24 → S10 35.5**
(gated slices: 3-anchor 46.98→30.65, 4+-anchor 92.38→48.71, massive-2-anchor
50.58→31.62); **standard backgame benchmark: S9 23.10 → S10 23.05** (the gated
regular slices improve slightly too; both even/odd halves ≤ 23.06). Gate
footprint: ~43% of pasko-game backgame evals, ~9% of standard-game ones — so
standard-game play is byte-equal to S9 outside rare deep backgames.

**Where it lives.** `backgame_gate` + `backgame_blend_weight` +
`blended_backgame_probs` + `select_nn_idx` in `cpp/src/neural_net.cpp` (sentinel
handling in the batch methods and in `cube_eval.cpp`'s `eval_groups_pair`);
constants in `neural_net.h`; the `stage10` registry entry (`extra_backgame`
list) and 21-length `plan_names` in `python/bgsage/weights.py`. Weight files
(**v3** — the pasko nets retrained on the S10-gated benchmark/train sets and
warm-restarted twice): `sl_s10_player_bg_v3.weights.best`,
`sl_s10_opponent_bg_v3.weights.best`. The original shipped nets
(`sl_s10_*_bg.weights.best`) are kept for A/B.

**Retrained pasko nets (v3).** The gate footprint (~9%/43% of standard/pasko
backgames) barely intersects the full-game benchmarks, so the pasko nets are
better re-benchmarked and re-trained on a **gated-only** backgame set — the
positions where the blend weight > 0 (`generate_pasko_data.py --gate`). On a new
20k/side gated benchmark, 1-ply cubeless ER: **S9 49.51 → S10 v1 31.62 → S10 v3
28.97** (−42% vs S9). v3 = SL on the gated 100k/side train set warm-started from
the shipped v1 nets, then one warm-restart. (Warm-starting from the TD net lands
~1.4 ER worse; a further restart gains <0.1 — converged.)

### Stage 11 (S11) — EXPERIMENTAL: the back-game-aware phased model

**The registry has ONE Stage 11 entry, `stage11`: the 24-NN phased layout**
(Stage 9's 17 standard nets + the category trio + P3 + containment + snake +
massive), i.e. what the sections below arrived at. The intermediate names
those sections use — the 20-NN trio, `stage11p` (22), `stage11s` (23),
`stage11m` (24) — are the history of how it was built and no longer
resolve; the C++ strategy still accepts 22 or 23 paths for A/B work.
`PRODUCTION_MODEL` is still `stage9`.

#### The categorized backgame trio (the first step)

Stage 11 replaces Stage 9's two backgame NNs (player/opponent) with **three,
selected by the backgame's CATEGORY — the same NN whichever side holds it**:

| Index | NN | Categories | Anchor rule |
|-------|----|-----------|-------------|
| 17 | `bg_deep` | 21, 31, 32 | both anchors on the 1/2/3 points |
| 18 | `bg_middle` | 41, 42, 51, 52 | one anchor on the 1/2 point, one higher |
| 19 | `bg_double` | 43, 53, 54 | exactly two anchors, none deeper than the 3-pt |

With 3+ anchors: deep when at least two sit on the 1/2/3 points, else middle
(never double). The 6-point counts as an anchor (Stage 9 detection convention);
pairs with a 6-anchor map by the same min/max rules ({1,6}/{2,6} middle, the
rest double). **Detection itself is Stage 9's, unchanged**: plan pair
(anchoring, racing), backgame side behind on pips, 2+ anchors in the opponent's
home board. The category is perspective-invariant.

**Where it lives.** `BackgameCategory` + free `backgame_category(board)` +
`category_from_anchor_mask` / `backgame_category_given_plans` in
`neural_net.h/.cpp`; `BackgameAwarePairStrategy` accepts **20** NNs
(`NUM_BACKGAME_PAIR_NNS_CATEGORIZED`) and routes detected backgames to
17/18/19 by category — with 19 or 21 NNs behaviour is exactly S9/S10. The
`stage11` registry entry in `weights.py` carries S9's 17 standard NNs
unchanged plus `sl_s11_bg_{deep,middle,double}.weights.best` via
`extra_backgame` (plans type `backgame_pair_categorized`; dispatches to the
same C++ class). Python: `bgbot_cpp.backgame_category(board)` returns
`"deep"/"middle"/"double"/"none"`. Tests: `tests/test_backgame_category.py`.

**Training step 1 — truncated TD** (`td_train_backgame_truncated` in
`training.h/.cpp`, driver `scripts/run_s11_backgame_td.py`): each backgame NN
starts from random small weights and trains by TD(0) self-play with games that

* start from the category's reference positions
  (`backgame_ref_positions/benchmark/<folder> starting.txt`, cycled, a coin
  picking the first mover);
* are played with 1-ply decisions by the training NN;
* END when a post-move position is no longer in ANY backgame category — the
  terminal TD target is then **Stage 9's 3-ply cubeless post-move eval** of
  that position, standing in for the game outcome exactly as the 0/1 outcome
  vector does at a real terminal (which still ends the game the ordinary way —
  a backgame can lose by bear-off with the anchors still held).

Progress is scored pasko-style (mean |equity − target| × 1000) against the
`data/*-backgame-benchmark-rollout` rows falling in the training category.
Outputs `models/td_s11_bg_<cat>.weights(.best)`; promote by copying the three
`.best` files to the `sl_s11_bg_*` names the registry points at.

```bash
py -3.14 scripts/run_s11_backgame_td.py --category deep --n-games 200000
py -3.14 scripts/run_s11_backgame_td.py --category all
```

**Training step 2 — SL on rollout targets** (the quality stage, mirroring how
Stage 9's backgame nets were made). ``scripts/segregate_s11_backgame_data.py``
splits every backgame-labelled rollout pile from the S9/S10 eras (the
player/opponent S9 sets plus the three pasko families; ``.s8`` targets and the
position-only ``*-data`` files skipped) into
``data/s11-bg-<cat>-{train,benchmark}-rollout`` by ``backgame_category`` —
about 245k/167k/94k train rows for deep/middle/double, benchmarks held out
and deduplicated against train. Every row in those piles passes S11 detection
(0 dropped), and ``pasko-gated-remainder`` is entirely duplicates of the gated
sets. ``scripts/run_s11_backgame_sl.py`` then trains each category NN with the
S9 backgame recipe (GPU ``cuda_supervised_train_preencoded``, 2,500-epoch
chunks at batch 4096, 100k ep @ a=3.1 then 250k @ a=1.0, best-ER
checkpointing), warm-started from the TD bootstrap by default, writing
``models/sl_s11_bg_<cat>.weights.best`` — the exact filenames the ``stage11``
registry loads. ~2 h per category on the RTX 4070S.

**Measured path lengths** (why truncated games are short): under S9 1-ply play
from the deep seeds, the region exits after a median of ~9 half-moves — mostly
because the PLAN PAIR flips (the racer starts `attacking`, or the backgame side
reads as `priming`) while the anchors are still physically held, which is
inherited Stage 9 detection behaviour. Under a random-init NN it is ~4
half-moves, lengthening as the NN learns. The S9 3-ply reference eval is ~10 ms
cold and cached across games (16 recurring seeds), so TD throughput stays high.

**The flee equilibrium — why `--anchor-boundary` exists** (measured on a 200k
deep run, 2026-08-29). Plain truncated TD converges CORRECTLY to the value of
the truncated game under its own play — and that game's optimal policy is
degenerate. Interior states are worth what the net's own (weak) continuation
makes them (net −1.02 vs an independent on-policy Monte-Carlo estimate −1.01,
corr 0.93), while crossing the boundary swaps the appraiser to Stage 9, whose
grades assume EXPERT continuation (backgame-side exits −0.79 vs the true
good-play interior value −0.60; measured play-quality wedge S9 − V_pi = +0.39).
So handing the position to the expert appraiser beats keeping it, the backgame
side (which holds exit moves on 77% of its rolls; the racer holds 0%) exits in
median 1 half-move, paths collapse (3.8 → 2.8 over training while the
stay-vs-exit value gap grows −0.03 → +0.24 and exit-choice rises 42% → 94%),
and the benchmark ER plateaus (~262 by game 15k, flat thereafter, vs S9's 22).
It is one-sided because the subsidy is proportional to each side's own
incompetence gap — the racer's decisions here cost only ~0.03/move vs S9, so it
has no wedge to chase and every true-equity reason to welcome the opponent's
exits. `anchor_boundary` adds one supervised update per game — the exit
position itself toward its S9 eval — which re-grounds the slice to expert
scale: at 15k games, exit-choice 72% → 21%, backgame-side move cost
0.216 → 0.046 equity, path median ~10 half-moves (S9-play territory), ER 250
and still falling where the plain run had plateaued. `boundary_extra_plies` (the grace period) attacks
the same disease from the other side: the path plays N more half-moves past
the first out-of-region position (updates continuing, the net still moving)
before the Stage 9 grade is taken, so the boundary zone becomes trained input
and fleeing means living with your own play N moves longer. Measured on
30k-game deep runs (play cost = mean equity lost per backgame-side seed
choice vs S9-3P; N=0 is the plain trainer, byte-verified):

    N=0: cost 0.202, exits 71%, ER 301 (the flee equilibrium)
    N=2/4/6/10: cost 0.045/0.043/0.042/0.041, exits ~19-26%, ER 260/252/225/222
    anchor only: cost 0.033, ER 226
    N=6 + anchor: cost 0.034, ER 206 (best) and still falling

Even N=2 removes the equilibrium; larger N mainly lowers target noise (the
grade lands where positions are settled and Stage 9 is reliable). The
recommended mode for any real S11 training run is anchor_boundary +
boundary_extra_plies ~6 — the driver's defaults. A 30k-game A/B/C
(`ref_move_games`: S9 picks the moves for the first N games) showed the
S9-move warm-up is not it: pure S9-move training gives the best VALUE fit
(ER 223, and in-region values within 0.04 of S9's) but 4x worse PLAY at the
seeds (0.122 equity lost per backgame-side choice vs 0.033 for anchored
self-play) — value accuracy on S9's trajectories does not calibrate the
off-distribution candidates greedy selection must compare, and the exit
ordering inverts again. A 15k warm-up then pivot lands exactly where pure
self-play lands (0.034). The net should make its own moves from game 0.

### Stage 11p — EXPERIMENTAL: phased layout (P3 + containment NNs)

`stage11p` is the trio model plus two more NNs, **22 in all**
(`NUM_BACKGAME_PAIR_NNS_PHASED`, strategy type `backgame_pair_phased`, the
`BackgameAwarePairStrategy` constructor flag `phase_containment`):

| Index | NN | Selected when |
|-------|----|---------------|
| 17-19 | `bg_deep` / `bg_middle` / `bg_double` | a detected backgame, by category (the trio above) |
| 20 | `bg_p3` | **early containment** phase: the backgame side still holds >= 2 anchors and is behind, has hit (>= 1 straggler), and the racer has <= 2 off (`backgame_phase()` == `EARLY_CONTAINMENT`) |
| 21 | `bg_containment` | the **containment rule** below |

`select_nn_idx` tests them in the order containment rule -> category trio ->
P3 phase -> standard pair NN, so the containment NN wins whenever its rule
fires. Everything else is Stage 9. Python: `bgbot_cpp.backgame_phase(board)`
(`"waiting"/"bear_in"/"early_containment"/"late_containment"/"none"`) and
`bgbot_cpp.containment_category(board)`.

**The containment rule is escaper-centric** (`containment_category` in
`neural_net.cpp`, mirrored by `scripts/containment_rule.py`, which the tests
hold the C++ to): the position is a containment game when one side, the
ESCAPER, has borne off >= 3 checkers (`CONTAINMENT_E_OFF_MIN`) and has 1-3
stragglers (`CONTAINMENT_STRAGGLERS_MAX`) — checkers on the bar or outside its
home board with a container checker still ahead of them — and there is
contact. Nothing is required of the container: no anchors, no particular
structure, blots allowed. The rule is flip-invariant. Measured footprint:
99% of the `containment` family folder, 39% of `Positions XG gets wrong`,
1.3% of the money benchmark, 2.6% of pasko, <= 1.2% of the ten backgame
folders (their windows sit in the waiting phase; a hit shot there routes to
P3, 8-21% of their positions, because the racer has <= 2 off). The P3
phase's single-anchor fallback was tried and rejected (money PR +0.75).

**Blend sentinels.** `BLENDED_PLAYER_BG_IDX` / `BLENDED_OPPONENT_BG_IDX`
were 21/22, compared with `>= NUM_BACKGAME_PAIR_NNS_HYBRID`; the phased
layout's slot 21 collided with them, so every batch evaluator treated a
containment decision as a gated S10 blend and slot 21 was never read (two
different containment nets scored byte-identically). They are now 1000/1001
(`BLENDED_SENTINEL_BASE`) and `tests/test_backgame_category.py` asserts a
containment board evaluated through the phased strategy equals the
containment NN evaluated directly.

**Data.** `scripts/harvest_containment_positions.py` plays 3,000 two-ply
stage11 self-play games from the containment seeds (every benchmark decision
and candidate board, both orientations, excluded) and keeps the rule's
positions; `scripts/split_containment_targets.py` moves the ones the S9-era
piles had already rolled into `data/s11-bg-containment-pile-rollout` (9,734)
and leaves `data/s11-bg-containment-data` (35,266 fresh boards), rolled out
under S9 play at 1,296 paths / 3-ply into `data/s11-bg-containment-rollout`
(Fargate Spot, $176 realized). `scripts/extract_general_containment_rows.py`
adds the rule's rows from the main corpus — 16k of `contact-train-data`
(1.3%) and 275k of `crashed-train-data` (46%: closeouts and late-hit
bear-offs), GNUbg targets — as `data/s11-bg-containment-general-rollout`.

**Training** is `scripts/run_s11_containment_sl.py` (GPU SL, 20k epochs @
alpha 3.1 then 60k @ 1.0, best-holdout-ER checkpointing). The installed net
is the S9 `prim_race` warm start trained on family rows x4 plus the general
rows (`--extra-data s11-bg-containment-general-rollout --family-weight 4`).
The family-only version was the narrow-subset failure this file warns about
elsewhere: on the money benchmark's 237 containment-routed decisions it
scored PR 4.62 with 6 blunders against 2.91 / 2 for Stage 9's ordinary nets,
while ordinary-game containment simply never appeared in its data. The
widened set restores that slice to 3.03 and improves the family folder too
(on the then-uncompleted reference, 1P PR 11.05 -> 10.12 against S9's 11.82;
the completed-reference numbers are in the table below). Warm-starting from
the S9-initialised deep net (14.59) or fine-tuning the family-only net on the
widened set (tied on the folder, worse on the money slice and on `Positions
XG gets wrong`) both lost. The general rows carry GNUbg's targets, so on ordinary-game
containment the net can at best match `prim_race`; re-rolling a subsample of
them under S9 is the open lever there.

**Candidate completion.** A folder reference rolled out only the candidates
the reference player's 2-ply filter kept; the phased model's picks fall
outside that set far more often than Stage 9's (56% of its containment-folder
error mass sat on filter-graded picks at 1P, 63% at 3P), which over-charges
it. The parent repo's `scripts/build_rollout_jobs_containment.py` (both
references of a family folder, `--category`, `--models`, `--key-suffix`) and
`scripts/build_rollout_jobs_folders_stage11p.py` force those picks into
rollout jobs; `scripts/merge_candidate_rollouts.py` splices the results in.
Read the `PR(RO)` and `filt mass` columns of `score_backgame_pr.py` before
trusting a headline PR on any of these folders.

**Results (2026-09-03).** Every pick of every model rollout-graded (one 3-3
decision in `Positions XG gets wrong` excepted — its 3-ply trials run into
hours-long containment endgames and the batch was abandoned). PR, with
blunder counts where they matter:

| Benchmark (1P unless noted) | Stage 9 | trio (`stage11`) | `stage11p` |
|---|---|---|---|
| containment folder, 3,000 decisions, 1P / 2P / 3P | 10.23 / 10.79 / 6.21 | 9.40 / 10.20 / 5.10 | **6.83 / 4.94 / 2.82** |
| ... blunders at 3P | 140 | 99 | **40** |
| containment, S11-play sample (300), 1P / 2P / 3P | 8.76 / 8.58 / 4.21 | 8.50 / 8.72 / 3.04 | **6.16 / 5.47 / 2.13** |
| Positions XG gets wrong (421), S9-play / S11-play ref | 16.22 / 16.63 | 14.20 / 16.32 | **10.01 / 12.26** |
| ten backgame folders pooled (10,021), blunders | 4.75 (278) | 3.28 (127) | **2.98 (89)** |
| money benchmark (17,535) | 2.59 | 2.56 | 2.56 |
| ... its 237 containment-routed decisions | 2.91 | 2.91 | 3.03 |
| Paskogammon benchmark (2,556) | 7.70 | 6.85 | **6.52** |

Two things to keep in mind when reading it. Completing the candidates moved
Stage 9's own containment number from 11.82 to 10.23 — its filter-graded
picks had been over-charged too — so never compare a completed reference
with an uncompleted one. And on this folder both Stage 9 and the trio score
WORSE at 2-ply than at 1-ply, on checker and cube errors alike (cube
2.77 -> 7.23 -> 2.77 for Stage 9 across 3P/2P/1P is not a typo); `stage11p`
improves monotonically. The Stage 11s section below has the diagnosis.

### Stage 11s — EXPERIMENTAL: the snake NN (index 22)

`stage11s` is `stage11p` plus a 23rd NN for the **snake**: a far-side prime
trapping a straggler against a crunched board. `snake_category(board)` —
mirrored by `scripts/snake_rule.py`, and the same rule as the benchmark's
`snake` family filter in `scripts/backgame_benchmark.py`, which the tests
hold all three to — fires when either side holds a run of >=
`SNAKE_PRIME_MIN_POINTS` (4) consecutive points, each with >= 2 checkers,
entirely on the opponent's half of the board, the opponent has >= 1
straggler on the bar or in the holder's home board, and >= `SNAKE_MIN_HOME`
(10) checkers already in its own home board. It is a priming / containment
structure, not a back game; it reads as one to Stage 9's plan-pair gate only
because the holder's points sit in the opponent's home board and the holder
trails in the race, which is why the trio's double and middle nets serve it
today at 1-ply PR 200 and 109. The phased strategy accepts 22 or 23 NNs;
with 23 a snake routes to NN 22 ahead of everything else.

**The region has no data and never occurs in play.** ~100 rule rows in the
1.8M-row GNUbg corpus, ~36 in the Stage 11 piles, none in the pasko or S9
backgame piles, and 0 positions in the money and pasko benchmarks and in
every other folder — so a snake NN carries no regression risk anywhere, and
everything it learns from has to be generated. Self-play from the benchmark
seeds cannot generate it: measured 2026-09-03, stage11p (snake PR ~54)
breaks the prime or runs within a move or two, so 900 games yielded ~2.4
snake decisions each. `scripts/harvest_snake_positions.py` therefore
samples the region directly — 5,000 random synthetic snakes (prime length
and location, spares spread from the far side to home, a crunched opponent
with 0-5 off and 1-3 stragglers) seed short 2-ply games in which the holder
prefers structure-keeping moves — and keeps, per decision, the trajectory
position plus its top-5 candidates and up to 3 region-leaving candidates
(routing is by the pre-move board, so the snake NN is the one asked to value
the release moves): 18,557 decisions -> 98,515 distinct boards,
`data/s11-bg-snake-data` holds 44,000 of them. Note the rule's own edge: a
straggler that reaches the holder's OUTER board is still trapped behind the
prime but no longer counts, so trajectories leave the region after ~3-4
decisions; the benchmark is defined the same way, so the routing matches it.

**Snake rollouts cost ~6x containment ones.** Measured on ten random
harvested boards at the reference convention (1,296 paths, 3-ply in-trial
play): 78.6 s each on the 32-thread dev box against ~12 s for containment
positions, because the holder must bring 15 checkers round before the
trapped side bears off. 648 paths / 3-ply takes 35.8 s, 1,296 / 2-ply
22.3 s. The first rollout (run `0d6c74fc8753`) therefore used 648 paths at
3-ply on the Batch backend — 3-ply in-trial play was kept because Stage 9's
2-ply play in these regions is markedly worse than its 3-ply (containment
folder 10.79 vs 6.21), which biases every target, whereas halving the paths
only widens the target SE from ~0.009 to ~0.013 — for the first 22,000
boards of the file. `rollout_backgame_positions.py --n-trials / --checker-ply`
(parent repo) bakes the configuration into the cloud-pickled task function.
The run realized ~$159 on Batch (64 workers, 9.4 h); the folder's
candidate-completion pass (2,062 boards at the reference convention) a
further ~$75 on Fargate.

**Training and results (2026-09-04).** `run_s11_containment_sl.py
--family-data s11-bg-snake-rollout --out-prefix sl_snake` from the S9
`prim_race` net: holdout ER 299.6 -> 53.95 in 2.5 minutes (a schedule is
passes over the data; a 4x schedule plateaus at 53.6 and scores slightly
worse on the folder, and the trio's deep net as warm start is worse again).
Installed as `models/sl_s11_bg_snake.weights.best` (replaced on 2026-09-05 by
the round-2 `sl_snake2_warm` net, below). Snake folder, every pick
rollout-graded, PR at 1P / 2P / 3P: Stage 9 63.56 / 48.06 / 32.18 (364 /
286 / 205 blunders); trio 52.45 / 34.72 / 28.49; `stage11s` **15.71 / 35.65 /
24.35** (141 / 251 / 165). The 1-ply figure is the first real competence any
Open Sage net has shown in the region. **Why 2-ply scores worse than 1-ply
— measured 2026-09-04** on the 795 checker decisions (cubeless 1-ply PR
16.0): two separate effects, one a bug and one not.

1. *A ranking bug in the multi-ply candidate list.* The filter keeps the top
   five 1-ply candidates within 0.08 and re-scores them at N-ply, and the
   list was then sorted with the PRUNED candidates still on their 1-ply
   values. Whenever the deeper search values every survivor below a pruned
   move's 1-ply number — exactly what happens where N-ply matters most —
   that move topped the list without ever being evaluated: 312 of the 795
   cubeless 2-ply picks were such moves, carrying 55 of that level's 68 PR
   points, against 31 for a faithful 2-ply of the survivors alone. The
   cubeful wrapper was immune (it re-scores every candidate through the
   cube-aware N-ply tree), so the benchmark tables never saw it, but the C++
   `batch_checker_play` overloads — the app's analytics path — promoted only
   the runner-up in the same way. Both now promote whichever of the top two
   still carries a 1-ply value until both are full-depth
   (`_first_stale_top_two` in `analyzer.py`, `stale_top` in `bindings.cpp`).
   The rollout path always had the right rule (promote #1 until it is
   rollout-grade); it is now the rule everywhere.

2. *The search walks out of the training distribution.* Against the rollout
   references the snake net's 2-ply values are BETTER than its 1-ply ones on
   candidates that keep the snake (RMSE 0.106 vs 0.136, holder on roll) and
   much WORSE on candidates that release it (0.204 vs 0.151): the leaves
   below a release are one half-move beyond every position the harvest
   rolled out, and there no net is any good — the router hands them to the
   standard nets, whose error on the harvest's own exit rows is RMSE 0.32
   (bias −0.10), while the snake net fits those same rows at 0.047. A
   hold-versus-release comparison therefore pits a good estimate against a
   bad one and the max picks the noise: pick-PR among rollout-graded
   candidates 16.3 at 1-ply, 27.7 at 2-ply. Routing the whole tree by the
   ROOT position's net helps only a little (27.8 vs 31.2 at 2-ply) because
   the snake net has never seen an exit+1 position either. The fix is data,
   not routing: roll out the opponent's replies to the exit boards (and
   their replies, for 3-ply) and train the snake net on them, under root
   routing. It is the containment net's lesson extended to depth — a
   specialist must cover the neighbourhood the search visits, not only its
   own region — and the containment folder's 2P-worse-than-1P shape for
   Stage 9 and the trio is the same effect.

**Round 2 (2026-09-04): root routing, the exit-reply data, and the widened
containment rule.** Both remedies for effect 2 are implemented; choosing
between them waits on the candidate-completion rollout of their picks (see
the table at the end of this section).

*Root routing.* A decision's whole tree is evaluated with the net chosen for
its ROOT position. `Strategy::root_pin_for(root)` (virtual, -1 = route per
node) is answered by `BackgameAwarePairStrategy` with `select_nn_idx(root)`
whenever that index is in `root_pinned_`, which defaults to `{22}` on the
23-NN layout and to nothing on every other layout, so Stage 9 through
`stage11p` are untouched; `BearoffStrategy` forwards it. The cubeful engine
carries the pin in `EvalCtx.pinned_nn`: `eval_groups_pair` (the batched
interior picks) and `pinned_post_probs` (leaves, dances, forced moves and the
1-ply entries) evaluate every contact position with the pinned net, while
races and exact bearoff positions keep their own values. It enters through
`root_board` on `cubeful_equity_nply_multi` / `cubeful_probs_and_equity_nply`
(bound as `root_board=`; the analyzer passes the decision's pre-move board)
and through `cubeless_tree_probs`, which passes its pre-move board;
`cube_decision_nply_multi` pins from the position itself. The cubeful eval
cache folds the pin into its fingerprint. Switch it off with
`set_root_pinned([])` on the strategy (bound on both classes — the analyzer's
`_strategy_1ply` is the bearoff wrapper) or process-wide with
`BGSAGE_NO_ROOT_ROUTING=1` in the environment at construction.

Two leaks made the first measurements wrong, and are why
`tests/test_backgame_category.py::test_root_routing_pins_leaves_and_cache_through_nested_wrappers`
exists: `pinned_post_probs` unwrapped ONE bearoff layer, but the analyzer
wraps its strategy in `BearoffStrategy` and the binding wraps that again, so
every dance and forced-move leaf silently fell back to the board's own
routing (the pinned 2-ply snake number moved 29.73 -> 37.27 when the widened
rule below changed where those leaves routed, with the pin verified on every
root); and the cache fingerprint ignored the pin, so a pinned node could be
served by an unpinned evaluation of the same board from an earlier call on
the same thread.

*Exit-reply data.* `scripts/harvest_snake_exit_replies.py` builds
`data/s11-bg-snake-exit1-data`: for every exit board in the snake harvest the
opponent's best reply under root routing, plus 4,000 replies picked with
per-node routing — 25,892 boards, rolled out at 648 paths / 3-ply in-trial
play (Fargate, 1,000 workers, 62 min, ~$218) into
`data/s11-bg-snake-exit1-rollout`. `models/s11_diag/sl_snake2_{primrace,warm}.weights.best`
are the snake recipe on the snake rows plus these (warm = from the installed
snake net, primrace = from S9 `prim_race`).

*The widened containment rule* — the proposal that a broken snake is a kind
of containment game, so the containment net should cover the hand-off.
`containment_category` gains a "crunched" arm: an escaper with fewer than
`CONTAINMENT_E_OFF_MIN` off still qualifies when it has >=
`CONTAINMENT_CRUNCHED_HOME_MIN` (10) checkers in its home board and the
container has >= `CONTAINMENT_CONTAINER_FAR_MIN` (8) checkers with >=
`CONTAINMENT_CONTAINER_FAR_POINTS_MIN` (3) made points on the escaper's half;
the straggler and contact conditions are unchanged, and
`scripts/containment_rule.py` mirrors it. Measured footprint: 92% of the
snake harvest's exit boards and 92% of the exit replies; 98.2% of the
containment folder's decisions; money 1.3% -> 1.3%; pasko 2.6% -> 2.9%; the
ten backgame folders 0.7%; massive 4.6%. (A first draft with the home
threshold alone claimed 23.5% of the money benchmark; the container clauses
are what confine it.) Its data: `data/s11-bg-containment-snakeexit-rollout`
(the 10,078 harvest exit rows and 21,522 exit replies the arm claims, with
their rollout targets) and `data/s11-bg-containment-general-wide-rollout`
(301,976 GNUbg rows under the widened rule;
`extract_general_containment_rows.py <output name>`).
`models/s11_diag/sl_cont2_{warm,primrace}.weights.best` train the containment
recipe (family x4) on those plus the original family rows.

*Results on completed references (2026-09-04 evening).* Two completion
passes rolled out every pick of every configuration below — the three snake
nets' root-routed picks (709 boards) and the two containment candidates'
per-node picks on the containment and snake folders (569 boards); ~$30 and
~$50 realized on Fargate against $169 estimates each, because single-board
completion tasks finish in minutes rather than the estimator's twenty — so
every number here is rollout-graded. PR at 1P / 2P / 3P, S9-play reference
first, S11-play twin second:

| Configuration (`stage11s`) | snake, S9-play | snake, S11-play twin |
|---|---|---|
| root routing, round-1 snake net | 16.75 / 17.34 / 25.22 | 18.79 / 18.66 / 15.15 |
| root routing, `sl_snake2_primrace` | 16.37 / 17.67 / 25.11 | 19.60 / 19.87 / 15.35 |
| root routing, `sl_snake2_warm` (**installed 2026-09-05**) | 15.13 / 16.80 / 24.20 | 17.58 / 19.61 / 15.05 |
| per-node, widened rule, old containment net | 16.75 / 46.35 / 39.18 | 18.79 / 45.00 / 29.69 |
| per-node, widened rule, `sl_cont2_warm` | 16.75 / 15.68 / 23.99 | 18.79 / 18.64 / 14.17 |
| per-node, widened rule, `sl_cont2_primrace` | 16.75 / 15.10 / 23.57 | 18.79 / 18.18 / 14.23 |
| per-node, widened rule, `sl_cont2_primrace` + `sl_snake2_warm` | 15.13 / 15.95 / 23.71 | 17.58 / 18.32 / 15.77 |

What the table says. (1) The 2-ply anomaly is gone under either remedy:
1-ply and 2-ply are level, and per-node routing collapses to ~45 only while
the exits reach a containment net that has never seen one. (2) The two
references disagree about 3-ply in opposite directions — the worst level
under S9-play rollouts, the best under S11-play ones — and the split sits
in the values a weak trial player distorts most, the release-versus-hold
checker picks and the cube (cube PR 13-15 against 6-9 at 3-ply). Stage 9
plays these positions at PR 63, so its rollouts are the shakier reference,
but neither is authoritative; a reference rolled out under `stage11s` play,
or an XG pass, would arbitrate. (3) `sl_snake2_warm` is the best snake net
at 1-ply on both references. (4) The widened rule's retrained containment
nets are the stronger snake mechanism at 2-ply and 3-ply, by about 1 PR.

Where the widened rule costs something. A control run (the arm disabled,
same completed references) reproduces the widened-rule containment-folder
numbers to the decimal — 6.72 / 5.67 / 3.62 on the S9-play reference and
6.16 / 5.47 / 2.13 on the twin for the installed net — so the rule itself
changes nothing there. (Those differ from the Stage 11p table above because
this folder's reference was completed further today: every model's headline
moves whenever a reference is completed, so the earlier table is stale on
the reference now on disk and needs rescoring before it is quoted again.)
The cost is on the money benchmark's containment slice, split at 1-ply by
whether the OLD rule already claimed the decision: on the 237 old-rule
decisions the installed net scores 3.03 (4 blunders) but `sl_cont2_warm`
4.27 (7) and `sl_cont2_primrace` 3.94 (7) — retraining on family x4 + the
snake exits + the wide general rows diluted ordinary-game containment —
while on the 8 decisions the arm adds, the old net scores 22.77 (3
blunders) and the retrained ones 8.5-9.4 (1). Both candidates therefore
regress a slice of ordinary games by about 1 PR. The widened rule is the
better snake mechanism only once a containment net exists that also keeps
the old slice at 3.03, which is a training question (weight the
crunched-arm rows as their own family, or give them their own slot) rather
than a routing one.

*Shipped (2026-09-05).* Root routing, with `sl_snake2_warm` installed as
`models/sl_s11_bg_snake.weights.best` (the round-1 net it replaced is the
first row of the table). The widened containment rule is NOT in the code:
its crunched arm was removed again, because with the old containment net it
costs the money slice 3.03 -> 3.67 and with either retrained net 3.03 -> ~4.
The definition, footprint and numbers above are the record of that
experiment; its data (`data/s11-bg-containment-general-wide-rollout`,
`data/s11-bg-containment-snakeexit-rollout`) and the `sl_cont2_*` candidates
stay on disk, untracked. The two mechanisms are compatible — root routing
acts only on decisions whose ROOT is a snake, the arm on everything else it
claims (a broken snake's later decisions, which no folder measures) — so the
arm is worth re-adding once a containment net exists that keeps the old
money slice at 3.03.

### Stage 11m — EXPERIMENTAL: the massive-backgame NN (index 23)

`stage11m` is `stage11s` plus a 24th NN for the benchmark's **massive
backgame** family: a holder behind in the race with >= `MASSIVE_ALT_ANCHORS`
(3) anchors in the opponent's home board, or >= `MASSIVE_MIN_ANCHORS` (2)
anchors and >= `MASSIVE_BACK_MIN` (7) checkers back (opponent's home board
plus the bar), that is neither a containment game nor a snake.
`massive_category(board)` in `neural_net.cpp` mirrors the family filter in
`scripts/backgame_benchmark.py` (`_massive`), which `tests/test_backgame_category.py`
holds it to on the folder references; the phased strategy accepts 22, 23 or
24 NNs and routes snake (22) -> containment (21) -> massive (23) -> trio ->
P3 -> standard. One known difference: the benchmark's own `_containment`
filter is older than the engine's containment rule and disagrees with it on
21 of 7,400 folder boards, which the engine therefore routes to the
containment net while the folder lists them as massive; the test tolerates
that, the benchmark filter is what needs aligning.

**The region was never short of data** — unlike the snake, massive backgames
are 62% of the deep category's rollout pile, 38% of the middle's and 17% of
the double's — so the question was whether a net trained on "massiveness"
across the categories beats the category trio, which serves those decisions
today (a massive-folder root goes to the deep net 51% of the time, the
middle net 22%, P3 20%, the double net 7%). `data/s11-bg-massive-rollout`
(240,173 region rows from the three train piles) and
`data/s11-bg-massive-nbhd-rollout` (their 385,501 non-massive rows plus the
three exit piles) — both built by `scripts/build_massive_data.py` and, like
the category piles they come from, not tracked — feed the containment recipe (`run_s11_containment_sl.py`,
family x4); warm starts from the deep net and from the P3 net converge to
the same net (1-ply ER on the benchmark piles' massive rows 26.59 / 26.55
against the routed trio's 27.38 — deep-category rows 25.6 vs 26.1, middle
27.6 vs 28.7, double 31.6 vs 35.0 — and 30.0 vs 25.9 on the non-massive
neighbourhood), so the data decides and the warm start does not. The
deep-warm net is installed as `models/sl_s11_bg_massive.weights.best`.

**Root routing is OFF for this net** (`root_pinned_` stays `{22}` on the 24-NN
layout): it values the non-massive positions below a massive root worse than
per-node routing does, and pinning it bought nothing on the folder (2-ply
pinned 6.25 / 4.94 vs per-node 6.14 / 4.84; 3-ply 4.34 / 2.94 vs 4.48 / 3.07,
S9-play reference / S11-play twin). The lesson is the snake's in reverse: a
specialist whose neighbourhood is already well served by the nets it would
displace should route per node.

**Results (2026-09-05).** The massive folder's references were completed
twice: first for Stage 9's and the previous layout's picks (run
f364980e6a9b, 508 of 512 positions pooled), then for the massive net's own
(run 2b0459a83c9e, 106 of 110; both tails abandoned once they were retrying
three-hour task timeouts on a handful of positions). That moved the previous
layout from the 7.28 the page still quotes to 6.30 at 1-ply and Stage 9 from
9.43 to 9.78. PR at 1P / 2P / 3P on the final references, S9-play reference
| S11-play twin, every pick rollout-graded, blunders in the second row:

| Model | massive folder |
|---|---|
| Stage 9 | 9.78 / 9.38 / 6.18 \| 9.94 / 8.11 / 4.89 |
| previous layout (23 NNs, `stage11s`) | 6.30 / 5.89 / 4.06 \| 6.66 / 4.78 / 2.83 |
| ... blunders | 85 / 76 / 33 \| 86 / 63 / 29 |
| with the massive net (24 NNs), per-node | 5.39 / 5.77 / 4.14 \| 5.96 / 4.71 / 2.95 |
| ... blunders | 63 / 75 / 41 \| 83 / 64 / 29 |

So: clearly better at 1-ply, slightly better at 2-ply, level at 3-ply, and
on the ordinary-game slice a gain rather than a regression — the money
benchmark's 52 massive-rule decisions score 3.73 with no blunders against
the previous layout's 5.89 and Stage 9's 5.47 — while the ten backgame
folders, where the rule claims up to 18% of a folder, pool to 2.98 at 1-ply
with it, the same as without. A small, clean win at the leaves that the
search does not amplify.

## Glossary

- **ER**: Error Rate — mean equity loss per decision vs GNUbg best, millipips (x1000)
- **PR**: Performance Rating - equal to ER / 2 (total error / # of decisions * 500)
- **ppg**: Points per game
- **PubEval**: Tesauro's linear evaluator, standard weak opponent
- **TD(0)**: Temporal Difference learning (no eligibility trace)
- **gpw**: Game plan weight — gradient multiplier for matching positions in SL
- **TINY filter**: Default move filter (5 moves, 0.08 threshold)
- **VR**: Variance Reduction — luck-tracking for rollout noise reduction (always 1-ply, decoupled from decision ply). Internal to rollouts; distinct from the app-facing per-roll **Luck** metric.
- **Luck** (per-roll): How lucky an actual roll was, in equity units — `actual_equity - average_equity` from the roller's perspective (see Interfaces §7 Luck). Positive = lucky, averages to zero over many rolls. Not the same as VR.
- **BMI**: Best Move Index — the core function that selects the best move from legal candidates (1-ply score + filter + N-ply rescore)
- **Move0Cache/Move1Cache**: Pre-computed move decisions for the first/second half-moves of a rollout trial, shared across all trials
- **SharedPosCache**: Lock-free cross-thread N-ply position evaluation cache for rollout trials
- **Jacoby rule**: Optional money game rule — gammons/backgammons count as single while cube is centered. Default on in Python API, auto-disabled for match play. `CubeInfo::jacoby_active()` checks `jacoby && is_money() && owner == CENTERED`.
- **Beaver rule**: Optional money game rule — opponent can redouble while retaining cube after being doubled. Punishes bad doubles (DT < 0). DB = 2*DT. Default on in Python API, auto-disabled for match play.
- **Janowski**: Cubeless-to-cubeful equity interpolation
- **ND/DT/DP/DB**: No Double / Double-Take / Double-Pass / Double-Beaver
- **MET**: Match Equity Table — lookup table of match-winning probabilities at each score
- **MWC**: Match Winning Chance — probability of winning the match from a given score
- **Crawford**: First game after a player reaches 1-away; no doubling allowed
- **MatchInfo**: `{away1, away2, is_crawford}` — match state; `{0,0,false}` = money game
