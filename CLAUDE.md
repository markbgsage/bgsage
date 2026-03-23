# bgsage — Open Sage Bot Engine Library

Neural-network backgammon engine with C++ core and Python interface. MIT licensed.

## Git Worktree Rules

**CRITICAL: When working in a git worktree, ALL file operations (reads, edits,
writes, new files, builds, script execution) MUST use the worktree path — never
the main repo path.** The worktree path is shown in the environment as "Worktree
path" and is the primary working directory for the session.

- The worktree has its own branch. Commit and push from the worktree, then merge
  to main via PR or local merge — do NOT commit directly to main.
- Use relative paths or the worktree absolute path for all tool calls. If you see
  yourself using the main repo path (e.g. `C:/.../bgsage/` instead of
  `C:/.../bgsage/.claude/worktrees/<name>/`), STOP and fix it.
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
```

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
| `CubeActionResult` | `cube_action()` | `.equity_nd/dt/dp`, `.should_double`, `.should_take`, `.optimal_action`, `.probs` |
| `PositionEval` | `batch_evaluate()` | `.probs`, `.cubeless_equity`, `.cubeful_equity`, `.equity_nd/dt/dp`, `.optimal_action` |
| `GamePlanResult` | `classify_game_plans()` | `.player`, `.opponent` |
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

## Building

### Windows (MSVC — required for Python 3.14)

**Python 3.14 is compiled with MSVC.** MinGW-compiled pybind11 modules crash due
to incompatible C runtime. Always use MSVC.

```powershell
# One-time CMake configure
cd build_msvc
cmake ..\cpp -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -Dpybind11_DIR=C:/path/to/pybind11/share/cmake/pybind11

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
Training is long-running and must survive past Claude Code's ~1h timeout:
```bash
# IMPORTANT: Use python -u for unbuffered output, full path to python
powershell -Command "Start-Process -FilePath 'C:\Users\mghig\AppData\Local\Programs\Python\Python314\python.exe' -ArgumentList '-u','bgsage\scripts\run_stage5small_training.py' -WorkingDirectory 'C:\Users\mghig\Dropbox\agents\bgbot' -WindowStyle Hidden -RedirectStandardOutput 'C:\Users\mghig\Dropbox\agents\bgbot\logs\training.log' -RedirectStandardError 'C:\Users\mghig\Dropbox\agents\bgbot\logs\training_err.log'"
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
stat -c '%Y' bgsage/models/td_s5s_racing.weights && date +%s

# Check history CSV
cat bgsage/models/td_s5s.history.csv

# Check log output (may be delayed due to C++ internal buffering)
powershell -Command "Get-Content 'logs\training.log' -Tail 20"
```

**Step 5: After training completes**, run benchmarks:
```bash
python bgsage/scripts/run_stage5small_benchmarks.py
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
the chain, calling `evaluate_probs_nply_impl()` at each step's ply level.

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

### Truncated Rollouts (XG Roller-style)

Truncated rollouts are short Monte Carlo simulations truncated at a fixed depth
with N-ply evaluation at the truncation point. They are stronger than pure N-ply
search but faster than full rollouts, making them the best speed/accuracy tradeoff
for position evaluation.

**Key parameters:**
- `n_trials`: Number of trial games per candidate (42-360 typical for truncated rollouts)
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
- Truncation evaluation: `truncation_strat_` (defaults to `decision_ply`, configurable via `truncation_ply`)

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

**App level names**: `truncated1` = XG Roller, `truncated2` = XG Roller+,
`truncated3` = XG Roller++ Checker, `rollout` = full rollout (1296 trials,
play to completion).

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
with separate late/ultra-late fallbacks. Backward compatible — when no per-purpose
configs are set, checker uses `decision_ply` and cube uses 1-ply (identical to
previous behavior).

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
  perspective). For each of 21 dice rolls: generate moves, pick best by cubeless
  1-ply equity (shared across all cube states), flip to opponent perspective, recurse
  at plies-1. Average over 36 total weight, flip perspective back, collapse via
  get_ecf3.

**Top-level entry points:**

- `cube_decision_nply`: Starts with cci=2 (ND state + DT state), fTop=true. Returns
  both ND and DT equities from a single tree traversal.
- `cubeful_equity_nply`: Starts with cci=1, fTop=false. The internal expansion/collapse
  handles all cube branching automatically.

**Key properties:**
- Cube decisions at every level use full recursive values (not heuristic predictions)
- Move selection is cubeless (negligible impact, much cheaper than per-state cubeful)
- Janowski `x` is only applied at 1-ply leaf nodes
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

## Glossary

- **ER**: Error Rate — mean equity loss per decision vs GNUbg best, millipips (x1000)
- **PR**: Performance Rating - equal to ER / 2 (total error / # of decisions * 500)
- **ppg**: Points per game
- **PubEval**: Tesauro's linear evaluator, standard weak opponent
- **TD(0)**: Temporal Difference learning (no eligibility trace)
- **gpw**: Game plan weight — gradient multiplier for matching positions in SL
- **TINY filter**: Default move filter (5 moves, 0.08 threshold)
- **VR**: Variance Reduction — luck-tracking for rollout noise reduction (always 1-ply, decoupled from decision ply)
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
