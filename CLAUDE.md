# bgsage — Backgammon Bot Engine Library

Neural-network backgammon engine with C++ core and Python interface. MIT licensed.

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
data/                        # GNUbg benchmark + training data
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

## Benchmark Scripts

All benchmark scripts default to the production model and accept `--model <name>`
to override. Scripts live in `scripts/`.

| Script | What it measures | Key args |
|--------|-----------------|----------|
| `run_full_benchmark.py` | Full suite: per-plan ER + contact/crashed/race ER + vs PubEval ppg + self-play distribution. Supports 0-ply through N-ply. | `--model`, `--ply N`, `--scenarios N`, `--threads N`, `--games N` |
| `run_rollout_benchmark.py` | Top-N worst 0-ply errors compared at 1-ply, 2-ply, 3-ply, rollout | `--model`, `--top N`, `--threads N` |
| `score_benchmark_pr.py` | Benchmark PR (equity error vs rollout reference, 103k decisions) | `--model`, `--plies N`, `--all-models`, `--all-plies` |
| `score_benchmark_pr_gnubg.py` | GNUbg's Benchmark PR (parallel GNUbg CLI subprocesses) | `--plies N` |
| `test_evaluate_probs.py` | Single position eval at 0-3 ply + GNUbg + rollouts | `--model`, `--checkers`, `--ply N` |
| `test_cube_decision.py` | Cube decisions vs 3 reference positions at 0-3 ply + rollout | `--model` |

```bash
# Full benchmark with production model (0-ply)
python scripts/run_full_benchmark.py

# Compare two models
python scripts/run_full_benchmark.py --model stage5
python scripts/run_full_benchmark.py --model stage3

# Multi-ply benchmark
python scripts/run_full_benchmark.py --ply 1
python scripts/run_full_benchmark.py --ply 2 --scenarios 500

# Score all registered models on Benchmark PR
python scripts/score_benchmark_pr.py --all-models
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

### Key Training Scripts

```bash
# TD self-play (5-NN)
python scripts/run_td_gameplan_training.py --games 200000 --alpha 0.1

# GPU supervised learning
python scripts/run_gpu_sl_training.py --type racing --epochs 500 --alpha 2.0
```

See "Benchmark Scripts" section above for all benchmarking commands.

## Multi-Ply Search

- 0-ply: Direct NN evaluation
- 1-ply: Average over 21 opponent rolls (~60x slower with TINY filter)
- 2-ply: Recursive (~800-1000x slower than 0-ply)

**Move filter**: After 0-ply scoring, keep top `max_moves` within `threshold` equity.
Default TINY: 5 moves, 0.08 threshold.

**Optimizations**: AVX2 FMA intrinsics, fast sigmoid LUT, open-addressing position
cache, incremental delta evaluation, transposed weight matrix.

## Rollout

Monte Carlo evaluation with XG-style variance reduction. Stratified first roll
(36 dice pairs). Parallelized trial execution.

## Doubling Cube

Janowski interpolation for money games. Cube efficiency: 0.68 contact,
pip-dependent for race.

## Current Best Scores (Production Model: stage5)

| Metric | 0-ply | Target |
|--------|-------|--------|
| Contact ER | 9.87 | < 10.5 |
| Race ER | 0.95 | < 0.643 |
| vs PubEval | +0.633 | > +0.63 |

Benchmark PR (103k decisions): 0-ply=2.47, 1-ply=1.85, 2-ply=1.53.

The production model is defined in `python/bgsage/weights.py` — see "Production Model"
section above. See `MODEL_BENCHMARKS.md` for full comparison of all trained models.

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

We (and GNUbg) call raw NN evaluation "0-ply". XG calls it "1-ply". So XG's 2-ply
= our 1-ply, XG's 4-ply = our 3-ply. Keep this in mind when comparing results.

## C++ Gotchas

- `std::fixed` is sticky — always reset with `std::defaultfloat` at start of
  functions that need default formatting
- `MultiPlyStrategy::get_cache()` uses `thread_local static PosCache` — ALL instances
  share the same cache. Call `clear_cache()` between strategy comparisons.
- Rollout `NeuralNetwork` transposed-weight init uses `std::call_once` for thread safety

## Glossary

- **ER**: Error Rate — mean equity loss per decision vs GNUbg best, millipips (x1000)
- **ppg**: Points per game
- **PubEval**: Tesauro's linear evaluator, standard weak opponent
- **TD(0)**: Temporal Difference learning (no eligibility trace)
- **gpw**: Game plan weight — gradient multiplier for matching positions in SL
- **TINY filter**: Default move filter (5 moves, 0.08 threshold)
- **VR**: Variance Reduction — luck-tracking for rollout noise reduction
- **Janowski**: Cubeless-to-cubeful equity interpolation
- **ND/DT/DP**: No Double / Double-Take / Double-Pass
