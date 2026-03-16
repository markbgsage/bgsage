"""Benchmark: XG Roller++ Cube truncated rollout.

Runs a single cube_decision_rollout call (XG Roller++ Cube settings)
and reports all values with standard errors, plus wall-clock time.

The computation is 100% C++ via pybind11 — Python just calls the function.
"""
import os
import sys
import time

# Ensure bgbot_cpp can find CUDA DLLs and is importable
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/x64')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import bgbot_cpp
from bgsage.weights import WeightConfig, PRODUCTION_MODEL

# Position
CHECKERS = [0,0,0,2,2,-2,3,2,2,0,0,0,-3,4,0,0,0,-3,0,-3,-2,-2,0,0,0,0]

# XG Roller++ Cube settings
N_TRIALS = 360
TRUNCATION_DEPTH = 7
DECISION_PLY = 3
LATE_PLY = 2
LATE_THRESHOLD = 2

# Cube settings: centered at 1, money game, jacoby on, beaver on
CUBE_VALUE = 1
OWNER = bgbot_cpp.CubeOwner.CENTERED

# Load production model weights
w = WeightConfig.default()
w.validate()
paths = w.weight_paths
hidden = w.hidden_sizes

print("=" * 70)
print("TRUNCATED ROLLOUT BENCHMARK — XG Roller++ Cube")
print("=" * 70)
print(f"Position: {CHECKERS}")
print(f"Cube: value={CUBE_VALUE}, owner=CENTERED, money game")
print(f"Jacoby: on, Beaver: on")
print(f"Model: {PRODUCTION_MODEL}")
print(f"Settings: {N_TRIALS} trials, depth={TRUNCATION_DEPTH}, "
      f"decision_ply={DECISION_PLY}, late_ply={LATE_PLY}, "
      f"late_threshold={LATE_THRESHOLD}")
print(f"Filter: TINY (5, 0.08)")
print(f"Threads: 0 (auto = all CPUs)")
print()

# Run the benchmark
start = time.perf_counter()
result = bgbot_cpp.cube_decision_rollout(
    checkers=CHECKERS,
    cube_value=CUBE_VALUE,
    owner=OWNER,
    purerace_weights=paths['purerace'],
    racing_weights=paths['racing'],
    attacking_weights=paths['attacking'],
    priming_weights=paths['priming'],
    anchoring_weights=paths['anchoring'],
    n_hidden_purerace=hidden[0],
    n_hidden_racing=hidden[1],
    n_hidden_attacking=hidden[2],
    n_hidden_priming=hidden[3],
    n_hidden_anchoring=hidden[4],
    n_trials=N_TRIALS,
    truncation_depth=TRUNCATION_DEPTH,
    decision_ply=DECISION_PLY,
    filter_max_moves=5,
    filter_threshold=0.08,
    n_threads=0,
    seed=42,
    late_ply=LATE_PLY,
    late_threshold=LATE_THRESHOLD,
    away1=0, away2=0, is_crawford=False,
    cube_x_override=-1.0,
    enable_vr=True,
    jacoby=True,
    beaver=True,
    max_cube_value=0,
)
elapsed = time.perf_counter() - start

# Extract values
probs = result['probs']
prob_se = result['prob_std_errors']
cl_eq = result['cubeless_equity']
cl_se = result['cubeless_se']
eq_nd = result['equity_nd']
eq_nd_se = result['equity_nd_se']
eq_dt = result['equity_dt']
eq_dt_se_raw = result['equity_dt_se']
eq_dp = result['equity_dp']
is_beaver = result['is_beaver']

# Fix beaver SE: if beaver applied, equity_dt = 2*actual_dt, so SE should be 2*raw_se
if is_beaver:
    eq_dt_se = 2.0 * eq_dt_se_raw
else:
    eq_dt_se = eq_dt_se_raw

should_double = result['should_double']
should_take = result['should_take']
optimal_eq = result['optimal_equity']
is_race = result['is_race']

# Print results
print(f"Wall clock time: {elapsed:.3f} s")
print(f"Position type: {'Race' if is_race else 'Contact'}")
print()

print("--- Cubeless Probabilities ---")
prob_names = ['P(win)', 'P(gw)', 'P(bw)', 'P(gl)', 'P(bl)']
for name, p, se in zip(prob_names, probs, prob_se):
    print(f"  {name:8s} = {p:+.5f}  (SE: {se:.5f})")

print()
print("--- Cubeless Equity ---")
print(f"  Equity   = {cl_eq:+.5f}  (SE: {cl_se:.5f})")

print()
print("--- Cubeful Equities ---")
print(f"  E(ND)    = {eq_nd:+.5f}  (SE: {eq_nd_se:.5f})")
dt_label = "E(DB)" if is_beaver else "E(DT)"
print(f"  {dt_label:8s} = {eq_dt:+.5f}  (SE: {eq_dt_se:.5f})")
print(f"  E(DP)    = {eq_dp:+.5f}  (constant)")

print()
print("--- Cube Decision ---")
action = "No Double"
if should_double:
    if is_beaver:
        action = "Double/Beaver" if should_take else "Double/Pass"
    else:
        action = "Double/Take" if should_take else "Double/Pass"
print(f"  Should double: {should_double}")
print(f"  Should take:   {should_take}")
print(f"  Is beaver:     {is_beaver}")
print(f"  Optimal:       {action}  (equity: {optimal_eq:+.5f})")

print()
print("=" * 70)
print(f"BENCHMARK TIME: {elapsed:.3f} seconds")
print("=" * 70)
