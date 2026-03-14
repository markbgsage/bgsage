"""Profile rollout cube action with higher decision plies."""
import sys
import os
import time

build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Add DLL search directories for Windows
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(build_dir)
    # Add CUDA toolkit if available
    for cuda_dir in [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin',
    ]:
        if os.path.isdir(cuda_dir):
            os.add_dll_directory(cuda_dir)

import bgbot_cpp
from bgsage.weights import WeightConfig, PRODUCTION_MODEL

checkers = [0, -2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, -2, -3, -3, -3, -2, 1, 0, 0]

w = WeightConfig.default()
w.validate()

print(f"Position: {checkers}")
print(f"Model: {PRODUCTION_MODEL}")
print(f"Rollout: 36 trials, decision_ply=2, late_ply=1, late_threshold=5")
print(f"Cube: centered, value=1, money game")
print()

t0 = time.time()
result = bgbot_cpp.cube_decision_rollout(
    checkers,
    cube_value=1,
    owner=bgbot_cpp.CubeOwner.CENTERED,
    purerace_weights=w.purerace,
    racing_weights=w.racing,
    attacking_weights=w.attacking,
    priming_weights=w.priming,
    anchoring_weights=w.anchoring,
    n_hidden_purerace=w.n_hidden_purerace,
    n_hidden_racing=w.n_hidden_racing,
    n_hidden_attacking=w.n_hidden_attacking,
    n_hidden_priming=w.n_hidden_priming,
    n_hidden_anchoring=w.n_hidden_anchoring,
    n_trials=36,
    truncation_depth=0,
    decision_ply=2,
    filter_max_moves=5,
    filter_threshold=0.08,
    n_threads=0,
    seed=42,
    late_ply=1,
    late_threshold=5,
    enable_vr=True,
)
elapsed = time.time() - t0

probs = result['probs']
print(f"=== Results ===")
print(f"Cubeless probabilities:")
print(f"  P(win)           = {probs[0]:.6f}")
print(f"  P(gammon_win)    = {probs[1]:.6f}")
print(f"  P(backgammon_win)= {probs[2]:.6f}")
print(f"  P(gammon_loss)   = {probs[3]:.6f}")
print(f"  P(backgammon_loss)= {probs[4]:.6f}")
print()
print(f"Cubeless equity:  {result['cubeless_equity']:+.6f}  (SE: {result['cubeless_se']:.6f})")
print()
print(f"Cubeful equities:")
print(f"  No Double (ND):   {result['equity_nd']:+.6f}  (SE: {result['equity_nd_se']:.6f})")
print(f"  Double/Take (DT): {result['equity_dt']:+.6f}  (SE: {result['equity_dt_se']:.6f})")
print(f"  Double/Pass (DP): {result['equity_dp']:+.6f}")
print()
print(f"Decision: {'Double' if result['should_double'] else 'No Double'}, "
      f"{'Take' if result['should_take'] else 'Pass'}")
print(f"Optimal equity:   {result['optimal_equity']:+.6f}")
print()
print(f"Calculation time: {elapsed:.3f}s")
