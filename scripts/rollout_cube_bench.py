"""Benchmark cubeful rollout cube decisions at various settings.

Tests the reference position with different evaluation strengths,
with ultra_late_threshold=9999 (no ply reductions) to match XG behavior.
Uses progress callback to report incremental trial completion.
"""
import sys
import os
import time

# Unbuffered stdout for real-time log visibility
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# CUDA DLL
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64")

import bgbot_cpp
from bgsage.weights import WeightConfig

w = WeightConfig.default()

POSITION = [2,2,0,0,0,0,4,1,2,0,1,0,-4,3,0,0,0,-3,0,-4,-2,0,1,0,1,0]
N_TRIALS = 1296
N_THREADS = 16

configs = [
    ("3p", {
        "enable_vr": True,
        "ultra_late_threshold": 9999,
        "checker": bgbot_cpp.TrialEvalConfig(ply=3),
        "cube": bgbot_cpp.TrialEvalConfig(ply=3),
    }),
]

print(f"Position: {POSITION}")
print(f"Trials: {N_TRIALS}, Threads: {N_THREADS}")
print(f"ultra_late_threshold: 9999 (no ply reductions)")
print()

header = f"{'Config':<16} {'ND eq':>8} {'ND SE':>7} {'DT eq':>8} {'DT SE':>7} {'CL eq':>8} {'CL SE':>7} {'P(win)':>7} {'P(gw)':>7} {'P(gl)':>7} {'Time':>8}"
print(header)
print("-" * len(header))

for name, cfg in configs:
    # Progress callback: print progress periodically
    t_start = time.perf_counter()
    last_pct = [0]

    def make_progress_cb(config_name, t_start_ref):
        last_report = [0.0]
        def progress_cb(completed, total):
            pct = completed * 100 // total
            now = time.perf_counter()
            elapsed = now - t_start_ref
            # Report every 10% or every 5 seconds, whichever comes first
            if pct >= last_pct[0] + 10 or (now - last_report[0]) >= 5.0:
                last_pct[0] = pct
                last_report[0] = now
                print(f"  [{config_name}] {completed}/{total} trials ({pct}%) - {elapsed:.1f}s elapsed")
        return progress_cb

    progress_cb = make_progress_cb(name, t_start)

    kwargs = {
        "checkers": POSITION,
        "cube_value": 1,
        "owner": bgbot_cpp.CubeOwner.CENTERED,
        **(dict(zip(
            ["purerace_weights", "racing_weights", "attacking_weights",
             "priming_weights", "anchoring_weights"],
            [w.weight_paths["purerace"], w.weight_paths["racing"],
             w.weight_paths["attacking"], w.weight_paths["priming"],
             w.weight_paths["anchoring"]]
        ))),
        "n_hidden_purerace": w.hidden_sizes[0],
        "n_hidden_racing": w.hidden_sizes[1],
        "n_hidden_attacking": w.hidden_sizes[2],
        "n_hidden_priming": w.hidden_sizes[3],
        "n_hidden_anchoring": w.hidden_sizes[4],
        "n_trials": N_TRIALS,
        "truncation_depth": 0,  # full rollout
        "decision_ply": 1,
        "n_threads": N_THREADS,
        "seed": 42,
        "enable_vr": cfg.get("enable_vr", True),
        "jacoby": True,
        "beaver": True,
        "ultra_late_threshold": cfg.get("ultra_late_threshold", 2),
        "progress": progress_cb,
    }
    if "checker" in cfg:
        kwargs["checker"] = cfg["checker"]
    if "cube" in cfg:
        kwargs["cube"] = cfg["cube"]

    print(f"  [{name}] Starting...")
    t0 = time.perf_counter()
    last_pct[0] = 0
    r = bgbot_cpp.cube_decision_rollout(**kwargs)
    elapsed = time.perf_counter() - t0

    probs = r["probs"]
    print(f"{name:<16} {r['equity_nd']:>+8.3f} {r['equity_nd_se']:>7.3f} "
          f"{r['equity_dt']:>+8.3f} {r['equity_dt_se']:>7.3f} "
          f"{r['cubeless_equity']:>+8.3f} {r['cubeless_se']:>7.3f} "
          f"{probs[0]:>7.4f} {probs[1]:>7.4f} {probs[3]:>7.4f} "
          f"{elapsed:>7.1f}s")
    print()

print("XG reference (Default equiv): ND +0.598, DT +0.606, CL eq +0.449 SE 0.014")
print("  P(win)=0.6007, P(gw)=0.3801, P(gl)=0.1156")
print("\nDone.")
