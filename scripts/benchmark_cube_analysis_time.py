"""Compare cube analysis time: bgbot Stage 5 vs GNUbg at 3-ply.

Evaluates cubeless probabilities from the starting position (pre-roll) at 3-ply
for both our NN (via MultiPlyStrategy) and GNUbg CLI.

Usage:
    python python/benchmark_cube_analysis_time.py [--build-dir build]
"""

import sys
import os
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=str, default='build')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(project_dir, args.build_dir)
    sys.path.insert(0, build_dir)
    sys.path.insert(0, project_dir)

    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)

    import bgbot_cpp
    from bgsage.gnubg import _build_cube_analytics_command, _run_gnubg, _parse_cube_analytics

    # Starting position (pre-roll, symmetric)
    starting_pos = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0,
                    -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]

    # Model paths (Stage 5)
    models_dir = os.path.join(project_dir, 'models')
    pr_w = os.path.join(models_dir, 'sl_s5_purerace.weights.best')
    rc_w = os.path.join(models_dir, 'sl_s5_racing.weights.best')
    at_w = os.path.join(models_dir, 'sl_s5_attacking.weights.best')
    pm_w = os.path.join(models_dir, 'sl_s5_priming.weights.best')
    an_w = os.path.join(models_dir, 'sl_s5_anchoring.weights.best')

    for path in [pr_w, rc_w, at_w, pm_w, an_w]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            sys.exit(1)

    print("=== Cube Analysis Time Benchmark: Starting Position, 3-ply ===\n")
    print(f"Position: {starting_pos}")
    print(f"This is a pre-roll position (player about to roll).\n")

    # ---------------------------------------------------------------
    # bgbot Stage 5 — cubeless probabilities at 3-ply
    # ---------------------------------------------------------------
    # For a pre-roll position: flip → evaluate (post-move semantics) → invert
    # The MultiPlyStrategy.evaluate_board() expects a post-move board.
    # Flipping the pre-roll board gives the opponent's post-move perspective.

    N_RUNS = 5

    print("--- bgbot Stage 5 (3-ply, TINY filter) ---\n")

    # Create 3-ply strategy
    strat = bgbot_cpp.create_multipy_5nn(
        purerace_weights=pr_w,
        racing_weights=rc_w,
        attacking_weights=at_w,
        priming_weights=pm_w,
        anchoring_weights=an_w,
        n_hidden_purerace=200,
        n_hidden_racing=400,
        n_hidden_attacking=400,
        n_hidden_priming=400,
        n_hidden_anchoring=400,
        n_plies=3,
        filter_max_moves=5,
        filter_threshold=0.08,
        parallel_evaluate=True,
        parallel_threads=0,
    )

    flipped = bgbot_cpp.flip_board(starting_pos)

    # Warm-up run
    _ = strat.evaluate_board(flipped, flipped)
    strat.clear_cache()

    bgbot_times = []
    for i in range(N_RUNS):
        strat.clear_cache()
        t0 = time.perf_counter()
        result = strat.evaluate_board(flipped, flipped)
        dt = time.perf_counter() - t0
        bgbot_times.append(dt)
        print(f"  Run {i+1}: {dt:.3f}s")

    post_probs = result['probs']
    pre_roll_probs = bgbot_cpp.invert_probs_py(post_probs)
    pre_roll_equity = -result['equity']
    t_bgbot = sum(bgbot_times) / len(bgbot_times)

    print(f"\n  Average: {t_bgbot:.3f}s  (min={min(bgbot_times):.3f}, max={max(bgbot_times):.3f})")
    print(f"  P(win):  {pre_roll_probs[0]:.4f}")
    print(f"  P(gw):   {pre_roll_probs[1]:.4f}")
    print(f"  P(bw):   {pre_roll_probs[2]:.4f}")
    print(f"  P(gl):   {pre_roll_probs[3]:.4f}")
    print(f"  P(bl):   {pre_roll_probs[4]:.4f}")
    print(f"  Equity:  {pre_roll_equity:+.4f}")
    print()

    # ---------------------------------------------------------------
    # GNUbg — cubeless probabilities at 3-ply
    # ---------------------------------------------------------------
    # For a pre-roll position, we call GNUbg's cube analytics directly
    # (GNUbg hint gives pre-roll cubeless probs for the player on roll)

    print("--- GNUbg (3-ply) ---\n")

    cmd = _build_cube_analytics_command(starting_pos, n_plies=3)

    gnubg_times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        output = _run_gnubg(cmd, timeout=300)
        dt = time.perf_counter() - t0
        gnubg_times.append(dt)
        print(f"  Run {i+1}: {dt:.3f}s")

    gnubg_result = _parse_cube_analytics(output, n_plies=3)
    t_gnubg = sum(gnubg_times) / len(gnubg_times)

    print(f"\n  Average: {t_gnubg:.3f}s  (min={min(gnubg_times):.3f}, max={max(gnubg_times):.3f})")
    print(f"  P(win):  {gnubg_result['p_win']:.4f}")
    print(f"  P(gw):   {gnubg_result['p_gw']:.4f}")
    print(f"  P(bw):   {gnubg_result['p_bw']:.4f}")
    print(f"  P(gl):   {gnubg_result['p_gl']:.4f}")
    print(f"  P(bl):   {gnubg_result['p_bl']:.4f}")
    print(f"  Equity:  {gnubg_result['equity_cubeless']:+.4f}")
    print()

    # ---------------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------------
    print("--- Comparison ---\n")

    speedup = t_gnubg / t_bgbot if t_bgbot > 0 else float('inf')

    print(f"  bgbot avg:    {t_bgbot:.3f}s")
    print(f"  GNUbg avg:    {t_gnubg:.3f}s")
    print(f"  Speedup:      {speedup:.1f}x")
    print()

    # Probability differences
    gnubg_probs = [gnubg_result['p_win'], gnubg_result['p_gw'], gnubg_result['p_bw'],
                   gnubg_result['p_gl'], gnubg_result['p_bl']]
    labels = ['P(win)', 'P(gw)', 'P(bw)', 'P(gl)', 'P(bl)']

    print(f"  {'':12s}  {'bgbot':>8s}  {'GNUbg':>8s}  {'Diff':>8s}")
    for label, ours, theirs in zip(labels, pre_roll_probs, gnubg_probs):
        print(f"  {label:12s}  {ours:8.4f}  {theirs:8.4f}  {ours - theirs:+8.4f}")
    print(f"  {'Equity':12s}  {pre_roll_equity:+8.4f}  {gnubg_result['equity_cubeless']:+8.4f}  "
          f"{pre_roll_equity - gnubg_result['equity_cubeless']:+8.4f}")
    print()

    # Starting position is symmetric, so equity should be ~0
    print(f"  Note: starting position is symmetric; equity should be ~0.000")


if __name__ == '__main__':
    main()
