"""Test script: evaluate a position at 0-ply, 1-ply, 2-ply, 3-ply, and rollout.

Displays a table of the 5 probabilities, equity, and computation time for each.

Usage:
    python python/test_evaluate_probs.py [--checkers 0,0,0,...] [--build-dir build_msvc]
    python python/test_evaluate_probs.py --ply 3 --stage 4  # 3-ply only benchmark
"""

import sys
import os
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkers', type=str, default=None,
                        help='Comma-separated 26-element board array')
    parser.add_argument('--build-dir', type=str, default='build_msvc',
                        help='Directory containing bgbot_cpp .pyd')
    parser.add_argument('--skip-rollout', action='store_true',
                        help='Skip rollout evaluations')
    parser.add_argument('--stage', type=int, default=5,
                        help='Model stage (4 or 5, default: 5)')
    parser.add_argument('--ply', type=int, default=None,
                        help='Run only this ply depth (and GNUbg at same depth)')
    parser.add_argument('--skip-gnubg', action='store_true',
                        help='Skip GNUbg evaluations')
    args = parser.parse_args()

    # Add the build dir and python dir to sys.path
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(project_dir, args.build_dir)
    build_dir_std = os.path.join(project_dir, 'build')
    sys.path.insert(0, build_dir)
    sys.path.insert(0, build_dir_std)
    sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

    # Add CUDA bin and build dir to DLL search path
    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)
    if os.path.isdir(build_dir_std):
        os.add_dll_directory(build_dir_std)

    import bgbot_cpp

    # Default checkers if not provided
    if args.checkers:
        checkers = list(map(int, args.checkers.split(',')))
    else:
        checkers = [0, 0, 0, 0, 0, 1, 3, 0, 4, 0, 0, 1, -2, 2, -1, 0, 2, -4, -2, -4, -2, 0, 0, 0, 2, 0]

    assert len(checkers) == 26, f"Need 26 elements, got {len(checkers)}"

    # Model paths and hidden sizes by stage
    models_dir = os.path.join(project_dir, 'models')
    stage = args.stage
    prefix = f'sl_s{stage}_'
    pr_w = os.path.join(models_dir, f'{prefix}purerace.weights.best')
    rc_w = os.path.join(models_dir, f'{prefix}racing.weights.best')
    at_w = os.path.join(models_dir, f'{prefix}attacking.weights.best')
    pm_w = os.path.join(models_dir, f'{prefix}priming.weights.best')
    an_w = os.path.join(models_dir, f'{prefix}anchoring.weights.best')

    if stage == 5:
        nh_pr, nh_rc, nh_at, nh_pm, nh_an = 200, 400, 400, 400, 400
    else:
        nh_pr, nh_rc, nh_at, nh_pm, nh_an = 120, 250, 250, 250, 250

    for path in [pr_w, rc_w, at_w, pm_w, an_w]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            sys.exit(1)

    # Classify the position
    gp = bgbot_cpp.classify_game_plan(checkers)
    print(f"Position: {checkers}")
    print(f"Game plan: {gp}")
    print()

    # Build strategies
    # 0-ply: GamePlanStrategy (5-NN)
    gps = bgbot_cpp.GamePlanStrategy(pr_w, rc_w, at_w, pm_w, an_w,
                                      nh_pr, nh_rc, nh_at, nh_pm, nh_an)

    results = []

    # Determine which plies to evaluate
    if args.ply is not None:
        ply_depths = [args.ply]
    else:
        ply_depths = [0, 1, 2, 3]

    # ---- N-ply evaluations ----
    for depth in ply_depths:
        if depth == 0:
            t0 = time.perf_counter()
            r0 = gps.evaluate_board(checkers, checkers)
            t0_elapsed = time.perf_counter() - t0
            results.append({'name': '0-ply', 'probs': r0['probs'], 'equity': r0['equity'],
                            'time': t0_elapsed})
        else:
            mp = bgbot_cpp.create_multipy_5nn(pr_w, rc_w, at_w, pm_w, an_w,
                                               nh_pr, nh_rc, nh_at, nh_pm, nh_an,
                                               n_plies=depth)
            t0 = time.perf_counter()
            r = mp.evaluate_board(checkers, checkers)
            t_elapsed = time.perf_counter() - t0
            cache_h = mp.cache_hits()
            cache_m = mp.cache_misses()
            cache_total = cache_h + cache_m
            hit_pct = 100.0 * cache_h / cache_total if cache_total > 0 else 0
            results.append({'name': f'{depth}-ply', 'probs': r['probs'], 'equity': r['equity'],
                            'time': t_elapsed})
            print(f"  {depth}-ply cache: {cache_h} hits, {cache_m} misses, {hit_pct:.1f}% hit rate")
            mp.clear_cache()

    # ---- gnubg evaluations ----
    if not args.skip_gnubg:
        from bgsage.gnubg import post_move_analytics
        gnubg_plies = ply_depths if args.ply is not None else list(range(4))
        for n_plies in gnubg_plies:
            label = f'gnubg {n_plies}-ply'
            t_ = time.perf_counter()
            r_ = post_move_analytics(checkers, n_plies=n_plies, timeout=600)
            t_elapsed = time.perf_counter() - t_
            results.append({'name': label, 'probs': r_['probs'], 'equity': r_['equity'],
                            'time': t_elapsed})

    # ---- Rollouts at different truncation depths (all dp=1, VR=0) ----
    if not args.skip_rollout and args.ply is None:
        for trunc, label in [(7, 'RO t=7'), (8, 'RO t=8'), (11, 'RO t=11'),
                              (14, 'RO t=14'), (15, 'RO t=15'), (16, 'RO t=16'),
                              (20, 'RO t=20'), (50, 'RO t=50'), (0, 'RO full')]:
            ro = bgbot_cpp.create_rollout_5nn(pr_w, rc_w, at_w, pm_w, an_w,
                                               nh_pr, nh_rc, nh_at, nh_pm, nh_an,
                                               n_trials=360,
                                               truncation_depth=trunc,
                                               decision_ply=1,
                                               vr_ply=0,
                                               n_threads=0)
            t0_ = time.perf_counter()
            r_ = ro.evaluate_board(checkers, checkers)
            t_ = time.perf_counter() - t0_
            results.append({'name': label, 'probs': r_['probs'], 'equity': r_['equity'],
                            'time': t_, 'std_error': r_.get('std_error'),
                            'prob_se': r_.get('prob_std_errors'),
                            'svr_eq': r_.get('scalar_vr_equity'),
                            'svr_se': r_.get('scalar_vr_se')})

        # ---- Same rollouts but with VR disabled to see raw odd/even pattern ----
        for trunc, label in [(7, 'noVR t=7'), (8, 'noVR t=8'), (15, 'noVR t=15'),
                              (16, 'noVR t=16'), (0, 'noVR full')]:
            ro = bgbot_cpp.create_rollout_5nn(pr_w, rc_w, at_w, pm_w, an_w,
                                               nh_pr, nh_rc, nh_at, nh_pm, nh_an,
                                               n_trials=360,
                                               truncation_depth=trunc,
                                               decision_ply=1,
                                               vr_ply=-1,
                                               n_threads=0)
            t0_ = time.perf_counter()
            r_ = ro.evaluate_board(checkers, checkers)
            t_ = time.perf_counter() - t0_
            results.append({'name': label, 'probs': r_['probs'], 'equity': r_['equity'],
                            'time': t_, 'std_error': r_.get('std_error'),
                            'prob_se': r_.get('prob_std_errors')})

    # ---- Display table ----
    def fmt_time(t):
        if t < 1.0: return f"{t*1000:.1f}ms"
        elif t < 60: return f"{t:.2f}s"
        else: return f"{t/60:.1f}min"

    # Main table
    W = 12  # name column width
    print(f"{'Method':<{W}} {'P(win)':>8} {'P(gw)':>8} {'P(bw)':>8} {'P(gl)':>8} {'P(bl)':>8} {'ProbVR Eq':>18} {'ScalarVR Eq':>18} {'Time':>10}")
    print('-' * (W + 99))
    for r in results:
        p = r['probs']
        eq_str = f"{r['equity']:+.4f}"
        se = r.get('std_error')
        if se is not None:
            eq_str += f"+/-{se:.4f}"

        svr_str = ""
        svr_eq = r.get('svr_eq')
        svr_se = r.get('svr_se')
        if svr_eq is not None:
            svr_str = f"{svr_eq:+.4f}"
            if svr_se is not None:
                svr_str += f"+/-{svr_se:.4f}"

        print(f"{r['name']:<{W}} {p[0]:8.4f} {p[1]:8.4f} {p[2]:8.4f} "
              f"{p[3]:8.4f} {p[4]:8.4f} {eq_str:>18} {svr_str:>18} {fmt_time(r['time']):>10}")
        pse = r.get('prob_se')
        if pse:
            print(f"{'  +/- SE':<{W}} {pse[0]:8.4f} {pse[1]:8.4f} {pse[2]:8.4f} "
                  f"{pse[3]:8.4f} {pse[4]:8.4f}")

if __name__ == '__main__':
    main()
