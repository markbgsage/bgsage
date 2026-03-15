"""Test that cubeful(max_cube=1, Jacoby=off) produces cubeless-equivalent results.

Verifies the unification of cubeless and cubeful code paths by comparing:
  1. N-ply cubeful equity (max_cube_value=1, Jacoby=off) vs cubeless equity at 1-4 ply
  2. Cubeful rollout (max_cube_value=1) vs (max_cube_value=0): cubeless probs/equity match
  3. Cubeful rollout (max_cube=1) equity_nd == cubeless_equity from same result
  4. Rollout determinism: re-running with same seed gives identical results
  5. Dead cube semantics: should_double=false when max_cube=1
  6. Performance: cubeful(max_cube=1) rollout has comparable speed to cubeless rollout

The N-ply test uses relaxed tolerance (2e-3) at 2-4 ply because cubeful and cubeless
use different recursive implementations (cubeful_recursive_multi vs
evaluate_probs_nply_impl) that select moves slightly differently at intermediate
nodes (different game plan classification, pre-filtering). At 1-ply both use direct
NN evaluation, so tolerance is tight (1e-6).

Usage:
    python bgsage/scripts/test_unified_rollout.py [--build-dir build] [--model stage5]
"""

import sys
import os
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=str, default='build')

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))

    # Pre-parse to get build_dir before importing bgsage (which needs bgbot_cpp)
    args_pre, _ = parser.parse_known_args()
    build_dir = os.path.join(project_dir, args_pre.build_dir)
    build_dir_std = os.path.join(project_dir, 'build')

    if sys.platform == 'win32':
        cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
        if os.path.isdir(cuda_x64):
            os.add_dll_directory(cuda_x64)
        if os.path.isdir(build_dir):
            os.add_dll_directory(build_dir)
        if os.path.isdir(build_dir_std):
            os.add_dll_directory(build_dir_std)

    sys.path.insert(0, build_dir)
    sys.path.insert(0, build_dir_std)
    sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

    import bgbot_cpp
    bgbot_cpp.init_escape_tables()

    from bgsage.weights import WeightConfig
    WeightConfig.add_model_arg(parser)
    args = parser.parse_args()

    w = WeightConfig.from_args(args)
    w.validate()
    w.print_summary('Model')

    # Reference position
    checkers = [0, 0, 0, 0, 2, 2, 3, 0, 3, 0, 0, 0, -4, 3, 0, 0, 0, -2, -2, -3, 2, -2, -2, 0, 0, 0]

    # Helper: call cube_decision_nply with keyword args
    def nply_cube(ply, max_cube_value=0):
        return bgbot_cpp.cube_decision_nply(
            checkers, cube_value=1, owner=bgbot_cpp.CubeOwner.CENTERED,
            n_plies=ply,
            purerace_weights=w.purerace, racing_weights=w.racing,
            attacking_weights=w.attacking, priming_weights=w.priming,
            anchoring_weights=w.anchoring,
            n_hidden_purerace=w.n_hidden_purerace, n_hidden_racing=w.n_hidden_racing,
            n_hidden_attacking=w.n_hidden_attacking, n_hidden_priming=w.n_hidden_priming,
            n_hidden_anchoring=w.n_hidden_anchoring,
            jacoby=False, beaver=False, max_cube_value=max_cube_value)

    all_pass = True

    # =====================================================================
    # Test 1: N-ply equivalence (1-4 ply)
    # =====================================================================
    # At 1-ply: both code paths use direct NN evaluation → exact match (1e-6).
    # At 2-4 ply: cubeful uses cubeful_recursive_multi, cubeless uses
    # evaluate_probs_nply_impl — different move selection at intermediate
    # nodes causes small differences (2e-4 to 1e-3). Use relaxed tolerance.
    print("=" * 70)
    print("Test 1: N-ply cubeful(max_cube=1) vs cubeless equity")
    print("=" * 70)

    for ply in [1, 2, 3, 4]:
        # Get cubeless equity at N-ply
        if ply == 1:
            result_cl = bgbot_cpp.evaluate_cube_decision(
                checkers, 1, bgbot_cpp.CubeOwner.CENTERED,
                *w.weight_args, jacoby=False, beaver=False)
            cl_eq = result_cl['cubeless_equity']
        else:
            result_nply = nply_cube(ply, max_cube_value=0)
            cl_eq = result_nply['cubeless_equity']

        # Cubeful N-ply with max_cube_value=1 (should approximate cubeless)
        if ply == 1:
            result_cf = bgbot_cpp.evaluate_cube_decision(
                checkers, 1, bgbot_cpp.CubeOwner.CENTERED,
                *w.weight_args, jacoby=False, beaver=False, max_cube_value=1)
            cf_nd = result_cf['equity_nd']
        else:
            result_cf = nply_cube(ply, max_cube_value=1)
            cf_nd = result_cf['equity_nd']

        diff = abs(cf_nd - cl_eq)
        # Tight tolerance at 1-ply (same code path), relaxed at higher plies
        tol = 1e-6 if ply == 1 else 2e-3
        ok = diff < tol
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {ply}-ply: cubeless_eq={cl_eq:+.10f}  cubeful_nd(max=1)={cf_nd:+.10f}  diff={diff:.2e}  tol={tol:.0e}  [{status}]")

    # =====================================================================
    # Test 2: Cubeful rollout — cubeless results independent of max_cube_value
    # =====================================================================
    # The cubeless probs/equity from cube_decision_rollout should be
    # identical regardless of max_cube_value, because the cubeless VR
    # and move selection code paths don't depend on cube state.
    print()
    print("=" * 70)
    print("Test 2: Rollout cubeless results: max_cube=1 vs max_cube=0")
    print("=" * 70)

    # Cubeful rollout with max_cube_value=0 (unlimited — standard cubeful)
    cf_result_0 = bgbot_cpp.cube_decision_rollout(
        checkers, 1, bgbot_cpp.CubeOwner.CENTERED,
        *w.weight_args,
        n_trials=36, truncation_depth=7,
        decision_ply=1, n_threads=0, seed=42,
        jacoby=False, beaver=False, max_cube_value=0)
    cf0_cl_eq = cf_result_0['cubeless_equity']
    cf0_cl_probs = list(cf_result_0['probs'])

    # Cubeful rollout with max_cube_value=1 (dead cube — cubeless equivalent)
    cf_result_1 = bgbot_cpp.cube_decision_rollout(
        checkers, 1, bgbot_cpp.CubeOwner.CENTERED,
        *w.weight_args,
        n_trials=36, truncation_depth=7,
        decision_ply=1, n_threads=0, seed=42,
        jacoby=False, beaver=False, max_cube_value=1)
    cf1_cl_eq = cf_result_1['cubeless_equity']
    cf1_cl_probs = list(cf_result_1['probs'])

    print(f"  max_cube=0: cubeless_eq={cf0_cl_eq:+.15f}")
    print(f"    probs: {cf0_cl_probs}")
    print(f"  max_cube=1: cubeless_eq={cf1_cl_eq:+.15f}")
    print(f"    probs: {cf1_cl_probs}")

    # Cubeless equity should match exactly
    eq_diff = abs(cf1_cl_eq - cf0_cl_eq)
    ok_eq = eq_diff < 1e-10
    status = "PASS" if ok_eq else "FAIL"
    if not ok_eq:
        all_pass = False
    print(f"  Cubeless equity diff: {eq_diff:.2e}  [{status}]")

    # Cubeless probs should match exactly
    max_prob_diff = max(abs(cf1_cl_probs[k] - cf0_cl_probs[k]) for k in range(5))
    ok_probs = max_prob_diff < 1e-10
    status = "PASS" if ok_probs else "FAIL"
    if not ok_probs:
        all_pass = False
    print(f"  Max cubeless prob diff: {max_prob_diff:.2e}  [{status}]")

    # =====================================================================
    # Test 3: equity_nd(max_cube=1) == cubeless_equity from same rollout
    # =====================================================================
    # When cube is dead, cubeful equity_nd should equal cubeless equity
    # since Janowski is bypassed at every point.
    print()
    print("=" * 70)
    print("Test 3: Rollout equity_nd(max_cube=1) == cubeless_equity")
    print("=" * 70)

    cf1_nd = cf_result_1['equity_nd']
    cf1_cl = cf_result_1['cubeless_equity']
    nd_diff = abs(cf1_nd - cf1_cl)
    ok_nd = nd_diff < 1e-6
    status = "PASS" if ok_nd else "FAIL"
    if not ok_nd:
        all_pass = False
    print(f"  equity_nd:       {cf1_nd:+.15f}")
    print(f"  cubeless_equity: {cf1_cl:+.15f}")
    print(f"  Diff: {nd_diff:.2e}  [{status}]")

    # Also verify equity_nd is different when max_cube=0 (Janowski applied)
    cf0_nd = cf_result_0['equity_nd']
    nd_gap = abs(cf0_nd - cf0_cl_eq)
    print(f"  equity_nd(max=0): {cf0_nd:+.15f}  (gap from cubeless: {nd_gap:.4f})")
    if nd_gap < 1e-3:
        print(f"  WARNING: equity_nd(max=0) unexpectedly close to cubeless — Janowski may not be applied")

    # =====================================================================
    # Test 4: Rollout determinism (run-to-run)
    # =====================================================================
    # Verify that running with the same seed produces identical results.
    print()
    print("=" * 70)
    print("Test 4: Rollout determinism (same seed -> same result)")
    print("=" * 70)

    # Run cubeless rollout (via evaluate_board)
    rollout_cl = bgbot_cpp.create_rollout_5nn(
        *w.weight_args,
        n_trials=36, truncation_depth=7,
        decision_ply=1, n_threads=0, seed=42)

    run1 = rollout_cl.evaluate_board(checkers, checkers)
    run2 = rollout_cl.evaluate_board(checkers, checkers)

    eq1 = run1['equity']
    eq2 = run2['equity']
    probs1 = list(run1['probs'])
    probs2 = list(run2['probs'])

    eq_det_diff = abs(eq1 - eq2)
    ok_det_eq = eq_det_diff < 1e-15
    status = "PASS" if ok_det_eq else "FAIL"
    if not ok_det_eq:
        all_pass = False
    print(f"  Run 1 equity: {eq1:+.15f}")
    print(f"  Run 2 equity: {eq2:+.15f}")
    print(f"  Diff: {eq_det_diff:.2e}  [{status}]")

    max_det_pdiff = max(abs(probs1[k] - probs2[k]) for k in range(5))
    ok_det_p = max_det_pdiff < 1e-15
    status = "PASS" if ok_det_p else "FAIL"
    if not ok_det_p:
        all_pass = False
    print(f"  Max prob diff: {max_det_pdiff:.2e}  [{status}]")

    # Also verify cubeful rollout determinism
    cf_run2 = bgbot_cpp.cube_decision_rollout(
        checkers, 1, bgbot_cpp.CubeOwner.CENTERED,
        *w.weight_args,
        n_trials=36, truncation_depth=7,
        decision_ply=1, n_threads=0, seed=42,
        jacoby=False, beaver=False, max_cube_value=1)
    cf_det_diff = abs(cf_run2['cubeless_equity'] - cf1_cl_eq)
    ok_cf_det = cf_det_diff < 1e-15
    status = "PASS" if ok_cf_det else "FAIL"
    if not ok_cf_det:
        all_pass = False
    print(f"  Cubeful rollout determinism diff: {cf_det_diff:.2e}  [{status}]")

    # =====================================================================
    # Test 5: Dead cube semantics
    # =====================================================================
    print()
    print("=" * 70)
    print("Test 5: Dead cube: should_double=false when max_cube=1")
    print("=" * 70)

    for ply in [1, 2]:
        if ply == 1:
            r = bgbot_cpp.evaluate_cube_decision(
                checkers, 1, bgbot_cpp.CubeOwner.CENTERED,
                *w.weight_args, jacoby=False, beaver=False, max_cube_value=1)
        else:
            r = nply_cube(ply, max_cube_value=1)

        sd = r['should_double']
        ok_sd = not sd
        status = "PASS" if ok_sd else "FAIL"
        if not ok_sd:
            all_pass = False
        print(f"  {ply}-ply: should_double={sd}  [{status}]")

    # Rollout — reuse cf_result_1 from Test 2
    sd_r = cf_result_1['should_double']
    ok_r = not sd_r
    status = "PASS" if ok_r else "FAIL"
    if not ok_r:
        all_pass = False
    print(f"  rollout: should_double={sd_r}  [{status}]")

    # =====================================================================
    # Test 6: Performance comparison
    # =====================================================================
    print()
    print("=" * 70)
    print("Test 6: Performance comparison (cubeless vs cubeful(max=1))")
    print("=" * 70)

    # Run 3 times each for more stable timing
    times_cl = []
    times_cf = []
    for i in range(3):
        t0 = time.perf_counter()
        rollout_cl.evaluate_board(checkers, checkers)
        times_cl.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        bgbot_cpp.cube_decision_rollout(
            checkers, 1, bgbot_cpp.CubeOwner.CENTERED,
            *w.weight_args,
            n_trials=36, truncation_depth=7,
            decision_ply=1, n_threads=0, seed=42,
            jacoby=False, beaver=False, max_cube_value=1)
        times_cf.append(time.perf_counter() - t0)

    median_cl = sorted(times_cl)[1]
    median_cf = sorted(times_cf)[1]
    ratio = median_cf / median_cl if median_cl > 0 else float('inf')
    ok_perf = ratio < 1.5  # Allow up to 50% overhead (conservative, expect < 5%)
    status = "PASS" if ok_perf else "FAIL"
    if not ok_perf:
        all_pass = False
    print(f"  Cubeless median:       {median_cl:.3f}s")
    print(f"  Cubeful(max=1) median: {median_cf:.3f}s")
    print(f"  Ratio: {ratio:.2f}x  [{status}]")

    # =====================================================================
    # Summary
    # =====================================================================
    print()
    print("=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
