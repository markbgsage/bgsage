"""Test doubling cube decisions against reference positions.

Tests the Janowski cubeful equity model at 0-ply against three reference
positions covering all three cube decision types:
  1. Double/Take (centered cube)
  2. No Double (player owns cube)
  3. Double/Pass (player owns cube)

Usage:
    python bgsage/scripts/test_cube_decision.py [--build-dir build_msvc] [--model stage5]
"""

import sys
import os
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=str, default='build_msvc')

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

    from bgsage.weights import WeightConfig
    WeightConfig.add_model_arg(parser)
    args = parser.parse_args()

    build_dir = os.path.join(project_dir, args.build_dir)
    build_dir_std = os.path.join(project_dir, 'build')
    sys.path.insert(0, build_dir)
    sys.path.insert(0, build_dir_std)

    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)
    if os.path.isdir(build_dir_std):
        os.add_dll_directory(build_dir_std)

    import bgbot_cpp

    w = WeightConfig.from_args(args)
    w.validate()

    # =====================================================================
    # Reference positions
    # =====================================================================

    positions = [
        {
            'name': 'Position 1: Double/Take (centered cube=1)',
            'checkers': [0, 0, 0, 2, 2, -2, 5, 0, 3, 0, 0, 0,
                         -3, 3, 0, 0, 0, -3, 0, -3, -2, 0, -2, 0, 0, 0],
            'cube_value': 1,
            'cube_owner': bgbot_cpp.CubeOwner.CENTERED,
            'ref_probs': [0.7668, 0.0329, 0.0006, 0.0189, 0.0006],
            'ref_cubeless_eq': 0.548,
            'ref_nd': 0.913,
            'ref_dt': 0.983,
            'ref_dp': 1.000,
            'ref_should_double': True,
            'ref_should_take': True,
        },
        {
            'name': 'Position 2: No Double (player owns cube=2)',
            'checkers': [0, 0, 0, 2, 2, -2, 4, 0, 2, 0, 1, 0,
                         -3, 4, 0, 0, 0, -1, 0, -3, -2, -2, -2, 0, 0, 0],
            'cube_value': 2,
            'cube_owner': bgbot_cpp.CubeOwner.PLAYER,
            'ref_probs': [0.6808, 0.0185, 0.0003, 0.0147, 0.0002],
            'ref_cubeless_eq': 0.366,
            'ref_nd': 0.615,
            'ref_dt': 0.511,
            'ref_dp': 1.000,
            'ref_should_double': False,
            'ref_should_take': True,  # opponent would take if doubled
        },
        {
            'name': 'Position 3: Double/Pass (player owns cube=4)',
            'checkers': [0, 0, 2, 2, 2, 2, 1, 0, 2, 3, 0, 0,
                         0, 0, 0, 0, 0, -2, -4, -5, 0, -2, -2, 0, 0, 0],
            'cube_value': 4,
            'cube_owner': bgbot_cpp.CubeOwner.PLAYER,
            'ref_probs': [0.8214, 0.0000, 0.0000, 0.0000, 0.0000],
            'ref_cubeless_eq': 0.643,
            'ref_nd': 0.934,
            'ref_dt': 1.159,
            'ref_dp': 1.000,
            'ref_should_double': True,
            'ref_should_take': False,  # opponent should pass
        },
    ]

    print(f"=== Doubling Cube Decision Test ({args.model}) ===\n")

    # First, test with the REFERENCE probs to validate formulas
    print("--- Formula Validation (using reference cubeless probs) ---\n")
    all_ok = True

    for pos in positions:
        ref_probs = pos['ref_probs']
        is_race = bgbot_cpp.is_race(pos['checkers'])

        # Cube efficiency
        cube_x = bgbot_cpp.cube_efficiency(pos['checkers'], is_race)

        # Cube decision using reference probs
        cd = bgbot_cpp.cube_decision_0ply(ref_probs, pos['cube_value'],
                                           pos['cube_owner'], cube_x)
        cl_eq = bgbot_cpp.cubeless_equity(ref_probs)

        print(f"  {pos['name']}")
        print(f"    Cube efficiency x = {cube_x:.3f}")
        print(f"    Cubeless equity:  ref={pos['ref_cubeless_eq']:+.3f}  ours={cl_eq:+.3f}  "
              f"err={abs(cl_eq - pos['ref_cubeless_eq']):.3f}")
        print(f"    No Double:        ref={pos['ref_nd']:+.3f}  ours={cd.equity_nd:+.3f}  "
              f"err={abs(cd.equity_nd - pos['ref_nd']):.3f}")
        print(f"    Double/Take:      ref={pos['ref_dt']:+.3f}  ours={cd.equity_dt:+.3f}  "
              f"err={abs(cd.equity_dt - pos['ref_dt']):.3f}")
        print(f"    Double/Pass:      ref={pos['ref_dp']:+.3f}  ours={cd.equity_dp:+.3f}  "
              f"err={abs(cd.equity_dp - pos['ref_dp']):.3f}")
        print(f"    Should double:    ref={pos['ref_should_double']}  ours={cd.should_double}")
        print(f"    Should take:      ref={pos['ref_should_take']}  ours={cd.should_take}")

        # Check for large errors or wrong decisions
        errors = []
        for label, ref, ours in [('ND', pos['ref_nd'], cd.equity_nd),
                                  ('DT', pos['ref_dt'], cd.equity_dt)]:
            if abs(ours - ref) > 0.10:
                errors.append(f"{label} error {abs(ours - ref):.3f} > 0.10")
        if cd.should_double != pos['ref_should_double']:
            errors.append(f"wrong double decision")
        if cd.should_take != pos['ref_should_take']:
            errors.append(f"wrong take decision")

        if errors:
            print(f"    *** ISSUES: {', '.join(errors)} ***")
            all_ok = False
        else:
            print(f"    OK")
        print()

    # Now test with our NN's own probs
    print(f"--- Full Evaluation (using our NN probs, {args.model}) ---\n")

    for pos in positions:
        result = bgbot_cpp.evaluate_cube_decision(
            pos['checkers'], pos['cube_value'], pos['cube_owner'],
            *w.weight_args)

        probs = result['probs']
        gp = bgbot_cpp.classify_game_plan(pos['checkers'])

        print(f"  {pos['name']}")
        print(f"    Game plan: {gp}, is_race: {result['is_race']}, cube_x: {result['cube_x']:.3f}")
        print(f"    Our probs:  P(w)={probs[0]:.4f}  P(gw)={probs[1]:.4f}  P(bw)={probs[2]:.4f}  "
              f"P(gl)={probs[3]:.4f}  P(bl)={probs[4]:.4f}")
        print(f"    Ref probs:  P(w)={pos['ref_probs'][0]:.4f}  P(gw)={pos['ref_probs'][1]:.4f}  "
              f"P(bw)={pos['ref_probs'][2]:.4f}  P(gl)={pos['ref_probs'][3]:.4f}  "
              f"P(bl)={pos['ref_probs'][4]:.4f}")
        print(f"    Cubeless equity:  ref={pos['ref_cubeless_eq']:+.3f}  ours={result['cubeless_equity']:+.3f}")
        print(f"    No Double:        ref={pos['ref_nd']:+.3f}  ours={result['equity_nd']:+.3f}  "
              f"err={abs(result['equity_nd'] - pos['ref_nd']):.3f}")
        print(f"    Double/Take:      ref={pos['ref_dt']:+.3f}  ours={result['equity_dt']:+.3f}  "
              f"err={abs(result['equity_dt'] - pos['ref_dt']):.3f}")
        print(f"    Double/Pass:      ref={pos['ref_dp']:+.3f}  ours={result['equity_dp']:+.3f}")
        print(f"    Should double:    ref={pos['ref_should_double']}  ours={result['should_double']}")
        print(f"    Should take:      ref={pos['ref_should_take']}  ours={result['should_take']}")
        print(f"    Optimal equity:   {result['optimal_equity']:+.3f}")

        # Flag issues
        errors = []
        for label, ref, ours in [('ND', pos['ref_nd'], result['equity_nd']),
                                  ('DT', pos['ref_dt'], result['equity_dt'])]:
            if abs(ours - ref) > 0.15:
                errors.append(f"{label} error {abs(ours - ref):.3f} > 0.15")
        if result['should_double'] != pos['ref_should_double']:
            errors.append(f"wrong double decision")
        if result['should_take'] != pos['ref_should_take']:
            errors.append(f"wrong take decision")

        if errors:
            print(f"    *** ISSUES: {', '.join(errors)} ***")
            all_ok = False
        else:
            print(f"    OK")
        print()

    # N-ply cube decisions
    print("--- N-ply Cube Decisions (1-ply, using our NN) ---\n")

    for pos in positions:
        t0 = time.time()
        result = bgbot_cpp.cube_decision_nply(
            pos['checkers'], pos['cube_value'], pos['cube_owner'],
            n_plies=1,
            purerace_weights=w.purerace, racing_weights=w.racing,
            attacking_weights=w.attacking, priming_weights=w.priming,
            anchoring_weights=w.anchoring,
            n_hidden_purerace=w.n_hidden_purerace, n_hidden_racing=w.n_hidden_racing,
            n_hidden_attacking=w.n_hidden_attacking, n_hidden_priming=w.n_hidden_priming,
            n_hidden_anchoring=w.n_hidden_anchoring)
        dt = time.time() - t0

        print(f"  {pos['name']}  ({dt:.1f}s)")
        print(f"    No Double:        ref={pos['ref_nd']:+.3f}  ours={result['equity_nd']:+.3f}  "
              f"err={abs(result['equity_nd'] - pos['ref_nd']):.3f}")
        print(f"    Double/Take:      ref={pos['ref_dt']:+.3f}  ours={result['equity_dt']:+.3f}  "
              f"err={abs(result['equity_dt'] - pos['ref_dt']):.3f}")
        print(f"    Double/Pass:      ref={pos['ref_dp']:+.3f}  ours={result['equity_dp']:+.3f}")
        print(f"    Should double:    ref={pos['ref_should_double']}  ours={result['should_double']}")
        print(f"    Should take:      ref={pos['ref_should_take']}  ours={result['should_take']}")
        print(f"    Optimal equity:   {result['optimal_equity']:+.3f}")

        errors = []
        if result['should_double'] != pos['ref_should_double']:
            errors.append(f"wrong double decision")
        if result['should_take'] != pos['ref_should_take']:
            errors.append(f"wrong take decision")

        if errors:
            print(f"    *** ISSUES: {', '.join(errors)} ***")
            all_ok = False
        else:
            print(f"    OK")
        print()

    # N-ply cube decisions at 2-ply and 3-ply
    for n_ply in [2, 3]:
        print(f"--- N-ply Cube Decisions ({n_ply}-ply, using our NN) ---\n")

        for pos in positions:
            t0 = time.time()
            result = bgbot_cpp.cube_decision_nply(
                pos['checkers'], pos['cube_value'], pos['cube_owner'],
                n_plies=n_ply,
                purerace_weights=w.purerace, racing_weights=w.racing,
                attacking_weights=w.attacking, priming_weights=w.priming,
                anchoring_weights=w.anchoring,
                n_hidden_purerace=w.n_hidden_purerace, n_hidden_racing=w.n_hidden_racing,
                n_hidden_attacking=w.n_hidden_attacking, n_hidden_priming=w.n_hidden_priming,
                n_hidden_anchoring=w.n_hidden_anchoring)
            dt = time.time() - t0

            print(f"  {pos['name']}  ({dt:.1f}s)")
            print(f"    No Double:        ref={pos['ref_nd']:+.3f}  ours={result['equity_nd']:+.3f}  "
                  f"err={abs(result['equity_nd'] - pos['ref_nd']):.3f}")
            print(f"    Double/Take:      ref={pos['ref_dt']:+.3f}  ours={result['equity_dt']:+.3f}  "
                  f"err={abs(result['equity_dt'] - pos['ref_dt']):.3f}")
            print(f"    Double/Pass:      ref={pos['ref_dp']:+.3f}  ours={result['equity_dp']:+.3f}")
            print(f"    Should double:    ref={pos['ref_should_double']}  ours={result['should_double']}")
            print(f"    Should take:      ref={pos['ref_should_take']}  ours={result['should_take']}")
            print(f"    Optimal equity:   {result['optimal_equity']:+.3f}")

            errors = []
            if result['should_double'] != pos['ref_should_double']:
                errors.append(f"wrong double decision")
            if result['should_take'] != pos['ref_should_take']:
                errors.append(f"wrong take decision")

            if errors:
                print(f"    *** ISSUES: {', '.join(errors)} ***")
                all_ok = False
            else:
                print(f"    OK")
            print()

    # Rollout-based cube decisions (cubeless rollout + Janowski)
    print("--- Rollout Cube Decisions (1296 trials, 0-ply, VR) ---\n")

    for pos in positions:
        t0 = time.time()
        result = bgbot_cpp.cube_decision_rollout(
            pos['checkers'], pos['cube_value'], pos['cube_owner'],
            purerace_weights=w.purerace, racing_weights=w.racing,
            attacking_weights=w.attacking, priming_weights=w.priming,
            anchoring_weights=w.anchoring,
            n_hidden_purerace=w.n_hidden_purerace, n_hidden_racing=w.n_hidden_racing,
            n_hidden_attacking=w.n_hidden_attacking, n_hidden_priming=w.n_hidden_priming,
            n_hidden_anchoring=w.n_hidden_anchoring,
            n_trials=1296, truncation_depth=0, decision_ply=0, vr_ply=0,
            n_threads=0, seed=42)
        dt = time.time() - t0

        probs = result['probs']
        print(f"  {pos['name']}  ({dt:.1f}s)")
        print(f"    Rollout probs:  P(w)={probs[0]:.4f}  P(gw)={probs[1]:.4f}  "
              f"P(bw)={probs[2]:.4f}  P(gl)={probs[3]:.4f}  P(bl)={probs[4]:.4f}")
        print(f"    Cubeless equity:  ref={pos['ref_cubeless_eq']:+.3f}  "
              f"ours={result['cubeless_equity']:+.3f}  SE={result['cubeless_se']:.3f}")
        print(f"    No Double:        ref={pos['ref_nd']:+.3f}  ours={result['equity_nd']:+.3f}  "
              f"err={abs(result['equity_nd'] - pos['ref_nd']):.3f}")
        print(f"    Double/Take:      ref={pos['ref_dt']:+.3f}  ours={result['equity_dt']:+.3f}  "
              f"err={abs(result['equity_dt'] - pos['ref_dt']):.3f}")
        print(f"    Double/Pass:      ref={pos['ref_dp']:+.3f}  ours={result['equity_dp']:+.3f}")
        print(f"    Should double:    ref={pos['ref_should_double']}  ours={result['should_double']}")
        print(f"    Should take:      ref={pos['ref_should_take']}  ours={result['should_take']}")
        print(f"    Optimal equity:   {result['optimal_equity']:+.3f}")

        errors = []
        if result['should_double'] != pos['ref_should_double']:
            errors.append(f"wrong double decision")
        if result['should_take'] != pos['ref_should_take']:
            errors.append(f"wrong take decision")

        if errors:
            print(f"    *** ISSUES: {', '.join(errors)} ***")
            all_ok = False
        else:
            print(f"    OK")
        print()

    # Sanity checks
    print("--- Sanity Checks ---\n")

    # x=0 should give cubeless equity
    test_probs = [0.6, 0.05, 0.01, 0.03, 0.005]
    cl_eq = bgbot_cpp.cubeless_equity(test_probs)
    cf_eq_x0 = bgbot_cpp.cl2cf_money(test_probs, bgbot_cpp.CubeOwner.CENTERED, 0.0)
    print(f"  x=0 test: cubeless={cl_eq:+.4f}  cl2cf(x=0)={cf_eq_x0:+.4f}  "
          f"match={'OK' if abs(cl_eq - cf_eq_x0) < 0.001 else 'FAIL'}")

    # P(win)=0.5 no gammons → equity ~ 0
    sym_probs = [0.5, 0.0, 0.0, 0.0, 0.0]
    sym_cl = bgbot_cpp.cubeless_equity(sym_probs)
    sym_cf = bgbot_cpp.cl2cf_money(sym_probs, bgbot_cpp.CubeOwner.CENTERED, 0.68)
    print(f"  50/50 no gammons: cubeless={sym_cl:+.4f}  cubeful={sym_cf:+.4f}  "
          f"match={'OK' if abs(sym_cf) < 0.01 else 'FAIL'}")

    # Player equity + opponent equity should be close to 0 for centered cube
    probs_a = [0.65, 0.03, 0.001, 0.02, 0.001]
    cf_player = bgbot_cpp.cl2cf_money(probs_a, bgbot_cpp.CubeOwner.PLAYER, 0.68)
    cf_opp = bgbot_cpp.cl2cf_money(probs_a, bgbot_cpp.CubeOwner.OPPONENT, 0.68)
    cf_center = bgbot_cpp.cl2cf_money(probs_a, bgbot_cpp.CubeOwner.CENTERED, 0.68)
    print(f"  Ownership comparison: player_owns={cf_player:+.4f}  "
          f"centered={cf_center:+.4f}  opp_owns={cf_opp:+.4f}")
    print(f"    player > centered > opponent: "
          f"{'OK' if cf_player > cf_center > cf_opp else 'FAIL'}")

    print()
    if all_ok:
        print("All tests passed!")
    else:
        print("Some tests had issues — see above.")


if __name__ == '__main__':
    main()
