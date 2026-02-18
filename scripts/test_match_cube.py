"""Test match play cube decisions.

Tests cube decisions at various match scores to verify correct behavior:
  1. DMP (1-away, 1-away): gammons irrelevant, equity ≈ 2*P(win)-1
  2. Crawford game: can_double returns false, no doubling
  3. Post-Crawford 2-away/1-away: free drop / mandatory double
  4. Normal match scores: 0-ply cube decisions
  5. Money game regression: verify existing money decisions unchanged
  6. Batch evaluation with match play
  7. N-ply match cube decisions (1-ply)

Usage:
    python bgsage/scripts/test_match_cube.py [--build-dir build] [--model stage5]
"""

import sys
import os
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=str, default='build')

    # Setup paths — must set DLL dirs before any bgsage imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bgsage_dir = os.path.dirname(script_dir)
    project_dir = os.path.dirname(bgsage_dir)

    # Pre-parse build-dir to set paths before importing bgsage
    build_dir_arg = 'build'
    for i, arg in enumerate(sys.argv):
        if arg == '--build-dir' and i + 1 < len(sys.argv):
            build_dir_arg = sys.argv[i + 1]

    build_dir = os.path.join(project_dir, build_dir_arg)
    build_dir_std = os.path.join(project_dir, 'build')
    sys.path.insert(0, os.path.join(bgsage_dir, 'python'))
    sys.path.insert(0, build_dir)
    sys.path.insert(0, build_dir_std)

    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)
    if os.path.isdir(build_dir_std):
        os.add_dll_directory(build_dir_std)

    from bgsage.weights import WeightConfig
    WeightConfig.add_model_arg(parser)
    args = parser.parse_args()

    import bgbot_cpp
    from bgsage import BgBotAnalyzer, STARTING_BOARD, batch_evaluate, batch_cube_action

    w = WeightConfig.from_args(args)
    w.validate()

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} — {detail}")
            failed += 1

    CubeOwner = bgbot_cpp.CubeOwner

    # Reference positions for testing
    # Position 1: Starting position (equal, slight roller advantage)
    starting = STARTING_BOARD

    # Position 2: Strong position (near bear-off race)
    # Player has 5 checkers borne off, opponent still in full game
    strong_pos = [0, 0, 0, 0, 0, 3, 4, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]

    # Position 3: Race position (contact broken)
    race_pos = [0, 0, 0, 0, 4, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, -4, -3, -2, -1, 0, 0]

    # =========================================================================
    print("\n=== Test 1: DMP (1-away, 1-away) ===")
    # =========================================================================
    # At DMP, gammons are irrelevant. Cube is dead.
    # Cubeful equity should approximate 2*P(win)-1

    r = bgbot_cpp.evaluate_cube_decision(
        starting, 1, CubeOwner.CENTERED, *w.weight_args,
        away1=1, away2=1, is_crawford=False,
    )
    probs = list(r["probs"])
    p_win = probs[0]
    dmp_eq = 2.0 * p_win - 1.0

    check("DMP: should not double (cube is dead)",
          not r["should_double"],
          f"should_double={r['should_double']}")
    check("DMP: equity_nd close to 2*P(win)-1",
          abs(r["equity_nd"] - dmp_eq) < 0.05,
          f"equity_nd={r['equity_nd']:.4f}, 2*P(win)-1={dmp_eq:.4f}")
    print(f"    P(win)={p_win:.4f}, ND={r['equity_nd']:.4f}, 2*Pwin-1={dmp_eq:.4f}")

    # =========================================================================
    print("\n=== Test 2: Crawford Game (5-away, 1-away) ===")
    # =========================================================================
    # In Crawford game, nobody can double. Cube action should show No Double.

    r_craw = bgbot_cpp.evaluate_cube_decision(
        starting, 1, CubeOwner.CENTERED, *w.weight_args,
        away1=5, away2=1, is_crawford=True,
    )
    check("Crawford: should not double",
          not r_craw["should_double"],
          f"should_double={r_craw['should_double']}")
    print(f"    ND={r_craw['equity_nd']:.4f}")

    # Also test via BgBotAnalyzer
    analyzer = BgBotAnalyzer(weights=w, eval_level='0ply', cubeful=True)
    cube_craw = analyzer.cube_action(starting, cube_value=1, cube_owner='centered',
                                      away1=5, away2=1, is_crawford=True)
    check("Crawford via BgBotAnalyzer: No Double",
          cube_craw.optimal_action == "No Double",
          f"action={cube_craw.optimal_action}")

    # =========================================================================
    print("\n=== Test 3: Post-Crawford 2-away/1-away ===")
    # =========================================================================
    # Trailer (2-away) should always double immediately.
    # Leader (1-away) has free drop at 2-away.

    # From trailer's perspective (needs 2 points, opponent needs 1):
    r_pc = bgbot_cpp.evaluate_cube_decision(
        starting, 1, CubeOwner.CENTERED, *w.weight_args,
        away1=2, away2=1, is_crawford=False,
    )
    check("Post-Crawford 2a1a: trailer should double",
          r_pc["should_double"],
          f"should_double={r_pc['should_double']}")
    print(f"    ND={r_pc['equity_nd']:.4f}, DT={r_pc['equity_dt']:.4f}, DP={r_pc['equity_dp']:.4f}")

    # =========================================================================
    print("\n=== Test 4: Normal Match Scores (0-ply) ===")
    # =========================================================================

    test_scores = [
        (5, 5, False, "5a5a"),
        (3, 3, False, "3a3a"),
        (7, 3, False, "7a3a"),
        (3, 7, False, "3a7a"),
        (5, 3, False, "5a3a"),
    ]

    for a1, a2, craw, label in test_scores:
        r = bgbot_cpp.evaluate_cube_decision(
            starting, 1, CubeOwner.CENTERED, *w.weight_args,
            away1=a1, away2=a2, is_crawford=craw,
        )
        action = "D/T" if r["should_double"] and r["should_take"] else \
                 "D/P" if r["should_double"] else "ND"
        print(f"    {label}: ND={r['equity_nd']:+.4f} DT={r['equity_dt']:+.4f} "
              f"DP={r['equity_dp']:+.4f} -> {action}")

        # Basic sanity: equities should be in plausible range
        check(f"{label}: equity_nd in [-2, 2]",
              -2.0 < r['equity_nd'] < 2.0,
              f"equity_nd={r['equity_nd']:.4f}")
        check(f"{label}: equity_dp in [-2, 2]",
              -2.0 < r['equity_dp'] < 2.0,
              f"equity_dp={r['equity_dp']:.4f}")

    # =========================================================================
    print("\n=== Test 5: Money Game Regression ===")
    # =========================================================================
    # Reference positions from test_cube_decision.py
    # Position 1: double/take borderline (centered cube=1)
    ref1_board = [0, 0, -2, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]

    r_money = bgbot_cpp.evaluate_cube_decision(
        ref1_board, 1, CubeOwner.CENTERED, *w.weight_args,
    )
    r_money_explicit = bgbot_cpp.evaluate_cube_decision(
        ref1_board, 1, CubeOwner.CENTERED, *w.weight_args,
        away1=0, away2=0, is_crawford=False,
    )

    # No match params should give same result as explicit away1=0, away2=0
    check("Money regression: default == explicit away1=0, away2=0",
          abs(r_money["equity_nd"] - r_money_explicit["equity_nd"]) < 0.0001,
          f"default ND={r_money['equity_nd']:.4f}, explicit ND={r_money_explicit['equity_nd']:.4f}")

    # DP should be 1.0 for money game
    check("Money regression: equity_dp == 1.0",
          abs(r_money["equity_dp"] - 1.0) < 0.001,
          f"equity_dp={r_money['equity_dp']:.4f}")

    print(f"    Money: ND={r_money['equity_nd']:+.4f} DT={r_money['equity_dt']:+.4f} "
          f"DP={r_money['equity_dp']:+.4f}")

    # =========================================================================
    print("\n=== Test 6: BgBotAnalyzer Match Play ===")
    # =========================================================================

    analyzer = BgBotAnalyzer(weights=w, eval_level='0ply', cubeful=True)

    # Checker play with match state
    cp_money = analyzer.checker_play(starting, 3, 1, cube_value=1, cube_owner='centered')
    cp_match = analyzer.checker_play(starting, 3, 1, cube_value=1, cube_owner='centered',
                                      away1=5, away2=3)
    check("Checker play: match vs money gives different equities",
          abs(cp_money.moves[0].equity - cp_match.moves[0].equity) > 0.001,
          f"money={cp_money.moves[0].equity:.4f}, match={cp_match.moves[0].equity:.4f}")
    print(f"    Best move equity: money={cp_money.moves[0].equity:+.4f}, match 5a3a={cp_match.moves[0].equity:+.4f}")

    # Post-move analytics with match state
    pma_money = analyzer.post_move_analytics(starting, cube_owner='centered')
    pma_match = analyzer.post_move_analytics(starting, cube_owner='centered',
                                              away1=5, away2=3)
    check("Post-move: match vs money gives different cubeful equity",
          abs(pma_money.cubeful_equity - pma_match.cubeful_equity) > 0.001,
          f"money={pma_money.cubeful_equity:.4f}, match={pma_match.cubeful_equity:.4f}")
    print(f"    Post-move cubeful: money={pma_money.cubeful_equity:+.4f}, match 5a3a={pma_match.cubeful_equity:+.4f}")

    # =========================================================================
    print("\n=== Test 7: Batch Evaluation with Match Play ===")
    # =========================================================================

    positions = [
        {"board": starting, "cube_value": 1, "cube_owner": "centered"},
        {"board": starting, "cube_value": 1, "cube_owner": "centered",
         "away1": 5, "away2": 3, "is_crawford": False},
        {"board": starting, "cube_value": 1, "cube_owner": "centered",
         "away1": 1, "away2": 1},
    ]

    results = batch_evaluate(positions, eval_level="0ply", weights=w)
    check("Batch: 3 results returned",
          len(results) == 3,
          f"got {len(results)}")

    # Money game DP should be 1.0
    check("Batch[0] money: DP=1.0",
          abs(results[0].equity_dp - 1.0) < 0.001,
          f"got {results[0].equity_dp:.4f}")

    # Match play DP is also 1.0 in normalized equity space (mwc2eq of winning
    # cube_value points always gives 1.0, by definition of the linear normalization).
    # Instead check that match play ND differs from money ND.
    check("Batch[1] match 5a3a: ND differs from money",
          abs(results[1].equity_nd - results[0].equity_nd) > 0.001,
          f"match ND={results[1].equity_nd:.4f}, money ND={results[0].equity_nd:.4f}")

    # DMP: should not double
    check("Batch[2] DMP: should not double",
          not results[2].should_double,
          f"should_double={results[2].should_double}")

    print(f"    Batch money: ND={results[0].equity_nd:+.4f} DP={results[0].equity_dp:+.4f}")
    print(f"    Batch 5a3a:  ND={results[1].equity_nd:+.4f} DP={results[1].equity_dp:+.4f}")
    print(f"    Batch DMP:   ND={results[2].equity_nd:+.4f} DP={results[2].equity_dp:+.4f}")

    # batch_cube_action
    cube_results = batch_cube_action(positions, eval_level="0ply", weights=w)
    check("batch_cube_action: 3 results",
          len(cube_results) == 3)
    check("batch_cube_action DMP: No Double",
          cube_results[2].optimal_action == "No Double",
          f"got {cube_results[2].optimal_action}")

    # =========================================================================
    print("\n=== Test 8: 1-ply Match Cube Decision ===")
    # =========================================================================

    t0 = time.time()
    r_1ply = bgbot_cpp.cube_decision_nply(
        starting, 1, CubeOwner.CENTERED, 1, *w.weight_args,
        away1=5, away2=5, is_crawford=False,
    )
    t1 = time.time()
    action_1ply = "D/T" if r_1ply["should_double"] and r_1ply["should_take"] else \
                  "D/P" if r_1ply["should_double"] else "ND"
    print(f"    1-ply 5a5a: ND={r_1ply['equity_nd']:+.4f} DT={r_1ply['equity_dt']:+.4f} "
          f"DP={r_1ply['equity_dp']:+.4f} -> {action_1ply} ({t1-t0:.2f}s)")

    check("1-ply match: equity_nd in [-2, 2]",
          -2.0 < r_1ply['equity_nd'] < 2.0,
          f"equity_nd={r_1ply['equity_nd']:.4f}")

    # 1-ply DMP
    r_1ply_dmp = bgbot_cpp.cube_decision_nply(
        starting, 1, CubeOwner.CENTERED, 1, *w.weight_args,
        away1=1, away2=1, is_crawford=False,
    )
    check("1-ply DMP: should not double",
          not r_1ply_dmp["should_double"],
          f"should_double={r_1ply_dmp['should_double']}")

    # 1-ply Crawford
    r_1ply_craw = bgbot_cpp.cube_decision_nply(
        starting, 1, CubeOwner.CENTERED, 1, *w.weight_args,
        away1=5, away2=1, is_crawford=True,
    )
    check("1-ply Crawford: should not double",
          not r_1ply_craw["should_double"],
          f"should_double={r_1ply_craw['should_double']}")

    # =========================================================================
    print("\n=== Test 9: cl2cf Match vs Money ===")
    # =========================================================================

    # Direct cl2cf test: match should differ from money
    probs_test = [0.55, 0.12, 0.01, 0.08, 0.005]
    x_test = 0.68
    eq_money = bgbot_cpp.cl2cf_money(probs_test, CubeOwner.CENTERED, x_test)
    eq_match = bgbot_cpp.cl2cf(probs_test, 1, CubeOwner.CENTERED, x_test,
                                5, 5, False)
    check("cl2cf: match 5a5a != money for same probs",
          abs(eq_money - eq_match) > 0.001,
          f"money={eq_money:.4f}, match={eq_match:.4f}")
    print(f"    cl2cf money={eq_money:+.4f}, match 5a5a={eq_match:+.4f}")

    # cl2cf with away1=0, away2=0 should equal cl2cf_money
    eq_money2 = bgbot_cpp.cl2cf(probs_test, 1, CubeOwner.CENTERED, x_test,
                                 0, 0, False)
    check("cl2cf: away1=0, away2=0 == cl2cf_money",
          abs(eq_money - eq_money2) < 0.0001,
          f"cl2cf_money={eq_money:.6f}, cl2cf(0,0)={eq_money2:.6f}")

    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'='*60}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
