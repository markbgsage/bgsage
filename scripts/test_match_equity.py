"""Test Match Equity Table (MET) utilities.

Validates the hardcoded Kazaross-XG2 MET data and utility functions:
  - MET symmetry: get_met(a,b) + get_met(b,a) == 1.0
  - Known MET values
  - eq2mwc/mwc2eq roundtrip consistency
  - can_double_match: Crawford/post-Crawford/dead cube rules

Usage:
    python bgsage/scripts/test_match_equity.py [--build-dir build]
"""

import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=str, default='build')
    args = parser.parse_args()

    # Setup paths — find project root (parent of bgsage/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bgsage_dir = os.path.dirname(script_dir)
    project_dir = os.path.dirname(bgsage_dir)

    build_dir = os.path.join(project_dir, args.build_dir)
    sys.path.insert(0, os.path.join(bgsage_dir, 'python'))
    sys.path.insert(0, build_dir)

    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)

    import bgbot_cpp

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

    # =========================================================================
    print("\n=== MET Symmetry ===")
    # =========================================================================
    sym_errors = []
    for a in range(1, 26):
        for b in range(1, 26):
            s = bgbot_cpp.get_met(a, b) + bgbot_cpp.get_met(b, a)
            if abs(s - 1.0) > 0.001:
                sym_errors.append((a, b, s))

    check("Symmetry: get_met(a,b) + get_met(b,a) == 1.0 for all 1<=a,b<=25",
          len(sym_errors) == 0,
          f"{len(sym_errors)} failures: {sym_errors[:5]}")

    # =========================================================================
    print("\n=== Known MET Values ===")
    # =========================================================================

    # DMP: 1-away, 1-away = 0.5 exactly
    check("get_met(1,1) == 0.5",
          abs(bgbot_cpp.get_met(1, 1) - 0.5) < 0.0001,
          f"got {bgbot_cpp.get_met(1, 1):.6f}")

    # Boundary: player already won
    check("get_met(0, 5) == 1.0 (player won)",
          abs(bgbot_cpp.get_met(0, 5) - 1.0) < 0.0001,
          f"got {bgbot_cpp.get_met(0, 5):.6f}")

    # Boundary: opponent already won
    check("get_met(5, 0) == 0.0 (opponent won)",
          abs(bgbot_cpp.get_met(5, 0) - 0.0) < 0.0001,
          f"got {bgbot_cpp.get_met(5, 0):.6f}")

    # Known Kazaross-XG2 values (from the XML source)
    # MET_PRE[0][0] = 1-away, 1-away = 0.5
    check("get_met(1, 1, False) == 0.5",
          abs(bgbot_cpp.get_met(1, 1, False) - 0.5) < 0.001,
          f"got {bgbot_cpp.get_met(1, 1, False):.6f}")

    # get_met(1, 2): player is 1-away. Routes to post-Crawford table.
    # = 1.0 - MET_POST_CRAWFORD[1] = 1.0 - 0.48803 = 0.51197
    met_1_2 = bgbot_cpp.get_met(1, 2)
    check("get_met(1, 2) == 0.51197 (1-away leader, post-Crawford routing)",
          abs(met_1_2 - 0.51197) < 0.001,
          f"got {met_1_2:.6f}")

    # get_met(2, 1): player is trailer (2-away), opponent is 1-away leader.
    # = MET_POST_CRAWFORD[1] = 0.48803
    met_2_1 = bgbot_cpp.get_met(2, 1)
    check("get_met(2, 1) == 0.48803 (2-away trailer, post-Crawford)",
          abs(met_2_1 - 0.48803) < 0.001,
          f"got {met_2_1:.6f}")

    # Verify symmetry of 1-away values
    check("get_met(1,2) + get_met(2,1) == 1.0",
          abs(met_1_2 + met_2_1 - 1.0) < 0.001,
          f"sum = {met_1_2 + met_2_1:.6f}")

    # 5-away vs 5-away should be exactly 0.5 (symmetric)
    check("get_met(5, 5) == 0.5",
          abs(bgbot_cpp.get_met(5, 5) - 0.5) < 0.001,
          f"got {bgbot_cpp.get_met(5, 5):.6f}")

    # 5-away vs 3-away: pre-Crawford, player (5-away) is trailing
    met_5_3 = bgbot_cpp.get_met(5, 3)
    check("get_met(5, 3) == MET_PRE[4][2] == 0.35205",
          abs(met_5_3 - 0.35205) < 0.001,
          f"got {met_5_3:.6f}")

    # =========================================================================
    print("\n=== eq2mwc / mwc2eq Roundtrip ===")
    # =========================================================================

    test_cases = [
        # (equity, away1, away2, cube_value, is_crawford)
        (0.0, 5, 3, 1, False),
        (0.5, 7, 7, 1, False),
        (-0.3, 3, 5, 2, False),
        (0.8, 10, 2, 1, False),
        (0.0, 5, 1, 1, True),   # Crawford game
    ]
    for eq, a1, a2, cv, craw in test_cases:
        mwc = bgbot_cpp.eq2mwc(eq, a1, a2, cv, craw)
        eq_back = bgbot_cpp.mwc2eq(mwc, a1, a2, cv, craw)
        check(f"eq2mwc/mwc2eq roundtrip eq={eq:.1f} ({a1}a{a2}a cv={cv} craw={craw})",
              abs(eq_back - eq) < 0.001,
              f"eq={eq:.4f} -> mwc={mwc:.6f} -> eq_back={eq_back:.4f}")

    # mwc range checks
    for eq, a1, a2, cv, craw in test_cases:
        mwc = bgbot_cpp.eq2mwc(eq, a1, a2, cv, craw)
        check(f"eq2mwc in [0,1] for eq={eq:.1f} ({a1}a{a2}a)",
              0.0 <= mwc <= 1.0,
              f"mwc={mwc:.6f}")

    # =========================================================================
    print("\n=== can_double_match ===")
    # =========================================================================

    CubeOwner = bgbot_cpp.CubeOwner

    # Crawford game: nobody can double
    check("Crawford: can't double (centered)",
          not bgbot_cpp.can_double_match(5, 1, 1, CubeOwner.CENTERED, True))
    check("Crawford: can't double (player owns)",
          not bgbot_cpp.can_double_match(5, 1, 1, CubeOwner.PLAYER, True))

    # Post-Crawford: leader (1-away) can't double when cube is dead
    # At post-Crawford 2a1a, trailer owns the cube action
    # Leader (1-away) shouldn't double because winning more points is irrelevant
    # (Already at 1-away, any win completes the match)
    # The can_double_match function checks: if player is 1-away (post-Crawford leader),
    # cannot double.
    # Player is at 1-away: away1=1, post-Crawford (not crawford, someone is 1-away)
    check("Post-Crawford: 1-away leader can't double",
          not bgbot_cpp.can_double_match(1, 2, 1, CubeOwner.CENTERED, False))

    # Post-Crawford: trailer (2-away) CAN double
    check("Post-Crawford: 2-away trailer can double (centered)",
          bgbot_cpp.can_double_match(2, 1, 1, CubeOwner.CENTERED, False))

    # Normal game: centered cube, both can double
    check("Normal: centered cube, can double",
          bgbot_cpp.can_double_match(5, 3, 1, CubeOwner.CENTERED, False))

    # Normal game: player owns cube, can double
    check("Normal: player owns, can double",
          bgbot_cpp.can_double_match(5, 3, 2, CubeOwner.PLAYER, False))

    # Normal game: opponent owns cube, can't double
    check("Normal: opponent owns, can't double",
          not bgbot_cpp.can_double_match(5, 3, 2, CubeOwner.OPPONENT, False))

    # Dead cube: both 2-away, cube=2, doubling is dead (would win/lose match either way)
    check("Dead cube: 2a2a cv=2, can't double",
          not bgbot_cpp.can_double_match(2, 2, 2, CubeOwner.CENTERED, False))

    # =========================================================================
    print("\n=== cubeless_mwc ===")
    # =========================================================================

    # At DMP (1a1a), gammons don't matter. MWC should equal P(win).
    probs_60 = [0.60, 0.10, 0.01, 0.05, 0.005]  # 60% win
    mwc_dmp = bgbot_cpp.cubeless_mwc(probs_60, 1, 1, 1, False)
    check("DMP: cubeless_mwc == P(win) (gammons irrelevant)",
          abs(mwc_dmp - 0.60) < 0.01,
          f"got {mwc_dmp:.6f}, expected ~0.60")

    # At 5-away, 5-away, cubeless MWC should be close to 50% for equal probs
    probs_50 = [0.50, 0.10, 0.01, 0.10, 0.01]
    mwc_5a5a = bgbot_cpp.cubeless_mwc(probs_50, 5, 5, 1, False)
    check("5a5a: cubeless_mwc ~0.50 for symmetric probs",
          0.45 < mwc_5a5a < 0.55,
          f"got {mwc_5a5a:.6f}")

    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'='*60}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
