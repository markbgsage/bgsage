"""Tests for cube action points (Take / Double / Cash / Too Good).

Covers the user-supplied match-play reference values:

  - 4-away vs 2-away, centered cv=1, gammonless, live TP = 18.56%
  - 5-away vs 3-away, opp owns cv=2, gammonless, live TP = 15.82%
  - 5-away vs 3-away, opp owns cv=2, gammonless, dead TP = 21.07%
  - 5-away vs 3-away, player owns cv=2, gammonless, dead DP = 38.68%

Plus basic money-game sanity checks.

Run with:
    python -m pytest bgsage/tests/test_cube_points.py -v
"""

import os
import sys
import unittest

# Setup paths (same pattern as other bgsage tests)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, "build")

if sys.platform == "win32":
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, "bgsage", "python"))

from bgsage.cube_points import cube_action_points  # noqa: E402


# Gammonless: only the P(win) matters; all gammon/BG components are zero.
GAMMONLESS = (0.5, 0.0, 0.0, 0.0, 0.0)


class TestMatchPlayReferenceValues(unittest.TestCase):
    """User-supplied reference values verified against the recursive algorithm."""

    def test_test1_centered_cv1_live_tp(self):
        """4-away vs 2-away, centered cv=1, gammonless, live TP = 18.56%."""
        r = cube_action_points(GAMMONLESS, "centered", 1, 4, 2, False, 1.0)
        self.assertAlmostEqual(r.take.live, 0.18564, places=4)

    def test_test2a_opp_owns_cv2_live_tp(self):
        """5-away vs 3-away, opp owns cv=2, gammonless, live TP = 15.82%."""
        r = cube_action_points(GAMMONLESS, "opponent", 2, 5, 3, False, 1.0)
        self.assertAlmostEqual(r.take.live, 0.15821, places=4)

    def test_test2b_opp_owns_cv2_dead_tp(self):
        """5-away vs 3-away, opp owns cv=2, gammonless, dead TP = 21.07%."""
        r = cube_action_points(GAMMONLESS, "opponent", 2, 5, 3, False, 0.0)
        self.assertAlmostEqual(r.take.dead, 0.21073, places=4)

    def test_test3_player_owns_cv2_dead_dp(self):
        """5-away vs 3-away, player owns cv=2, gammonless, dead DP = 38.68%."""
        r = cube_action_points(GAMMONLESS, "player", 2, 5, 3, False, 0.0)
        self.assertAlmostEqual(r.double.dead, 0.38685, places=4)


class TestMatchPlayOwnership(unittest.TestCase):
    def test_player_owns_cannot_take(self):
        r = cube_action_points(GAMMONLESS, "player", 2, 5, 3, False, 0.68)
        self.assertTrue(r.cannot_take)
        self.assertFalse(r.cannot_double)
        self.assertEqual(r.double_label, "Redouble")

    def test_opp_owns_cannot_double(self):
        r = cube_action_points(GAMMONLESS, "opponent", 2, 5, 3, False, 0.68)
        self.assertFalse(r.cannot_take)
        self.assertTrue(r.cannot_double)
        self.assertTrue(r.cannot_cash)
        self.assertTrue(r.cannot_too_good)
        self.assertIsNone(r.double_label)

    def test_centered_all_rows(self):
        r = cube_action_points(GAMMONLESS, "centered", 1, 4, 2, False, 0.68)
        self.assertFalse(r.cannot_take)
        self.assertFalse(r.cannot_double)
        self.assertEqual(r.double_label, "Double")

    def test_crawford_all_na(self):
        r = cube_action_points(GAMMONLESS, "centered", 1, 3, 1, True, 0.68)
        self.assertTrue(r.cannot_take)
        self.assertTrue(r.cannot_double)


class TestMatchPlayCollapses(unittest.TestCase):
    """In the recursive linear-equity model the LIVE thresholds collapse:
    live CP = live TG = live DP = player_cp. The DEAD thresholds are distinct.
    """

    def test_live_collapse_centered(self):
        r = cube_action_points(GAMMONLESS, "centered", 1, 4, 2, False, 1.0)
        # CP == TG == DP at x=1.
        self.assertAlmostEqual(r.cash.live, r.too_good.live, places=5)
        self.assertAlmostEqual(r.cash.live, r.double.live, places=5)

    def test_live_collapse_player_owns(self):
        r = cube_action_points(GAMMONLESS, "player", 2, 5, 3, False, 1.0)
        self.assertAlmostEqual(r.cash.live, r.too_good.live, places=5)
        self.assertAlmostEqual(r.cash.live, r.double.live, places=5)

    def test_dead_distinct_centered(self):
        r = cube_action_points(GAMMONLESS, "centered", 1, 4, 2, False, 0.0)
        # Dead: TG strictly above CP, both above DP.
        self.assertGreater(r.too_good.dead, r.cash.dead)
        self.assertGreater(r.cash.dead, r.double.dead)


class TestMoneyGame(unittest.TestCase):
    """Money-game path (away1=0) should follow the closed-form Janowski
    formulas. Verified against documented gammonless / money values."""

    def test_money_gammonless_centered_x068(self):
        """Money game, centered, no Jacoby, gammonless, x=0.68.

        Closed-form Janowski values (W=L=1):
            TP = 0.5 / 2.34 = 0.21368
            CP = 1.84 / 2.34 = 0.78632
            TG = 2.0 / 2.34 = 0.85470
            ID (no Jacoby) with G(0.68) = 0.68*2.32/(2*1.32) = 0.59758,
                           = (1 + 0.59758) / 2.34 = 0.68273
        """
        r = cube_action_points(
            GAMMONLESS, "centered", 1,
            away1=0, away2=0, is_crawford=False,
            cube_life_index=0.68,
            jacoby=False,  # Turn off Jacoby for the pure ID formula.
        )
        self.assertAlmostEqual(r.take.janowski, 0.21368, places=4)
        self.assertAlmostEqual(r.cash.janowski, 0.78632, places=4)
        self.assertAlmostEqual(r.too_good.janowski, 0.85470, places=4)
        self.assertAlmostEqual(r.double.janowski, 0.68273, places=4)

    def test_money_gammonless_dead(self):
        """At x=0: TP = 0.25, CP = 0.75, TG = 1.0, ID = 0.5."""
        r = cube_action_points(
            GAMMONLESS, "centered", 1,
            away1=0, away2=0,
            cube_life_index=0.0, jacoby=False,
        )
        self.assertAlmostEqual(r.take.janowski, 0.25, places=5)
        self.assertAlmostEqual(r.cash.janowski, 0.75, places=5)
        self.assertAlmostEqual(r.too_good.janowski, 1.0, places=5)
        self.assertAlmostEqual(r.double.janowski, 0.5, places=5)

    def test_money_live_collapses(self):
        """At x=1: ID = CP = TG."""
        r = cube_action_points(
            GAMMONLESS, "centered", 1,
            away1=0, away2=0,
            cube_life_index=1.0, jacoby=False,
        )
        self.assertAlmostEqual(r.double.janowski, r.cash.janowski, places=5)
        self.assertAlmostEqual(r.cash.janowski, r.too_good.janowski, places=5)


class TestPostCrawford(unittest.TestCase):
    """Spot-check post-Crawford scores route through the MET correctly."""

    def test_post_crawford_does_not_crash(self):
        """Post-Crawford (player 1-away) with centered cube — shouldn't raise.

        At post-Crawford with leader at 1-away: leader can't meaningfully
        double. We don't set a specific expected value here — just that the
        call completes without crashing.
        """
        r = cube_action_points(GAMMONLESS, "centered", 1, 1, 4, True, 0.68)
        self.assertTrue(r.cannot_double)


class TestNoGammonsToggle(unittest.TestCase):
    """Regression: ``no_gammons=True`` used to forget to zero out the
    cumulative gammon field (probs[1] was being set to P(win) instead of 0),
    which displayed W=L=2 for match play because every win counted as a
    gammon."""

    def test_no_gammons_wl_is_1_match_play(self):
        """Real probs with gammons → no_gammons=True should give W=L=1."""
        probs_with_gammons = (0.55, 0.20, 0.03, 0.12, 0.02)
        r = cube_action_points(
            probs_with_gammons, "centered", 1,
            away1=5, away2=3, is_crawford=False,
            cube_life_index=0.68, no_gammons=True,
        )
        self.assertAlmostEqual(r.W, 1.0, places=5)
        self.assertAlmostEqual(r.L, 1.0, places=5)

    def test_no_gammons_wl_is_1_money(self):
        """Same check for money game."""
        probs_with_gammons = (0.55, 0.20, 0.03, 0.12, 0.02)
        r = cube_action_points(
            probs_with_gammons, "centered", 1,
            cube_life_index=0.68, no_gammons=True,
        )
        self.assertAlmostEqual(r.W, 1.0, places=5)
        self.assertAlmostEqual(r.L, 1.0, places=5)

    def test_no_gammons_reproduces_user_test_values(self):
        """With real-probs input + no_gammons=True, match-play points should
        match the gammonless reference values we use elsewhere."""
        probs_with_gammons = (0.5, 0.10, 0.01, 0.10, 0.01)
        r = cube_action_points(
            probs_with_gammons, "centered", 1,
            away1=4, away2=2, is_crawford=False,
            cube_life_index=1.0, no_gammons=True,
        )
        self.assertAlmostEqual(r.take.live, 0.18564, places=4)


class TestJanowskiExactAtIntermediateX(unittest.TestCase):
    """The Janowski ("Estimate") column at intermediate x comes from binary
    searching ``cube_decision_1ply`` at that x — i.e. the exact Janowski
    interpolation of the piecewise-linear ``cl2cf_match`` equity curves.

    These are the match-play analogues of the money-game closed-form
    formulas like ``TP(x) = (L-0.5)/(W+L+0.5x)``. We verify that:

    1. The Janowski value at x=0 equals the dead-cube column.
    2. The Janowski value at x=1 equals the live-cube column.
    3. At intermediate x (0.68) the values match an independent empirical
       probe of ``cube_decision_1ply`` — i.e. they are _not_ a linear
       interpolation of dead and live.
    """

    def test_janowski_x0_matches_dead(self):
        r = cube_action_points(GAMMONLESS, "centered", 1, 4, 2, False, 0.0)
        self.assertAlmostEqual(r.take.janowski, r.take.dead, places=4)
        self.assertAlmostEqual(r.cash.janowski, r.cash.dead, places=4)

    def test_janowski_x1_matches_live(self):
        r = cube_action_points(GAMMONLESS, "centered", 1, 4, 2, False, 1.0)
        self.assertAlmostEqual(r.take.janowski, r.take.live, places=4)
        self.assertAlmostEqual(r.cash.janowski, r.cash.live, places=4)

    def test_janowski_x068_exact_dp_test3(self):
        """5-away vs 3-away, player owns cv=2, gammonless, x=0.68.

        Independent derivation: at x=0.68 the double point solves
        ``equity_nd(p, 0.68) = equity_dt(p, 0.68)`` in the
        cl2cf_match_owned piecewise linear (cv=2 player-owns, cv=4 opp-owns
        dead), which gives p ≈ 54.106%.

        Linear interpolation of the dead endpoint (38.68%) and live endpoint
        (66.60%) gives 57.67% — we explicitly check we're _not_ doing that.
        """
        r = cube_action_points(GAMMONLESS, "player", 2, 5, 3, False, 0.68)
        self.assertAlmostEqual(r.double.janowski, 0.54106, places=3)
        # Sanity: significantly different from a linear interp of endpoints.
        linear_interp = 0.32 * r.double.dead + 0.68 * r.double.live
        self.assertGreater(abs(r.double.janowski - linear_interp), 0.02)

    def test_janowski_x068_exact_tp_test1(self):
        """Test 1 centered cv=1 at x=0.68: TP ≈ 22.10%."""
        r = cube_action_points(GAMMONLESS, "centered", 1, 4, 2, False, 0.68)
        self.assertAlmostEqual(r.take.janowski, 0.22100, places=3)

    def test_janowski_x068_exact_tp_test2(self):
        """Test 2 opp-owns cv=2 at x=0.68: TP ≈ 17.19%."""
        r = cube_action_points(GAMMONLESS, "opponent", 2, 5, 3, False, 0.68)
        self.assertAlmostEqual(r.take.janowski, 0.17192, places=3)


if __name__ == "__main__":
    unittest.main()
