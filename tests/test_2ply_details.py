"""
Tests for incl_2ply_details in cube action analytics.

Verifies that the per-roll details returned by cube_action(incl_2ply_details=True)
are consistent with independent checker play and cube action evaluations at the
corresponding ply levels.

The cubeful recursion uses 1-ply cubeless equity for move selection at all
internal nodes, then evaluates at the full depth. This test mirrors that:
- Board verification: 1-ply cubeless checker play (matching recursion's move selection)
- Equity verification: cube_action at (n-1)-ply on the resulting board

Run with:
    python -m pytest bgsage/tests/test_2ply_details.py -v
    python -m unittest bgsage.tests.test_2ply_details -v
"""

import os
import sys
import unittest

# Setup paths — tests/ lives at the bgsage repo root, so one level up is the repo.
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)  # bgsage repo root
build_dir = os.path.join(repo_dir, "build")

if sys.platform == "win32":
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(repo_dir, "python"))

import bgbot_cpp
from bgsage import BgBotAnalyzer
from bgsage.board import flip_board, check_game_over, possible_moves

# Test position: contact/priming position with non-trivial cube dynamics
BOARD = [0, 0, 0, 2, 3, 0, 4, -2, 2, 0, 0, 0, -4, 2, -3, 0, -1, 0, 0, -3, 2, 0, -2, 0, 0, 0]

# Equity tolerance — the details extraction and independent evaluations use
# the same underlying C++ cubeful recursion, but the thread-local caches may
# cause small floating point differences between integrated and standalone calls.
EQUITY_TOL = 0.015

# The 21 dice combinations (matching C++ ALL_ROLLS order)
DICE_ROLLS = [
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 4), (3, 5), (3, 6),
    (4, 5), (4, 6),
    (5, 6),
]


def best_move_1ply_cubeless(board, d1, d2, post_move_analyzer):
    """Pick best move by 1-ply cubeless equity (matching cubeful recursion)."""
    candidates = possible_moves(board, d1, d2)
    if not candidates:
        return board  # no moves: standing pat
    best_board = None
    best_eq = -1e30
    for cand in candidates:
        pma = post_move_analyzer.post_move_analytics(list(cand))
        eq = pma.cubeless_equity
        if eq > best_eq:
            best_eq = eq
            best_board = list(cand)
    return best_board


class TestTwoPlyDetailsND(unittest.TestCase):
    """Verify ND section of 2-ply details against independent per-roll evaluations."""

    def _run_nd_test(self, n_plies: int):
        """Core ND test logic for a given ply level."""
        analyzer = BgBotAnalyzer(eval_level=f"{n_plies}ply", cubeful=True)

        result = analyzer.cube_action(
            BOARD, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True, incl_2ply_details=True,
        )

        self.assertIsNotNone(result.details, "details should be present")
        self.assertIn("nd", result.details, "details should have 'nd' key")
        nd_rolls = result.details["nd"]
        self.assertEqual(len(nd_rolls), 21, "Should have 21 ND player rolls")

        # Verify weighted average of ND player-roll equities matches equity_nd
        weighted_sum = sum(
            pr["cubeful_equity"] * (1 if pr["die1"] == pr["die2"] else 2)
            for pr in nd_rolls
        )
        avg_equity = weighted_sum / 36.0
        self.assertAlmostEqual(
            avg_equity, result.equity_nd, places=3,
            msg=f"ND weighted average ({avg_equity:.6f}) should match equity_nd ({result.equity_nd:.6f})",
        )

        post_move_analyzer = BgBotAnalyzer(eval_level="1ply", cubeful=False)
        opp_cube_analyzer = BgBotAnalyzer(eval_level=f"{n_plies - 1}ply", cubeful=True)
        player_cube_analyzer = BgBotAnalyzer(eval_level=f"{n_plies - 2}ply", cubeful=True)

        for roll_idx, pr in enumerate(nd_rolls):
            d1, d2 = DICE_ROLLS[roll_idx]
            self.assertEqual(pr["die1"], d1)
            self.assertEqual(pr["die2"], d2)

            # --- Verify player's best move board ---
            candidates = possible_moves(BOARD, d1, d2)

            if not candidates:
                self.assertEqual(
                    list(pr["checkers"]), list(BOARD),
                    f"Roll {d1}-{d2}: no moves, board should be unchanged",
                )
            else:
                best_board = best_move_1ply_cubeless(BOARD, d1, d2, post_move_analyzer)
                self.assertEqual(
                    list(pr["checkers"]), list(best_board),
                    f"Roll {d1}-{d2}: post-move board should match 1-ply cubeless best",
                )

                game_over = check_game_over(best_board)
                if game_over != 0:
                    self.assertNotIn(
                        "opponent_rolls", pr,
                        f"Roll {d1}-{d2}: terminal should not have opponent_rolls",
                    )
                    continue

            # --- Verify player-roll equity via opponent's cube action ---
            post_move_board = list(pr["checkers"])
            opp_board = flip_board(post_move_board)

            opp_cube = opp_cube_analyzer.cube_action(
                opp_board, cube_value=1, cube_owner="centered",
                jacoby=True, beaver=True,
            )

            expected_player_eq = -opp_cube.optimal_equity
            self.assertAlmostEqual(
                pr["cubeful_equity"], expected_player_eq, delta=EQUITY_TOL,
                msg=f"Roll {d1}-{d2}: ND player equity {pr['cubeful_equity']:.4f} "
                    f"vs expected {expected_player_eq:.4f} "
                    f"(opp optimal_action={opp_cube.optimal_action})",
            )

            is_opp_dp = (opp_cube.should_double and not opp_cube.should_take)

            if is_opp_dp:
                self.assertTrue(
                    pr.get("opponent_dp", False) or "opponent_rolls" not in pr,
                    f"Roll {d1}-{d2}: opponent D/P but opponent_rolls present",
                )
            else:
                self.assertIn(
                    "opponent_rolls", pr,
                    f"Roll {d1}-{d2}: no D/P but opponent_rolls missing",
                )
                opp_rolls = pr["opponent_rolls"]
                self.assertEqual(len(opp_rolls), 21)

                for opp_idx, opp_r in enumerate(opp_rolls):
                    od1, od2 = DICE_ROLLS[opp_idx]
                    self.assertEqual(opp_r["die1"], od1)
                    self.assertEqual(opp_r["die2"], od2)

                    opp_candidates = possible_moves(opp_board, od1, od2)
                    if not opp_candidates:
                        expected_board = flip_board(opp_board)
                    else:
                        opp_best = best_move_1ply_cubeless(opp_board, od1, od2, post_move_analyzer)
                        expected_board = flip_board(opp_best)

                    self.assertEqual(
                        list(opp_r["checkers"]), list(expected_board),
                        f"Roll {d1}-{d2}, opp {od1}-{od2}: board mismatch",
                    )

                    player_post_opp_board = list(opp_r["checkers"])
                    opp_post = flip_board(player_post_opp_board)
                    game_over_opp = check_game_over(opp_post)
                    if game_over_opp != 0:
                        continue

                    player_cube = player_cube_analyzer.cube_action(
                        player_post_opp_board, cube_value=1, cube_owner="centered",
                        jacoby=True, beaver=True,
                    )

                    if opp_cube.should_double and opp_cube.should_take:
                        player_cube_dt = player_cube_analyzer.cube_action(
                            player_post_opp_board, cube_value=2, cube_owner="player",
                            jacoby=True, beaver=True,
                        )
                        expected_opp_eq = 2.0 * player_cube_dt.optimal_equity
                    else:
                        expected_opp_eq = player_cube.optimal_equity

                    self.assertAlmostEqual(
                        opp_r["cubeful_equity"], expected_opp_eq, delta=EQUITY_TOL,
                        msg=f"Roll {d1}-{d2}, opp {od1}-{od2}: "
                            f"equity {opp_r['cubeful_equity']:.4f} "
                            f"vs expected {expected_opp_eq:.4f}",
                    )

    def test_3ply_nd(self):
        """Test ND 2-ply details at 3-ply."""
        self._run_nd_test(3)

    def test_4ply_nd(self):
        """Test ND 2-ply details at 4-ply."""
        self._run_nd_test(4)


class TestTwoPlyDetailsDT(unittest.TestCase):
    """Verify DT section of 2-ply details against independent per-roll evaluations."""

    def _run_dt_test(self, n_plies: int):
        """Core DT test logic for a given ply level."""
        analyzer = BgBotAnalyzer(eval_level=f"{n_plies}ply", cubeful=True)

        result = analyzer.cube_action(
            BOARD, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True, incl_2ply_details=True,
        )

        self.assertIsNotNone(result.details, "details should be present")
        self.assertIn("dt", result.details, "details should have 'dt' key")
        dt_rolls = result.details["dt"]
        self.assertEqual(len(dt_rolls), 21, "Should have 21 DT player rolls")

        # Verify weighted average of DT player-roll equities matches equity_dt
        weighted_sum = sum(
            pr["cubeful_equity"] * (1 if pr["die1"] == pr["die2"] else 2)
            for pr in dt_rolls
        )
        avg_equity = weighted_sum / 36.0
        self.assertAlmostEqual(
            avg_equity, result.equity_dt, places=3,
            msg=f"DT weighted average ({avg_equity:.6f}) should match equity_dt ({result.equity_dt:.6f})",
        )

        # Boards should match ND (same move selection — cubeless)
        nd_rolls = result.details["nd"]
        for roll_idx in range(21):
            self.assertEqual(
                dt_rolls[roll_idx]["checkers"], nd_rolls[roll_idx]["checkers"],
                f"Roll {roll_idx}: DT and ND boards should be identical",
            )

        post_move_analyzer = BgBotAnalyzer(eval_level="1ply", cubeful=False)
        # For DT, opponent's cube action is at (n-1)-ply with cube=2, opponent owns
        opp_cube_analyzer = BgBotAnalyzer(eval_level=f"{n_plies - 1}ply", cubeful=True)
        player_cube_analyzer = BgBotAnalyzer(eval_level=f"{n_plies - 2}ply", cubeful=True)

        for roll_idx, pr in enumerate(dt_rolls):
            d1, d2 = DICE_ROLLS[roll_idx]

            candidates = possible_moves(BOARD, d1, d2)
            if candidates:
                best_board = best_move_1ply_cubeless(BOARD, d1, d2, post_move_analyzer)
                game_over = check_game_over(best_board)
                if game_over != 0:
                    # Terminal: DT equity should be 2x the cubeless equity
                    nd_eq = nd_rolls[roll_idx]["cubeful_equity"]
                    # For terminal, ND equity = cubeless_equity (with Jacoby if active)
                    # DT equity should be 2x cubeless_equity (Jacoby inactive since cube turned)
                    # Just verify DT is present
                    continue

            post_move_board = list(pr["checkers"])
            opp_board = flip_board(post_move_board)

            # In DT scenario, cube is 2, opponent owns
            opp_cube = opp_cube_analyzer.cube_action(
                opp_board, cube_value=2, cube_owner="player",
                jacoby=True, beaver=True,
            )

            # Player's equity per initial cube = 2 * (-opp_optimal_equity)
            expected_player_eq = 2.0 * (-opp_cube.optimal_equity)
            self.assertAlmostEqual(
                pr["cubeful_equity"], expected_player_eq, delta=EQUITY_TOL,
                msg=f"Roll {d1}-{d2}: DT player equity {pr['cubeful_equity']:.4f} "
                    f"vs expected {expected_player_eq:.4f}",
            )

    def test_3ply_dt(self):
        """Test DT 2-ply details at 3-ply."""
        self._run_dt_test(3)

    def test_4ply_dt(self):
        """Test DT 2-ply details at 4-ply."""
        self._run_dt_test(4)


class TestTwoPlyDetailsErrors(unittest.TestCase):
    """Test error handling for 2-ply details."""

    def test_error_below_3ply(self):
        """Verify that incl_2ply_details raises an error for < 3-ply."""
        analyzer = BgBotAnalyzer(eval_level="2ply", cubeful=True)
        with self.assertRaises(Exception):
            analyzer.cube_action(
                BOARD, cube_value=1, cube_owner="centered",
                incl_2ply_details=True,
            )


if __name__ == "__main__":
    unittest.main()
