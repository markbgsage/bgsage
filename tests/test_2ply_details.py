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


class TestTwoPlyDetails(unittest.TestCase):
    """Verify 2-ply details against independent per-roll evaluations."""

    def _run_detail_test(self, n_plies: int):
        """Core test logic for a given ply level."""
        analyzer = BgBotAnalyzer(eval_level=f"{n_plies}ply", cubeful=True)

        # Get cube action with 2-ply details
        result = analyzer.cube_action(
            BOARD, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True, incl_2ply_details=True,
        )

        self.assertIsNotNone(result.player_rolls, "player_rolls should be present")
        self.assertEqual(len(result.player_rolls), 21, "Should have 21 player rolls")

        # Verify weighted average of player-roll equities matches equity_nd
        weighted_sum = sum(
            pr["cubeful_equity"] * (1 if pr["die1"] == pr["die2"] else 2)
            for pr in result.player_rolls
        )
        avg_equity = weighted_sum / 36.0
        self.assertAlmostEqual(
            avg_equity, result.equity_nd, places=3,
            msg=f"Weighted average of player roll equities ({avg_equity:.6f}) "
                f"should match equity_nd ({result.equity_nd:.6f})",
        )

        # 1-ply cubeless post-move analyzer (matches cubeful recursion's move selection)
        post_move_analyzer = BgBotAnalyzer(eval_level="1ply", cubeful=False)

        # Cube action analyzers at lower ply levels
        opp_cube_analyzer = BgBotAnalyzer(eval_level=f"{n_plies - 1}ply", cubeful=True)
        player_cube_analyzer = BgBotAnalyzer(eval_level=f"{n_plies - 2}ply", cubeful=True)

        def best_move_1ply_cubeless(board, d1, d2):
            """Pick best move by 1-ply cubeless equity (matching cubeful recursion)."""
            candidates = possible_moves(board, d1, d2)
            if not candidates:
                return None, board  # no moves: standing pat
            best_board = None
            best_eq = -1e30
            for cand in candidates:
                pma = post_move_analyzer.post_move_analytics(list(cand))
                eq = pma.cubeless_equity
                if eq > best_eq:
                    best_eq = eq
                    best_board = list(cand)
            return best_board, best_board

        for roll_idx, pr in enumerate(result.player_rolls):
            d1, d2 = DICE_ROLLS[roll_idx]
            self.assertEqual(pr["die1"], d1)
            self.assertEqual(pr["die2"], d2)

            # --- Verify player's best move board ---
            candidates = possible_moves(BOARD, d1, d2)

            if not candidates:
                # No legal moves (dancing) — board should be unchanged
                self.assertEqual(
                    list(pr["checkers"]), list(BOARD),
                    f"Roll {d1}-{d2}: no moves, board should be unchanged",
                )
            else:
                _, best_move_board = best_move_1ply_cubeless(BOARD, d1, d2)
                self.assertEqual(
                    list(pr["checkers"]), list(best_move_board),
                    f"Roll {d1}-{d2}: post-move board should match 1-ply cubeless best",
                )

                game_over = check_game_over(best_move_board)
                if game_over != 0:
                    # Terminal: no opponent_rolls expected
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
                msg=f"Roll {d1}-{d2}: player equity {pr['cubeful_equity']:.4f} "
                    f"vs expected {expected_player_eq:.4f} "
                    f"(opp optimal_action={opp_cube.optimal_action})",
            )

            is_opp_dp = (opp_cube.should_double and not opp_cube.should_take)

            if is_opp_dp:
                # Opponent has D/P — no opponent_rolls
                self.assertTrue(
                    pr.get("opponent_dp", False) or "opponent_rolls" not in pr,
                    f"Roll {d1}-{d2}: opponent D/P but opponent_rolls present",
                )
            else:
                # Should have opponent_rolls
                self.assertIn(
                    "opponent_rolls", pr,
                    f"Roll {d1}-{d2}: no D/P but opponent_rolls missing",
                )
                opp_rolls = pr["opponent_rolls"]
                self.assertEqual(
                    len(opp_rolls), 21,
                    f"Roll {d1}-{d2}: should have 21 opponent rolls",
                )

                # --- Verify each opponent roll ---
                for opp_idx, opp_r in enumerate(opp_rolls):
                    od1, od2 = DICE_ROLLS[opp_idx]
                    self.assertEqual(opp_r["die1"], od1)
                    self.assertEqual(opp_r["die2"], od2)

                    # Verify opponent's best move board (1-ply cubeless selection)
                    opp_candidates = possible_moves(opp_board, od1, od2)

                    if not opp_candidates:
                        expected_board = flip_board(opp_board)
                    else:
                        _, opp_best_board = best_move_1ply_cubeless(opp_board, od1, od2)
                        expected_board = flip_board(opp_best_board)

                    self.assertEqual(
                        list(opp_r["checkers"]), list(expected_board),
                        f"Roll {d1}-{d2}, opp {od1}-{od2}: board mismatch",
                    )

                    # Verify opponent-roll equity via player's cube action
                    # The board in opp_r is in player's perspective (pre-roll)
                    player_post_opp_board = list(opp_r["checkers"])

                    # Check if terminal
                    # The opponent moved on opp_board; the result board from player's
                    # perspective is player_post_opp_board. To check game-over, we need
                    # the board from the mover's (opponent's) perspective.
                    opp_post = flip_board(player_post_opp_board)
                    game_over_opp = check_game_over(opp_post)
                    if game_over_opp != 0:
                        continue  # Terminal — equity computed from terminal probs

                    player_cube = player_cube_analyzer.cube_action(
                        player_post_opp_board, cube_value=1, cube_owner="centered",
                        jacoby=True, beaver=True,
                    )

                    # The equity should be the player's optimal equity at this position.
                    # The details report per-initial-cube, so if opponent doubled (DT),
                    # the equity is scaled by 2x. If ND, it's direct.
                    if opp_cube.should_double and opp_cube.should_take:
                        # D/T: cube is now at 2, player owns. Run cube_action with cv=2.
                        player_cube_dt = player_cube_analyzer.cube_action(
                            player_post_opp_board, cube_value=2, cube_owner="player",
                            jacoby=True, beaver=True,
                        )
                        # Scale to initial cube: 2 * per-cube-2 equity
                        expected_opp_eq = 2.0 * player_cube_dt.optimal_equity
                    else:
                        # ND: cube unchanged
                        expected_opp_eq = player_cube.optimal_equity

                    self.assertAlmostEqual(
                        opp_r["cubeful_equity"], expected_opp_eq, delta=EQUITY_TOL,
                        msg=f"Roll {d1}-{d2}, opp {od1}-{od2}: "
                            f"equity {opp_r['cubeful_equity']:.4f} "
                            f"vs expected {expected_opp_eq:.4f}",
                    )

    def test_3ply_details(self):
        """Test 2-ply details at 3-ply evaluation depth."""
        self._run_detail_test(3)

    def test_4ply_details(self):
        """Test 2-ply details at 4-ply evaluation depth."""
        self._run_detail_test(4)

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
