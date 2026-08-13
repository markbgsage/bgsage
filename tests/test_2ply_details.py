# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""
Tests for incl_2ply_details in cube action analytics.

Verifies that the per-roll details returned by cube_action(incl_2ply_details=True)
are consistent with independent checker play and cube action evaluations at the
corresponding ply levels.

The details path (pick_best_move_for_roll in cpp/src/cube.cpp) selects moves by
1-ply cubeless equity, after narrowing big candidate sets (>16) to the top 15 by
PubEval — the same prefilter the cubeful recursion applies at interior nodes
(cube_decision_nply_unified always passes the PubEval filter). This test mirrors
that:
- Board verification: PubEval keep-15 prefilter + 1-ply cubeless checker play
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
from bgsage.board import flip_board, check_game_over, is_race, possible_moves

# Test position: contact/priming position with non-trivial cube dynamics
BOARD = [0, 0, 0, 2, 3, 0, 4, -2, 2, 0, 0, 0, -4, 2, -3, 0, -1, 0, 0, -3, 2, 0, -2, 0, 0, 0]

# Per-roll equity tolerance, by analysis ply. The details path picks moves at
# its two captured levels by 1-ply cubeless equity (required so the ND and DT
# sections share one board per roll), while the standalone cube_action used as
# the reference picks interior moves by 1-ply cubeful equity (cube_eval.cpp).
# When the two pick rules choose different moves somewhere in the subtree, the
# per-roll equities legitimately diverge — more at deeper ply (observed max
# ~0.013 at 3-ply request depth, ~0.030 at 4-ply on the test position). The
# headline equity_nd/equity_dt still agree with the plain (no-details) call to
# within ~0.003. A plumbing bug (wrong perspective, scaling, or roll mapping)
# produces diffs of 0.1+, so these tolerances still catch real breakage.
# At 2-ply the reference subtree is a single 1-ply Janowski leaf below the
# pick, so there is no interior-pick divergence — observed exact agreement.
EQUITY_TOL = {2: 0.001, 3: 0.02, 4: 0.05}

# The 21 dice combinations (matching C++ ALL_ROLLS order)
DICE_ROLLS = [
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 4), (3, 5), (3, 6),
    (4, 5), (4, 6),
    (5, 6),
]


# PubEval prefilter constants — mirror pick_best_move_for_roll in cpp/src/cube.cpp.
PREFILTER_THRESHOLD = 16
PREFILTER_KEEP = 15


def _load_pubeval_weights():
    """Parse the Tesauro PubEval weights from cpp/src/pubeval.cpp (the default
    weights of the PubEval prefilter passed by cube_decision_nply_unified)."""
    import re
    src_path = os.path.join(repo_dir, "cpp", "src", "pubeval.cpp")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    weights = {}
    for name in ("CONTACT_TESAURO", "RACE_TESAURO"):
        m = re.search(name + r"\[122\]\s*=\s*\{(.*?)\};", src, re.S)
        vals = [float(x) for x in m.group(1).split(",") if x.strip()]
        assert len(vals) == 122, f"{name}: parsed {len(vals)} weights"
        weights[name] = vals
    return weights["CONTACT_TESAURO"], weights["RACE_TESAURO"]


_PUBEVAL_CONTACT, _PUBEVAL_RACE = _load_pubeval_weights()


def _pubeval_score(board, pre_move_is_race):
    """Python port of PubEval::evaluate (cpp/src/pubeval.cpp), Tesauro weights."""
    go = check_game_over(board)
    if go != 0:
        # Terminals always survive the prefilter (scored 1e30 in C++).
        return 1e30
    x = [0.0] * 122
    for i in range(24):
        n = board[24 - i]
        if n == 0:
            continue
        if n == -1:
            x[5 * i] = 1.0
        if n == 1:
            x[5 * i + 1] = 1.0
        if n >= 2:
            x[5 * i + 2] = 1.0
        if n == 3:
            x[5 * i + 3] = 1.0
        if n >= 4:
            x[5 * i + 4] = (n - 3) / 2.0
    x[120] = board[0] / 2.0
    borne_off = 15 - sum(v for v in board[1:25] if v > 0) - board[25]
    x[121] = borne_off / 15.0
    w = _PUBEVAL_RACE if pre_move_is_race else _PUBEVAL_CONTACT
    return sum(wi * xi for wi, xi in zip(w, x))


def best_move_1ply_cubeless(board, d1, d2, post_move_analyzer):
    """Pick best move mirroring pick_best_move_for_roll in cpp/src/cube.cpp:
    PubEval keep-15 prefilter for >16 candidates, then 1-ply cubeless argmax."""
    candidates = [list(c) for c in possible_moves(board, d1, d2)]
    if not candidates:
        return board  # no moves: standing pat
    if len(candidates) > PREFILTER_THRESHOLD:
        pre_race = is_race(board)
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-_pubeval_score(candidates[i], pre_race), i),
        )
        candidates = [candidates[i] for i in order[:PREFILTER_KEEP]]
    best_board = None
    best_eq = -1e30
    for cand in candidates:
        go = check_game_over(cand)
        if go != 0:
            eq = float(go)
        else:
            eq = post_move_analyzer.post_move_analytics(cand).cubeless_equity
        if eq > best_eq:
            best_eq = eq
            best_board = cand
    return best_board


class TestTwoPlyDetailsND(unittest.TestCase):
    """Verify ND section of 2-ply details against independent per-roll evaluations."""

    def _run_nd_test(self, n_plies: int):
        """Core ND test logic for a given ply level."""
        tol = EQUITY_TOL[n_plies]
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
        # The opponent-roll level is captured only at 3-ply and above (at
        # 2-ply it would sit below the 1-ply leaf).
        player_cube_analyzer = (
            BgBotAnalyzer(eval_level=f"{n_plies - 2}ply", cubeful=True)
            if n_plies >= 3 else None
        )

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
                pr["cubeful_equity"], expected_player_eq, delta=tol,
                msg=f"Roll {d1}-{d2}: ND player equity {pr['cubeful_equity']:.4f} "
                    f"vs expected {expected_player_eq:.4f} "
                    f"(opp optimal_action={opp_cube.optimal_action})",
            )

            if n_plies == 2:
                # No opponent-roll level at 2-ply: the field must be absent.
                self.assertNotIn(
                    "opponent_rolls", pr,
                    f"Roll {d1}-{d2}: 2-ply details should not carry opponent_rolls",
                )
                continue

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
                        opp_r["cubeful_equity"], expected_opp_eq, delta=tol,
                        msg=f"Roll {d1}-{d2}, opp {od1}-{od2}: "
                            f"equity {opp_r['cubeful_equity']:.4f} "
                            f"vs expected {expected_opp_eq:.4f}",
                    )

    def test_2ply_nd(self):
        """Test ND 2-ply details at 2-ply (true 2-ply, no opponent rolls)."""
        self._run_nd_test(2)

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
        tol = EQUITY_TOL[n_plies]
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
                pr["cubeful_equity"], expected_player_eq, delta=tol,
                msg=f"Roll {d1}-{d2}: DT player equity {pr['cubeful_equity']:.4f} "
                    f"vs expected {expected_player_eq:.4f}",
            )

    def test_2ply_dt(self):
        """Test DT 2-ply details at 2-ply (true 2-ply, no opponent rolls)."""
        self._run_dt_test(2)

    def test_3ply_dt(self):
        """Test DT 2-ply details at 3-ply."""
        self._run_dt_test(3)

    def test_4ply_dt(self):
        """Test DT 2-ply details at 4-ply."""
        self._run_dt_test(4)


class TestDetailsHeadlineMatchesPlain(unittest.TestCase):
    """The headline equities of a details call must match the plain call.

    Regression for the n_plies=2 bug where cube_decision_nply_with_details
    always ran its two manual recursion levels, so a 2-ply details request
    silently returned 3-ply-grade headline equities (~0.06 off plain 2-ply on
    this position) at ~21x the cost. The only legitimate divergence is the
    interior pick rule (1-ply cubeless in the details path vs 1-ply cubeful in
    the plain recursion), observed <= ~0.003.
    """

    def _run_headline_test(self, n_plies: int):
        analyzer = BgBotAnalyzer(eval_level=f"{n_plies}ply", cubeful=True)
        plain = analyzer.cube_action(
            BOARD, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True,
        )
        detailed = analyzer.cube_action(
            BOARD, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True, incl_2ply_details=True,
        )
        for field in ("equity_nd", "equity_dt", "equity_dp"):
            self.assertAlmostEqual(
                getattr(detailed, field), getattr(plain, field), delta=0.005,
                msg=f"{n_plies}-ply {field}: details {getattr(detailed, field):.6f} "
                    f"vs plain {getattr(plain, field):.6f}",
            )

    def test_2ply_headline(self):
        """2-ply details headline == plain 2-ply (the original bug)."""
        self._run_headline_test(2)

    def test_3ply_headline(self):
        """3-ply details headline == plain 3-ply."""
        self._run_headline_test(3)

    def test_2ply_headline_dancing(self):
        """2-ply headline equality on a position with dancing rolls.

        Player on the bar against a 5-point board (only a 1 enters), so the
        no-legal-moves branch of the 2-ply details path is exercised.
        """
        dancing_board = [
            0,
            0, 0, 0, 0, -2, 5, 0, 4, 0, 0, 0, 0,
            3, 0, 0, 0, -3, 0, -2, -2, -2, -2, -2, 2,
            1,
        ]
        analyzer = BgBotAnalyzer(eval_level="2ply", cubeful=True)
        plain = analyzer.cube_action(
            dancing_board, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True,
        )
        detailed = analyzer.cube_action(
            dancing_board, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True, incl_2ply_details=True,
        )
        for field in ("equity_nd", "equity_dt"):
            self.assertAlmostEqual(
                getattr(detailed, field), getattr(plain, field), delta=0.005,
                msg=f"dancing {field}: details {getattr(detailed, field):.6f} "
                    f"vs plain {getattr(plain, field):.6f}",
            )
        # A roll without a 1 dances: standing pat, board unchanged.
        for pr in detailed.details["nd"]:
            if pr["die1"] == 2 and pr["die2"] == 3:
                self.assertEqual(list(pr["checkers"]), dancing_board)
                self.assertNotIn("opponent_rolls", pr)
                break
        else:
            self.fail("Roll 2-3 not found in nd details")


class TestTwoPlyDetailsErrors(unittest.TestCase):
    """Test error handling for 2-ply details."""

    def test_error_below_2ply(self):
        """Verify that incl_2ply_details raises an error for 1-ply.

        2-ply and above are allowed: at 2-ply the details path captures only
        the player-roll level, evaluating each post-move position at the
        1-ply Janowski leaf (true 2-ply — headline matches the plain call).
        """
        analyzer = BgBotAnalyzer(eval_level="1ply", cubeful=True)
        with self.assertRaises(Exception):
            analyzer.cube_action(
                BOARD, cube_value=1, cube_owner="centered",
                incl_2ply_details=True,
            )

    def test_2ply_details_allowed(self):
        """Verify that incl_2ply_details works at 2-ply.

        At 2-ply only the player-roll level is captured (the opponent-roll
        level would sit below the 1-ply leaf), so opponent_rolls is absent
        from every entry.
        """
        analyzer = BgBotAnalyzer(eval_level="2ply", cubeful=True)
        result = analyzer.cube_action(
            BOARD, cube_value=1, cube_owner="centered",
            jacoby=True, beaver=True, incl_2ply_details=True,
        )
        self.assertIsNotNone(result.details)
        self.assertEqual(len(result.details["nd"]), 21)
        self.assertEqual(len(result.details["dt"]), 21)
        for section in ("nd", "dt"):
            for pr in result.details[section]:
                self.assertNotIn("opponent_rolls", pr)


if __name__ == "__main__":
    unittest.main()
