# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""
Tests for the Stage 11 backgame category (backgame_category + the 20-NN
BackgameAwarePairStrategy selection).

The category rules under test (see neural_net.h):
  - exactly two anchors: both on the 1/2/3 points -> deep; none deeper than
    the 3-point -> double; otherwise (one on the 1/2 point, one higher) middle
  - three or more anchors: deep when at least two sit on the 1/2/3 points,
    else middle (never double)
  - the 6-point counts as an anchor (Stage 9 detection convention)
  - same answer from either perspective, and for either side's backgame

Boards are built synthetically so the anchor set is exact and Stage 9's
detection (plan pair (anchoring, racing), backgame side behind on pips,
2+ anchors) demonstrably fires — each case asserts detection fired at all
before asserting the category.

Run with:
    python -m pytest bgsage/tests/test_backgame_category.py -v
"""

import os
import sys
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
build_dir = os.path.join(repo_dir, "build")
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(repo_dir, "python"))
if sys.platform == "win32":
    cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(cuda):
        os.add_dll_directory(cuda)
    os.add_dll_directory(build_dir)

import bgbot_cpp  # noqa: E402


def backgame_board(anchor_depths):
    """A player-backgame board with anchors exactly on ``anchor_depths``.

    Player 1: two checkers on the opponent's d-point (index 25-d) per depth,
    the rest split between the 5- and 6-points (a flat stack, so the plan
    classifier reads anchoring rather than priming). Player 2: 15 checkers
    racing at home on whatever of 19-24 the anchors leave free, spilling to
    18, 17, ... when needed. P1 is far behind on pips by construction.
    """
    board = [0] * 26
    for d in anchor_depths:
        board[25 - d] = 2
    rest = 15 - 2 * len(anchor_depths)
    board[6] = rest - rest // 2
    board[5] = rest // 2

    # One straggler just outside P2's home keeps contact for EVERY anchor
    # set (index 18 < any anchor index, which is >= 19), the rest race at
    # home on the free points, lowest first.
    board[18] = -1
    free = [pt for pt in range(19, 25) if board[pt] == 0]
    per = -(-14 // len(free))          # ceil: big anchor sets leave few points
    remaining = 14
    for pt in free:
        take = min(per, remaining)
        board[pt] -= take
        remaining -= take
    assert remaining == 0
    return board


class TestBackgameCategory(unittest.TestCase):
    # (anchor depths, expected category)
    CASES = [
        # The ten named pairs.
        ({1, 2}, "deep"), ({1, 3}, "deep"), ({2, 3}, "deep"),
        ({1, 4}, "middle"), ({2, 4}, "middle"),
        ({1, 5}, "middle"), ({2, 5}, "middle"),
        ({3, 4}, "double"), ({3, 5}, "double"), ({4, 5}, "double"),
        # Pairs involving the 6-point (an anchor for detection, so it needs a
        # category): with a 1/2 anchor -> middle, otherwise double.
        ({1, 6}, "middle"), ({2, 6}, "middle"),
        ({3, 6}, "double"), ({4, 6}, "double"), ({5, 6}, "double"),
        # Three or more anchors: deep iff two or more sit on the 1/2/3 points.
        ({1, 2, 3}, "deep"), ({1, 2, 4}, "deep"), ({2, 3, 5}, "deep"),
        ({1, 2, 4, 5}, "deep"),
        ({3, 4, 5}, "middle"), ({2, 4, 5}, "middle"), ({1, 4, 5}, "middle"),
        ({4, 5, 6}, "middle"), ({3, 5, 6}, "middle"),
    ]

    def assert_detected(self, board, depths):
        """The case is only meaningful if Stage 9's detection fires."""
        gp_p = bgbot_cpp.classify_game_plan(board)
        gp_o = bgbot_cpp.classify_game_plan(bgbot_cpp.flip_board(board))
        pips_p, pips_o = bgbot_cpp.pip_counts(board)
        self.assertEqual(gp_p, "anchoring",
                         f"depths {sorted(depths)}: player plan {gp_p}")
        self.assertIn(gp_o, ("racing", "purerace"),
                      f"depths {sorted(depths)}: opponent plan {gp_o}")
        self.assertGreater(pips_p, pips_o)

    def test_categories(self):
        for depths, expect in self.CASES:
            board = backgame_board(depths)
            self.assert_detected(board, depths)
            got = bgbot_cpp.backgame_category(board)
            self.assertEqual(
                got, expect,
                f"anchors {sorted(depths)}: got {got}, want {expect}")

    def test_perspective_invariant(self):
        for depths, expect in self.CASES:
            board = backgame_board(depths)
            flipped = bgbot_cpp.flip_board(board)
            self.assertEqual(bgbot_cpp.backgame_category(board),
                             bgbot_cpp.backgame_category(flipped),
                             f"anchors {sorted(depths)}: perspective changed "
                             f"the category")

    def test_single_anchor_is_none(self):
        board = backgame_board({1})
        self.assertEqual(bgbot_cpp.backgame_category(board), "none")

    def test_no_backgame_positions(self):
        from bgsage.board import STARTING_BOARD
        self.assertEqual(bgbot_cpp.backgame_category(STARTING_BOARD), "none")
        # A pure race: no contact, no backgame.
        race = [0] * 26
        race[1] = race[2] = race[3] = 5
        race[22] = race[23] = race[24] = -5
        self.assertEqual(bgbot_cpp.backgame_category(race), "none")


class TestStage11Selection(unittest.TestCase):
    """The 20-NN strategy routes backgames by category, everything else as S9."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        from bgsage.weights import WeightConfigPair

        cls.s9 = WeightConfigPair.from_model("stage9")
        try:
            cls.s9.validate()
        except FileNotFoundError:
            raise unittest.SkipTest("stage9 weights not present")

        # Three random-init backgame NNs, written by a 0-game training run
        # (it saves the freshly initialised weights at its final checkpoint).
        cls.tmp = tempfile.TemporaryDirectory()
        seed_board = backgame_board({1, 2})
        extras = []
        for cat in ("deep", "middle", "double"):
            bgbot_cpp.td_train_backgame_truncated(
                n_games=0, model_name=f"td_test_bg_{cat}",
                models_dir=cls.tmp.name, start_boards=[seed_board],
                ref_weight_paths=cls.s9.paths, ref_hidden_sizes=cls.s9.hiddens)
            extras.append(os.path.join(cls.tmp.name, f"td_test_bg_{cat}.weights"))

        cls.paths = cls.s9.paths[:17] + extras
        cls.hiddens = cls.s9.hiddens[:17] + [400, 400, 400]
        cls.s11 = bgbot_cpp.BackgameAwarePairStrategy(cls.paths, cls.hiddens)
        cls.s9_strat = bgbot_cpp.BackgameAwarePairStrategy(
            cls.s9.paths, cls.s9.hiddens)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_backgame_routing(self):
        for depths, expect in TestBackgameCategory.CASES:
            board = backgame_board(depths)
            idx = self.s11.select_nn_idx(board)
            want = {"deep": 17, "middle": 18, "double": 19}[expect]
            self.assertEqual(idx, want,
                             f"anchors {sorted(depths)}: NN {idx}, want {want}")
            # Same NN from the other side's perspective.
            self.assertEqual(self.s11.select_nn_idx(bgbot_cpp.flip_board(board)),
                             want)

    def test_non_backgame_matches_stage9(self):
        from bgsage.board import STARTING_BOARD

        boards = [list(STARTING_BOARD)]
        # A few random playouts' worth of early positions.
        import random
        rng = random.Random(7)
        b = list(STARTING_BOARD)
        for _ in range(30):
            moves = bgbot_cpp.possible_moves(b, rng.randint(1, 6), rng.randint(1, 6))
            if not moves:
                break
            b = list(moves[rng.randrange(len(moves))])
            if bgbot_cpp.check_game_over(b) != 0:
                break
            boards.append(list(b))
            b = bgbot_cpp.flip_board(b)
        checked = 0
        for board in boards:
            s9_idx = self.s9_strat.select_nn_idx(board)
            if s9_idx in (17, 18):   # backgames route differently by design
                continue
            self.assertEqual(self.s11.select_nn_idx(board), s9_idx)
            checked += 1
        self.assertGreater(checked, 5)


if __name__ == "__main__":
    unittest.main()
