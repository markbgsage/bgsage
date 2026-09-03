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

    def test_exit_candidates_route_by_pre_move_board(self):
        """Candidate evaluation is decision-level: with an in-region pre-move
        board as context, an out-of-region candidate is valued by the CATEGORY
        NN (select_nn_idx(pre_move_board) in every candidate path). The
        exit-descendant training data exists because of this property — the
        category NN is the appraiser of exit candidates, so it must be
        trained on them."""
        pre = backgame_board({1, 2})
        self.assertEqual(bgbot_cpp.backgame_category(pre), "deep")

        # A candidate-like exit: the deepest anchor abandoned (its checkers
        # brought home), leaving a single anchor -> no backgame category.
        exit_board = list(pre)
        exit_board[24] = 0
        exit_board[6] += 2
        self.assertEqual(bgbot_cpp.backgame_category(exit_board), "none")

        routed = self.s11.evaluate_board(exit_board, pre)["probs"]
        deep_nn = bgbot_cpp.NNStrategy(self.paths[17], 400, 244)
        direct = deep_nn.evaluate_board(exit_board, exit_board)["probs"]
        for r, d in zip(routed, direct):
            self.assertAlmostEqual(r, d, places=5)

        # By its own classification the same candidate routes to a standard
        # pair NN, whose value differs from the random-init deep NN's.
        self_routed = self.s11.evaluate_board(exit_board, exit_board)["probs"]
        self.assertGreater(
            max(abs(a - b) for a, b in zip(self_routed, direct)), 1e-4)

    def test_phased_layout_routes_out_of_region_containment(self):
        """The 21-NN phased layout keeps the trio's routing everywhere the
        plan-pair gate detects a backgame, and sends ONLY out-of-region
        early-containment positions to NN 20."""
        seed_board = backgame_board({1, 2})
        extras = []
        for name in ("td_test_bg_p3", "td_test_bg_containment"):
            bgbot_cpp.td_train_backgame_truncated(
                n_games=0, model_name=name, models_dir=self.tmp.name,
                start_boards=[seed_board], ref_weight_paths=self.s9.paths,
                ref_hidden_sizes=self.s9.hiddens)
            extras.append(os.path.join(self.tmp.name, f"{name}.weights"))
        phased = bgbot_cpp.BackgameAwarePairStrategy(
            self.paths + extras, self.hiddens + [400, 400], True)
        # The flag demands exactly the phased count.
        with self.assertRaises(RuntimeError):
            bgbot_cpp.BackgameAwarePairStrategy(self.paths, self.hiddens, True)

        # Snake: a far-side prime holding one straggler, opponent crunched,
        # nobody off -> early containment. Lamford 01: 12 off -> late.
        snake = [0,0,0,-1,0,0,0,0,0,0,0,0,0,1,0,1,0,2,3,3,2,2,1,-7,-7,0]
        lamford = [0,-1,2,2,0,-1,2,1,0,0,0,-1,0,0,0,0,0,0,0,2,0,2,1,0,2,1]
        self.assertEqual(bgbot_cpp.backgame_phase(snake), "early_containment")
        self.assertEqual(bgbot_cpp.backgame_phase(lamford), "late_containment")
        # backgame_board() parks the racer's 14 checkers at home: bear-in.
        self.assertEqual(bgbot_cpp.backgame_phase(seed_board), "bear_in")
        from bgsage.board import STARTING_BOARD
        self.assertEqual(bgbot_cpp.backgame_phase(STARTING_BOARD), "none")
        self.assertEqual(bgbot_cpp.backgame_phase(
            bgbot_cpp.flip_board(snake)), "early_containment")

        # Every detected backgame routes exactly as the 20-NN trio does
        # (none of the synthetic boards is a containment game: the racer
        # has nothing off).
        for depths, _ in TestBackgameCategory.CASES:
            board = backgame_board(depths)
            self.assertFalse(bgbot_cpp.containment_category(board))
            self.assertEqual(phased.select_nn_idx(board), self.s11.select_nn_idx(board))

        # Containment games route to NN 21 whatever the container holds —
        # ahead of the trio for the anchored ones — and the C++ rule agrees
        # with the Python reference on every containment-folder decision.
        cont = os.path.join(repo_dir, "backgame_ref_positions", "benchmark",
                            "containment rollout.jsonl")
        if not os.path.exists(cont):
            raise unittest.SkipTest("containment reference not present")
        sys.path.insert(0, os.path.join(repo_dir, "scripts"))
        import containment_rule as cr
        import json
        n_cont = n_anchored = 0
        with open(cont, encoding="utf-8") as f:
            for line in f:
                b = json.loads(line)["board"]
                is_c = bgbot_cpp.containment_category(b)
                self.assertEqual(is_c, cr.containment(b))
                self.assertEqual(bgbot_cpp.containment_category(bgbot_cpp.flip_board(b)), is_c)
                if not is_c:
                    continue
                n_cont += 1
                self.assertEqual(phased.select_nn_idx(b), 21)
                if bgbot_cpp.backgame_category(b) != "none":
                    n_anchored += 1
                    self.assertGreaterEqual(self.s11.select_nn_idx(b), 17)
                # Routing is not enough: the EVALUATION must read slot 21.
                # (Slot 21 once collided with the Stage 10 blend sentinel, so
                # the evaluator blended NNs 17/19 and never touched it.)
                if n_cont <= 20:
                    want = bgbot_cpp.NNStrategy(extras[1], 400, 244).evaluate_board(b, b)["probs"]
                    got = phased.evaluate_board(b, b)["probs"]
                    for g, w in zip(got, want):
                        self.assertAlmostEqual(g, w, places=5)
        self.assertGreater(n_cont, 2500)
        self.assertGreater(n_anchored, 100)

        # Out-of-region early containment -> 20, from either perspective,
        # while the trio (no phase NN) still uses a standard pair net there.
        # Real cases come from the committed folder reference: the plan pair
        # has flipped (the racer's block reads as priming) with a straggler
        # still being contained. Snake itself is detected (middle), so it
        # keeps its trio routing under both layouts.
        self.assertEqual(phased.select_nn_idx(snake), self.s11.select_nn_idx(snake))
        import json
        ref = os.path.join(repo_dir, "backgame_ref_positions", "benchmark",
                           "21 backgame rollout.jsonl")
        if not os.path.exists(ref):
            raise unittest.SkipTest("folder reference not present")
        found = 0
        with open(ref, encoding="utf-8") as f:
            for line in f:
                b = json.loads(line)["board"]
                if (bgbot_cpp.backgame_category(b) != "none"
                        or bgbot_cpp.backgame_phase(b) != "early_containment"):
                    continue
                self.assertEqual(phased.select_nn_idx(b), 20)
                self.assertEqual(phased.select_nn_idx(bgbot_cpp.flip_board(b)), 20)
                self.assertLess(self.s11.select_nn_idx(b), 17)
                found += 1
                if found >= 25:
                    break
        self.assertGreater(found, 0)

    def test_snake_layout_routes_snakes_first(self):
        """With a 23rd NN the phased layout sends every snake — a far-side
        prime trapping a straggler against a crunched board — to NN 22,
        ahead of the trio and the containment rule, and leaves everything
        else exactly as the 22-NN layout routes it."""
        seed_board = backgame_board({1, 2})
        extras = []
        for name in ("td_test_bg_p3s", "td_test_bg_containments", "td_test_bg_snake"):
            bgbot_cpp.td_train_backgame_truncated(
                n_games=0, model_name=name, models_dir=self.tmp.name,
                start_boards=[seed_board], ref_weight_paths=self.s9.paths,
                ref_hidden_sizes=self.s9.hiddens)
            extras.append(os.path.join(self.tmp.name, f"{name}.weights"))
        phased = bgbot_cpp.BackgameAwarePairStrategy(
            self.paths + extras[:2], self.hiddens + [400, 400], True)
        snaked = bgbot_cpp.BackgameAwarePairStrategy(
            self.paths + extras, self.hiddens + [400, 400, 400], True)
        with self.assertRaises(RuntimeError):
            bgbot_cpp.BackgameAwarePairStrategy(
                self.paths + extras + extras[:1], self.hiddens + [400] * 4, True)

        from bgsage.board import STARTING_BOARD
        snake = [0,0,0,-1,0,0,0,0,0,0,0,0,0,1,0,1,0,2,3,3,2,2,1,-7,-7,0]
        self.assertTrue(bgbot_cpp.snake_category(snake))
        self.assertTrue(bgbot_cpp.snake_category(bgbot_cpp.flip_board(snake)))
        self.assertEqual(snaked.select_nn_idx(snake), 22)
        self.assertEqual(snaked.select_nn_idx(bgbot_cpp.flip_board(snake)), 22)
        # Not snakes: no crunched opponent (the synthetic backgames), and the
        # starting position; the 22-NN layout is untouched there.
        for board in [backgame_board({1, 2}), backgame_board({4, 5}), STARTING_BOARD]:
            self.assertFalse(bgbot_cpp.snake_category(board))
            self.assertEqual(snaked.select_nn_idx(board), phased.select_nn_idx(board))
        self.assertNotEqual(phased.select_nn_idx(snake), 22)

        # The C++ rule agrees with the Python reference and the benchmark
        # family filter on every snake-folder decision, from both sides, and
        # the EVALUATION reads slot 22.
        ref = os.path.join(repo_dir, "backgame_ref_positions", "benchmark",
                           "snake rollout.jsonl")
        if not os.path.exists(ref):
            raise unittest.SkipTest("snake reference not present")
        sys.path.insert(0, os.path.join(repo_dir, "scripts"))
        import snake_rule as sr
        from backgame_benchmark import _snake as family_snake
        import json
        n_snake = n_total = 0
        with open(ref, encoding="utf-8") as f:
            for line in f:
                b = json.loads(line)["board"]
                n_total += 1
                is_s = bgbot_cpp.snake_category(b)
                self.assertEqual(is_s, sr.snake(b))
                self.assertEqual(is_s, family_snake(tuple(b)))
                self.assertEqual(bgbot_cpp.snake_category(bgbot_cpp.flip_board(b)), is_s)
                if not is_s:
                    self.assertEqual(snaked.select_nn_idx(b), phased.select_nn_idx(b))
                    continue
                n_snake += 1
                self.assertEqual(snaked.select_nn_idx(b), 22)
                self.assertEqual(snaked.select_nn_idx(bgbot_cpp.flip_board(b)), 22)
                if n_snake <= 20:
                    want = bgbot_cpp.NNStrategy(extras[2], 400, 244).evaluate_board(b, b)["probs"]
                    got = snaked.evaluate_board(b, b)["probs"]
                    for g, w in zip(got, want):
                        self.assertAlmostEqual(g, w, places=5)
        self.assertGreater(n_snake, 0.9 * n_total)

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
