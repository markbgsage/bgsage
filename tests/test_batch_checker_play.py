"""
Tests for batch_checker_play.

Verifies that batch_checker_play results match serial (single-position)
BgBotAnalyzer.checker_play for both 0-ply and 1-ply, using random positions
from contact.bm with random dice.

Run with:
    python -m pytest bgsage/tests/test_batch_checker_play.py -v
    python -m unittest bgsage.tests.test_batch_checker_play -v
"""

import os
import random
import sys
import unittest

# Setup paths
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

from bgsage import BgBotAnalyzer
from bgsage.batch import batch_checker_play
from bgsage.data import board_from_gnubg_position_string
from bgsage.weights import WeightConfig

DATA_DIR = os.path.join(project_dir, "bgsage", "data")
CONTACT_BM = os.path.join(DATA_DIR, "contact.bm")

SEED = 54321
N_POSITIONS = 50


def _load_positions_from_bm(
    filepath: str, n: int, seed: int,
) -> list[dict]:
    """Load n random positions (board + dice) from a .bm file."""
    with open(filepath, "r") as f:
        move_lines = [line for line in f if line.startswith("m ")]

    rng = random.Random(seed)
    chosen = rng.sample(range(len(move_lines)), min(n, len(move_lines)))

    positions = []
    for idx in chosen:
        bits = move_lines[idx].split()
        board = board_from_gnubg_position_string(bits[1])
        die1, die2 = int(bits[2]), int(bits[3])
        positions.append({
            "board": board,
            "die1": die1,
            "die2": die2,
            "cube_value": 1,
            "cube_owner": "centered",
        })
    return positions


class TestBatchCheckerPlay(unittest.TestCase):
    """Verify batch_checker_play matches serial BgBotAnalyzer.checker_play."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(CONTACT_BM):
            raise unittest.SkipTest(f"contact.bm not found at {CONTACT_BM}")
        cls.positions = _load_positions_from_bm(CONTACT_BM, N_POSITIONS, SEED)
        cls.weights = WeightConfig.default()
        cls.weights.validate()

    # ------------------------------------------------------------------
    # 0-ply tests
    # ------------------------------------------------------------------

    def test_0ply_batch_matches_serial(self):
        """batch_checker_play at 0-ply matches BgBotAnalyzer.checker_play."""
        analyzer = BgBotAnalyzer(eval_level="0ply", cubeful=True)
        batch_results = batch_checker_play(
            self.positions, eval_level="0ply",
            weights=self.weights, n_threads=0,
        )

        self.assertEqual(len(batch_results), len(self.positions))
        for i, (br, pos) in enumerate(zip(batch_results, self.positions)):
            with self.subTest(position=i):
                sr = analyzer.checker_play(
                    pos["board"], pos["die1"], pos["die2"],
                    cube_value=pos["cube_value"],
                    cube_owner=pos["cube_owner"],
                )
                # Same number of moves
                self.assertEqual(
                    len(br.moves), len(sr.moves),
                    msg=f"pos {i}: move count mismatch "
                        f"({len(br.moves)} vs {len(sr.moves)})",
                )
                # Best move equity matches
                if br.moves:
                    self.assertAlmostEqual(
                        br.moves[0].equity, sr.moves[0].equity, places=4,
                        msg=f"pos {i}: best equity mismatch",
                    )
                    # Best move board matches
                    self.assertEqual(
                        br.moves[0].board, sr.moves[0].board,
                        msg=f"pos {i}: best move board mismatch",
                    )
                    # Check equity_diff of best is 0
                    self.assertAlmostEqual(
                        br.moves[0].equity_diff, 0.0, places=6,
                    )

    def test_0ply_parallel_matches_serial(self):
        """batch_checker_play at 0-ply: parallel vs n_threads=1."""
        serial = batch_checker_play(
            self.positions, eval_level="0ply",
            weights=self.weights, n_threads=1,
        )
        parallel = batch_checker_play(
            self.positions, eval_level="0ply",
            weights=self.weights, n_threads=0,
        )

        self.assertEqual(len(serial), len(parallel))
        for i, (s, p) in enumerate(zip(serial, parallel)):
            with self.subTest(position=i):
                self.assertEqual(len(s.moves), len(p.moves))
                if s.moves:
                    self.assertAlmostEqual(
                        s.moves[0].equity, p.moves[0].equity, places=6,
                    )
                    self.assertEqual(s.moves[0].board, p.moves[0].board)

    # ------------------------------------------------------------------
    # 1-ply tests
    # ------------------------------------------------------------------

    def test_1ply_batch_matches_serial(self):
        """batch_checker_play at 1-ply matches BgBotAnalyzer.checker_play exactly."""
        analyzer = BgBotAnalyzer(eval_level="1ply", cubeful=True)
        # Use fewer positions for 1-ply (slower)
        positions = self.positions[:10]
        batch_results = batch_checker_play(
            positions, eval_level="1ply",
            weights=self.weights, n_threads=0,
        )

        self.assertEqual(len(batch_results), len(positions))
        for i, (br, pos) in enumerate(zip(batch_results, positions)):
            with self.subTest(position=i):
                sr = analyzer.checker_play(
                    pos["board"], pos["die1"], pos["die2"],
                    cube_value=pos["cube_value"],
                    cube_owner=pos["cube_owner"],
                )
                # Same number of moves
                self.assertEqual(
                    len(br.moves), len(sr.moves),
                    msg=f"pos {i}: move count mismatch",
                )
                if not br.moves:
                    continue
                # Best move board matches
                self.assertEqual(
                    br.moves[0].board, sr.moves[0].board,
                    msg=f"pos {i}: best move board mismatch",
                )
                # Best move equity matches
                self.assertAlmostEqual(
                    br.moves[0].equity, sr.moves[0].equity, places=4,
                    msg=f"pos {i}: best equity mismatch",
                )
                # Best move cubeless equity matches
                self.assertAlmostEqual(
                    br.moves[0].cubeless_equity,
                    sr.moves[0].cubeless_equity, places=4,
                    msg=f"pos {i}: best cubeless equity mismatch",
                )
                # Probabilities match
                for j, (bp, sp) in enumerate(zip(
                    br.moves[0].probs.to_list(),
                    sr.moves[0].probs.to_list(),
                )):
                    self.assertAlmostEqual(
                        bp, sp, places=4,
                        msg=f"pos {i}: prob[{j}] mismatch",
                    )

    def test_1ply_parallel_matches_serial(self):
        """batch_checker_play at 1-ply: parallel vs n_threads=1."""
        positions = self.positions[:10]
        serial = batch_checker_play(
            positions, eval_level="1ply",
            weights=self.weights, n_threads=1,
        )
        parallel = batch_checker_play(
            positions, eval_level="1ply",
            weights=self.weights, n_threads=0,
        )

        self.assertEqual(len(serial), len(parallel))
        for i, (s, p) in enumerate(zip(serial, parallel)):
            with self.subTest(position=i):
                if s.moves and p.moves:
                    self.assertAlmostEqual(
                        s.moves[0].equity, p.moves[0].equity, places=4,
                    )
                    self.assertEqual(s.moves[0].board, p.moves[0].board)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_input(self):
        """batch_checker_play with empty list returns empty list."""
        results = batch_checker_play([], eval_level="0ply", weights=self.weights)
        self.assertEqual(results, [])

    def test_result_structure(self):
        """Verify CheckerPlayResult fields are populated correctly."""
        pos = self.positions[0]
        results = batch_checker_play(
            [pos], eval_level="0ply", weights=self.weights,
        )
        self.assertEqual(len(results), 1)
        r = results[0]

        # CheckerPlayResult fields
        self.assertEqual(r.board, list(pos["board"]))
        self.assertEqual(r.die1, pos["die1"])
        self.assertEqual(r.die2, pos["die2"])
        self.assertEqual(r.eval_level, "0-ply")

        # MoveAnalysis fields
        self.assertGreater(len(r.moves), 0)
        m = r.moves[0]
        self.assertEqual(len(m.board), 26)
        self.assertIsInstance(m.equity, float)
        self.assertIsInstance(m.cubeless_equity, float)
        self.assertAlmostEqual(m.equity_diff, 0.0, places=6)
        self.assertEqual(m.eval_level, "0-ply")
        self.assertIsNotNone(m.probs)
        self.assertGreaterEqual(m.probs.win, 0.0)
        self.assertLessEqual(m.probs.win, 1.0)

    def test_moves_sorted_descending(self):
        """Moves are sorted by equity descending (best first)."""
        results = batch_checker_play(
            self.positions[:20], eval_level="0ply", weights=self.weights,
        )
        for i, r in enumerate(results):
            with self.subTest(position=i):
                for j in range(len(r.moves) - 1):
                    self.assertGreaterEqual(
                        r.moves[j].equity, r.moves[j + 1].equity,
                        msg=f"pos {i}: moves not sorted at index {j}",
                    )


if __name__ == "__main__":
    unittest.main()
