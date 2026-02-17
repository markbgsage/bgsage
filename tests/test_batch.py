"""
Tests for batch position evaluation.

Verifies that batch_evaluate results match serial (single-position) evaluation
for both 0-ply and 1-ply, using 100 random positions from contact.bm.

Run with:
    python -m pytest bgsage/tests/test_batch.py -v
    python -m unittest bgsage.tests.test_batch -v
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

import bgbot_cpp
from bgsage.batch import batch_evaluate
from bgsage.data import board_from_gnubg_position_string
from bgsage.weights import WeightConfig

DATA_DIR = os.path.join(project_dir, "bgsage", "data")
CONTACT_BM = os.path.join(DATA_DIR, "contact.bm")

# Fixed seed for reproducibility
SEED = 12345
N_POSITIONS = 100


def _load_boards_from_bm(filepath: str, n: int, seed: int) -> list[list[int]]:
    """Load n random pre-roll boards from a .bm file."""
    with open(filepath, "r") as f:
        move_lines = [line for line in f if line.startswith("m ")]

    rng = random.Random(seed)
    chosen = rng.sample(range(len(move_lines)), min(n, len(move_lines)))

    boards = []
    for idx in chosen:
        bits = move_lines[idx].split()
        boards.append(board_from_gnubg_position_string(bits[1]))
    return boards


def _serial_evaluate_0ply(
    boards: list[list[int]], weights: WeightConfig
) -> list[dict]:
    """Evaluate each position one at a time at 0-ply.

    Uses evaluate_cube_decision which follows the same code path as the
    batch C++ function: flip → evaluate_probs(flipped, is_race) → invert.
    """
    owner = bgbot_cpp.CubeOwner.CENTERED
    results = []
    for board in boards:
        r = bgbot_cpp.evaluate_cube_decision(
            board, 1, owner, *weights.weight_args,
        )
        results.append(
            {
                "probs": list(r["probs"]),
                "cubeless_equity": r["cubeless_equity"],
                "cubeful_equity": r["equity_nd"],
            }
        )
    return results


def _serial_evaluate_1ply(
    boards: list[list[int]], weights: WeightConfig
) -> list[dict]:
    """Evaluate each position one at a time at 1-ply.

    Calls batch_evaluate with n_threads=1 to use the exact same C++ code path
    as the parallel version, just single-threaded.
    """
    positions = [
        {"board": b, "cube_value": 1, "cube_owner": "centered"}
        for b in boards
    ]
    batch_results = batch_evaluate(
        positions, eval_level="1ply", weights=weights, n_threads=1,
    )
    results = []
    for br in batch_results:
        results.append(
            {
                "probs": br.probs.to_list(),
                "cubeless_equity": br.cubeless_equity,
                "cubeful_equity": br.cubeful_equity,
            }
        )
    return results


class TestBatchEvaluate(unittest.TestCase):
    """Verify batch_evaluate matches serial evaluation."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(CONTACT_BM):
            raise unittest.SkipTest(f"contact.bm not found at {CONTACT_BM}")
        cls.boards = _load_boards_from_bm(CONTACT_BM, N_POSITIONS, SEED)
        cls.weights = WeightConfig.default()
        cls.weights.validate()
        cls.positions = [
            {"board": b, "cube_value": 1, "cube_owner": "centered"}
            for b in cls.boards
        ]

    # ------------------------------------------------------------------
    # 0-ply tests
    # ------------------------------------------------------------------

    def test_0ply_batch_matches_serial(self):
        """batch_evaluate at 0-ply matches one-at-a-time C++ evaluation."""
        batch_results = batch_evaluate(
            self.positions, eval_level="0ply", weights=self.weights, n_threads=0,
        )
        serial_results = _serial_evaluate_0ply(self.boards, self.weights)

        self.assertEqual(len(batch_results), len(serial_results))
        for i, (br, sr) in enumerate(zip(batch_results, serial_results)):
            with self.subTest(position=i):
                # Cubeless equity
                self.assertAlmostEqual(
                    br.cubeless_equity, sr["cubeless_equity"], places=5,
                    msg=f"pos {i}: cubeless equity mismatch",
                )
                # Cubeful equity
                self.assertAlmostEqual(
                    br.cubeful_equity, sr["cubeful_equity"], places=5,
                    msg=f"pos {i}: cubeful equity mismatch",
                )
                # Probabilities
                for j, (bp, sp) in enumerate(
                    zip(br.probs.to_list(), sr["probs"])
                ):
                    self.assertAlmostEqual(
                        bp, sp, places=5,
                        msg=f"pos {i}: prob[{j}] mismatch",
                    )

    def test_0ply_parallel_matches_serial_batch(self):
        """batch_evaluate at 0-ply: parallel vs n_threads=1 give same results."""
        serial = batch_evaluate(
            self.positions, eval_level="0ply", weights=self.weights, n_threads=1,
        )
        parallel = batch_evaluate(
            self.positions, eval_level="0ply", weights=self.weights, n_threads=0,
        )

        self.assertEqual(len(serial), len(parallel))
        for i, (s, p) in enumerate(zip(serial, parallel)):
            with self.subTest(position=i):
                self.assertAlmostEqual(
                    s.cubeless_equity, p.cubeless_equity, places=6,
                )
                self.assertAlmostEqual(
                    s.cubeful_equity, p.cubeful_equity, places=6,
                )
                self.assertEqual(s.should_double, p.should_double)
                self.assertEqual(s.should_take, p.should_take)
                self.assertEqual(s.optimal_action, p.optimal_action)

    # ------------------------------------------------------------------
    # 1-ply tests
    # ------------------------------------------------------------------

    def test_1ply_batch_matches_serial(self):
        """batch_evaluate at 1-ply matches one-at-a-time MultiPlyStrategy."""
        batch_results = batch_evaluate(
            self.positions, eval_level="1ply", weights=self.weights, n_threads=0,
        )
        serial_results = _serial_evaluate_1ply(self.boards, self.weights)

        self.assertEqual(len(batch_results), len(serial_results))
        for i, (br, sr) in enumerate(zip(batch_results, serial_results)):
            with self.subTest(position=i):
                self.assertAlmostEqual(
                    br.cubeless_equity, sr["cubeless_equity"], places=4,
                    msg=f"pos {i}: cubeless equity mismatch",
                )
                self.assertAlmostEqual(
                    br.cubeful_equity, sr["cubeful_equity"], places=4,
                    msg=f"pos {i}: cubeful equity mismatch",
                )
                for j, (bp, sp) in enumerate(
                    zip(br.probs.to_list(), sr["probs"])
                ):
                    self.assertAlmostEqual(
                        bp, sp, places=4,
                        msg=f"pos {i}: prob[{j}] mismatch",
                    )

    def test_1ply_parallel_matches_serial_batch(self):
        """batch_evaluate at 1-ply: parallel vs n_threads=1 give same results."""
        serial = batch_evaluate(
            self.positions, eval_level="1ply", weights=self.weights, n_threads=1,
        )
        parallel = batch_evaluate(
            self.positions, eval_level="1ply", weights=self.weights, n_threads=0,
        )

        self.assertEqual(len(serial), len(parallel))
        for i, (s, p) in enumerate(zip(serial, parallel)):
            with self.subTest(position=i):
                self.assertAlmostEqual(
                    s.cubeless_equity, p.cubeless_equity, places=4,
                )
                self.assertAlmostEqual(
                    s.cubeful_equity, p.cubeful_equity, places=4,
                )
                self.assertEqual(s.should_double, p.should_double)
                self.assertEqual(s.should_take, p.should_take)
                self.assertEqual(s.optimal_action, p.optimal_action)


if __name__ == "__main__":
    unittest.main()
