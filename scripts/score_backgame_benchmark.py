"""Score S9 back game benchmark ER against S9 rollout targets.

Evaluates each position with the full 19-NN BackgameAwarePairStrategy at 1-ply,
compares cubeless equity against the rollout target equity.

Usage:
    python bgsage/scripts/score_backgame_benchmark.py
    python bgsage/scripts/score_backgame_benchmark.py --side player
    python bgsage/scripts/score_backgame_benchmark.py --side opponent
"""

import argparse
import os
import sys

sys.path.insert(0, "build")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64")
import bgbot_cpp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from bgsage.weights import WeightConfigPair


def load_rollout(filepath):
    """Load positions + rollout equities from a rollout file."""
    boards, equities = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 31:
                continue
            board = [int(x) for x in parts[:26]]
            probs = [float(x) for x in parts[26:31]]
            eq = 2 * probs[0] - 1 + probs[1] - probs[3] + probs[2] - probs[4]
            boards.append(board)
            equities.append(eq)
    return boards, equities


def score_er(boards, target_equities, strategy):
    """Compute ER (mean |equity error| * 1000) using BackgameAwarePairStrategy."""
    total_err = 0.0
    for board, target_eq in zip(boards, target_equities):
        result = strategy.evaluate_board(board, board)
        model_eq = result["equity"]
        total_err += abs(model_eq - target_eq)
    return (total_err / len(boards)) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["player", "opponent", "both"], default="both")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    w = WeightConfigPair.from_model("stage9")
    w.validate()
    strategy = bgbot_cpp.BackgameAwarePairStrategy(w.paths, w.hiddens)

    sides = ["player", "opponent"] if args.side == "both" else [args.side]

    for side in sides:
        bench_file = os.path.join(data_dir, f"{side}-backgame-benchmark-rollout")
        boards, equities = load_rollout(bench_file)
        er = score_er(boards, equities, strategy)
        print(f"{side:>8s} back game ER (1-ply S9 vs S9 rollout): {er:.2f}  ({len(boards)} positions)")


if __name__ == "__main__":
    main()
