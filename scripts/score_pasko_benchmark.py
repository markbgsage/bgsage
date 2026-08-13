# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Score model benchmark PR against the Paskogammon rollout benchmark.

Reads bgsage/data/pasko-benchmark-rollout (positions + cubeless rolled-out probs)
and, for each model, evaluates every position at 1-ply cubeless and reports the
equity error rate (ER = mean |equity - rollout_equity| * 1000 millipips) and the
Performance Rating (PR = ER / 2) against the rollout targets.

Only finished positions are scored: a "-rollout" line has 26 ints + 5 probs, so
positions still rolling out (absent from the file) or a half-written trailing
line (< 31 fields) are skipped automatically.

Models scored:
  * Stage 9 production model (19-NN BackgameAwarePairStrategy), 1-ply cubeless.
  * Paskogammon TD net (single 400h / 244-input extended-contact net), 1-ply cubeless.

Usage:
    python bgsage/scripts/score_pasko_benchmark.py
    python bgsage/scripts/score_pasko_benchmark.py --pasko-weights models/td_pasko.weights
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BGSAGE_ROOT = os.path.dirname(_SCRIPT_DIR)
_PARENT_ROOT = os.path.dirname(_BGSAGE_ROOT)

if sys.platform == 'win32':
    _cuda = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    for _d in (os.path.join(_PARENT_ROOT, 'build'), os.path.join(_BGSAGE_ROOT, 'build')):
        if os.path.isdir(_d):
            os.add_dll_directory(_d)
# Prefer bgsage/build (kept fresh); parent build/ is a fallback.
sys.path.insert(0, os.path.join(_PARENT_ROOT, 'build'))
sys.path.insert(0, os.path.join(_BGSAGE_ROOT, 'build'))
sys.path.insert(0, os.path.join(_BGSAGE_ROOT, 'python'))

import bgbot_cpp
from bgsage.weights import WeightConfigPair

DATA_DIR = os.path.join(_BGSAGE_ROOT, 'data')
MODELS_DIR = os.path.join(_BGSAGE_ROOT, 'models')
PASKO_HIDDEN = 400
PASKO_INPUTS = 244


def load_rollout(filepath):
    """Boards + cubeless target equities from a *-rollout file.

    Each valid line is 26 ints (board) + 5 rolled-out probs (W, Gw, Bw, Gl, Bl).
    Lines with < 31 fields (unfinished / half-written) are skipped.
    """
    boards, equities = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 31:
                continue
            board = [int(x) for x in parts[:26]]
            p = [float(x) for x in parts[26:31]]
            eq = 2 * p[0] - 1 + p[1] - p[3] + p[2] - p[4]
            boards.append(board)
            equities.append(eq)
    return boards, equities


def score_er(boards, target_equities, strategy):
    """Mean |1-ply cubeless equity - rollout equity| * 1000 (millipips)."""
    total_err = 0.0
    for board, target_eq in zip(boards, target_equities):
        model_eq = strategy.evaluate_board(board, board)["equity"]
        total_err += abs(model_eq - target_eq)
    n = len(boards)
    return (total_err / n) * 1000.0 if n else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Score model benchmark PR vs the Paskogammon rollout benchmark")
    parser.add_argument("--rollout", default="pasko-benchmark-rollout",
                        help="Rollout file name in bgsage/data (default: pasko-benchmark-rollout)")
    parser.add_argument("--pasko-weights", default="models/td_pasko.weights.best",
                        help="Paskogammon net weights, abs or relative to bgsage/ "
                             "(default: models/td_pasko.weights.best)")
    args = parser.parse_args()

    rollout_path = args.rollout if os.path.isabs(args.rollout) else os.path.join(DATA_DIR, args.rollout)
    pasko_path = args.pasko_weights if os.path.isabs(args.pasko_weights) \
        else os.path.join(_BGSAGE_ROOT, args.pasko_weights)

    boards, targets = load_rollout(rollout_path)
    n = len(boards)
    print(f"Benchmark: {os.path.basename(rollout_path)} — {n:,} finished positions scored\n")
    if n == 0:
        print("No finished positions yet.")
        return

    # Stage 9 production model (19-NN), 1-ply cubeless.
    w9 = WeightConfigPair.from_model("stage9")
    w9.validate()
    strat9 = bgbot_cpp.BackgameAwarePairStrategy(w9.paths, w9.hiddens)
    er9 = score_er(boards, targets, strat9)

    # Paskogammon TD net (single 400h / 244-input), 1-ply cubeless.
    stratp = bgbot_cpp.NNStrategy(pasko_path, PASKO_HIDDEN, PASKO_INPUTS)
    erp = score_er(boards, targets, stratp)

    label_p = f"Paskogammon TD ({os.path.basename(pasko_path)})"
    print(f"{'Model (1-ply cubeless)':<40s} {'ER (mpips)':>11s} {'PR':>8s}")
    print(f"{'-'*40} {'-'*11} {'-'*8}")
    print(f"{'Stage 9 (19-NN, production)':<40s} {er9:>11.2f} {er9/2:>8.2f}")
    print(f"{label_p:<40s} {erp:>11.2f} {erp/2:>8.2f}")


if __name__ == "__main__":
    main()
