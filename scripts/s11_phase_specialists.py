#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Phase-specialist test for the Stage 11 "route by phase first" proposal.

The remaining S11 error is phase-driven, not anchor-driven (2026-09-02
research report): PR climbs as the racer bears in and off, and the early
containment phase is where the holder's play is worst. Before building phase
routing into the C++ selector, test whether phase-specialist nets beat the
installed category trio on their own phase slices — the anchor-pair
specialists did NOT, so this is a real gate, not a formality.

Phases (B = the back-game holder, R = the other side; all boards
mover-positive, both orientations resolved):

    stragglers = R on the bar + R checkers in B's home board
    P3 early containment : stragglers >= 1 and R_off <= 2
    P4 late containment  : stragglers >= 1 and R_off >= 3   (not trained: no data)
    P2 bear-in / bear-off: stragglers == 0 and (R_home >= 10 or R_off >= 1)
    P1 waiting           : everything else

Training rows: the three category train piles plus the 90% split of the
three exit piles, filtered by phase; holdout: the category benchmark piles
plus the 10% exit split. Each specialist starts from the S9-init deep net
(the installed deep net) and trains 30k epochs at alpha 1.0 with best-ER
checkpointing on its phase holdout.

Scoring: every checker decision of the ten folder benchmarks whose pre-move
board is in-region is classified by phase; P2 / P3 decisions are played at
1-ply by (a) the installed trio and (b) a trio with the specialist in all
three slots, and PR is compared per slice against the rollout references.
Out-of-region containment (plan pair flipped) cannot be routed to a category
net without C++ changes and is excluded here.

Usage (GPU; ~1 h):
    py -3.14 scripts/s11_phase_specialists.py
    py -3.14 scripts/s11_phase_specialists.py --score-only
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT / "build", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if sys.platform == "win32":
    _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    os.add_dll_directory(str(_PROJECT_ROOT / "build"))

import bgbot_cpp  # noqa: E402
import numpy as np  # noqa: E402

_DATA = _PROJECT_ROOT / "data"
_MODELS = _PROJECT_ROOT / "models"
_DIAG = _MODELS / "s11_diag"
_BENCH = _PROJECT_ROOT / "backgame_ref_positions" / "benchmark"
FOLDERS = ["21 backgame", "31 backgame", "32 backgame", "41 backgame",
           "42 backgame", "51 backgame", "52 backgame", "43 backgame",
           "53 backgame", "54 backgame"]
BASE = _DIAG / "sl_deep_s9init_best.weights"
EPOCHS, ALPHA, CHUNK, BATCH = 30_000, 1.0, 2500, 4096


# ---------------------------------------------------------------------------
# Phase features
# ---------------------------------------------------------------------------
def _pips(board, p1: bool) -> int:
    if p1:
        return sum(board[i] * i for i in range(1, 26) if board[i] > 0)
    return sum(-board[i] * (25 - i) for i in range(1, 25) if board[i] < 0) + board[0] * 25


def holder_is_p1(board) -> bool | None:
    """Which side plays the back game. None when neither side qualifies."""
    a1 = sum(1 for i in range(19, 25) if board[i] >= 2)
    a2 = sum(1 for i in range(1, 7) if board[i] <= -2)
    p1, p2 = _pips(board, True), _pips(board, False)
    if a1 >= 2 and p1 > p2:
        return True
    if a2 >= 2 and p2 > p1:
        return False
    if a1 >= 1 and p1 > p2 and a1 >= a2:
        return True
    if a2 >= 1 and p2 > p1:
        return False
    return None


def phase(board) -> str | None:
    hp1 = holder_is_p1(board)
    if hp1 is None:
        return None
    if hp1:
        r_bar = board[0]
        r_in_b_home = sum(-board[i] for i in range(1, 7) if board[i] < 0)
        r_on_board = sum(-board[i] for i in range(1, 25) if board[i] < 0) + board[0]
        r_home = sum(-board[i] for i in range(19, 25) if board[i] < 0)
    else:
        r_bar = board[25]
        r_in_b_home = sum(board[i] for i in range(19, 25) if board[i] > 0)
        r_on_board = sum(board[i] for i in range(1, 26) if board[i] > 0)
        r_home = sum(board[i] for i in range(1, 7) if board[i] > 0)
    r_off = 15 - r_on_board
    stragglers = r_bar + r_in_b_home
    if stragglers >= 1:
        return "P3" if r_off <= 2 else "P4"
    if r_home >= 10 or r_off >= 1:
        return "P2"
    return "P1"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _rows(path: Path):
    for line in path.open(encoding="utf-8"):
        parts = line.split()
        if len(parts) >= 31:
            yield [int(x) for x in parts[:26]], [float(x) for x in parts[26:31]]


def _exit_holdout(board) -> bool:
    key = " ".join(str(x) for x in board).encode()
    return hashlib.md5(key).digest()[0] % 10 == 0


def build_sets(target: str):
    train_b, train_p, hold_b, hold_p = [], [], [], []
    for cat in ("deep", "middle", "double"):
        for b, p in _rows(_DATA / f"s11-bg-{cat}-train-rollout"):
            if phase(b) == target:
                train_b.append(b); train_p.append(p)
        for b, p in _rows(_DATA / f"s11-bg-{cat}-benchmark-rollout"):
            if phase(b) == target:
                hold_b.append(b); hold_p.append(p)
        for b, p in _rows(_DATA / f"s11-bg-{cat}-exit-rollout"):
            if phase(b) != target:
                continue
            if _exit_holdout(b):
                hold_b.append(b); hold_p.append(p)
            else:
                train_b.append(b); train_p.append(p)
    return (train_b, np.array(train_p, dtype=np.float32),
            hold_b, np.array(hold_p, dtype=np.float32))


def _eq(p: np.ndarray) -> np.ndarray:
    return 2 * p[:, 0] - 1 + p[:, 1] - p[:, 3] + p[:, 2] - p[:, 4]


def er(boards, eq: np.ndarray, weights_path: str) -> float:
    nn = bgbot_cpp.NNStrategy(weights_path, 400, 244)
    tot = 0.0
    for b, e in zip(boards, eq):
        tot += abs(nn.evaluate_board(b, b)["equity"] - float(e))
    return tot / len(boards) * 1000.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_specialist(target: str) -> Path:
    print(f"=== specialist {target} ===", flush=True)
    tb, tp, hb, hp = build_sets(target)
    heq = _eq(hp)
    print(f"  {len(tb)} train rows, {len(hb)} holdout rows", flush=True)
    wpath = str(_DIAG / f"phase_{target}.weights")
    best = wpath + ".best"
    shutil.copy2(BASE, wpath)
    shutil.copy2(BASE, best)
    inputs = bgbot_cpp.encode_boards_batch(np.array(tb, dtype=np.int32), 244)
    best_er = er(hb, heq, best)
    print(f"  base ER on {target} holdout: {best_er:.2f}", flush=True)
    done, t0 = 0, time.time()
    while done < EPOCHS:
        chunk = min(CHUNK, EPOCHS - done)
        done += chunk
        bgbot_cpp.cuda_supervised_train_preencoded(
            inputs=inputs, targets=tp, weights_path=wpath, n_hidden=400,
            n_inputs=244, alpha=ALPHA, epochs=chunk, batch_size=BATCH,
            seed=777 + done, print_interval=chunk + 1, save_path=wpath)
        e = er(hb, heq, wpath)
        gc.collect()
        tag = ""
        if e < best_er:
            best_er = e
            shutil.copy2(wpath, best)
            tag = "  *BEST*"
        print(f"  {done:6d}/{EPOCHS}  ER={e:.2f}  best={best_er:.2f}  "
              f"{time.time() - t0:.0f}s{tag}", flush=True)
    print(f"  {target}: best ER {best_er:.2f} -> {os.path.basename(best)}\n", flush=True)
    return Path(best)


# ---------------------------------------------------------------------------
# Scoring: phase slices of the ten folders, installed trio vs specialist trio
# ---------------------------------------------------------------------------
def score(specialists: dict[str, Path]) -> None:
    from bgsage import BgBotAnalyzer
    from bgsage import weights as W
    from bgsage.weights import WeightConfigPair
    from benchmark_money import BLUNDER_THRESHOLD

    def trio(name: str, deep_file: str) -> WeightConfigPair:
        # The 20-NN categorized trio (the registry's stage11 is now the full
        # 24-NN phased layout, so the trio is built here explicitly).
        e = dict(W.MODELS["stage11"])
        e["hidden"] = (100,) + (400,) * 19
        e["plans"] = "backgame_pair_categorized"
        e["extra_backgame"] = [deep_file] * 3
        W.MODELS[name] = e
        return WeightConfigPair.from_model(name)

    analyzers = {"installed": BgBotAnalyzer(weights=WeightConfigPair.from_model("stage11"),
                                            eval_level="1ply", cubeful=True, parallel_threads=0)}
    for ph, path in specialists.items():
        analyzers[ph] = BgBotAnalyzer(
            weights=trio(f"phase_{ph}", f"s11_diag/{path.name}"),
            eval_level="1ply", cubeful=True, parallel_threads=0)

    stats: dict = {}
    for cat in FOLDERS:
        for line in (_BENCH / f"{cat} rollout.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r["kind"] != "checker":
                continue
            if bgbot_cpp.backgame_category(r["board"]) == "none":
                continue
            ph = phase(r["board"])
            if ph not in specialists:
                continue
            ref = {tuple(m["board"]): m["equity"] for m in r["moves"]}
            best_eq = r["moves"][0]["equity"]
            for who in ("installed", ph):
                res = analyzers[who].checker_play(
                    r["board"], *r["dice"], cube_value=r["cube_value"],
                    cube_owner=r["cube_owner"], jacoby=True, beaver=True)
                if not res.moves:
                    continue
                chosen = tuple(res.moves[0].board)
                err = max(0.0, best_eq - ref.get(chosen, best_eq))
                s = stats.setdefault((ph, who), {"n": 0, "err": 0.0, "bl": 0})
                s["n"] += 1
                s["err"] += err
                s["bl"] += err > BLUNDER_THRESHOLD
    print(f"\n{'phase':>6} {'model':>11} {'n':>6} {'PR':>7} {'blunders':>9}")
    for (ph, who), s in sorted(stats.items()):
        print(f"{ph:>6} {who:>11} {s['n']:6d} {s['err'] / s['n'] * 500:7.2f} {s['bl']:9d}")
    out = _DIAG / "phase_specialists_scores.json"
    out.write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in stats.items()}, indent=2),
                   encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    specialists = {}
    for target in ("P2", "P3"):
        best = _DIAG / f"phase_{target}.weights.best"
        if not args.score_only:
            best = train_specialist(target)
        specialists[target] = best
    score(specialists)


if __name__ == "__main__":
    main()
