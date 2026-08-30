#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Stage 11 backgame TD bootstrap: truncated self-play per backgame category.

Stage 11 replaces Stage 9's two backgame NNs (player/opponent) with THREE,
selected by the backgame's category — the same NN whichever side holds it:

    deep      21, 31, 32          (both anchors on the 1/2/3 points)
    middle    41, 42, 51, 52      (one anchor on the 1/2 point, one higher)
    double    43, 53, 54          (two anchors, none deeper than the 3-point)

(3+ anchors: deep when at least two sit on the 1/2/3 points, else middle.)

This script runs the FIRST training step for one of those NNs: truncated TD
self-play (``bgbot_cpp.td_train_backgame_truncated``). Starting from random
small weights, each game

  * starts from one of the category's reference positions
    (``backgame_ref_positions/benchmark/<folder> starting.txt``), cycling
    through them, a coin deciding which side moves first;
  * is played with 1-ply decisions by the training NN;
  * ends the moment a post-move position is no longer in ANY backgame
    category, at which point Stage 9's 3-ply cubeless post-move evaluation of
    that position stands in for the game outcome — the TD chain's terminal
    target, exactly as the 0/1 outcome vector does in ordinary TD. A game that
    genuinely ends inside the region (the opponent bears off through the
    anchors) uses the real outcome as usual.

Progress is scored the way the Paskogammon trainer scores: mean |equity -
target| x 1000 over an equity benchmark — here the S9-rollout backgame
benchmark positions (``data/*-backgame-benchmark-rollout``) that fall in the
training category. The best-scoring weights are kept as ``.weights.best``.

Outputs (in ``models/``): ``td_s11_bg_<cat>.weights`` (latest),
``td_s11_bg_<cat>.weights.best`` (best benchmark ER), and a history CSV.
To try the trained NN inside the full Stage 11 model, promote all three:

    copy models\\td_s11_bg_deep.weights.best   models\\sl_s11_bg_deep.weights.best
    copy models\\td_s11_bg_middle.weights.best models\\sl_s11_bg_middle.weights.best
    copy models\\td_s11_bg_double.weights.best models\\sl_s11_bg_double.weights.best

then ``WeightConfigPair.from_model("stage11")`` loads the 20-NN model (Stage
9's 17 standard NNs + the trio).

Usage (long-running — launch detached per the CLAUDE.md pattern)::

    py -3.14 scripts/run_s11_backgame_td.py --category deep --n-games 200000
    py -3.14 scripts/run_s11_backgame_td.py --category middle --resume
    py -3.14 scripts/run_s11_backgame_td.py --category all --n-games 200000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT / "build", _SCRIPT_DIR):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

if sys.platform == "win32":
    _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    if (_PROJECT_ROOT / "build").is_dir():
        os.add_dll_directory(str(_PROJECT_ROOT / "build"))

import bgbot_cpp  # noqa: E402

from backgame_benchmark import read_start_positions  # noqa: E402
from bgsage.weights import WeightConfigPair  # noqa: E402

#: Which reference-position folders seed each category's games.
CATEGORY_FOLDERS: dict[str, list[str]] = {
    "deep": ["21 backgame", "31 backgame", "32 backgame"],
    "middle": ["41 backgame", "42 backgame", "51 backgame", "52 backgame"],
    "double": ["43 backgame", "53 backgame", "54 backgame"],
}

#: Hidden size of each backgame NN (matches Stage 9's backgame/contact NNs).
N_HIDDEN = 400

_DATA_DIR = _PROJECT_ROOT / "data"
_MODELS_DIR = _PROJECT_ROOT / "models"


def load_seeds(category: str) -> list[list[int]]:
    """The category's seed boards, deduped, with a per-seed category report."""
    boards: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for folder in CATEGORY_FOLDERS[category]:
        for pos in read_start_positions(folder):
            if pos.board in seen:
                continue
            seen.add(pos.board)
            boards.append(list(pos.board))

    counts: dict[str, int] = {}
    for b in boards:
        cat = bgbot_cpp.backgame_category(b)
        counts[cat] = counts.get(cat, 0) + 1
    print(f"  {len(boards)} seed positions; category at the seed itself: {counts}")
    if counts.get(category, 0) == 0:
        print("  WARNING: no seed classifies as the training category — "
              "check the folders/filters.")
    return boards


def load_benchmark(category: str, limit: int) -> tuple[list[list[int]], list[float]]:
    """S9-rollout backgame benchmark rows falling in this category.

    ``data/{player,opponent}-backgame-benchmark-rollout``: 26 board ints + 5
    cubeless probs per line, post-move, positive-player perspective. The
    target equity is computed from the probs; rows are filtered by
    ``backgame_category`` so each NN is scored on its own region.
    """
    boards: list[list[int]] = []
    targets: list[float] = []
    for side in ("player", "opponent"):
        path = _DATA_DIR / f"{side}-backgame-benchmark-rollout"
        if not path.exists():
            print(f"  (no {path.name} — skipping)")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 31:
                continue
            board = [int(x) for x in parts[:26]]
            if bgbot_cpp.backgame_category(board) != category:
                continue
            w, gw, bw, gl, bl = (float(x) for x in parts[26:31])
            boards.append(board)
            targets.append(2 * w - 1 + gw - gl + bw - bl)
            if len(boards) >= limit:
                return boards, targets
    return boards, targets


def train_category(category: str, args: argparse.Namespace) -> None:
    print(f"=== {category} ===")
    seeds = load_seeds(category)
    bench_boards, bench_targets = load_benchmark(category, args.bench_limit)
    print(f"  {len(bench_boards)} benchmark rows in the {category} region")

    ref = WeightConfigPair.from_model("stage9")
    ref.validate()

    model_name = f"td_s11_bg_{category}"
    resume_from = ""
    if args.resume:
        existing = _MODELS_DIR / f"{model_name}.weights"
        if existing.exists():
            resume_from = str(existing)
            print(f"  resuming from {existing.name}")
        else:
            print(f"  --resume: no {existing.name} yet, starting fresh")

    t0 = time.perf_counter()
    result = bgbot_cpp.td_train_backgame_truncated(
        n_games=args.n_games,
        alpha=args.alpha,
        n_hidden=N_HIDDEN,
        eps=args.eps,
        seed=args.seed,
        benchmark_interval=args.benchmark_interval,
        model_name=model_name,
        models_dir=str(_MODELS_DIR),
        resume_from=resume_from,
        start_boards=seeds,
        randomize_first_mover=not args.no_randomize_first_mover,
        max_half_moves=args.max_half_moves,
        anchor_boundary=args.anchor_boundary,
        ref_weight_paths=ref.paths,
        ref_hidden_sizes=ref.hiddens,
        ref_plies=args.ref_plies,
        ref_threads=args.ref_threads,
        bench_boards=bench_boards,
        bench_targets=bench_targets,
    )
    minutes = (time.perf_counter() - t0) / 60
    print(f"  {result.games_played} games in {minutes:.1f} min -> "
          f"models/{model_name}.weights(.best)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True,
                        choices=[*CATEGORY_FOLDERS, "all"],
                        help="Which backgame NN to train ('all' = the three in turn)")
    parser.add_argument("--n-games", type=int, default=200_000,
                        help="TD games (default 200k, the standard phase-1 count)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="TD learning rate (default 0.1, the phase-1 rate)")
    parser.add_argument("--eps", type=float, default=0.1,
                        help="Random-init weight scale for a fresh NN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-interval", type=int, default=5000)
    parser.add_argument("--bench-limit", type=int, default=2000,
                        help="Benchmark rows scored per interval (default 2000)")
    parser.add_argument("--ref-plies", type=int, default=3,
                        help="Reference (Stage 9) eval depth at truncation")
    parser.add_argument("--ref-threads", type=int, default=0,
                        help="Threads for the reference eval (0 = auto)")
    parser.add_argument("--max-half-moves", type=int, default=2000)
    parser.add_argument("--no-randomize-first-mover", action="store_true",
                        help="Always start a game with the seed's own side to "
                             "move. By default a coin decides, so the NN sees "
                             "each seed from both perspectives.")
    parser.add_argument("--anchor-boundary", action="store_true",
                        help="Also train each game's exit position itself "
                             "toward its Stage 9 eval. Without this the flee "
                             "equilibrium develops (see CLAUDE.md Stage 11).")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from models/td_s11_bg_<cat>.weights")
    args = parser.parse_args()

    categories = list(CATEGORY_FOLDERS) if args.category == "all" else [args.category]
    for category in categories:
        train_category(category, args)


if __name__ == "__main__":
    main()
