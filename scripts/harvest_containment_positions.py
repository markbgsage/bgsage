#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Harvest containment-game positions for rollout (training targets).

The piles hold ~1% containment rows, so the containment NN needs its own
targets. Positions come from self-play games seeded from the containment
seed file (the same seeds the benchmark used, but different games: a
different master seed, and every board that appears among the benchmark's
decisions or their candidate moves — both orientations — is excluded), plus
the existing pile rows that satisfy the rule. Games are played 2-ply by
stage11 in N worker processes; every recorded decision whose pre-move board
is a containment position becomes one training position — the board
FLIPPED, i.e. in the post-move convention the rollout runner and the SL
trainer use (the side that just moved is positive).

Output: data/s11-bg-containment-data (26 ints per line), for
    python scripts/rollout_backgame_positions.py s11-bg-containment-data ...
(parent repo), which appends the rolled-out probabilities as
data/s11-bg-containment-rollout.

Usage:
    py -3.14 scripts/harvest_containment_positions.py --games 3000 --workers 6
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
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

_DATA = _PROJECT_ROOT / "data"
_BENCH = _PROJECT_ROOT / "backgame_ref_positions" / "benchmark"
MASTER_SEED = 7          # the benchmark folder used seed 1
LEVEL = "2ply"
MODEL = "stage11"


def _flip(b):
    return [b[25]] + [-b[25 - i] for i in range(1, 25)] + [b[0]]


def excluded_boards() -> set:
    """Benchmark decision + candidate boards, both orientations."""
    ex = set()
    for path in _BENCH.glob("* rollout.jsonl"):
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            for b in [r["board"], *[m["board"] for m in r.get("moves", [])]]:
                ex.add(tuple(b))
                ex.add(tuple(_flip(b)))
    for name in ("money_benchmark", "pasko_money_benchmark"):
        for d in json.load((_DATA / name / "benchmark.json").open(encoding="utf-8"))["decisions"]:
            ex.add(tuple(d["board"]))
            ex.add(tuple(_flip(d["board"])))
    return ex


def play_worker(args):
    """Play games [start, start+count) and return containment positions."""
    start, count = args
    from backgame_benchmark import _play_game, _start_for_game, read_start_positions
    from bgsage import BgBotAnalyzer
    from bgsage.weights import WeightConfig
    import containment_rule as cr

    starts = read_start_positions("containment")
    analyzer = BgBotAnalyzer(weights=WeightConfig.from_model(MODEL),
                             eval_level=LEVEL, cubeful=True, parallel_threads=4)
    found: list[tuple[int, ...]] = []

    def record(decision) -> bool:
        if cr.containment(decision.board):
            found.append(tuple(_flip(list(decision.board))))
        return True

    for g in range(start, start + count):
        st = _start_for_game(starts, g, MASTER_SEED, "containment")
        rng = random.Random(f"{MASTER_SEED}:containment:game{g}")
        _play_game(st, rng, analyzer, cr.containment, record)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=45_000)
    args = parser.parse_args()

    import containment_rule as cr

    ex = excluded_boards()
    print(f"exclusion set: {len(ex)} boards", flush=True)

    # 1. existing pile rows (already in post-move convention)
    seen: set = set()
    pile: list[tuple[int, ...]] = []
    for path in sorted(glob.glob(str(_DATA / "*-rollout"))):
        for line in open(path, encoding="utf-8"):
            p = line.split()
            if len(p) < 31:
                continue
            b = tuple(int(x) for x in p[:26])
            if b in seen or b in ex or not cr.containment(list(b)):
                continue
            seen.add(b)
            pile.append(b)
    print(f"pile rows matching the rule: {len(pile)}", flush=True)

    # 2. self-play harvest
    per = args.games // args.workers
    jobs = [(i * per, per) for i in range(args.workers)]
    fresh: list[tuple[int, ...]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex_pool:
        for res in ex_pool.map(play_worker, jobs):
            fresh.extend(res)
    n_raw = len(fresh)
    fresh = [b for b in dict.fromkeys(fresh) if b not in seen and b not in ex]
    print(f"self-play: {n_raw} containment decisions -> {len(fresh)} new distinct boards",
          flush=True)

    rng = random.Random(MASTER_SEED)
    rng.shuffle(fresh)
    out = pile + fresh
    out = out[:args.limit]
    path = _DATA / "s11-bg-containment-data"
    with path.open("w", encoding="utf-8") as f:
        for b in out:
            f.write(" ".join(str(x) for x in b) + "\n")
    print(f"wrote {len(out)} positions -> {path.name} ({len(pile)} pile + "
          f"{len(out) - len(pile)} self-play)", flush=True)


if __name__ == "__main__":
    main()
