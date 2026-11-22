"""Experiment only: time two-ply screening, then re-evaluate close moves at three plies.

Run separately for exact and shortlist to give each arm a fresh process.
Both use the same checked-out Python and chosen native binary. This driver
requires the direct cube-aware checker path in the optimized analyzer.
"""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / "data/multi_ply_benchmark"
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import load_engine


def order(move):
    return -move.equity, -move.cubeless_equity, tuple(move.board)


def screened(engine2, engine3, cpp, inputs):
    preliminary = engine2.checker_play(**inputs)
    best = preliminary.moves[0].equity
    live = [m for i, m in enumerate(preliminary.moves) if i < 2 or m.equity >= best - .08]
    initial = len(live)
    by_board = {tuple(m.board): m for m in preliminary.moves}
    deep = set()
    while live:
        endpoints = [m.board for m in live]
        # Only intercept root move enumeration. The public analyzer performs
        # its normal cube-aware tree evaluation for each selected endpoint.
        with patch.object(cpp, "possible_moves", return_value=endpoints):
            result = engine3.checker_play(**inputs)
        if {tuple(m.board) for m in result.moves} != {tuple(b) for b in endpoints}:
            raise RuntimeError("Root shortlist differs from evaluated candidates")
        for m in result.moves:
            key = tuple(m.board)
            deep.add(key)
            by_board[key] = m
        ranked = sorted(by_board.values(), key=order)
        live = [m for m in ranked[:2] if tuple(m.board) not in deep]
    for m in ranked:
        m.equity_diff = m.equity - ranked[0].equity
    preliminary.moves = ranked
    preliminary.eval_level = "3-ply"
    return preliminary, initial, len(deep)-initial


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", choices=["exact", "shortlist"])
    p.add_argument("extension", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--affinity", type=lambda v: int(v, 0))
    args = p.parse_args()
    if args.affinity:
        import psutil
        psutil.Process().cpu_affinity([i for i in range(args.affinity.bit_length()) if args.affinity & (1 << i)])
    path = args.output.resolve()
    if not path.is_relative_to(ROOT) or path.exists():
        raise ValueError("Choose a new in-repository output")
    path.parent.mkdir(parents=True, exist_ok=True)
    engine, extension = load_engine(args.extension)
    import bgbot_cpp
    a2 = engine.BgBotAnalyzer(eval_level="2ply", parallel_threads=2)
    a3 = engine.BgBotAnalyzer(eval_level="3ply", parallel_threads=2)
    positions=json.loads((DATA/"checker_play.json").read_bytes())["positions"]
    with gzip.open(path, "wt", encoding="utf-8") as out:
        for pos in positions:
            started=perf_counter()
            if args.arm == "exact":
                result=a3.checker_play(**pos["inputs"])
                initial, extra=len(result.moves), 0
            else:
                result, initial, extra = screened(a2, a3, bgbot_cpp, pos["inputs"])
            elapsed=perf_counter()-started
            row={"id":pos["id"],"mode":pos["mode"],"source_game_plan":pos["game_plan"],
                 "seconds":elapsed,"initial_deep":initial,"promotions":extra,"analytics":asdict(result)}
            out.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
    print(args.arm, len(positions), path.name, hashlib.sha256(extension.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
