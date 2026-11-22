# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Build/check the fixed checker and cube/post-move timing position sets.

Generation uses only the standard library. Optional engine verification checks
classification, legal moves and every public analytics call at 2-ply.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "multi_ply_benchmark"
SOURCES = {
    "money": ROOT / "data" / "money_benchmark" / "benchmark.json.gz",
    "match": ROOT / "data" / "match_benchmark" / "5pt" / "benchmark.json.gz",
}
PLANS = ("attacking", "priming", "anchoring", "racing", "purerace")
SEED = 20261121
PER_CELL = 10
FLIP_OWNER = {"centered": "centered", "player": "opponent", "opponent": "player"}


def flip_board(board: list[int]) -> list[int]:
    return [board[25], *(-value for value in board[24:0:-1]), board[0]]


def call_inputs(decision: dict, mode: str) -> dict:
    inputs = {
        "board": decision["board"],
        "cube_value": decision["cube_value"],
        "cube_owner": decision["cube_owner"],
        "away1": decision["away1"] if mode == "match" else 0,
        "away2": decision["away2"] if mode == "match" else 0,
        "is_crawford": decision["is_crawford"] if mode == "match" else False,
        "jacoby": mode == "money",
        "beaver": mode == "money",
        "max_cube_value": 0,
    }
    if decision["kind"] == "checker":
        inputs["die1"], inputs["die2"] = decision["dice"]
    return inputs


def post_move_inputs(inputs: dict) -> dict:
    # The previous mover is the opponent of the source's player on roll.
    # post_move_analytics has no beaver or max_cube_value parameters.
    return {
        "board": flip_board(inputs["board"]),
        "cube_value": inputs["cube_value"],
        "cube_owner": FLIP_OWNER[inputs["cube_owner"]],
        "away1": inputs["away2"],
        "away2": inputs["away1"],
        "is_crawford": inputs["is_crawford"],
        "jacoby": inputs["jacoby"],
    }


def sample_rank(mode: str, kind: str, key: str) -> str:
    return hashlib.sha256(f"{SEED}:{mode}:{kind}:{key}".encode("ascii")).hexdigest()


def build_sets() -> dict[str, dict]:
    sources, metadata = {}, {}
    for mode, path in SOURCES.items():
        raw = path.read_bytes()
        source = json.loads(gzip.decompress(raw))
        # Keep only call inputs and provenance between files. The candidate
        # analytics account for most of the decompressed source size.
        fields = ("kind", "key", "tier", "board", "dice", "cube_value", "cube_owner",
                  "game_plan", "seed", "turn", "player")
        if mode == "match":
            fields += ("away1", "away2", "is_crawford", "game_number")
        sources[mode] = [{key: decision[key] for key in fields} for decision in source["decisions"]]
        metadata[mode] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "meta": source["meta"],
        }
        del source, raw

    sets = {}
    for kind, filename in (("checker", "checker_play.json"), ("cube", "cube_action.json")):
        positions, counts, seen_boards = [], {}, set()
        for mode, decisions in sources.items():
            counts[mode] = {}
            for plan in PLANS:
                pool = [(index, d) for index, d in enumerate(decisions)
                        if d["kind"] == kind and d["game_plan"] == plan]
                counts[mode][plan] = len(pool)
                pool.sort(key=lambda item: (sample_rank(mode, kind, item[1]["key"]),
                                            item[1]["key"]))
                selected = 0
                for index, decision in pool:
                    board_key = tuple(decision["board"])
                    if board_key in seen_boards:
                        continue
                    seen_boards.add(board_key)
                    inputs = call_inputs(decision, mode)
                    record = {
                        "id": f"{kind}-{mode}-{plan}-{selected + 1:02d}",
                        "mode": mode,
                        "game_plan": plan,
                        "source": {
                            "decision_index": index,
                            **{key: decision[key] for key in
                               ("key", "tier", "seed", "turn", "player", "game_number")
                               if key in decision},
                        },
                        "inputs": inputs,
                    }
                    if kind == "cube":
                        record["post_move_inputs"] = post_move_inputs(inputs)
                    positions.append(record)
                    selected += 1
                    if selected == PER_CELL:
                        break
                if selected != PER_CELL:
                    raise ValueError(f"Not enough distinct boards for {kind}/{mode}/{plan}")
        sets[filename] = {
            "schema_version": 1,
            "kind": kind,
            "selection": {
                "seed": SEED,
                "per_mode_game_plan": PER_CELL,
                "reference_tiers": ["3P", "3T", "rollout"],
                "method": "Ascending SHA-256(seed:mode:kind:source_key); skip duplicate boards",
                "game_plan_perspective": "source player on roll (before checker move)",
                "eligible_counts": counts,
            },
            "sources": metadata,
            "positions": positions,
        }
    return sets


def validate_set(dataset: dict) -> None:
    positions = dataset["positions"]
    expected = Counter({(mode, plan): PER_CELL for mode in SOURCES for plan in PLANS})
    if Counter((p["mode"], p["game_plan"]) for p in positions) != expected:
        raise ValueError("Incorrect source/game-plan balance")
    if len({tuple(p["inputs"]["board"]) for p in positions}) != len(positions):
        raise ValueError("Duplicate board in position set")
    for position in positions:
        inputs = position["inputs"]
        board = inputs["board"]
        if (len(board) != 26 or any(type(v) is not int for v in board)
                or board[0] < 0 or board[25] < 0
                or not 0 < board[25] + sum(max(v, 0) for v in board[1:25]) <= 15
                or not 0 < board[0] + sum(max(-v, 0) for v in board[1:25]) <= 15):
            raise ValueError(f"Invalid board: {position['id']}")
        cv = inputs["cube_value"]
        if type(cv) is not int or cv < 1 or cv & (cv - 1):
            raise ValueError(f"Invalid cube value: {position['id']}")
        if inputs["cube_owner"] not in FLIP_OWNER:
            raise ValueError(f"Invalid cube owner: {position['id']}")
        if position["source"]["tier"] not in dataset["selection"]["reference_tiers"]:
            raise ValueError(f"Unknown source tier: {position['id']}")
        if dataset["kind"] == "checker":
            if not all(type(inputs[k]) is int and 1 <= inputs[k] <= 6
                       for k in ("die1", "die2")):
                raise ValueError(f"Invalid dice: {position['id']}")
        elif position["post_move_inputs"] != post_move_inputs(inputs):
            raise ValueError(f"Incorrect post-move perspective: {position['id']}")


def verify_engine(sets: dict[str, dict], extension: Path) -> None:
    extension = extension.resolve()
    if not extension.is_relative_to(ROOT):
        raise ValueError("The compiled extension must be inside the bgsage repository")
    spec = importlib.util.spec_from_file_location("bgbot_cpp", extension)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load engine extension: {extension}")
    cpp = importlib.util.module_from_spec(spec)
    sys.modules["bgbot_cpp"] = cpp
    spec.loader.exec_module(cpp)
    sys.path.insert(0, str(ROOT / "python"))
    from bgsage import BgBotAnalyzer, classify_game_plan, flip_board as engine_flip, possible_moves
    from bgsage.weights import PRODUCTION_MODEL

    analyzer = BgBotAnalyzer(eval_level="2ply", cubeful=True, parallel_threads=1)
    calls = Counter()
    for dataset in sets.values():
        for position in dataset["positions"]:
            inputs = position["inputs"]
            if classify_game_plan(inputs["board"]) != position["game_plan"]:
                raise ValueError(f"Game-plan classification changed: {position['id']}")
            if dataset["kind"] == "checker":
                legal = {tuple(b) for b in possible_moves(inputs["board"], inputs["die1"], inputs["die2"])}
                result = analyzer.checker_play(**inputs)
                if len(legal) < 2 or {tuple(m.board) for m in result.moves} != legal:
                    raise ValueError(f"Legal move mismatch: {position['id']}")
                values = [m.equity for m in result.moves]
                calls["checker_play"] += 1
            else:
                result = analyzer.cube_action(**inputs)
                post = position["post_move_inputs"]
                if engine_flip(post["board"]) != inputs["board"]:
                    raise ValueError(f"Board flip mismatch: {position['id']}")
                post_result = analyzer.post_move_analytics(**post)
                values = [result.equity_nd, result.equity_dt, result.equity_dp,
                          result.cubeless_equity, post_result.cubeless_equity,
                          post_result.cubeful_equity]
                calls["cube_action"] += 1
                calls["post_move_analytics"] += 1
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite analytics: {position['id']}")
        print(f"Verified {dataset['kind']} classification and 2-ply calls", flush=True)
    print(f"Engine verification passed: model={PRODUCTION_MODEL}, calls={dict(calls)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Regenerate in memory and verify saved sets byte for byte; do not write")
    parser.add_argument("--verify-engine", type=Path, metavar="EXTENSION",
                        help="Also verify all 300 analytics calls with this in-repository bgbot_cpp extension")
    args = parser.parse_args()
    sets = build_sets()
    for filename, dataset in sets.items():
        validate_set(dataset)
        content = (json.dumps(dataset, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        path = OUTPUT / filename
        if args.check:
            if path.read_bytes() != content:
                raise ValueError(f"Saved set differs from deterministic generation: {path}")
        else:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        tiers = Counter(p["source"]["tier"] for p in dataset["positions"])
        print(f"{'Checked' if args.check else 'Wrote'} {filename}: 100 positions, tiers={dict(tiers)}", flush=True)
    if args.verify_engine:
        verify_engine(sets, args.verify_engine)


if __name__ == "__main__":
    main()
