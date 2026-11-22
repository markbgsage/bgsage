"""Test Stage 11 post-move equity on disjoint historical rollout games."""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import psutil
from experiment import DATA, ROOT, load_engine, score, sha, tree_method
from build_multi_ply_benchmark import PLANS, SOURCES, call_inputs, flip_board, post_move_inputs


def sample():
    seen_boards, primary_games, positions, counts, sources = set(), set(), [], {}, {}
    for dataset in ("checker_play", "cube_action"):
        for row in json.loads((DATA / f"{dataset}.json").read_bytes())["positions"]:
            board = row["inputs"]["board"]
            seen_boards.update((tuple(board), tuple(flip_board(board))))
            primary_games.add((row["mode"], row["source"]["seed"]))
    for mode, path in SOURCES.items():
        data = path.read_bytes()
        source = json.loads(gzip.decompress(data))
        sources[mode] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(data),
                         "meta": source["meta"]}
        for plan in PLANS:
            eligible = [r for r in source["decisions"]
                        if r["kind"] == "cube" and r["tier"] == "rollout" and
                        r["game_plan"] == plan and (mode, r["seed"]) not in primary_games
                        and tuple(r["board"]) not in seen_boards]
            ranked = sorted(eligible, key=lambda r: sha(
                f'post_move_tree_heldout_v1:{mode}:{plan}:{r["key"]}'.encode("ascii")))
            counts[f"{mode}/{plan}"] = len(ranked)
            n = 0
            for row in ranked:
                if tuple(row["board"]) in seen_boards:
                    continue
                assert row["eval_level"] == "Rollout" and row["tier"] == "rollout"
                seen_boards.update((tuple(row["board"]), tuple(flip_board(row["board"]))))
                ref = row["equity_nd"] if not row["should_double_ref"] else (
                    row["equity_dt"] if row["should_take_ref"] else row["equity_dp"])
                action = ("No Double" if not row["should_double_ref"] else "Double/Beaver"
                          if row["is_beaver"] else "Double/Take" if row["should_take_ref"]
                          else "Double/Pass")
                assert action == row["optimal_action"]
                p = row["probs"]
                previous_probs = dict(zip(
                    ("win", "gammon_win", "backgammon_win", "gammon_loss", "backgammon_loss"),
                    (1 - p[0], p[3], p[4], p[1], p[2]), strict=True))
                positions.append({"id": row["key"], "mode": mode, "game_plan": plan,
                    "seed": row["seed"], "tier": row["tier"], "action": action,
                    "inputs": post_move_inputs(call_inputs(row, mode)), "truth": {
                        "probs": previous_probs, "cubeless_equity": -(2*p[0]-1+p[1]-p[3]+p[2]-p[4]),
                        "cubeful_equity": -ref}})
                n += 1
                if n == 10:
                    break
        del source
    return positions, counts, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT) or not 1 <= len(args.cpus) <= 2:
        raise ValueError("Use an in-repo output directory and one or two CPUs")
    args.output.mkdir(parents=True, exist_ok=False)
    psutil.Process().cpu_affinity(args.cpus)
    positions, counts, sources = sample()
    engine, extension = load_engine(args.extension)
    a = engine.BgBotAnalyzer(weights=engine.WeightConfig.from_model("stage11"), eval_level="3ply",
                             cubeful=True, parallel_threads=len(args.cpus),
                             bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
    assert a._bearoff_db is not None
    direct = tree_method(a, bounded=False)
    bounded = tree_method(a, bounded=True)
    for row in positions:
        for arm, fn in (("public", a.post_move_analytics), ("tree", direct), ("bounded", bounded)):
            row[arm] = asdict(fn(**row["inputs"]))
            assert row[arm]["probs"] == row["public"]["probs"]
            assert row[arm]["cubeless_equity"] == row["public"]["cubeless_equity"]
    def equity(rows):
        return {arm: score(rows, "cubeful_equity", arm) for arm in ("tree", "bounded")}
    results = {"positions": len(positions), "selected_by_stratum":
               dict(Counter(f'{r["mode"]}/{r["game_plan"]}' for r in positions)),
               "eligible_by_stratum_before_final_dedup": counts, "sources": sources,
               "heldout_games": len({(r["mode"], r["seed"]) for r in positions}),
               "extension_sha256": sha(extension.read_bytes()),
               "analyzer_sha256": sha((ROOT / "python/bgsage/analyzer.py").read_bytes()),
               "equity": equity(positions),
               "by_mode": {mode: equity([r for r in positions if r["mode"] == mode]) for mode in SOURCES},
               "by_cube_decision": {s: equity([r for r in positions if r["action"] == s])
                                    for s in sorted({r["action"] for r in positions})}}
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    with gzip.open(args.output / "outputs.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in positions:
            f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps(results["equity"]["tree"]))
    print(results["selected_by_stratum"])


if __name__ == "__main__":
    main()
