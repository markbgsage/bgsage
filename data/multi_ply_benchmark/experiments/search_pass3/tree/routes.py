# SPDX-License-Identifier: MPL-2.0
"""Compare the filtered cube survivor cache on blended, pair and fallback models."""
import argparse
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import sys
from prepare import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import file_identity, load_engine, write_json
from multi_ply_rollout_truth import set_cpu_mask


def worker(conn, ext):
    set_cpu_mask(15)
    engine, _ = load_engine(ext)
    import bgbot_cpp as cpp
    sys.path.append(str(ROOT / "tests"))
    from test_backgame_category import backgame_board
    boards = [backgame_board({1, 2, 3}), list(engine.board.STARTING_BOARD)]
    db, empty = cpp.BearoffDB(), cpp.BearoffDB()
    assert db.load(str(ROOT / "data/bearoff_1sided.db"))
    for model in ("stage10", "standard_pair", "stage5"):
        weights = engine.WeightConfig.from_model("stage9" if model == "standard_pair" else model)
        kind = "pair" if model == "standard_pair" else weights.strategy_type
        count = 17 if model == "standard_pair" else len(weights.weight_paths_list)
        native = cpp.create_strategy(kind, weights.weight_paths_list[:count], weights.hidden_sizes_list[:count])
        routed = native.select_nn_idx(boards[0]) if model == "stage10" else None
        if model == "stage10" and routed < 1000:
            raise AssertionError("The Stage 10 fixture must select a blended network")
        wrapped = cpp.BearoffStrategy(cpp.BearoffStrategy(native, db), empty)
        for board in boards:
            for width in (1, 4):
                inputs = dict(checkers=board, cube_value=1, owner=cpp.CubeOwner.CENTERED,
                              n_plies=3, strategy_type=kind, weight_paths=[], hidden_sizes=[],
                              n_threads=width, away1=5, away2=7, loaded_strategy=wrapped)
                values = cpp.cube_decision_nply_unified(**inputs)
                conn.send(dict(model=model, route=routed, board=board, threads=width, analytics=values))
    conn.close()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("control", type=Path)
    cli.add_argument("candidate", type=Path)
    cli.add_argument("output", type=Path)
    a = cli.parse_args()
    a.output.mkdir(exist_ok=False)
    ref = {}
    ctx = mp.get_context("spawn")
    with gzip.open(a.output / "calls.jsonl.gz", "wt") as f:
        for name, ext in (("control", a.control), ("partial", a.candidate)):
            rx, tx = ctx.Pipe(False)
            proc = ctx.Process(target=worker, args=(tx, ext))
            proc.start()
            tx.close()
            for _ in range(12):
                row = rx.recv()
                key = row["model"], tuple(row["board"]), row["threads"]
                if name == "control":
                    ref[key] = row["analytics"]
                elif row["analytics"] != ref[key] or row["analytics"] != ref[key[:2] + (1,)]:
                    raise AssertionError(f"Different routed cube result for {key}")
                f.write(json.dumps(dict(arm=name, **row), separators=(",", ":")) + "\n")
            proc.join()
            if proc.exitcode:
                raise RuntimeError(proc.exitcode)
    write_json(a.output / "summary.json", dict(exact_comparisons=12,
        control=file_identity(a.control), candidate=file_identity(a.candidate),
        models=["stage10", "standard_pair", "stage5"], threads=[1, 4], cpu_mask="0xf"))


if __name__ == "__main__":
    main()
