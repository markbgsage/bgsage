# SPDX-License-Identifier: MPL-2.0
"""Replay a few custom and endgame rollout configurations in an isolated process per binary."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import load_engine


def make_board(points):
    board = [0] * 26
    for point, count in points.items():
        board[point] = count
    return board


def worker(source, path):
    binary = next(source.glob("bgbot_cpp.*.so"))
    engine, _ = load_engine(binary, source / "python")
    import bgbot_cpp
    base = dict(eval_level="rollout", decision_ply=1, truncation_ply=1, n_trials=31,
                truncation_depth=4, seed=717, ultra_late_threshold=9999,
                parallel_threads=2, target_se=0, max_batches=1)
    home = make_board({1: 1, 3: 2, 19: -2, 23: -1})
    boundary = make_board({1: 1, 7: 1, 19: -3, 21: -2})
    near = make_board({2: 2, 5: 2, 9: 1, 17: -2, 21: -1, 24: -3})
    bar = make_board({25: 1, 12: 4, 16: 10, 19: -3, 20: -2, 21: -3, 22: -2, 23: -3, 24: -2})
    done = make_board({9: -1})
    child = bgbot_cpp.TrialEvalConfig(rollout_trials=2, rollout_depth=1, rollout_ply=1)
    cases = [
        ("all bearoff", home, {}, {}),
        ("mixed bearoff", boundary, {}, {"away1": 3, "away2": 5}),
        ("all opening rolls pass", bar, {}, {}),
        ("cubeless checker", near, {"cubeful_trial_moves": False}, {}),
        ("late cubeless checker", bar, {"cubeful_late_threshold": 2}, {}),
        ("terminal postmove", done, {}, {}),
        ("no database", near, {"bearoff_db": False}, {}),
        ("nested checker", boundary, {"checker": child, "n_trials": 4}, {}),
        ("nested cube", near, {"cube": child, "n_trials": 4}, {}),
        ("batched target SE", near, {"target_se": .01, "max_batches": 2}, {}),
    ]
    rows = []
    for name, board, config, context in cases:
        setting = base | config
        model = engine.BgBotAnalyzer(**setting)
        api = "post_move_analytics" if name == "terminal postmove" else "cube_action"
        result = asdict(getattr(model, api)(board, **context))
        rows.append(dict(case=name, board=board, api=api, config=setting | (
            {key: {"rollout_trials": 2, "rollout_depth": 1, "rollout_ply": 1}
             for key in ("checker", "cube") if key in setting}), context=context, analytics=result))
        path.write_text(json.dumps(dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), rows=rows)))


def compare(baseline, candidate, destination):
    os.sched_setaffinity(0, [4, 5])
    with TemporaryDirectory(dir=ROOT / "build") as tmp:
        calls = []
        with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
            for name, source in [("baseline", baseline), ("candidate", candidate)]:
                path = Path(tmp) / (name + ".json")
                subprocess.run([sys.executable, "-B", __file__, "--worker", str(source), str(path)],
                               check=True, timeout=90)
                archive.write(path, path.name)
                calls.append(json.loads(path.read_bytes())["rows"])
            report = {}
            for old, new in zip(*calls, strict=True):
                assert old["case"] == new["case"] and old["config"] == new["config"]
                a, b = old["analytics"], new["analytics"]
                assert a.keys() == b.keys()
                delta = max(abs(a["probs"][p] - b["probs"][p]) for p in a["probs"])
                eq = max(abs(a[k] - b[k]) for k in a if "equity" in k and not k.endswith("_se"))
                report[old["case"]] = dict(identical=a == b, same_action=a.get("optimal_action") == b.get("optimal_action"),
                                           max_probability_change=delta, max_equity_change=eq)
            archive.writestr("summary.json", json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    (worker if args.worker else compare)(*[path.resolve() for path in args.paths])
