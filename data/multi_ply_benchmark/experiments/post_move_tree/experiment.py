"""Score the native post-move tree equity against the same rollout as the public API.

Run from the engine checkout with a compatible native extension and psutil:

    python data/multi_ply_benchmark/experiments/post_move_tree/experiment.py \
        --extension build/multi_ply_reviewed/bgbot_cpp.cp314-win_amd64.pyd \
        --output logs/post_move_tree --cpus 14 15
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import inspect
import json
from math import ceil
from pathlib import Path
from statistics import mean
import subprocess
import sys
import textwrap
from time import perf_counter_ns
from types import MethodType
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / "data/multi_ply_benchmark"
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import load_engine


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tree_method(analyzer, bounded):
    """Compile a private method that changes only the 3-ply cubeful estimator."""
    method = type(analyzer).post_move_analytics
    source = textwrap.dedent(inspect.getsource(method))
    changes = [
        ("opp_probs = bgbot_cpp.cubeful_probs_nply(",
         "native = bgbot_cpp.cubeful_probs_and_equity_nply("),
        ("probs = [1.0 - opp_probs[0],",
         'opp_probs = native["probs"]\n            probs = [1.0 - opp_probs[0],'),
    ]
    condition = 'isinstance(inner, _MultiPlyAnalyzer) and owner is not None and n_plies > 1'
    if bounded:
        changes.append(('    return PostMoveAnalysis(',
                        f'    if {condition}:\n'
                        '        adjustment = -native["equity"] - cf_eq\n'
                        '        cf_eq += max(-0.0049, min(0.0049, adjustment))\n\n'
                        '    return PostMoveAnalysis('))
    else:
        changes.append(('if use_cubeful_rollout:\n        # cubeful_evaluate_board',
                        f'if {condition}:\n'
                        '        cf_eq = -native["equity"]\n    elif use_cubeful_rollout:\n'
                        '        # cubeful_evaluate_board'))
    for old, new in changes:
        if source.count(old) != 1:
            raise RuntimeError("Reviewed post_move_analytics source changed: " + old)
        source = source.replace(old, new)
    scope = dict(method.__globals__)
    exec(compile(source, "<post_move_tree_experiment>", "exec"), scope)
    return MethodType(scope[method.__name__], analyzer)


def worker(args):
    import psutil
    psutil.Process().cpu_affinity(args.cpus)
    engine, _ = load_engine(args.extension)
    weights = engine.WeightConfig.from_model("stage11")
    analyzer = engine.BgBotAnalyzer(weights=weights, eval_level="3ply", cubeful=True,
                                    parallel_threads=len(args.cpus),
                                    bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
    if analyzer._bearoff_db is None:
        raise RuntimeError("Bearoff database failed to load")
    call = (analyzer.post_move_analytics if args.arm == "public"
            else tree_method(analyzer, args.arm == "bounded"))
    rows = []
    for position in json.loads((DATA / "cube_action.json").read_bytes())["positions"]:
        inputs = position["post_move_inputs"]
        start = perf_counter_ns()
        result = call(**inputs)
        elapsed = (perf_counter_ns() - start) / 1e9
        rows.append({"id": position["id"], "seconds": elapsed, "analytics": asdict(result)})
    (args.output / f"{args.pair}-{args.arm}.json").write_text(json.dumps(rows))


def scalars(analytics):
    return {**analytics["probs"], "cubeless_equity": analytics["cubeless_equity"],
            "cubeful_equity": analytics["cubeful_equity"]}


def distribution(values):
    ordered = sorted(values)
    return {f"p{percent}": ordered[ceil(percent / 100 * len(ordered)) - 1]
            for percent in (10, 25, 50, 75, 90, 95, 100)}


def score(rows, key, arm):
    public, candidate, delta = [], [], []
    for row in rows:
        ref = scalars(row["truth"])[key]
        public.append(abs(scalars(row["public"])[key] - ref))
        candidate.append(abs(scalars(row[arm])[key] - ref))
        delta.append(candidate[-1] - public[-1])
    limit = .005 if key.endswith("equity") else .0025
    return {"n": len(rows), "threshold": limit, "public_mae": mean(public),
            "candidate_mae": mean(candidate), "materially_better": sum(x < -limit for x in delta),
            "materially_worse": sum(x > limit for x in delta),
            "within_threshold": sum(abs(x) <= limit for x in delta),
            "closer": sum(x < 0 for x in delta), "farther": sum(x > 0 for x in delta),
            "public_absolute_error": distribution(public), "candidate_absolute_error": distribution(candidate),
            "paired_error_change": distribution(delta), "largest_improvement": -min(delta),
            "largest_regression": max(delta),
            "identical_predictions": sum(scalars(r[arm])[key] == scalars(r["public"])[key] for r in rows)}


def summarize(args, runs):
    positions = json.loads((DATA / "cube_action.json").read_bytes())["positions"]
    records = [json.loads(line) for line in
               (DATA / "rollout_3p_vr1296/results.jsonl").read_bytes().splitlines()]
    truth = {(r["api"], r["position_id"]): r for r in records if r["api"] != "checker_play"}
    if len(truth) != 200 or len(positions) != 100:
        raise ValueError("Expected 100 paired cube-action and post-move references")
    combined, times = {}, []
    for pair, arm in runs:
        rows = json.loads((args.output / f"{pair}-{arm}.json").read_bytes())
        if [r["id"] for r in rows] != [p["id"] for p in positions]:
            raise ValueError("Incomplete or reordered results")
        times.append({"pair": pair, "arm": arm, "seconds": sum(r["seconds"] for r in rows)})
        for position, row in zip(positions, rows):
            id_ = position["id"]
            reference = truth["post_move_analytics", id_]
            cube = truth["cube_action", id_]
            assert position["post_move_inputs"] == reference["inputs"]
            target = combined.setdefault(id_, {"id": id_, "mode": position["mode"],
                "source_game_plan": position["game_plan"], "truth": reference["analytics"],
                "rollout_cube_action": cube["analytics"]["optimal_action"], "timings": {}})
            if arm in target and target[arm] != row["analytics"]:
                raise RuntimeError(f"Repeat changed outputs: {id_} {arm}")
            target[arm] = row["analytics"]
            target["timings"].setdefault(arm, []).append(row["seconds"])
    rows = list(combined.values())
    with ZipFile(DATA / "experiments/paired_3ply_outputs.zip") as archive:
        saved = [json.loads(line) for line in archive.read("2t-1-candidate/results.jsonl").splitlines()]
    archived = {(r["api"], r["position_id"]): r["analytics"] for r in saved}
    public_exact = sum(r["public"] == archived["post_move_analytics", r["id"]] for r in rows)
    prior_actions_correct = sum(archived["cube_action", r["id"]]["optimal_action"] ==
                                r["rollout_cube_action"] for r in rows)
    arm = args.candidate
    stats = {key: score(rows, key, arm) for key in scalars(rows[0]["truth"])}
    groups = {key: {value: score([r for r in rows if r[key] == value], "cubeful_equity", arm)
                    for value in sorted({r[key] for r in rows})}
              for key in ("mode", "rollout_cube_action", "source_game_plan")}
    pairs = [{name: next(t["seconds"] for t in times if t["pair"] == i and t["arm"] == name)
              for name in ("public", arm)} for i in range(1, args.pairs + 1)]
    output = {"candidate": arm, "method": "Fresh process per arm. AB/BA, public first in odd pairs. "
                        "Each process uses fixed corpus order, one Stage 11 analyzer, two "
                        "logical CPUs or fewer. Elapsed sums exclude setup and output I/O.",
              "cpus": args.cpus, "model": "stage11",
              "extension_sha256": sha(args.extension.read_bytes()),
              "analyzer_sha256": sha((ROOT / "python/bgsage/analyzer.py").read_bytes()),
              "cube_corpus_sha256": sha((DATA / "cube_action.json").read_bytes()),
              "paired_truth_sha256": sha(json.dumps([truth[k] for k in sorted(truth)],
                                         sort_keys=True, separators=(",", ":")).encode()),
              "scalars": stats, "cubeful_equity_groups": groups, "times": times,
              "public_rows_equal_to_archived_2thread_analytics": public_exact,
              "archived_cube_actions_matching_rollout": prior_actions_correct,
              "paired_public_over_candidate": [t["public"] / t[arm] for t in pairs],
              "pooled_public_over_candidate": sum(t["public"] for t in pairs) / sum(t[arm] for t in pairs),
              "opponent_action_counts_in_cube_truth": dict(Counter(r["rollout_cube_action"] for r in rows))}
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    with gzip.open(args.output / "outputs.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps(output["scalars"]["cubeful_equity"]), flush=True)
    print(json.dumps(output["paired_public_over_candidate"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", nargs="+", type=int, required=True)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--candidate", choices=("tree", "bounded"), default="tree")
    parser.add_argument("--arm", choices=("public", "tree", "bounded"))
    parser.add_argument("--pair", type=int)
    parser.add_argument("--score-only", action="store_true", help="Re-score the stored arm outputs")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.extension = args.extension.resolve()
    if not args.output.is_relative_to(ROOT) or not 1 <= len(args.cpus) <= 2 or args.pairs < 2:
        raise ValueError("Select an engine path, 1 or 2 CPUs, and at least two pairs")
    if args.arm:
        worker(args)
        return
    if not args.score_only:
        args.output.mkdir(parents=True, exist_ok=False)
    runs = []
    for pair in range(1, args.pairs + 1):
        for arm in (("public", args.candidate) if pair % 2 else (args.candidate, "public")):
            if not args.score_only:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--extension",
                                str(args.extension), "--output", str(args.output), "--cpus",
                                *map(str, args.cpus), "--arm", arm, "--pair", str(pair)], check=True)
            runs.append((pair, arm))
    summarize(args, runs)


if __name__ == "__main__":
    main()
