# SPDX-License-Identifier: MPL-2.0
"""Measure warm 2T calls at the analyzer's default seed, with and without checker progress."""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
from importlib.machinery import EXTENSION_SUFFIXES
import inspect
import json
from multiprocessing import get_context
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import DATA, file_identity, inside_repo, load_engine, write_json
from compare_rollout_performance import compare
from compare_truncated_benchmark import score_api
from multi_ply_rollout_truth import read, set_cpu_mask

ARMS = ("original", "intermediate", "selected")


def worker(pipe, source, warmup):
    set_cpu_mask(0xffff)
    binary = next(source / ("bgbot_cpp" + suffix) for suffix in EXTENSION_SUFFIXES
                  if (source / ("bgbot_cpp" + suffix)).is_file())
    engine, _ = load_engine(binary, source / "python")
    from bgsage.weights import bearoff_db_path
    analyzer = engine.BgBotAnalyzer(eval_level="truncated2", cubeful=True, parallel_threads=16)
    cfg = analyzer._analyzer._inner._rollout_strategy.config()
    weights = engine.default_weights()
    if (engine.PRODUCTION_MODEL, cfg.seed, cfg.n_trials, cfg.truncation_depth,
        cfg.n_threads, cfg.enable_vr) != ("stage11", 42, 360, 7, 16, True) or analyzer._bearoff_db is None:
        raise ValueError("Different preset or missing bearoff database")
    sig = inspect.signature(analyzer.checker_play)
    if not {"progress_callback", "finalize_progress"} <= sig.parameters.keys():
        raise ValueError("Checker callback API missing")
    analyzer.cube_action(**warmup)
    pipe.send(dict(extension=file_identity(binary), python=file_identity(Path(engine.__file__)),
                   analyzer=file_identity(Path(inspect.getfile(type(analyzer)))),
                   weight_slots=[file_identity(Path(p)) for p in weights.weight_paths_list],
                   database=file_identity(Path(bearoff_db_path())), threads=cfg.n_threads, seed=cfg.seed))
    while (request := pipe.recv()) is not None:
        api, inputs, progress = request
        counts = [0, 0]

        def on_progress(completed, total, partial):
            counts[0] += 1

        def on_finalize(completed, total):
            counts[1] += 1

        callbacks = dict(progress_callback=on_progress, finalize_progress=on_finalize) if progress else {}
        call = getattr(analyzer, api)
        start = perf_counter()
        result = call(**inputs, **callbacks)
        seconds = perf_counter() - start
        pipe.send((seconds, asdict(result), *counts))


def summarize(rows):
    references = {(r["api"], r["position_id"]): r for r in
                  (json.loads(line) for line in (DATA / "rollout_positions/results.jsonl").read_bytes().splitlines())}
    report = {}
    for api, callbacks in sorted({(r["api"], r["progress_requested"]) for r in rows}):
        sides = [[r for r in rows if (r["api"], r["progress_requested"], r["arm"]) == (api, callbacks, arm)]
                 for arm in ARMS]
        name = api + ("_with_progress" if callbacks else "_without_progress")
        report[name] = dict(positions=len(sides[-1]), seconds={arm: sum(r["elapsed_seconds"] for r in side)
                            for arm, side in zip(ARMS, sides)}, comparisons={},
                            callback_counts={arm: [sum(r[key] for r in side) for key in ("progress", "finalize")]
                                             for arm, side in zip(ARMS, sides)})
        for arm, side in zip(ARMS[:-1], sides[:-1], strict=True):
            pairs = []
            for a, b in zip(side, sides[-1], strict=True):
                if a["position_id"] != b["position_id"] or a["inputs"] != b["inputs"]:
                    raise ValueError("Unpaired position or inputs")
                pairs.append((a, b, references[api, a["position_id"]]))
            stats = score_api(api, pairs)
            diffs = [compare(a, b) for a, b, _ in pairs]
            report[name]["comparisons"][arm + "_to_selected"] = dict(
                speedup=stats["timing"]["speedup"],
                material_farther=sum(e["change_in_absolute_error"] > 0 for e in stats["material_changes_in_rollout_error"]),
                material_closer=sum(e["change_in_absolute_error"] < 0 for e in stats["material_changes_in_rollout_error"]),
                same_best=sum(d["best_action_same"] for d in diffs),
                max_abs_probability_shift=max(d["max_abs_probability"] for d in diffs),
                max_abs_equity_shift=max(d["max_abs_equity"] for d in diffs))
    for arm in ARMS:
        checker = [r for r in rows if r["arm"] == arm and r["api"] == "checker_play"]
        plain = {r["position_id"]: r["analytics"] for r in checker if not r["progress_requested"]}
        if any(r["analytics"] != plain[r["position_id"]] for r in checker if r["progress_requested"]):
            raise ValueError(f"Callbacks changed checker analytics for {arm}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ROOT / "build/truncated_pass2/original-reserved")
    parser.add_argument("--intermediate", type=Path, default=ROOT / "build/det_eval2/linux_release/guarded")
    parser.add_argument("--selected", type=Path, default=ROOT / "build/truncated_pass2/shipped")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = inside_repo(args.output)
    out.mkdir(parents=True, exist_ok=False)
    os.environ["BGBOT_MULTIPLY_THREADS"] = "16"
    paths = {api: DATA / (api + ".json") for api in ("checker_play", "cube_action")}
    corpus = {api: read(path)["positions"] for api, path in paths.items()}
    roster = []
    for api, positions in corpus.items():
        sample = [p for p in positions if p["id"].endswith("-01")]
        cells = Counter((p["mode"], p["game_plan"]) for p in sample)
        if len(cells) != 10 or set(cells.values()) != {1}:
            raise ValueError("Incomplete mode/plan sample")
        roster.extend((api, p) for p in sample)
    random.Random(8071).shuffle(roster)
    warmup = next(p for p in corpus["cube_action"] if p["id"] == "cube-money-racing-02")
    workers, identities, rows = {}, {}, []
    ctx = get_context("spawn")
    try:
        for name in ARMS:
            for mode in ("interactive", "plain"):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=worker, args=(child, inside_repo(getattr(args, name)), warmup["inputs"]))
                proc.start()
                child.close()
                workers[name, mode] = parent, proc
                info = parent.recv()
                if name in identities and identities[name] != info:
                    raise ValueError("Callback arms loaded different assets")
                identities[name] = info
        original = read(HERE.parent / "historical-linux/provenance.json")
        if identities["original"]["extension"]["sha256"] != original["extension"]["sha256"]:
            raise ValueError("Original build differs from corrected historical release")
        identities["original"]["commit"] = original["commit"]
        for name, archive, side in (
            ("intermediate", HERE.parents[1] / "search_pass3/release/linux/full_outputs.zip", "baseline"),
            ("selected", HERE.parent / "qualified/linux/full_outputs.zip", "candidate"),
        ):
            with ZipFile(archive) as saved:
                qualified = json.loads(saved.read(side + "/2T/metadata.json"))
            analyzer = next(p for p in qualified["source_files"] if p["path"].endswith("/python/bgsage/analyzer.py"))
            info = identities[name]
            if (info["extension"]["sha256"], info["analyzer"]["sha256"]) != (
                    qualified["extension"]["sha256"], analyzer["sha256"]):
                raise ValueError(f"{name} binary or API differs from the qualified release")
            if name == "intermediate":
                info["commit"] = qualified["git_commit"]
        if len({tuple(p["sha256"] for p in a["weight_slots"]) for a in identities.values()}) != 1 or len(
                {a["database"]["sha256"] for a in identities.values()}) != 1:
            raise ValueError("Arms used different model slots or database")
        write_json(out / "settings.json", dict(arms=identities, cpu_mask="0xffff", default_seed=42,
                   python=sys.version, affinity_workers=16, preset="truncated2", games=360, reseed=False,
                   cold_start_timed=False, warmup_id=warmup["id"], archive_sources={k:file_identity(v) for k,v in paths.items()},
                   order=[p["id"] for _, p in roster], timing="public call including counter callbacks"))
        checker_count = 0
        arm_count = Counter()
        with gzip.open(out / "calls.jsonl.gz", "wt", encoding="utf-8") as sink:
            for api, p in roster:
                modes = ["interactive"]
                if api == "checker_play":
                    modes = ["interactive", "plain"] if checker_count % 2 == 0 else ["plain", "interactive"]
                    checker_count += 1
                for mode in modes:
                    progress = api == "checker_play" and mode == "interactive"
                    key = api, mode
                    turn = arm_count[key]
                    offset = (turn // 2 + (mode == "plain")) % len(ARMS)
                    order = ARMS[offset:] + ARMS[:offset]
                    if turn % 2:
                        order = order[::-1]
                    arm_count[key] += 1
                    for name in order:
                        pipe, _ = workers[name, mode]
                        pipe.send((api, p["inputs"], progress))
                        seconds, result, normal, final = pipe.recv()
                        row = dict(arm=name, api=api, position_id=p["id"], source=p["source"], mode=p["mode"],
                                   source_game_plan=p["game_plan"], repeat=1, inputs=p["inputs"],
                                   progress_requested=progress, progress=normal, finalize=final,
                                   elapsed_seconds=seconds, analytics=result)
                        rows.append(row)
                        sink.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        report = summarize(rows)
        write_json(out / "summary.json", report)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        for pipe, proc in workers.values():
            if proc.is_alive():
                pipe.send(None)
            pipe.close()
            proc.join()


if __name__ == "__main__":
    main()
