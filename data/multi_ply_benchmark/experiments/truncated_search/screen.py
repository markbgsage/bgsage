# SPDX-License-Identifier: MPL-2.0
"""Screen rollout policies against the published 2T/3T builds and full-game targets."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import APIS, DATA, TRUNCATED_LEVELS, file_identity, inside_repo, load_engine
from compare_multi_ply_accuracy import records
from compare_truncated_benchmark import score_api

POLICIES = ("preset", "custom-control", "288-trials", "5-half-moves", "shallower-leaf")
CASES = {
    "checker_play": ("checker-money-attacking-01", "checker-match-priming-01"),
    "cube_action": ("cube-match-anchoring-01", "cube-money-racing-01"),
    "post_move_analytics": ("cube-money-purerace-01", "cube-match-attacking-01"),
}


def create_analyzer(engine, level, policy, threads):
    options = dict(weights=engine.WeightConfig.from_model("stage11"), parallel_threads=threads,
                   cubeful=True, bearoff_db=str(ROOT / "data" / "bearoff_1sided.db"))
    if policy == "preset":
        return engine.BgBotAnalyzer(eval_level=TRUNCATED_LEVELS[level], **options)
    ply = int(level[0])
    # The named presets fix these arguments. The public custom-rollout form
    # exposes the same settings, including its optional two-stage root filter.
    if ply == 2:
        import bgbot_cpp
        options.update(cube=bgbot_cpp.TrialEvalConfig(ply=2), cube_late=bgbot_cpp.TrialEvalConfig(ply=2))
    return engine.BgBotAnalyzer(eval_level="rollout", n_trials=288 if policy == "288-trials" else 360,
        truncation_depth=5 if policy == "5-half-moves" else 7, decision_ply=ply,
        truncation_ply=ply - 1 if policy == "shallower-leaf" else ply,
        late_ply=ply - 1, late_threshold=ply - 1, ultra_late_threshold=9999,
        prefilter_threshold=0.15, **options)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extension", type=Path, required=True)
    p.add_argument("--policy", choices=POLICIES, default="preset")
    p.add_argument("--levels", choices=("2T", "3T"), nargs="+", default=["2T", "3T"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cpus", type=int, nargs="+")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.cpus:
        import psutil
        psutil.Process().cpu_affinity(args.cpus)
    engine, dll = load_engine(args.extension)
    out = inside_repo(args.output)
    out.mkdir(parents=True, exist_ok=False)
    truth = {(r["api"], r["position_id"]): r for r in records(DATA / "rollout_positions/results.jsonl")}
    study = dict(policy=args.policy, threads=args.threads, cpus=args.cpus,
                 extension=file_identity(dll), levels={})
    archive = DATA / "experiments/truncated_16t/full_outputs.zip"
    with zipfile.ZipFile(archive) as baseline, (out / "results.jsonl").open("x", encoding="utf-8") as raw:
        for level in args.levels:
            analyzer = create_analyzer(engine, level, args.policy, args.threads)
            original = [json.loads(s) for s in baseline.read(f"{level}-candidate/results.jsonl").splitlines()]
            results = {}
            for api in APIS:
                pairs = []
                for old in original:
                    if old["api"] != api or old["position_id"] not in CASES[api]:
                        continue
                    analyzer.set_seed(old["seed"])
                    call = getattr(analyzer, api)
                    start = time.perf_counter()
                    public = call(**old["inputs"])
                    new = dict(old, elapsed_seconds=time.perf_counter() - start, analytics=asdict(public))
                    raw.write(json.dumps(new, separators=(",", ":"), allow_nan=False) + "\n")
                    raw.flush()
                    pairs.append((old, new, truth[api, old["position_id"]]))
                if len(pairs) != len(CASES[api]):
                    raise ValueError(f"Missing screen cases for {api}")
                score = score_api(api, pairs)
                changes = score["material_changes_in_rollout_error"]
                results[api] = dict(seconds=score["timing"]["candidate"]["total_seconds"],
                    exact=score["exact_analytics"], calls=len(pairs),
                    material_counts=dict(Counter("farther" if c["change_in_absolute_error"] > 0 else "closer" for c in changes)),
                    largest_regression=max((c for c in changes if c["change_in_absolute_error"] > 0),
                                           key=lambda c: c["change_in_absolute_error"], default=None))
                print(level, api, results[api]["seconds"], results[api]["material_counts"], flush=True)
            study["levels"][level] = results
            (out / "summary.json").write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
