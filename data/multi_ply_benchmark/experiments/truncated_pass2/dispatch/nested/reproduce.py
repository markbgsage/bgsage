# SPDX-License-Identifier: MPL-2.0
"""Compare seeded, nested rollout results from serial and parallel workers."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import load_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--python-dir", type=Path, default=ROOT / "python")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine, _ = load_engine(args.extension, args.python_dir)
    weights = engine.WeightConfig.from_model("stage11")
    positions = json.loads((ROOT / "data/multi_ply_benchmark/cube_action.json").read_bytes())
    position = next(row for row in positions["positions"] if row["id"] == "cube-money-attacking-01")
    records = []
    for purpose in ("checker", "checker_late", "cube"):
        evaluator = engine.TrialEvalConfig(rollout_trials=2, rollout_depth=1, rollout_ply=1)
        for workers in (1, 2):
            analyzer = engine.BgBotAnalyzer(
                weights=weights, eval_level="rollout", cubeful=True, parallel_threads=workers,
                n_trials=4, truncation_depth=3, decision_ply=1, truncation_ply=1,
                late_threshold=2, seed=1720888050, bearoff_db=str(ROOT / "data/bearoff_1sided.db"),
                **{purpose: evaluator},
            )
            result = analyzer.cube_action(**position["inputs"])
            records.append(dict(purpose=purpose, workers=workers, inputs=position["inputs"],
                                analytics=asdict(result)))
        print(purpose, records[-1]["analytics"] == records[-2]["analytics"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
