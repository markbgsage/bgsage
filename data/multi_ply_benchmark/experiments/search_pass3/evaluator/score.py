# SPDX-License-Identifier: MPL-2.0
"""Score the paired public API screen against frozen 1,296-game analytics."""
from collections import defaultdict
import json
from pathlib import Path
import sys
from zipfile import ZipFile
from build_linux import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from compare_truncated_benchmark import score_api


def analyze(folder):
    truth = {}
    for line in (ROOT / "data/multi_ply_benchmark/rollout_positions/results.jsonl").read_bytes().splitlines():
        row = json.loads(line)
        truth[row["api"], row["position_id"]] = row
    with ZipFile(folder / "outputs.zip") as archive:
        rows = [json.loads(line) for line in archive.read("results.jsonl").splitlines()]
    data = ROOT / "data/multi_ply_benchmark"
    positions = {}
    for dataset in ("checker_play", "cube_action"):
        for pos in json.loads((data / (dataset + ".json")).read_bytes())["positions"]:
            positions[dataset, pos["id"]] = pos
    for row in rows:
        dataset = "cube_action" if row["api"] == "post_move_analytics" else row["api"]
        pos = positions[dataset, row["position_id"]]
        row.update(source=pos["source"], mode=pos["mode"], source_game_plan=pos["game_plan"])
    def key(row):
        return tuple(row[k] for k in ("level", "repeat", "api", "position_id"))
    controls = {key(row):row for row in rows if row["arm"] == "control"}
    pairs = defaultdict(list)
    for row in rows:
        if row["arm"] != "control":
            pairs[row["arm"], row["level"], row["api"]].append(
                (controls[key(row)], row, truth[row["api"], row["position_id"]]))
    result = {}
    for (arm, level, api), group in pairs.items():
        stats = score_api(api, group)
        result.setdefault(arm, {}).setdefault(level, {})[api] = {
            "calls":len(group), "exact":stats["exact_analytics"],
            "material":stats["material_changes_in_rollout_error"], "seconds":stats["timing"]["candidate"]["total_seconds"],
            "baseline":stats["timing"]["baseline"]["total_seconds"], "speedup":stats["timing"]["speedup"]}
    (folder / "score.json").write_text(json.dumps(result,indent=2) + "\n")


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))
