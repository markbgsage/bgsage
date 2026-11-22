# SPDX-License-Identifier: MPL-2.0
"""Render the public benchmark scorecards as Markdown, without rounding intermediate values."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
ORDER = ("1P", "2P", "3P", "4P", "1T", "2T", "3T")
APIS = ("checker_play", "cube_action", "post_move_analytics")


def main():
    lines = ["# Full-corpus CPU benchmark results", "",
             "Times are wall seconds for 100 checker, 100 cube and 100 post-move calls. "
             "Each build uses 16 CPU workers, Stage 11, the same boards, match contexts and bearoff database. "
             "The original is `139427d` and the checkpoint is `33c4aa1`.", ""]
    for platform, label in (("linux", "Linux (GCC, x86-64)"), ("windows", "Windows (MSVC, x86-64)")):
        stats = json.loads((HERE / platform / "summary.json").read_bytes())
        detail = json.loads((HERE / platform / "checkpoint_comparison.json").read_bytes())["levels"]
        levels = [level for level in ORDER if level in stats]
        lines.extend([f"## {label}", "",
                      "| Preset | Original s | Checkpoint s | Final s | Time saved this pass | Time saved since start | Speedup since start |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for level in levels:
            data = stats[level]
            original = data["original"]["seconds"]["baseline"]
            before, after = (data["checkpoint"]["seconds"][k] for k in ("baseline", "candidate"))
            lines.append(f"| {level} | {original:.3f} | {before:.3f} | {after:.3f} | "
                         f"{1-after/before:.1%} | {1-after/original:.1%} | {original/after:.2f}× |")
        lines.extend(["", "Each entry below is checkpoint → final seconds per 100 calls.", "",
                      "| Preset | Checker play | Cube action | Post-move |",
                      "| --- | ---: | ---: | ---: |"])
        for level in levels:
            values = stats[level]["checkpoint"]["api_seconds"]
            cells = [f"{values[api][0]:.3f} → {values[api][1]:.3f}" for api in APIS]
            lines.append("| " + " | ".join([level, *cells]) + " |")
        lines.extend(["", "Single-call latency on the deeper deterministic searches, in milliseconds.", "",
                      "| Preset | API | Median before → after | p95 before → after |",
                      "| --- | --- | ---: | ---: |"])
        for level in ("3P", "4P"):
            for api in APIS:
                timing = detail[level]["by_api"][api]["timing"]
                cells = [" → ".join(f"{1000*timing[arm][field]:.1f}" for arm in ("baseline", "candidate"))
                         for field in ("median_seconds", "p95_seconds")]
                lines.append("| " + " | ".join([level, api, *cells]) + " |")
        lines.extend(["", "The accuracy columns count fields with material changes in absolute error "
                      "against the frozen rollout. Complete-response equality includes every field and the full order of legal moves.", "",
                      "| Preset | Scored fields | Farther / closer vs checkpoint | Farther / closer vs original | "
                      "Identical responses vs checkpoint | Same checker order / cube choice vs checkpoint |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for level in levels:
            before, original = stats[level]["checkpoint"], stats[level]["original"]
            lines.append(f"| {level} | {before['observations']:,} | {before['material_farther']} / "
                         f"{before['material_closer']} | {original['material_farther']} / {original['material_closer']} | "
                         f"{before['identical']}/300 | {before['same_checker_ranking']}/100, "
                         f"{before['same_cube_decision']}/100 |")
        lines.extend(["", "Agreement of the final engine with the rollout reference. The checker"
                      " equity test includes only selected moves with a rolled equity.", "",
                      "| Preset | Matches best checker | Within 0.005 of best / scorable | Matches optimal cube action |",
                      "| --- | ---: | ---: | ---: |"])
        for level in levels:
            apis = detail[level]["by_api"]
            checker = apis["checker_play"]["decision_accuracy"]["candidate"]
            cube = apis["cube_action"]["decision_accuracy"]["candidate"]
            lines.append(f"| {level} | {checker['agree_with_rollout']}/100 | "
                         f"{checker['within_0_005_equity_of_best']}/{checker['scorable_regrets']} | "
                         f"{cube['agree_with_rollout']}/100 |")
        lines.extend(["", f"[All timed analytics, inputs and group timings]({platform}/full_outputs.zip) · "
                      f"[Checkpoint accuracy by field and API]({platform}/checkpoint_comparison.json) · "
                      f"[Original accuracy by field and API]({platform}/original_comparison.json)", ""])
    embedded = json.loads((HERE / "linux-truncated/comparison.json").read_bytes())["levels"]
    lines.extend(["## Linux truncated evaluations", "",
                  "The 1T checkpoint was rerun and 2T/3T use the published 16-worker Linux runs. The Windows table above also covers "
                  "the original engine for all three truncated presets.", "",
                  "| Preset | Checkpoint s | Final s | Time saved | Identical responses |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for level in ("1T", "2T", "3T"):
        group = embedded[level]
        before, after = (group["total_call_seconds"][key] for key in ("baseline", "candidate"))
        identical = sum(a["exact_analytics"] for a in group["by_api"].values())
        lines.append(f"| {level} | {before:.3f} | {after:.3f} | {1-after/before:.1%} | {identical}/300 |")
    lines.extend(["", "[All Linux truncated outputs](linux-truncated/full_outputs.zip) · "
                  "[Accuracy by field](linux-truncated/comparison.json)", ""])
    repeat = HERE / "linux-repeat/comparison.json"
    if repeat.exists():
        primary = json.loads((HERE / "linux/summary.json").read_bytes())
        repeated = json.loads(repeat.read_bytes())["levels"]
        lines.extend(["## Repeatability on Linux", "",
                      "A second run started fresh processes and reversed the order of the two builds. Each row "
                      "covers 300 calls per build per pass. The pooled figure divides total seconds saved by "
                      "total checkpoint seconds across both passes.", "",
                      "| Preset | Time saved, first pass | Time saved, reverse order | Time saved, pooled |",
                      "| --- | ---: | ---: | ---: |"])
        for level, data in repeated.items():
            first = primary[level]["checkpoint"]["seconds"]
            second = data["total_call_seconds"]
            reduction = 1 - (first["candidate"] + second["candidate"]) / (first["baseline"] + second["baseline"])
            lines.append(f"| {level} | {1-first['candidate']/first['baseline']:.1%} | "
                         f"{1-second['candidate']/second['baseline']:.1%} | {reduction:.1%} |")
        lines.extend(["", "[All reverse-order outputs](linux-repeat/full_outputs.zip) · "
                      "[Repeat scorecard](linux-repeat/comparison.json)", ""])
    (HERE / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
