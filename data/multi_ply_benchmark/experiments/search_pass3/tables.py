# SPDX-License-Identifier: MPL-2.0
"""Render timing and accuracy tables from the paired comparisons."""
import json
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "search_pass2"
RELEASE = HERE / "release"
LEVELS = ("1P", "2P", "3P", "4P", "1T", "2T", "3T")
APIS = ("checker_play", "cube_action", "post_move_analytics")


def read(path):
    return json.loads(path.read_bytes())


def seconds(x):
    return f"{x:.3f}"


def render():
    text = ["# Full-corpus timings and accuracy", "",
            "Each row covers 100 checker, 100 cube and 100 post-move calls on the same 16-logical-CPU "
            "Ryzen 7 9700X. Linux timings were measured locally under WSL2 using GCC on x86-64. "
            "Windows uses MSVC. Each arm and preset starts in a fresh process. "
            "Timing excludes analyzer construction, serialization and disk writes.", ""]
    for platform, name in (("linux", "Linux"), ("windows", "Windows")):
        results = read(RELEASE / platform / "comparison.json")["levels"]
        original = read(PREVIOUS / platform / "original_comparison.json")["levels"]
        text += [f"## {name}", "",
                 "| Preset | Original (s) | Previous (s) | This pass (s) | Less time this pass | Speedup since the original |",
                 "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for level in LEVELS:
            if level not in results:
                continue
            old = original.get(level, {}).get("total_call_seconds", {}).get("baseline")
            pair = results[level]["total_call_seconds"]
            before, after = pair["baseline"], pair["candidate"]
            text.append(f"| {level} | {seconds(old) if old else '—'} | {seconds(before)} | "
                        f"{seconds(after)} | {100*(1-after/before):.1f}% | {f'{old/after:.2f}×' if old else '—'} |")
        text += ["", "The original deterministic engine is `139427d`. The previous engine is `82e6397`. "
                 "Original timings come from the published historical run with the same inputs. "
                 "The deterministic timings are freshly paired. All final truncated calls were retimed; "
                 "their control recordings come from the matched seven-preset run earlier in this study. "
                 "Dashes indicate that this study has no original measurement on that platform.", "",
                 "| Preset | Checker play, previous → current (s) | Cube action (s) | Post-move (s) |",
                 "| --- | ---: | ---: | ---: |"]
        for level in LEVELS:
            if level in results:
                cells = [results[level]["by_api"][api]["timing"] for api in APIS]
                text.append(f"| {level} | " + " | ".join(
                    f"{seconds(t['baseline']['total_seconds'])} → {seconds(t['candidate']['total_seconds'])}" for t in cells) + " |")
        text += ["", "| Deep search | API | Median, previous → current (ms) | p95, previous → current (ms) |",
                 "| --- | --- | ---: | ---: |"]
        for level in ("3P", "4P"):
            for api, stats in results[level]["by_api"].items():
                values = stats["timing"]
                cells = [" → ".join(f"{1000 * values[arm][field]:.1f}" for arm in ("baseline", "candidate"))
                         for field in ("median_seconds", "p95_seconds")]
                text.append(f"| {level} | {api} | " + " | ".join(cells) + " |")
        if platform == "linux":
            totals = {}
            with ZipFile(RELEASE / platform / "full_outputs.zip") as archive:
                for level in ("2P", "3P", "4P"):
                    for arm in ("baseline", "candidate"):
                        with archive.open(f"{arm}/{level}/results.jsonl") as stream:
                            for row in map(json.loads, stream):
                                part = totals.setdefault(row["source_game_plan"], {}).setdefault(level, {})
                                part[arm] = part.get(arm, 0) + row["elapsed_seconds"]
            text += ["", "Each game-plan cell includes 20 checker, 20 cube and 20 post-move calls.", "",
                     "| Game plan | 2P time saved | 3P time saved | 4P time saved |",
                     "| --- | ---: | ---: | ---: |"]
            for group, levels in totals.items():
                cells = [f"{100*(1-levels[l]['candidate']/levels[l]['baseline']):.1f}%" for l in ("2P", "3P", "4P")]
                label = {"racing": "racing with contact", "purerace": "pure race"}.get(group, group)
                text.append(f"| {label} | " + " | ".join(cells) + " |")
        text += ["", "The following counts compare absolute error with the existing full-game rollout for the "
                 "same position, checker endpoint and field. Material means a change exceeding 0.0025 for a "
                 "probability or 0.005 for an equity. The direct shift includes returned legal alternatives "
                 "with no rolled equity. Tiny changes in low-ranked alternatives can change their order.", "",
                 "| Preset | Rollout fields | Materially farther / closer | Max direct probability shift | Max direct equity shift | Same best checker / cube | Same full checker order |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for level in LEVELS:
            if level not in results:
                continue
            apis = results[level]["by_api"]
            fields = [f for a in apis.values() for f in a["rollout_error_by_field"].values()]
            shifts = [(k, f["max_abs_shift"]) for a in apis.values() for k, f in a["numeric_shifts"].items()]
            prob = max(v for k, v in shifts if k.startswith(("probs.", "p.")))
            eq = max(v for k, v in shifts if "equity" in k and "std_error" not in k)
            text.append(f"| {level} | {sum(f['n'] for f in fields):,} | {sum(f['materially_farther'] for f in fields)} / "
                        f"{sum(f['materially_closer'] for f in fields)} | {prob:.8f} | {eq:.8f} | "
                        f"{apis['checker_play']['same_best_action']}/100, {apis['cube_action']['same_best_action']}/100 | "
                        f"{apis['checker_play']['same_full_move_order']}/100 |")
        text.extend(["", f"[Full release inputs, timings and responses](release/{platform}/full_outputs.zip) · "
                     f"[Accuracy and latency distributions by API](release/{platform}/comparison.json) · "
                     f"[Earlier paired run, before the Linux 2P race adjustment]({platform}/full_outputs.zip)", ""])
    repeat = RELEASE / "linux-repeat/summary.json"
    if repeat.is_file():
        rows = read(repeat)
        text += ["## Independent Linux repeat", "",
                 "Fresh processes reversed the initial build order and reran the full corpus.", "",
                 "| Preset | Previous (s) | This pass (s) | Less time |", "| --- | ---: | ---: | ---: |"]
        for level, row in rows.items():
            times = row["seconds"]
            text.append(f"| {level} | {seconds(times['baseline'])} | {seconds(times['candidate'])} | "
                        f"{100*(1-times['candidate']/times['baseline']):.1f}% |")
        text += ["", "[Inputs and complete repeat outputs](release/linux-repeat/full_outputs.zip)", ""]
    (HERE / "results.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    render()
