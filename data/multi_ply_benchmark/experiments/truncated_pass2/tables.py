# SPDX-License-Identifier: MPL-2.0
"""Write the report tables from completed, independently verified benchmark archives."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVELS = ("1P", "2P", "3P", "4P", "1T", "2T", "3T")
TRUNCATED = ("1T", "2T", "3T")
APIS = {"checker_play": "Checker", "cube_action": "Cube", "post_move_analytics": "Post-move"}
PLANS = {"attacking": "Attacking", "priming": "Priming", "anchoring": "Anchoring",
         "racing": "Racing with contact", "purerace": "Pure race"}
REPEATS = {"linux": ("linux-3t-candidate-first",), "windows": ("windows-3p-control-first", "windows-3p-candidate-first")}


def read(path):
    return json.loads(path.read_bytes())


def wall_saved(seconds):
    return 100 * (1 - seconds["candidate"] / seconds["baseline"])


def pair(a, b, scale=1., decimals=3):
    return f"{a*scale:.{decimals}f} → {b*scale:.{decimals}f}"


def repeat_table(platform, primary, allow_pending):
    rows = []
    level = "3P" if platform == "windows" else "3T"
    for name in REPEATS[platform]:
        file = HERE / "qualified" / name / "verification.json"
        if allow_pending and not file.exists():
            return ["", f"The {level} repeat measurement is pending verification."]
        repeated = read(file)
        if repeated.get("repeat_of") != primary["archive_sha256"]:
            raise ValueError(f"Missing verification of the repeat against the main archive: {name}")
        rows.append(("[" + name + "](qualified/" + name + "/verification.json)", repeated["levels"][level]))
    original = primary["original_comparisons"][level]["seconds"]["baseline"]
    lines = ["", f"**{level} repeat measurements**", "",
             "| Measurement | First build | Checkpoint (s) | Selected (s) | Wall time saved | Speedup since start |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    totals = {"baseline": 0., "candidate": 0.}
    for label, row in [("Main measurement", primary["levels"][level]), *rows]:
        t = row["seconds"]
        first = min(row["recorded"], key=lambda arm: row["recorded"][arm]["started"])
        for arm in totals:
            totals[arm] += t[arm]
        speedup = f"{original/t['candidate']:.2f}×"
        lines.append(f"| {label} | {'Checkpoint' if first == 'baseline' else 'Selected'} | "
                     f"{t['baseline']:.3f} | {t['candidate']:.3f} | {wall_saved(t):.1f}% | {speedup} |")
    pooled_speedup = f"{original*(len(rows)+1)/totals['candidate']:.2f}×"
    lines.append(f"| Sum of all runs | — | {totals['baseline']:.3f} | {totals['candidate']:.3f} | "
                 f"{wall_saved(totals):.1f}% | {pooled_speedup} |")
    lines += ["", "Every repeated full response matched the corresponding build on the main measurement. "
              "The pooled percentage uses the summed times for all recorded runs. The pooled speedup divides the original "
              "time for one pass by the mean selected time per pass."]
    return lines


def section(platform, allow_pending):
    folder = HERE / "qualified" / platform
    audit = read(folder / "verification.json")
    groups = audit["levels"]
    comparison = read(folder / "comparison.json")["levels"]
    originals = audit["original_comparisons"]
    if set(groups) != set(LEVELS) or set(originals) != set(LEVELS):
        raise ValueError(f"Incomplete historical or paired measurements for {platform}")
    lines = [f"## {platform.title()}", "",
             "| Preset | Before all optimizations (s) | Published checkpoint (s) | Current control (s) | Selected (s) | Saved this pass | Speedup since start |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for level, r in groups.items():
        t = r["seconds"]
        original = originals[level]["seconds"]["baseline"]
        mark = " †" if originals[level]["origin"].startswith("new measurement") else ""
        orig = f"{original:.3f}{mark}"
        ratio = f"{original/t['candidate']:.2f}×"
        lines.append(f"| {level} | {orig} | {r['prior_published_seconds']:.3f} | {t['baseline']:.3f} | {t['candidate']:.3f} | "
                     f"{wall_saved(t):.1f}% | {ratio} |")
    if platform == "linux" and any(r["origin"].startswith("new measurement") for r in originals.values()):
        lines += ["", "† The original Linux truncated timings use the original engine with the isolated parallel "
                  "rollout cache publication fix. The original Linux deterministic timings use unmodified "
                  "`139427d`. [Historical Linux truncated control](historical-linux/README.md)"]
    if platform == "linux":
        lines += ["", f"An intermediate release at `{audit['intermediate_commit']}` ran this corpus in an earlier "
                  "session on the same machine. Timings below compare those independently recorded full panels "
                  "with the selected build. [Intermediate release recordings](../search_pass3/release/linux/full_outputs.zip)",
                  "", "| Preset | Intermediate total (s) | Selected (s) | Speedup vs intermediate | Materially farther / closer |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for level in LEVELS:
            r = audit["intermediate_comparisons"][level]
            lines.append(f"| {level} | {r['seconds']['baseline']:.3f} | {r['seconds']['candidate']:.3f} | "
                         f"{r['speedup']:.2f}× | {r['material_farther']} / {r['material_closer']} |")
    reused = [level for level, r in groups.items() if r["recorded"]["baseline"]["origin"] != "new measurement"]
    if reused:
        lines += ["", "The checkpoint recordings for " + ", ".join(reused) + " came from an earlier full-width "
                  "comparison in this study. Their inputs, timestamps, full results, binary and summaries were "
                  "verified byte for byte. All selected-build timings are fresh."]
    lines += repeat_table(platform, audit, allow_pending)
    lines += ["", "| Preset | API (100 calls) | Original total (s) | Total seconds, checkpoint → selected | Speedup since start | Median ms | p95 ms |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for level in LEVELS:
        for api, name in APIS.items():
            t = comparison[level]["by_api"][api]["timing"]
            cells = [pair(t["baseline"][field], t["candidate"][field], scale=scale, decimals=decimals)
                     for field, scale, decimals in (("total_seconds", 1., 3), ("median_seconds", 1000, 3), ("p95_seconds", 1000, 3))]
            original = originals[level]["api_seconds"][api][0]
            speedup = f"{original/t['candidate']['total_seconds']:.2f}×"
            lines.append(f"| {level} | {name} | {original:.3f} | {cells[0]} | {speedup} | {cells[1]} | {cells[2]} |")
    lines += ["", "| Game plan (60 calls per preset) | 1T wall time saved | 2T | 3T |",
              "| --- | ---: | ---: | ---: |"]
    for plan, name in PLANS.items():
        cells = [f"{wall_saved(groups[l]['grouped']['plan/' + plan]['seconds']):.1f}%" for l in TRUNCATED]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for mode, name in (("match", "Match (150 calls)"), ("money", "Unlimited (150 calls)")):
        cells = [f"{wall_saved(groups[l]['grouped']['mode/' + mode]['seconds']):.1f}%" for l in TRUNCATED]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Each probability MAE averages 2,890 equally weighted probability observations. Equity MAE "
              "averages 1,834 equity observations. The threshold applies separately to each observation. "
              "Direct shifts include all public checker alternatives, including those without a rollout.",
              "", "| Preset | Probability MAE, checkpoint → selected | Equity MAE | Materially farther / closer | "
              "Largest direct probability / equity shift | Whole outputs identical (/300) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for level, r in groups.items():
        err = r["grouped"]["all"]["errors"]
        prob, equity = err["probability"], err["equity"]
        direct = r["direct"]
        lines.append(f"| {level} | {pair(prob['baseline_mae'], prob['candidate_mae'], decimals=8)} | "
                     f"{pair(equity['baseline_mae'], equity['candidate_mae'], decimals=8)} | "
                     f"{r['material_farther']} / {r['material_closer']} | "
                     f"{direct['probability']['max_abs_change']:.8f} / {direct['equity']['max_abs_change']:.8f} | {r['exact']} |")
    lines += ["", "| Preset | Same best checker | Same full checker order | Same cube action | "
              "Checker agrees with rollout, checkpoint → selected | Cube agrees with rollout |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for level, r in groups.items():
        actions = r["decisions_vs_rollout"]
        cells = [f"{actions[api]['baseline']['agree_with_rollout']} → {actions[api]['candidate']['agree_with_rollout']}"
                 for api in ("checker_play", "cube_action")]
        lines.append(f"| {level} | {r['same_best_checker']}/100 | {r['same_checker_order']}/100 | {r['same_cube_action']}/100 | "
                     + " | ".join(cells) + " |")
    lines += ["", "Cumulative accuracy compares the selected build directly with the recorded original "
              "predictions using the same rollout references as the per-pass comparison.", "",
              "| Preset | Probability MAE, original → selected | Equity MAE | Materially farther / closer | "
              "Best checker agrees with rollout, original → selected | Cube agrees with rollout |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for level, r in originals.items():
        err = r["grouped"]["all"]["errors"]
        prob, eq = err["probability"], err["equity"]
        cells = [f"{r['decisions_vs_rollout'][api]['baseline']['agree_with_rollout']} → "
                 f"{r['decisions_vs_rollout'][api]['candidate']['agree_with_rollout']}" for api in ("checker_play", "cube_action")]
        lines.append(f"| {level} | {pair(prob['baseline_mae'], prob['candidate_mae'], decimals=8)} | "
                     f"{pair(eq['baseline_mae'], eq['candidate_mae'], decimals=8)} | "
                     f"{r['material_farther']} / {r['material_closer']} | " + " | ".join(cells) + " |")
    lines += ["", f"[Inputs, full analytics and per-call timings](qualified/{platform}/full_outputs.zip) · "
              f"[Per-field errors and latency distributions](qualified/{platform}/comparison.json) · "
              f"[Recomputed scores, hashes and subgroup errors](qualified/{platform}/verification.json)", ""]
    return lines


def render(allow_pending=False):
    lines = ["# All-level CPU timing and accuracy", "",
             "Each preset covers the same 100 checker plays, 100 cube actions and 100 post-move evaluations. "
             "The paired measurements use 16 logical CPUs on a Ryzen 7 9700X with GCC under local x86-64 WSL2 "
             "and MSVC on Windows. Both builds run their public Stage 11 analyzers with identical position seeds, "
             "bearoff database and weights. Timings cover each API call and exclude analyzer construction and "
             "serialization. The selected engine is `5de8023`. Changes are relative to fresh controls at `1ecca39`. The earlier published time "
             "for that commit is listed separately. Each paired total is one complete pass. The optimized "
             "1P panel takes about 20 ms for all 300 calls, so even small absolute timing fluctuations are visible as percentages.", "",
             "The runner calls `BgBotAnalyzer.checker_play`, `cube_action` and `post_move_analytics` directly. It reuses "
             "one analyzer per strength with ordinary engine caches, without a warmup, progress callback or optional "
             "cube-detail tree. Each truncated position gets a reproducible seed before timing. This clears the prior "
             "dice schedule, so its regeneration falls inside the timed call. A long-lived analyzer with a fixed seed "
             "can reuse that schedule. Application cache hits, queuing and request handling fall outside the measurement. "
             "[Benchmark runner](../../../../scripts/benchmark_multi_ply.py) · [Warm fixed-seed 2T calls with progress](app-style/README.md)", "",
             "The pre-optimization engine is `139427d`. Historical runs used the same hardware and corpus in "
             "separate sessions on November 22, 2026 (UTC). All original Windows timings and Linux 1P–4P timings "
             "are from unmodified builds. The Linux machine uses a Windows-mounted filesystem under WSL2. Repeated "
             "weight-file loading inside the original 1P cube calls makes that timing especially filesystem-dependent. "
             "[Original Windows recordings](../search_pass2/windows/full_outputs.zip) · "
             "[Original Linux deterministic recordings](../search_pass2/linux/full_outputs.zip) · "
             "[Two-thread comparison with Linux-native and mounted weights](inline/README.md)", "",
             "The original parallel truncated implementation could read an opening reply before another worker "
             "finished writing it. The archived Windows timings include this behavior. The original "
             "fourth-ranked move of `checker-money-priming-10` had a 2T win probability of 0.893650 against an "
             "independent full-game rollout of 0.819967. Some single-batch parallel full-game reference calls "
             "also used this historical code. [Cache-race investigation and serial replay](../truncated_16t/README.md#parallel-cache-correction)", "",
             "The rollout reference supplies 4,724 probability and equity fields per preset. It covers 378 "
             "rolled-out checker alternatives, 100 cube decisions and 100 post-move positions. All 1,709 "
             "legal checker alternatives are compared directly between the two builds. Material means a change "
             "in absolute error greater than 0.0025 in probability or 0.005 in equity. "
             "[Reference provenance](../../rollout_positions/README.md) · [Independent verification script](audit.py)", ""]
    lines += ["The fixed references use one 1,296-game batch with variance reduction and 3P checker and cube settings. "
              "For 37 checker positions the existing archival references used the same strength in "
              "1,296-game batches up to their target standard error. The archive omits their individual batch counts.", ""]
    for platform in ("linux", "windows"):
        lines.extend(section(platform, allow_pending))
    (HERE / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pending-repeats", action="store_true", help="Mark incomplete repeat runs in an interim report")
    render(parser.parse_args().allow_pending_repeats)
