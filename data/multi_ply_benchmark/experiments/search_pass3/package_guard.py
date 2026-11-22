# SPDX-License-Identifier: MPL-2.0
"""Archive the final 16-thread embedded 3P check without duplicating native binaries."""
import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true", help="Package the final, race-qualified release")
    cfg = parser.parse_args()
    folder = ROOT / ("logs/search_pass3_guard_release" if cfg.release else "logs/search_pass3_guard_16")
    settings = json.loads((folder / "settings.json").read_bytes())
    if (settings["threads"], settings["profiles"], settings["full_trials"], settings["repeats"]) != (16, ["full"], 36, 2):
        raise ValueError("Unexpected final full-game guard settings")
    pairs = json.loads((folder / "summary.json").read_bytes())["pairs"]
    if (len(pairs) != 12 or any(not p["best_action_same"] for p in pairs if p["api"] != "post_move_analytics")
            or any(not p["move_order_same"] for p in pairs if p["api"] == "checker_play")):
        raise ValueError("Incomplete full-game guard or changed decisions")
    runs = ROOT / ("logs/search_pass3_release" if cfg.release else "logs/search_pass3")
    official = json.loads((runs / "windows/candidate/3P/metadata.json").read_bytes())
    candidate = json.loads((folder / "candidate/hashes.json").read_bytes())
    if official["extension"]["sha256"] not in candidate.values():
        raise ValueError("Full-game guard used a different binary from the Windows release run")
    destination = HERE / "release" if cfg.release else HERE
    with ZipFile(destination / "fullgame_guard_16.zip", "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for name in ("0-baseline.jsonl", "0-candidate.jsonl", "1-baseline.jsonl", "1-candidate.jsonl",
                     "settings.json", "summary.json", "baseline/hashes.json", "candidate/hashes.json"):
            archive.write(folder / name, name)
        with ZipFile(HERE / "fullgame_guard.zip") as earlier:
            archive.writestr("full_profile.json", earlier.read("full_profile.json"))
    before = sum(p["baseline_seconds"] for p in pairs)
    after = sum(p["candidate_seconds"] for p in pairs)
    print(f"12 pairs, {before:.3f} -> {after:.3f} s, {100*(1-after/before):.1f}% saved; "
          f"max probability shift {max(p['max_abs_probability'] for p in pairs):.9g}, "
          f"max equity shift {max(p['max_abs_equity'] for p in pairs):.9g}")


if __name__ == "__main__":
    main()
