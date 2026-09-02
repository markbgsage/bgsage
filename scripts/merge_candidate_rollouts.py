#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Splice completed candidate rollouts into a folder's reference.

A folder reference (``<category> rollout.jsonl``) rolled out only the
candidates the reference player's 2-ply filter kept; every other legal move
carries the 1-/2-ply value the filter scored it at. A bot whose pick falls
outside that set is then compared at filter precision against a rolled-out
best — measured 2026-09-02, that over-charged stage11 by ~0.15 equity per
such pick (24% of its pooled error mass, 52% on "21 backgame").

``rollout_backgame_benchmark.py --jobs-file`` rolls those candidates out
under the reference convention (same paths, plies and trial player) into
``<category> candidates rollout.jsonl``. This script copies each rolled
candidate's equity / probs / std_error over the filter-level entry in the
reference, re-sorts the move list, stamps the record with how many
candidates were completed, and keeps a ``.pre-merge`` backup. Idempotent.

Usage:
    py -3.14 scripts/merge_candidate_rollouts.py --category "21 backgame"
    py -3.14 scripts/merge_candidate_rollouts.py --all
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from backgame_benchmark import benchmark_file  # noqa: E402

FOLDERS = ["21 backgame", "31 backgame", "32 backgame", "41 backgame",
           "42 backgame", "51 backgame", "52 backgame", "43 backgame",
           "53 backgame", "54 backgame"]


def merge(category: str) -> None:
    base_dir = benchmark_file(category).parent
    ref_path = base_dir / f"{category} rollout.jsonl"
    cand_path = base_dir / f"{category} candidates rollout.jsonl"
    if not cand_path.exists():
        print(f"{category}: no candidate rollouts ({cand_path.name} missing)")
        return

    rolled: dict[str, dict] = {}
    for line in cand_path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rolled[r["key"].removesuffix("|cand")] = r
    if not rolled:
        print(f"{category}: candidate file is empty")
        return

    backup = ref_path.with_suffix(".jsonl.pre-merge")
    if not backup.exists():
        shutil.copy2(ref_path, backup)

    out_lines, n_dec, n_moves = [], 0, 0
    for line in ref_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        cand = rolled.get(rec["key"])
        if cand and rec["kind"] == "checker":
            by_board = {tuple(m["board"]): m for m in cand["moves"]
                        if m["eval_level"] == "Rollout"}
            forced = {tuple(b) for b in cand.get("force_boards", [])}
            changed = 0
            for m in rec["moves"]:
                b = tuple(m["board"])
                new = by_board.get(b)
                if new is None or b not in forced or m["eval_level"] == "Rollout":
                    continue
                m.update(equity=new["equity"], cubeless_equity=new["cubeless_equity"],
                         probs=new["probs"], eval_level="Rollout",
                         std_error=new["std_error"])
                changed += 1
            if changed:
                rec["moves"].sort(key=lambda m: -m["equity"])
                rec["rollout_se"] = rec["moves"][0]["std_error"]
                rec["completed_candidates"] = rec.get("completed_candidates", 0) + changed
                n_dec += 1
                n_moves += changed
        out_lines.append(json.dumps(rec, separators=(",", ":")) + "\n")

    ref_path.write_text("".join(out_lines), encoding="utf-8")
    print(f"{category}: {n_moves} candidates upgraded to rollout grade across "
          f"{n_dec} decisions (backup: {backup.name})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", nargs="*", default=[])
    parser.add_argument("--all", action="store_true", help="All ten folders")
    args = parser.parse_args()
    cats = FOLDERS if args.all else args.category
    if not cats:
        parser.error("give --category or --all")
    for c in cats:
        merge(c)


if __name__ == "__main__":
    main()
