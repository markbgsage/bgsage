# SPDX-License-Identifier: MPL-2.0
"""Prepare both perspectives of the fixed corpus and targeted move-rule fixtures."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]

def generate(path):
    boards = set()
    for name in ("checker_play", "cube_action"):
        src = ROOT / "data/multi_ply_benchmark" / (name + ".json")
        boards.update(tuple(r["inputs"]["board"]) for r in json.loads(src.read_bytes())["positions"])
    stress = (0, -15, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0)
    boards.add(stress)
    for points in ({1: 1, 24: -1}, {6: 1, 24: -1}, {1: 1, 4: 1, 24: -1}, {24: -1},
                   {25: 2, 24: -2, 23: -2, 22: -2, 21: -2, 20: -2, 19: -2},
                   {25: 2, 24: -1, 23: -2, 22: -2, 21: -2, 20: -2, 19: -2},
                   {25: 1, 24: -1, 23: -1, 22: -1, 18: -1}, {2: 2, 23: -2}):
        boards.add(tuple(points.get(i, 0) for i in range(26)))
    boards.update((b[25], *(-b[i] for i in range(24, 0, -1)), b[0]) for b in tuple(boards))
    path.write_text("".join(" ".join(map(str, b)) + "\n" for b in sorted(boards)), encoding="ascii")
    print(f"{len(boards)} boards")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    generate(parser.parse_args().path)
