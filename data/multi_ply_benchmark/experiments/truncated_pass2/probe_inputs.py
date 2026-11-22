# SPDX-License-Identifier: MPL-2.0
"""Prepare the native dice-fan comparison from published engine inputs and checker endpoints."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    boards = set()
    with ZipFile(HERE.parent / "search_pass3/release/linux/full_outputs.zip") as archive:
        meta = json.loads(archive.read("candidate/2T/metadata.json"))
        for line in archive.read("candidate/2T/results.jsonl").splitlines():
            row = json.loads(line)
            boards.add(tuple(row["inputs"]["board"]))
            if row["api"] == "checker_play":
                for move in row["analytics"]["moves"]:
                    boards.add(tuple(move["board"]))
    # A player on the bar behind a completely closed home board.
    dance = [0] * 26
    dance[25], dance[5] = 1, 14
    dance[19:25] = [-2] * 6
    dance[10] = -3
    boards.add(tuple(dance))
    (out / "weights.txt").write_text("".join(f"{size} {w['path']}\n" for size, w in
        zip(meta["hidden_sizes"], meta["weights"], strict=True)))
    (out / "boards.txt").write_text("".join(" ".join(map(str, b)) + "\n" for b in sorted(boards)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    prepare(p.parse_args().output)
