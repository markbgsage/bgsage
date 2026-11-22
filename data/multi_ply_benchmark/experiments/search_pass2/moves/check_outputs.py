# SPDX-License-Identifier: MPL-2.0
"""Verify complete native board ordering and compare the two PubEval weight sets."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def summarize(build_root, output):
    import numpy as np
    fixture = build_root / "fixtures.txt"
    control = build_root / "base"
    base_boards = sha(control / "boards.bin")
    base_scores = np.memmap(control / "scores.bin", dtype="<f8", mode="r")
    n_calls, n_endpoints = 0, 0
    with (control / "boards.bin").open("rb") as f:
        while count := f.read(4):
            n, = struct.unpack("<I", count)
            n_calls += 1
            n_endpoints += n
            f.seek(26 * 4 * n, 1)
    assert len(base_scores) == n_endpoints * 4
    out = {"fixture_sha256": sha(fixture), "fixture_boards": len(fixture.read_bytes().splitlines()),
           "ordered_dice_calls": n_calls, "ordered_endpoints": n_endpoints,
           "evaluations_per_endpoint": ["Tesauro contact", "Tesauro race", "ListNet contact", "ListNet race"],
           "arms": {}}
    assert n_calls == out["fixture_boards"] * 36
    for arm in ("base", "sparse", "first", "inplace", "hybrid"):
        folder = build_root / arm
        board_hash = sha(folder / "boards.bin")
        if board_hash != base_boards:
            raise ValueError(f"{arm} changed legal endpoints or their order")
        scores = np.memmap(folder / "scores.bin", dtype="<f8", mode="r")
        if scores.shape != base_scores.shape:
            raise ValueError("PubEval score count differs")
        out["arms"][arm] = {
            "ordered_boards_sha256": board_hash, "pubeval_sha256": sha(folder / "scores.bin"),
            "pubeval_binary_differences": int((scores.view("<u8") != base_scores.view("<u8")).sum()),
            "largest_pubeval_difference": float(np.abs(np.asarray(scores) - base_scores).max()),
        }
    output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.build_root, args.output)
