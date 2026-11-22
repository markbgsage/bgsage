# SPDX-License-Identifier: MPL-2.0
"""Compare all input floats across independent, stratified random positions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import load_engine
from multi_ply_rollout_truth import set_cpu_mask


STRATA = ("race", "sparse", "blots", "blocks", "mixed",
          "one_on_bar", "multiple_on_bar", "both_on_bar")
SEED = 719605329
BATCH = 1024


def sample(rng, kind):
    rows = np.arange(BATCH)
    points = np.arange(1, 25)
    order = np.argsort(rng.random((BATCH, 24)), axis=1)
    # Each side has independent checker counts, with disjoint point ownership.
    # In race positions the mover has already passed every opponent checker.
    split = rng.integers(4, 21, size=BATCH)
    if kind == "race":
        order = np.concatenate((np.argsort(rng.random((BATCH, 12)), axis=1),
                                12 + np.argsort(rng.random((BATCH, 12)), axis=1)), axis=1)
        split[:] = 12
    elif kind == "blots":
        split = rng.integers(9, 16, size=BATCH)
    boards = np.zeros((BATCH, 26), dtype="<i4")
    for mover in (True, False):
        offset = 0 if mover else split
        available = split if mover else 24 - split
        bar = np.zeros(BATCH, dtype=np.int32)
        if kind == "one_on_bar" and mover:
            bar[:] = 1
        elif kind == "multiple_on_bar" and mover:
            bar = rng.integers(2, 9, size=BATCH)
        elif kind == "both_on_bar":
            bar = rng.integers(1, 6, size=BATCH)
        boards[:, 25 if mover else 0] = bar
        maximum = np.minimum(available, 15 - bar)
        if kind in ("blocks", "sparse"):
            maximum = np.minimum(maximum, 6)
        minimum = np.minimum(maximum, 8) if kind == "blots" else 1
        sites = rng.integers(minimum, maximum + 1)
        left = rng.integers(0, 16 - bar - sites)
        if kind == "sparse":
            left = np.minimum(left, 2)
        if kind == "blocks":
            left = 15 - bar - sites
        sign = 1 if mover else -1
        for step in range(15):
            active = rows[step < sites]
            indexes = order[rows, offset + (step % available)]
            boards[active, points[indexes[active]]] += sign
            active = rows[step < left]
            target = order[rows, offset + rng.integers(0, sites)]
            boards[active, points[target[active]]] += sign
    return boards


def evaluate(extension, path, blocks):
    load_engine(extension.resolve())
    import bgbot_cpp as cpp
    result = {"extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
              "seed": SEED, "strata": {}}
    for index, kind in enumerate(STRATA):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, index]))
        board_hash = hashlib.sha256()
        fields = {size: hashlib.sha256() for size in (196, 244)}
        counts = dict(contact=0, at_least_four_blots=0, at_least_four_blocks=0)
        for _ in range(blocks):
            boards = sample(rng, kind)
            board_hash.update(boards.tobytes())
            points = boards[:, 1:25]
            back = np.where(points > 0, np.arange(1, 25), 0).max(axis=1)
            opp = np.where(points < 0, np.arange(1, 25), 25).min(axis=1)
            counts["contact"] += int(((back > opp) | (boards[:, 0] > 0) | (boards[:, 25] > 0)).sum())
            counts["at_least_four_blots"] += int(((points == -1).sum(axis=1) >= 4).sum())
            counts["at_least_four_blocks"] += int(((points < -1).sum(axis=1) >= 4).sum())
            for size, digest in fields.items():
                data = cpp.encode_boards_batch(boards, size)
                assert data.shape == (BATCH, size) and data.dtype == np.float32
                digest.update(data.tobytes())
        result["strata"][kind] = {
            "rows": BATCH * blocks, **counts, "board_sha256": board_hash.hexdigest(),
            "feature_sha256": {k: d.hexdigest() for k, d in fields.items()}}
    path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--existing-release", type=Path,
                        help="Read the matching release arm output from an earlier identical run")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--cpu-mask", type=lambda s: int(s, 0), default=1)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    set_cpu_mask(args.cpu_mask)
    if args.worker:
        evaluate(args.candidate, args.output, args.blocks)
        return
    if args.release is None:
        parser.error("--release is required")
    arms = {}
    for name, extension in (("release", args.release), ("candidate", args.candidate)):
        if name == "release" and args.existing_release:
            saved = json.loads(args.existing_release.read_text())
            digest = hashlib.sha256(extension.read_bytes()).hexdigest()
            if saved["extension_sha256"] != digest or saved["seed"] != SEED:
                raise ValueError("The saved release hashes are from a different binary or seed")
            arms[name] = saved
            continue
        path = args.output.with_stem(args.output.stem + "-" + name)
        subprocess.run([sys.executable, "-B", __file__, "--worker", "--candidate", str(extension),
                        "--output", str(path), "--blocks", str(args.blocks),
                        "--cpu-mask", str(args.cpu_mask)], check=True)
        arms[name] = json.loads(path.read_text())
    matches = arms["release"]["strata"] == arms["candidate"]["strata"]
    summary = {"rows": len(STRATA) * BATCH * args.blocks, "seed": SEED,
               "all_features_bit_identical": matches, "fields_per_board": [196,244],
               "extensions": {k: v["extension_sha256"] for k,v in arms.items()},
               "strata": arms["release"]["strata"]}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if not matches:
        raise RuntimeError("Stratified feature mismatch. See the two separate arm outputs.")
    print(f"{summary['rows']} boards, {440*summary['rows']} identical float32 values")


if __name__ == "__main__":
    main()
