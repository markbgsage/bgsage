# SPDX-License-Identifier: MPL-2.0
"""Check all 196/244 feature bytes and all six backgame network layouts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np
import psutil
from screen import ROOT, binary, load_engine

sys.path.insert(0, str(ROOT / "data/multi_ply_benchmark/experiments/cpu_feature_masks"))
from verify_extension import directed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    psutil.Process().cpu_affinity([4, 5, 6, 7])
    engine, _ = load_engine(binary(args.arm), ROOT / "build/pass2_checkpoint/python")
    import bgbot_cpp as cpp
    boards = np.fromfile(ROOT / "build/cpu_feature_masks/all.bin", dtype="<i4").reshape(-1,26)
    boards = np.concatenate((boards, np.asarray(list(directed().values()), dtype=np.int32)))
    result = {"n":len(boards), "features":{}, "routes":{}}
    for size in (196, 244):
        sha = hashlib.sha256()
        for start in range(0,len(boards),512):
            sha.update(cpp.encode_boards_batch(boards[start:start+512], size).tobytes())
        result["features"][size] = sha.hexdigest()
    weights = engine.WeightConfig.from_model("stage11")
    cases = boards.tolist()
    for layout in (19, 20, 21, 22, 23, 24):
        model = cpp.BackgameAwarePairStrategy(weights.weight_paths_list[:layout], weights.hidden_sizes_list[:layout],
                                               layout >= 22)
        routes = [model.select_nn_idx(b) for b in cases]
        packed = struct.pack("<" + str(len(routes)) + "i", *routes)
        result["routes"][layout] = {"sha256":hashlib.sha256(packed).hexdigest(), "counts":Counter(routes)}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.arm, result["n"], result["features"], flush=True)
