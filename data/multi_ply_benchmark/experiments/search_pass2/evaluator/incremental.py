# SPDX-License-Identifier: MPL-2.0
"""Paired timing intervals for adding sparse board differences to routing and masks."""
from collections import defaultdict
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import psutil

if __name__ == "__main__":
    psutil.Process().cpu_affinity([4, 5, 6, 7])
    folder = Path(__file__).resolve().parent / "additive"
    totals = defaultdict(lambda: defaultdict(lambda: [0., 0.]))
    with ZipFile(folder/"full_outputs.zip") as archive:
        for line in archive.open("records.jsonl"):
            r = json.loads(line)
            if r["arm"] == "control":
                continue
            arm = ("combined", "all").index(r["arm"])
            case = (r["api"], r["position_id"])
            for api in (r["api"], "all"):
                totals[r["level"], api][case][arm] += r["elapsed_seconds"]

    # Treat the two timings of each case as paired repeats. Resample within
    # API, mode, and game plan to preserve the published corpus composition.
    rng = np.random.default_rng(20261122)
    result = {}
    for (level, api), cases in sorted(totals.items()):
        samples = np.zeros((5000, 2))
        groups = defaultdict(list)
        for (case_api, case), timings in sorted(cases.items()):
            groups[case_api, case.rsplit("-", 1)[0]].append(timings)
        for group in groups.values():
            pair = np.asarray(group)
            indices = rng.integers(0, len(pair), size=(len(samples), len(pair)))
            samples += pair[indices].sum(axis=1)
        first, second = np.asarray(list(cases.values())).sum(axis=0)
        cell = {"cases": len(cases), "speedup": first / second}
        # 4P has a single case per stratum. A conditional bootstrap cannot
        # estimate sampling variation in that panel.
        if all(len(group) > 1 for group in groups.values()):
            cell["bootstrap_95pct"] = np.quantile(samples[:, 0] / samples[:, 1], [.025, .975]).tolist()
        result.setdefault(level, {})[api] = cell
    (folder/"incremental.json").write_text(json.dumps(result, indent=2) + "\n")
