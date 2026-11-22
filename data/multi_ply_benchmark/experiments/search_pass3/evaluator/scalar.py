# SPDX-License-Identifier: MPL-2.0
"""Compare full native features and the independent shot functions without AVX2."""
import hashlib
import json
import subprocess
from build_linux import HERE, OLD, ROOT, WORK, build
from variants import prepare

results = {}
for arm, source in (("scalar_control", OLD / "guarded"), ("scalar_candidate", prepare("bar_reach_lanes"))):
    build(arm, source, True, scalar=True)
    exe = WORK / arm / "probe"
    dump = WORK / arm / "features.bin"
    subprocess.run([str(exe), str(ROOT / "build/cpu_feature_masks/all.bin"), str(dump)], check=True)
    audit = subprocess.check_output([str(exe), "audit"], text=True).strip()
    results[arm] = dict(sha256=hashlib.sha256(dump.read_bytes()).hexdigest(), bytes=dump.stat().st_size,
                        audit=audit)
    print(arm, results[arm], flush=True)
assert results["scalar_control"] == results["scalar_candidate"]
(HERE / "scalar.json").write_text(json.dumps(results, indent=2) + "\n")
