# SPDX-License-Identifier: MPL-2.0
"""Compile and compare isolated encoders on real descendants and adversarial boards."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from build_linux import build, HERE, ROOT, WORK
from variants import prepare

os.sched_setaffinity(0, [4,5,6,7])
corpus = ROOT / "build/cpu_feature_masks"
results = {}
for arm in sys.argv[1:]:
    if arm != "ctrl":
        build(arm, prepare(arm), True)
    exe = WORK / arm / "probe"
    dump = WORK / arm / "features.bin"
    subprocess.run([str(exe), str(corpus / "all.bin"), str(dump)], check=True)
    digest = hashlib.sha256(dump.read_bytes()).hexdigest()
    metrics = []
    for rep in range(4):
        output = subprocess.check_output(["taskset", "-c", "4", str(exe), str(corpus / "real.bin")], text=True)
        metrics.append({line.split()[0]:float(line.split()[1]) for line in output.splitlines()})
    results[arm] = dict(sha256=digest, measurements=metrics)
    print(arm, json.dumps(results[arm]), flush=True)
    (HERE / (arm + "_native.json")).write_text(json.dumps(results[arm], indent=2) + "\n")
