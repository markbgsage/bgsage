# SPDX-License-Identifier: MPL-2.0
"""Run and archive complete, isolated Linux 2T/3T comparisons."""
from collections import Counter
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BIN = ROOT / "build/trunc_eval_study/linux_final"
STUDY = HERE / "linux_final"
RUNS = ROOT / "build/trunc_eval_study/linux_final_qualification"
LEVELS = ("2T", "3T")
ARMS = ("control", "candidate")
APIS = ("checker_play", "cube_action", "post_move_analytics")


def main():
    # The compiler process has its own CPU affinity; a fresh parent starts this full benchmark.
    host_cpus = sorted(os.sched_getaffinity(0))
    cpus = host_cpus[:16]
    os.sched_setaffinity(0, cpus)
    RUNS.mkdir(parents=True, exist_ok=True)
    STUDY.mkdir(parents=True, exist_ok=True)
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    process = dict(os=platform.system(), kernel=platform.release(), architecture=platform.machine(),
                   python=sys.version, logical_cpu_count=os.cpu_count(),
                   available_cpus=host_cpus, affinity=sorted(os.sched_getaffinity(0)), trial_threads=len(cpus),
                   cgroup_cpu_max=cpu_max.read_text().strip() if cpu_max.is_file() else None,
                   memory_total=next(line for line in Path("/proc/meminfo").read_text().splitlines()
                                     if line.startswith("MemTotal:")),
                   storage="WSL2 /mnt/c", order="2T control then candidate; 3T control then candidate")
    (STUDY / "environment.json").write_text(json.dumps(process, indent=2) + "\n")
    for manifest in ("inputs.json", "source_identity.json"):
        shutil.copy2(BIN / manifest, STUDY / manifest)
    paths = {arm: [] for arm in ARMS}
    for level in LEVELS:
        for arm in ARMS:
            out = RUNS / f"{level}-{arm}"
            native = BIN / arm
            path = out / "results.jsonl"
            if not out.exists():
                ext, = native.glob("bgbot_cpp.*.so")
                subprocess.run([sys.executable, "-B", str(ROOT / "scripts/benchmark_multi_ply.py"),
                                "--levels", level, "--threads", str(len(cpus)), "--native-source", str(native),
                                "--extension", str(ext), "--output", str(out)], cwd=ROOT, check=True)
            metadata = json.loads((out / "metadata.json").read_bytes())
            expected = ("complete", 300, len(cpus))
            actual = (metadata["status"], metadata["completed_calls"], metadata["threads_requested"])
            if actual != expected:
                raise ValueError(f"Incomplete or wrong run: {out}, {actual}")
            count = Counter(json.loads(line)["api"] for line in path.read_bytes().splitlines())
            if count != dict.fromkeys(APIS, 100):
                raise ValueError(f"Unexpected workload: {out}, {count}")
            paths[arm].append(str(path))
    score = STUDY / "comparison.json"
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/compare_truncated_benchmark.py"),
                    "--levels", *LEVELS, "--baseline", *paths["control"], "--candidate", *paths["candidate"],
                    "--output", str(score)], cwd=ROOT, check=True)
    with ZipFile(STUDY / "full_outputs.zip", "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for level in LEVELS:
            for arm in ARMS:
                folder = RUNS / f"{level}-{arm}"
                for name in ("results.jsonl", "metadata.json", "summary.json"):
                    archive.write(folder / name, folder.name + "/" + name)
    report = json.loads(score.read_bytes())
    simple = {}
    for level, result in report["levels"].items():
        stats = list(result["by_api"].values())
        changes = [change["change_in_absolute_error"] for s in stats for change in s["material_changes_in_rollout_error"]]
        shifts = [(key, field["max_abs_shift"]) for s in stats for key, field in s["numeric_shifts"].items()]
        simple[level] = dict(total_call_seconds=result["total_call_seconds"], speedup=result["speedup"],
            identical_calls=sum(s["exact_analytics"] for s in stats),
            same_best_decisions=sum(s.get("same_best_action", 0) for s in stats),
            same_full_checker_rankings=result["by_api"]["checker_play"]["same_full_move_order"],
            max_probability_shift=max(v for k,v in shifts if k.startswith("probs.")),
            max_equity_shift=max(v for k,v in shifts if "equity" in k and not k.endswith("_se")),
            max_se_shift=max(v for k,v in shifts if k.endswith("_se") or "std_error" in k),
            scored_truth_fields=sum(f["n"] for s in stats for f in s["rollout_error_by_field"].values()),
            materially_farther=sum(change > 0 for change in changes),
            materially_closer=sum(change < 0 for change in changes))
    (STUDY / "qualification.json").write_text(json.dumps(simple, indent=2) + "\n")


if __name__ == "__main__":
    main()
