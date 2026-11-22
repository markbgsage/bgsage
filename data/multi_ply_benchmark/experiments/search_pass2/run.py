# SPDX-License-Identifier: MPL-2.0
"""Run one arm in a fresh interpreter for each preset with 16 CPU workers."""
import argparse
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import TRUNCATED_LEVELS, inside_repo
from compare_truncated_benchmark import level_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("original", "checkpoint", "candidate"))
    parser.add_argument("levels", nargs="+", type=level_name)
    parser.add_argument("--runs", type=Path, default=ROOT / "logs/search_pass2")
    args = parser.parse_args()
    linux = sys.platform.startswith("linux")
    orig = ROOT / "build/logs/linux_cpu"
    integrated = ROOT / "build/det_eval2/linux_release"
    if args.arm == "original":
        build = orig / "original-gcc" if linux else ROOT / "build/multi_ply_verified_control"
        source = orig / "original"
        python = ROOT / "logs/multi_ply_rollout_truth/frozen/python"
    elif args.arm == "checkpoint":
        build = (ROOT / "build/trunc_eval_study/linux_final/candidate" if linux
                 else ROOT / "build/multi_ply_truncated_final")
        checkpoint = ROOT / "build/pass2_checkpoint"
        source = build if linux else checkpoint
        python = checkpoint / "python"
    else:
        build = integrated / "guarded" if linux else ROOT / "build/multi_ply_second_final"
        source = integrated / "guarded" if linux else ROOT
        python = source / "python"
    binary = next((build / ("bgbot_cpp" + s) for s in EXTENSION_SUFFIXES
                   if (build / ("bgbot_cpp" + s)).is_file()), None)
    if binary is None:
        parser.error(f"No {sys.implementation.cache_tag}-compatible bgbot_cpp extension in {build}")
    platform = "linux" if linux else "windows"
    for level in args.levels:
        preset = TRUNCATED_LEVELS[level] if level in TRUNCATED_LEVELS else level[0] + "ply"
        output = inside_repo(args.runs) / platform / args.arm / level
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/benchmark_multi_ply.py"),
                        "--levels", preset, "--model", "stage11", "--threads", "16",
                        "--extension", str(binary), "--python-dir", str(python),
                        "--native-source", str(source), "--output", str(output)], check=True)


if __name__ == "__main__":
    main()
