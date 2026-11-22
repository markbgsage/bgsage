# SPDX-License-Identifier: MPL-2.0
"""Pair two builds on all 16 CPU threads with a fresh interpreter per arm and level."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import file_identity, inside_repo, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs=2, required=True, metavar=("BINARY_DIR", "SOURCE"))
    parser.add_argument("--candidate", type=Path, nargs=2, required=True, metavar=("BINARY_DIR", "SOURCE"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", required=True)
    parser.add_argument("--candidate-first", action="store_true")
    parser.add_argument("--reuse-truncated-controls", type=Path,
                        help="Directory of completed 16-thread controls for these same truncated presets")
    cfg = parser.parse_args()
    origins = {}
    for i, level in enumerate(cfg.levels):
        order = ["baseline", "candidate"]
        if bool(i % 2) != cfg.candidate_first:
            order.reverse()
        for arm in order:
            build, source = getattr(cfg, arm)
            if arm == "baseline" and level.endswith("T") and cfg.reuse_truncated_controls:
                old = inside_repo(cfg.reuse_truncated_controls) / arm / level
                dst = inside_repo(cfg.run_dir) / arm / level
                meta = json.loads((old / "metadata.json").read_bytes())
                binary = inside_repo(build) / Path(meta["extension"]["path"]).name
                if (meta["status"] != "complete" or meta["completed_calls"] != 300 or
                        meta["threads_requested"] != 16 or file_identity(binary)["sha256"] != meta["extension"]["sha256"]):
                    raise ValueError(f"Different build or unfinished historical control at {old}")
                dst.mkdir(parents=True)
                for name in ("metadata.json", "results.jsonl", "summary.json"):
                    shutil.copyfile(old / name, dst / name)
                origins[level] = file_identity(old / "results.jsonl")
                continue
            subprocess.run([sys.executable, "-B", str(HERE / "qualify.py"), "run", arm,
                            "--build", str(build), "--source", str(source), "--levels", level,
                            "--run-dir", str(cfg.run_dir)], cwd=ROOT, check=True)
        print("Completed paired level", level, flush=True)
    subprocess.run([sys.executable, "-B", str(HERE / "qualify.py"), "report",
                    "--run-dir", str(cfg.run_dir), "--output", str(cfg.output),
                    "--levels", *cfg.levels], cwd=ROOT, check=True)
    if origins:
        write_json(inside_repo(cfg.output) / "truncated_controls.json", origins)


if __name__ == "__main__":
    main()
