# SPDX-License-Identifier: MPL-2.0
"""Time old/new 3-ply builds on identical inputs, alternating fresh processes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from benchmark_multi_ply import APIS, ROOT, inside_repo, write_json
from multi_ply_rollout_truth import set_cpu_mask


def verify_control_sources(revision, native_source, python_dir):
    """Check the control's native tree and Python package against Git blobs."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", revision + "^{commit}"],
        cwd=ROOT, text=True,
    ).strip()
    tree = subprocess.check_output(
        ["git", "ls-tree", "-rz", commit, "--", "cpp", "python/bgsage"], cwd=ROOT,
    )
    roots = {"cpp": inside_repo(native_source) / "cpp", "python": inside_repo(python_dir)}
    checked = set()
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        directory, relative = name.decode().split("/", 1)
        path = roots[directory] / relative
        content = path.read_bytes()
        expected = header.split()[2].decode()
        # A Windows checkout may convert a text blob's LF to CRLF.
        for data in (content, content.replace(b"\r\n", b"\n")):
            blob = b"blob " + str(len(data)).encode() + b"\0" + data
            if hashlib.sha1(blob).hexdigest() == expected:
                break
        else:
            raise ValueError(f"Control source {path} differs from {commit}:{name.decode()}")
        checked.add(path)
    for directory in (roots["cpp"], roots["python"] / "bgsage"):
        for path in directory.rglob("*"):
            if path.suffix in {".py", ".cpp", ".cu", ".h", ".hpp", ".cmake"} and path not in checked:
                raise ValueError(f"Additional source in control tree: {path}")
    return {"commit": commit, "verified_native_and_python_files": len(checked)}


def read_arm(directory, apis):
    summary = json.loads((directory / "summary.json").read_bytes())
    if summary["status"] != "complete" or summary["completed_calls"] != 100 * len(apis):
        raise ValueError(f"Incomplete arm: {directory}")
    timings = {g["api"]: g["timing"] for g in summary["groups"]
               if g["mode"] is None and g["source_game_plan"] is None}
    if timings.keys() != set(apis):
        raise ValueError(f"Unexpected set of APIs: {directory}")
    # Hash every field of every record except its timing. List order, full
    # precision, legal endpoints, inputs and API options are all significant.
    digest = hashlib.sha256()
    for line in (directory / "results.jsonl").read_bytes().splitlines():
        row = json.loads(line)
        del row["elapsed_seconds"]
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8") + b"\n")
    return timings, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--baseline-extension", type=Path)
    parser.add_argument("--baseline-native-source", type=Path, default=ROOT)
    parser.add_argument("--baseline-revision", help="Check native and Python control sources against this Git revision")
    parser.add_argument("--candidate-python", type=Path, default=ROOT / "python")
    parser.add_argument("--candidate-extension", type=Path)
    parser.add_argument("--candidate-native-source", type=Path, default=ROOT)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", nargs="+", type=int, default=[2])
    parser.add_argument("--apis", nargs="+", choices=APIS, default=list(APIS))
    parser.add_argument("--cpu-mask", type=lambda s:int(s, 0))
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--summarize-only", action="store_true",
                        help="Rebuild paired.json from completed arms under --output")
    args = parser.parse_args()
    if args.pairs < 1 or min(args.threads) < 1:
        parser.error("Provide positive thread and pair counts")
    if len(args.apis) != len(set(args.apis)):
        parser.error("Select each API at most once")
    if args.baseline_revision and not args.baseline_python:
        parser.error("--baseline-revision requires --baseline-python")
    control = (verify_control_sources(args.baseline_revision, args.baseline_native_source,
                                      args.baseline_python) if args.baseline_revision else None)
    root = inside_repo(args.output)
    if not args.summarize_only:
        if not (args.baseline_python and args.baseline_extension and args.candidate_extension):
            parser.error("Measuring requires --baseline-python, --baseline-extension and --candidate-extension")
        root.mkdir(parents=True, exist_ok=False)
    if args.cpu_mask:
        set_cpu_mask(args.cpu_mask)
    study = {"affinity":args.cpu_mask, "ordering":"AB, BA (fresh process per arm)",
             "threads":args.threads, "apis":args.apis, "comparisons":[], "full_output_sha256":{}}
    if control:
        study["control_source_verification"] = control
    for threads in args.threads:
        for pair in range(args.pairs):
            summaries = {}
            order = ("baseline","candidate") if pair % 2 == 0 else ("candidate","baseline")
            for arm in order:
                output = root / f"{threads}t-{pair+1}-{arm}"
                if not args.summarize_only:
                    python = inside_repo(getattr(args,arm+"_python"))
                    ext = inside_repo(getattr(args,arm+"_extension"))
                    native = inside_repo(getattr(args,arm+"_native_source"))
                    subprocess.run([sys.executable, str(ROOT/"scripts/benchmark_multi_ply.py"),
                                    "--levels","3ply","--threads",str(threads),"--extension",str(ext),
                                    "--apis",*args.apis,
                                    "--python-dir",str(python),"--native-source",str(native),
                                    "--output",str(output)], check=True)
                summaries[arm], digest = read_arm(output, args.apis)
                study["full_output_sha256"][output.name] = digest
            for api in args.apis:
                b, c = (summaries[arm][api] for arm in ("baseline","candidate"))
                study["comparisons"].append({"api":api,"threads":threads,"pair":pair+1,
                    "baseline":b,"candidate":c,"speedup":b["total_seconds"]/c["total_seconds"]})
            write_json(root/"paired.json", study)
        for api in args.apis:
            pairs = [p["speedup"] for p in study["comparisons"] if p["api"]==api and p["threads"]==threads]
            print(f"{threads} threads, {api}: {statistics.geometric_mean(pairs):.3f}x across {len(pairs)} pairs")


if __name__=="__main__":
    main()
