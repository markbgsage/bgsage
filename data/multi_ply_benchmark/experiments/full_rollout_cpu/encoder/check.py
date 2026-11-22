# SPDX-License-Identifier: MPL-2.0
"""Compare the public dense encoder and native incremental inputs across engine trees."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", nargs=2, action="append", required=True, metavar=("LABEL", "ENGINE_ROOT"))
    parser.add_argument("--compile-commands", type=Path, help="Use flags from a Linux CMake release build")
    parser.add_argument("--portable", action="store_true", help="Exercise the scalar encoder on x86")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.compile_commands:
        entries = json.loads(args.compile_commands.read_bytes())
        entry = next(e for e in entries if Path(e["file"]).name == "encoding.cpp")
        command = shlex.split(entry["command"])
        flags = [f for f in command[:command.index("-o")] if not f.startswith("-I")]
    else:
        flags = [os.environ.get("CXX", "c++"), "-O3", "-ffast-math", "-std=c++17"]
        if not args.portable:
            flags += ["-mavx2", "-mfma"]
    if args.portable:
        flags = [f for f in flags if f not in ("-mavx2", "-mfma")]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "encoder_check"
        for label, path in args.tree:
            if label in results:
                raise ValueError(f"Repeated tree label {label}")
            tree = Path(path).resolve() / "cpp"
            sources = [tree / "src" / f"{name}.cpp" for name in ("board", "moves", "encoding")]
            subprocess.run([*flags, "-I", tree / "include", HERE / "encoder_check.cpp",
                            *sources, "-pthread", "-o", exe], check=True)
            output = subprocess.check_output([exe], text=True).strip()
            metrics = {k: int(v) for k, v in (pair.split("=") for pair in output.split())}
            results[label] = {"metrics": metrics, "encoding_cpp_sha256_lf":
                              hashlib.sha256(sources[-1].read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
            print(label, output, flush=True)
    baseline = next(iter(results.values()))["metrics"]
    if any(r["metrics"] != baseline for r in results.values()):
        raise ValueError("Dense feature fingerprints differ between engine trees")
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
