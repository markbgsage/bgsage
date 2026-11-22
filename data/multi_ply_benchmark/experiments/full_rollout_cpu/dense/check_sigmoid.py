# SPDX-License-Identifier: MPL-2.0
"""Compare the eight- and sixteen-lane sigmoid compiled from each supplied C++ engine export."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", action="append", nargs=2, required=True, metavar=("LABEL", "ENGINE_ROOT"))
    parser.add_argument("--compile-commands", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.compile_commands.read_bytes())
    entry = next(e for e in rows if Path(e["file"]).name == "neural_net.cpp")
    command = shlex.split(entry["command"])
    flags = [f for f in command[:command.index("-o")] if not f.startswith("-I")]
    reports = {}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for label, name in args.tree:
            if label in reports:
                raise ValueError(f"Duplicate tree label {label}")
            tree = Path(name).resolve() / "cpp"
            source = (tree / "src/neural_net.cpp").read_text()
            header, separator, _ = source.partition("// SIMD dot product:")
            if not separator or "sigmoid512_ps" not in header:
                raise ValueError("Tested neural_net.cpp does not expose the expected sigmoid source section")
            (out / "engine_sigmoid.inc").write_text(header + "} // namespace bgbot\n")
            executable = out / "check_sigmoid"
            subprocess.run([*flags, "-I", tree / "include", "-I", out, HERE / "sigmoid_probe.cpp",
                            "-o", executable], check=True)
            metrics = json.loads(subprocess.check_output([executable], text=True))
            reports[label] = dict(kernel_source_sha256=hashlib.sha256(header.encode()).hexdigest(), **metrics)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
