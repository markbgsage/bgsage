# SPDX-License-Identifier: MPL-2.0
"""Compare public incremental neural outputs from two built Linux engine exports, byte for byte."""
import argparse
import ctypes
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
INPUTS = (1, 39, 196, 244)
HIDDEN = (1, 7, 8, 19, 200, 400, 513, 544)
SEEDS = (2, 5, 9)
PATTERNS = ("mixed", "zeros", "dense")


def extension(tree):
    for suffix in EXTENSION_SUFFIXES:
        path = tree / ("bgbot_cpp" + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(f"No bgbot_cpp matching this interpreter in {tree}")


def case_sizes():
    for width in INPUTS:
        for hidden in HIDDEN:
            for _ in SEEDS:
                for _ in PATTERNS:
                    yield (("saved_hidden", hidden), ("saved_inputs", width), ("base_prediction", 5),
                           ("incremental_prediction", 5))


def compare(old, new):
    old_bits, new_bits = old.cast("I"), new.cast("I")
    old_values, new_values = old.cast("f"), new.cast("f")
    sections = {}
    offset = 0
    for counts in case_sizes():
        for name, length in counts:
            stat = sections.setdefault(name, dict(floats=0, unequal_floats=0, max_absolute_difference=0.0))
            for i in range(offset, offset + length):
                stat["unequal_floats"] += old_bits[i] != new_bits[i]
                stat["max_absolute_difference"] = max(stat["max_absolute_difference"], abs(old_values[i] - new_values[i]))
            stat["floats"] += length
            offset += length
    if offset != len(old_bits):
        raise ValueError("Unexpected probe buffer length")
    return sections


def measure(binary, shim, destination):
    # Each Python subprocess loads just one engine so the two extensions cannot
    # interpose one another's C++ symbols.
    ctypes.CDLL(str(binary), mode=ctypes.RTLD_GLOBAL)
    probe = ctypes.CDLL(str(shim)).probe_hidden
    array_arg = ctypes.POINTER(ctypes.c_float)
    probe.argtypes = [ctypes.c_int] * 3 + [array_arg] * 3
    probe.restype = ctypes.c_int
    with destination.open("wb") as out:
        for width in INPUTS:
            for hidden in HIDDEN:
                for seed in SEEDS:
                    for pattern in PATTERNS:
                        rng = random.Random(seed)
                        values = []
                        for _ in range(width):
                            zero = pattern == "zeros" or (pattern == "mixed" and rng.randrange(3) == 0)
                            values.append(0.0 if zero else rng.uniform(-3, 3))
                        array = ctypes.c_float * width
                        inputs = array(*values)
                        for i in range(0, width, 13):
                            values[i] += rng.uniform(-0.5, 0.5)
                        result = (ctypes.c_float * (width + hidden + 10))()
                        code = probe(width, hidden, seed, inputs, array(*values), result)
                        if code:
                            failed = {1: "hidden sums", 2: "saved inputs", 3: "delta prediction"}
                            raise ValueError(f"Direct preparation changed {failed[code]} in case "
                                             f"{width}, {hidden}, {seed}, {pattern}")
                        out.write(bytes(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True, help="Export containing cpp/ and compiled bgbot_cpp")
    parser.add_argument("--candidate", type=Path, required=True, help="Export with the save_hidden_base method")
    parser.add_argument("--compile-commands", type=Path, help="CMake Linux release compile_commands.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-differences", action="store_true",
                        help="Report discrepancies from an exploratory kernel without failing the command")
    args = parser.parse_args()
    if args.compile_commands:
        entries = json.loads(args.compile_commands.read_bytes())
        entry = next(row for row in entries if Path(row["file"]).name == "neural_net.cpp")
        command = shlex.split(entry["command"])
        flags = [f for f in command[:command.index("-o")] if not f.startswith("-I")]
    else:
        flags = [os.environ.get("CXX", "c++"), "-O3", "-std=c++17"]

    cases = len(INPUTS) * len(HIDDEN) * len(SEEDS) * len(PATTERNS)
    report = dict(cases=cases, input_widths=INPUTS, hidden_widths=HIDDEN, seeds=SEEDS,
                  input_patterns=PATTERNS, perturbation="one input in every thirteen",
                  comparison="saved hidden sums, saved inputs, five predictions, five delta predictions",
                  arms={})
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for label, tree in (("control", args.control.resolve()), ("candidate", args.candidate.resolve())):
            shim = directory / f"{label}.so"
            data = directory / f"{label}.bin"
            binary = extension(tree)
            extra = ["-DCHECK_DIRECT_PREPARATION"] if label == "candidate" else []
            subprocess.run([*flags, "-I", tree / "cpp/include", *extra, "-fPIC", "-shared", HERE / "probe.cpp",
                            "-o", shim], check=True)
            subprocess.run([sys.executable, "-B", __file__, "measure", binary, shim, data], check=True)
            report["arms"][label] = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                                          floats=data.stat().st_size // 4,
                                          results_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
        old, new = [memoryview((directory / f"{arm}.bin").read_bytes()) for arm in ("control", "candidate")]
        if len(old) != len(new):
            raise ValueError("The arms returned different buffer lengths")
        report["byte_identical"] = old == new
        sections = compare(old, new)
        report["sections"] = sections
        report["unequal_floats"] = sum(s["unequal_floats"] for s in sections.values())
        report["max_absolute_difference"] = max(s["max_absolute_difference"] for s in sections.values())
        report["direct_preparation_cases_matched"] = cases
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["byte_identical"] and not args.report_differences:
        raise SystemExit("The saved state or predictions differ")


if __name__ == "__main__":
    if sys.argv[1:2] == ["measure"]:
        measure(*map(Path, sys.argv[2:]))
    else:
        main()
