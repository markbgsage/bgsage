# SPDX-License-Identifier: MPL-2.0
"""Build the native-width regression against public 2P trees on either Linux arm."""
import argparse
import difflib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from linux_release import AREA, GCC, HERE, ROOT, sha, verify_native

WORK = ROOT / "build/det_eval2/slot_guard"
FILES = ("cube_eval.cpp", "neural_net.cpp")


def diff():
    result = []
    for name in FILES:
        def lines(path):
            return path.read_bytes().replace(b"\r\n", b"\n").splitlines(keepends=True)
        ref = f"cpp/src/{name}".encode()
        result.extend(difflib.diff_bytes(difflib.unified_diff, lines(AREA/"candidate/cpp/src"/name),
                                        lines(WORK/name), b"a/"+ref, b"b/"+ref))
    (HERE/"prebuilt_slot.patch").write_bytes(b"".join(result))


def snapshot():
    out = AREA/"guarded"
    out.mkdir(exist_ok=False)
    original = json.loads((AREA/"sources.json").read_bytes())["candidate"]
    guarded = {}
    for name, digest in original.items():
        path = AREA/"candidate"/name
        if sha(path) != digest:
            raise ValueError(f"Pre-correction candidate source mismatch at {name}")
        (out/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out/name)
    subprocess.run(["git", "apply", "--no-index", "--directory", out.relative_to(ROOT).as_posix(),
                    str(HERE/"prebuilt_slot.patch")], cwd=ROOT, check=True)
    for name in original:
        guarded[name] = sha(out/name)
        if guarded[name] != sha(ROOT/name):
            raise ValueError(f"Reviewed correction differs from working tree at {name}")
    (AREA/"guarded_sources.json").write_text(json.dumps(guarded, indent=2, sort_keys=True) + "\n")


def compile_probe():
    WORK.mkdir(exist_ok=True, parents=True)
    os.sched_setaffinity(0, [4, 5, 6, 7])
    tool = GCC/"toolchain/usr"
    os.environ.update(PATH=str(tool/"bin")+":/usr/bin:/bin", LD_LIBRARY_PATH=str(tool/"lib/x86_64-linux-gnu"),
                      TMPDIR=str(GCC/"tmp"))
    base = AREA/"candidate"
    guarded = AREA/"guarded"
    for name, digest in json.loads((AREA/"guarded_sources.json").read_bytes()).items():
        if sha(guarded/name) != digest:
            raise ValueError(f"Frozen correction differs at {name}")
    for name in ("models", "data"):
        if not (guarded/name).exists():
            (guarded/name).symlink_to(ROOT/name, target_is_directory=True)
    commands = json.loads((GCC/"final-gcc/compile_commands.json").read_text())
    by_name = {Path(item["file"]).name: item for item in commands}

    def compile(source, like):
        item = by_name[like]
        args = shlex.split(item["command"])
        args = [arg.replace(str(ROOT/"cpp")+"/", str(guarded/"cpp")+"/") for arg in args]
        args[args.index("-o")+1] = str(WORK/(source.stem+".o"))
        args[args.index("-c")+1] = str(source)
        subprocess.run(args, check=True, cwd=WORK)

    compile(HERE/"check_slot_width.cpp", "cube_eval.cpp")
    for name in FILES:
        compile(guarded/"cpp/src"/name, name)
    link = shlex.split((GCC/"final-gcc/CMakeFiles/bgbot_cpp.dir/link.txt").read_text())
    link = [arg for arg in link[:link.index("-o")] if not arg.startswith("-Wl,--dependency-file")]
    link = ["-flto=2" if arg == "-flto=auto" else arg for arg in link]
    program = [arg for arg in link if arg != "-shared"]
    report = {}
    for arm in ("original", "guarded"):
        objects = [obj for obj in sorted(base.glob("*.o")) if obj.name != "bindings.o"]
        if arm == "guarded":
            objects = [(WORK/obj.name if obj.stem+".cpp" in FILES else obj) for obj in objects]
        exe = WORK/("check-"+arm)
        subprocess.run([*program, "-o", str(exe), str(WORK/"check_slot_width.o"), *objects, "-pthread"], check=True)
        run = subprocess.run([exe], capture_output=True, text=True)
        report[arm] = {"returncode":run.returncode, "stdout":run.stdout, "stderr":run.stderr}
        print(arm, run.returncode, run.stdout, run.stderr, flush=True)
    (HERE/"prebuilt_slot_results.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["original"]["returncode"] != 1 or report["guarded"]["returncode"] != 0:
        raise ValueError("The diagnostic must reproduce the original error and pass on the guarded tree")
    objects = [(WORK/obj.name if obj.stem+".cpp" in FILES else obj) for obj in sorted(base.glob("*.o"))]
    suffix = next(base.glob("bgbot_cpp.*.so")).name
    subprocess.run([*link, "-o", str(guarded/suffix), *objects, "-pthread"], check=True)
    verify_native(guarded, suffix)
    print("Corrected release", sha(guarded/suffix, native=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["diff", "snapshot", "check"])
    {"diff": diff, "snapshot": snapshot, "check": compile_probe}[parser.parse_args().operation]()
