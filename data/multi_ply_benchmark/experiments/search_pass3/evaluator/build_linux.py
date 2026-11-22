# SPDX-License-Identifier: MPL-2.0
"""Build a private Linux evaluator or integrated source snapshot against the qualified release."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = ROOT / "build/search_pass3/evaluator"
GCC = ROOT / "build/logs/linux_cpu"
OLD = ROOT / "build/det_eval2/linux_release"
GUARD = ROOT / "build/det_eval2/slot_guard"


def checksum(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build(arm, source, probe, cores=(4, 5, 6, 7), scalar=False):
    os.sched_setaffinity(0, cores)
    tool = GCC / "toolchain/usr"
    os.environ.update(PATH=str(tool / "bin") + ":/usr/bin:/bin",
                      LD_LIBRARY_PATH=str(tool / "lib/x86_64-linux-gnu"), TMPDIR=str(GCC / "tmp"))
    cmake = GCC / "final-gcc"
    manifests = json.loads((OLD / "guarded_sources.json").read_bytes())
    commands = json.loads((cmake / "compile_commands.json").read_bytes())
    out = WORK / arm
    (out / "cpp").mkdir(exist_ok=True, parents=True)
    manifest = {}
    names = set(manifests)
    for folder, suffixes in (("cpp", {".h", ".hpp", ".inc", ".cpp"}), ("python", {".py"})):
        names.update(p.relative_to(source).as_posix() for p in (source / folder).rglob("*")
                     if p.is_file() and p.suffix in suffixes)
    for name in sorted(names):
        src = source / name
        manifest[name] = checksum(src)
        dst = out / name
        dst.parent.mkdir(exist_ok=True, parents=True)
        if not dst.exists() or checksum(dst) != manifest[name]:
            shutil.copyfile(src, dst)
    (out / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("data", "models"):
        if not (out / name).exists():
            (out / name).symlink_to(ROOT / name, target_is_directory=True)

    def compile_unit(item):
        src = Path(item["file"]).relative_to(ROOT)
        deps = Path(str(cmake / item["output"]) + ".d").read_text().replace("\\\n", " ").split()
        names = {Path(p).relative_to(ROOT).as_posix() for p in deps if p.startswith(str(ROOT / "cpp") + "/")}
        assert src.as_posix() in names
        obj = out / src.with_suffix(".o").name
        if all(manifests[n] == manifest[n] for n in names):
            base = GUARD if src.name in ("neural_net.cpp", "cube_eval.cpp") else OLD / "candidate"
            return base / obj.name
        args = shlex.split(item["command"])
        args = [p.replace(str(ROOT / "cpp") + "/", str(out / "cpp") + "/") for p in args]
        args[args.index("-o") + 1] = str(obj)
        print("compile", src, flush=True)
        subprocess.run(args, cwd=out, check=True)
        return obj

    if probe:
        item = next(i for i in commands if Path(i["file"]).name == "encoding.cpp")
        args = shlex.split(item["command"])
        args = args[:args.index("-o")]
        board = OLD / "candidate/board.o"
        if scalar:
            args = [arg for arg in args if arg not in ("-mavx2", "-mfma")]
            board = out / "cpp/src/board.cpp"
        subprocess.run([*args, "-I" + str(out / "cpp/src"), str(HERE / "probe.cpp"),
                        str(board), "-o", str(out / "probe")], check=True)
    else:
        if any(manifest[n] != manifests[n] for n in ("cpp/CMakeLists.txt", "CMakeLists.txt")):
            raise ValueError("Changed build rules require reconfiguration")
        with ThreadPoolExecutor(2) as pool:
            objects = list(pool.map(compile_unit, commands))
        cmd = shlex.split((cmake / "CMakeFiles/bgbot_cpp.dir/link.txt").read_text())
        cmd = [p for p in cmd[:cmd.index("-o")] if not p.startswith("-Wl,--dependency-file")]
        cmd = ["-flto=2" if p == "-flto=auto" else p for p in cmd]
        binary = out / next((OLD / "guarded").glob("bgbot_cpp.*.so")).name
        subprocess.run([*cmd, "-o", str(binary), *sorted(objects, key=lambda p:p.name), "-pthread"], check=True)
        print(binary.name, hashlib.sha256(binary.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm")
    parser.add_argument("source", type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--scalar", action="store_true", help="Compile probe without AVX2 and FMA")
    parser.add_argument("--cores", nargs="+", type=int, default=[4, 5, 6, 7])
    args = parser.parse_args()
    if args.scalar and not args.probe:
        parser.error("--scalar applies to the native probe")
    build(args.arm, args.source.resolve(), args.probe, args.cores, args.scalar)
