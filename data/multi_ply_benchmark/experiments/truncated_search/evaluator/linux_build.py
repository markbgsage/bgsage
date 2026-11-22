# SPDX-License-Identifier: MPL-2.0
"""Rebuild only GCC objects changed since the documented Linux benchmark."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[5]
ORIGINAL = ROOT / "build/logs/linux_cpu"
OLD = ORIGINAL / "final-gcc"
OUT = ROOT / "build/trunc_eval_study/linux"
FINAL = ROOT / "build/trunc_eval_study/linux_final"
TOOL = ORIGINAL / "toolchain/usr"
CONTROL = "60f93556137a7a5c4be671502f4b3f521998ff94"
FINAL_FILES = ("cpp/include/bgbot/multipy.h", "cpp/include/bgbot/rollout.h",
               "cpp/src/cube_eval.cpp", "cpp/src/neural_net.cpp", "cpp/src/rollout.cpp", "cpp/src/strategy.cpp")


def sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def prepare_final():
    FINAL.mkdir(parents=True, exist_ok=False)
    baseline = FINAL / "control"
    candidate = FINAL / "candidate"
    with tarfile.open(OUT / "control.tar") as tar:
        tar.extractall(baseline, filter="data")
    shutil.copytree(baseline / "cpp", candidate / "cpp")
    cmake = subprocess.check_output(["git", "show", f"{CONTROL}:CMakeLists.txt"], cwd=ROOT)
    for arm in (baseline, candidate):
        (arm / "CMakeLists.txt").write_bytes(cmake)
    manifest = {"control_commit": CONTROL, "files": {}}
    for source in FINAL_FILES:
        shutil.copy2(ROOT / source, candidate / source)
        committed = subprocess.check_output(["git", "show", f"{CONTROL}:{source}"], cwd=ROOT)
        control_sha = hashlib.sha256(committed.replace(b"\r\n", b"\n")).hexdigest()
        assert control_sha == sha(baseline / source)
        manifest["files"][source] = {"control": control_sha, "candidate": sha(candidate / source)}
    (FINAL / "inputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(final):
    os.sched_setaffinity(0, [4, 5, 6, 7])
    os.environ["PATH"] = str(TOOL / "bin") + ":/usr/bin:/bin"
    os.environ["LD_LIBRARY_PATH"] = str(TOOL / "lib/x86_64-linux-gnu")
    os.environ["TMPDIR"] = str(ORIGINAL / "tmp")
    location = FINAL if final else OUT
    baseline, candidate = location / "control", location / "candidate"
    if final:
        manifest = json.loads((FINAL / "inputs.json").read_text())
        for source, hashes in manifest["files"].items():
            for arm in (baseline, candidate):
                if sha(arm / source) != hashes[arm.name]:
                    raise ValueError(f"Frozen input changed: {arm.name}/{source}")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(OUT / "control.tar") as tar:
            tar.extractall(baseline, filter="data")
        shutil.copytree(baseline / "cpp", candidate / "cpp", dirs_exist_ok=True)
        for source in ("cpp/src/strategy.cpp", "cpp/src/neural_net.cpp", "cpp/src/cube_eval.cpp",
                       "cpp/include/bgbot/multipy.h"):
            shutil.copy2(ROOT / source, candidate / source)
    old_changes = json.loads((ORIGINAL / "input_identity.json").read_text())["changes_from_original"]
    commands = json.loads((OLD / "compile_commands.json").read_text())
    changed = {arm.name: {} for arm in (baseline, candidate)}

    def compile_unit(item):
        cached = OLD / item["output"]
        deptext = Path(str(cached) + ".d").read_text().replace("\\\n", " ").split()
        deps = sorted({str(Path(p).relative_to(ROOT)) for p in deptext if p.startswith(str(ROOT / "cpp") + "/")})
        assert str(Path(item["file"]).relative_to(ROOT)) in deps
        parts = shlex.split(item["command"])
        results = []
        basis = {f: (old_changes[f]["final_sha256"] if f in old_changes else sha(ORIGINAL / "original" / f)) for f in deps}
        for arm in (baseline, candidate):
            updates = {f: sha(arm / f) for f in deps if sha(arm / f) != basis[f]}
            obj = arm / Path(item["file"]).with_suffix(".o").name
            previous = OUT / arm.name
            if final and (previous / obj.name).exists() and all(sha(arm / f) == sha(previous / f) for f in deps):
                shutil.copy2(previous / obj.name, obj)
            elif not updates:
                shutil.copy2(cached, obj)
            elif results and results[-1][0] == updates:
                shutil.copy2(results[-1][1], obj)
            else:
                args = [p.replace(str(ROOT / "cpp") + "/", str(arm / "cpp") + "/") for p in parts]
                args[args.index("-o") + 1] = str(obj)
                print("compile", arm.name, obj.name, flush=True)
                subprocess.run(args, check=True, cwd=arm)
            changed[arm.name].update(updates)
            results.append((updates, obj))
        return results[0][1].name

    with ThreadPoolExecutor(max_workers=2) as pool:
        objects = list(pool.map(compile_unit, commands))
    link = shlex.split((OLD / "CMakeFiles/bgbot_cpp.dir/link.txt").read_text())
    link = [p for p in link[:link.index("-o")] if not p.startswith("-Wl,--dependency-file")]
    link = ["-flto=2" if p == "-flto=auto" else p for p in link]
    for arm in (baseline, candidate):
        ext = arm / next(OLD.glob("bgbot_cpp*.so")).name
        if final and arm == baseline:
            previous = OUT / "control"
            if all((arm / name).read_bytes() == (previous / name).read_bytes() for name in objects):
                shutil.copy2(previous / ext.name, ext)
                continue
        print("link", arm.name, flush=True)
        subprocess.run([*link, "-o", str(ext), *objects, "-pthread"], cwd=arm, check=True)
    (location / "source_identity.json").write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-final", action="store_true")
    mode.add_argument("--build-final", action="store_true")
    args = parser.parse_args()
    if args.prepare_final:
        prepare_final()
    else:
        main(args.build_final)
