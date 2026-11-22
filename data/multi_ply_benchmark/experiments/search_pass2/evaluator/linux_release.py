# SPDX-License-Identifier: MPL-2.0
"""Freeze and build the complete deterministic CPU candidate for Linux qualification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryFile
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ANCHOR = "33c4aa1f6c28d12b846e66e745eef6c2692c5052"
AREA = ROOT / "build/det_eval2/linux_release"
OLD = ROOT / "build/trunc_eval_study/linux_final/candidate"
GCC = ROOT / "build/logs/linux_cpu"


def sha(path, native=False):
    data = path.read_bytes()
    return hashlib.sha256(data if native else data.replace(b"\r\n", b"\n")).hexdigest()


def verify_native(folder, suffix):
    sys.path.insert(0, str(ROOT/"scripts"))
    from benchmark_multi_ply import load_engine
    engine, _ = load_engine(folder/suffix, folder/"python")
    import bgbot_cpp as cpp
    weights = engine.WeightConfig.from_model("stage11")
    weights.validate()
    assert len(weights.weight_paths_list) == 24 and (folder/"data/bearoff_1sided.db").is_file()
    try:
        cpp.cubeful_checker_moves_nply([0]*26, 2, 1, cpp.CubeOwner.CENTERED,
                                      None, 2, 1, 1, 0, 0, False, True, True, None, 0)
    except TypeError:
        return
    raise AssertionError("Missing strategy must be rejected by the binding")


def freeze():
    AREA.mkdir(parents=True, exist_ok=False)
    control, candidate = AREA / "control", AREA / "candidate"
    with TemporaryFile() as f:
        subprocess.run(["git", "archive", ANCHOR, "CMakeLists.txt", "cpp", "python"], cwd=ROOT, stdout=f, check=True)
        f.seek(0)
        with tarfile.open(fileobj=f) as tar:
            tar.extractall(control, filter="data")
    files = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
                                     "--", "CMakeLists.txt", "cpp", "python"], cwd=ROOT).decode().strip("\0").split("\0")
    for name in files:
        path = ROOT / name
        if not path.is_file(): continue
        (candidate / name).parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(path, candidate / name)
    digests = {}
    for source in (control, candidate):
        digests[source.name] = {p.relative_to(source).as_posix():sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
    (AREA / "sources.json").write_text(json.dumps(digests, indent=2, sort_keys=True)+"\n")
    differences = sorted(n for n in digests["candidate"] if digests["candidate"][n] != digests["control"].get(n))
    print("Frozen sources vs", ANCHOR, differences, flush=True)


def exact_sibling_snapshot():
    source, out = AREA / "candidate", AREA / "exact"
    digests = json.loads((AREA / "sources.json").read_bytes())["candidate"]
    out.mkdir(exist_ok=False)
    for name in digests:
        (out/name).parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(source/name, out/name)
    # The unchanged candidate headers contain declarations for the sparse
    # entry points. Keeping the same headers permits all 14 other object
    # files and the complete Python binding to be reused without ABI or ODR
    # differences. The extension exposes no sparse-only method to Python.
    for name in ("neural_net.cpp", "encoding.cpp"):
        shutil.copy2(ROOT/"build/det_eval2/combined"/name, out/"cpp/src"/name)
    name = "cpp/src/cube_eval.cpp"
    current = (source/name).read_bytes().replace(b"\r\n", b"\n")
    old = (AREA/"control"/name).read_bytes().replace(b"\r\n", b"\n")
    start = b"            if (done[k] || nn_idx_of[k] != nidx) continue;\n"
    end = b"            done[k] = 1;\n"
    assert current.count(start) == old.count(start) == 1
    prefix, rest = current.split(start)
    middle, suffix = rest.split(end, 1)
    assert b"nn.forward_board_from_base(" in middle
    (out/name).write_bytes(prefix + start + old.split(start)[1].split(end)[0] + end + suffix)
    result = {n:sha(out/n) for n in digests}
    (AREA/"exact_sources.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def build(exact=False):
    os.sched_setaffinity(0, [4, 5, 6, 7])
    tool = GCC / "toolchain/usr"
    os.environ.update(PATH=str(tool/"bin")+":/usr/bin:/bin", LD_LIBRARY_PATH=str(tool/"lib/x86_64-linux-gnu"),
                      TMPDIR=str(GCC/"tmp"))
    digests = json.loads((AREA / "sources.json").read_bytes())
    control, candidate = AREA / "control", AREA / ("exact" if exact else "candidate")
    if exact:
        digests["exact"] = exact_sibling_snapshot()
    for source in (control, candidate):
        for name, digest in digests[source.name].items():
            if sha(source / name) != digest: raise ValueError(f"Frozen source mismatch {source.name}/{name}")
        for name in ("models","data"):
            if not (source / name).exists(): (source / name).symlink_to(ROOT / name, target_is_directory=True)
    suffix = next(OLD.glob("bgbot_cpp.*.so")).name
    for name in digests["control"]:
        if name.startswith("cpp/") and sha(OLD/name) != digests["control"][name]:
            raise ValueError(f"Previous qualified Linux control differs from anchor at {name}")
    shutil.copy2(OLD/suffix, control/suffix)
    old_build = GCC / "final-gcc"
    commands = json.loads((old_build/"compile_commands.json").read_text())
    if any(digests["control"][n] != digests["candidate"].get(n) for n in digests["control"] if n.endswith(".txt")):
        raise ValueError("Build configuration changed; configure the candidate with CMake")
    def unit(item):
        src = Path(item["file"]).relative_to(ROOT)
        obj = candidate / src.with_suffix(".o").name
        dependencies = Path(str(old_build/item["output"])+".d").read_text().replace("\\\n"," ").split()
        names = {Path(p).relative_to(ROOT).as_posix() for p in dependencies if p.startswith(str(ROOT/"cpp")+"/")}
        assert src.as_posix() in names
        previous = "candidate" if exact else "control"
        if all(digests[previous][n] == digests[candidate.name].get(n) for n in names):
            shutil.copy2(((AREA/previous) if exact else OLD)/obj.name, obj)
        else:
            cmd = [p.replace(str(ROOT/"cpp")+"/", str(candidate/"cpp")+"/") for p in shlex.split(item["command"])]
            cmd[cmd.index("-o")+1] = str(obj)
            print("compile", src, flush=True)
            subprocess.run(cmd,cwd=candidate,check=True)
        return obj.name
    with ThreadPoolExecutor(max_workers=2) as pool:
        objects = list(pool.map(unit, commands))
    link = shlex.split((old_build/"CMakeFiles/bgbot_cpp.dir/link.txt").read_text())
    link = [p for p in link[:link.index("-o")] if not p.startswith("-Wl,--dependency-file")]
    link = ["-flto=2" if p == "-flto=auto" else p for p in link]
    subprocess.run([*link, "-o", suffix, *objects, "-pthread"], cwd=candidate, check=True)
    verify_native(candidate, suffix)
    print(json.dumps({arm.name:sha(arm/suffix, True) for arm in (control,candidate)},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["freeze","build","exact","verify"])
    parser.add_argument("arm", nargs="?", choices=["candidate","exact"])
    args = parser.parse_args()
    if args.operation == "freeze":
        freeze()
    elif args.operation == "verify":
        folder = AREA / args.arm
        verify_native(folder, next(folder.glob("bgbot_cpp.*.so")).name)
    else:
        build(exact=args.operation == "exact")
