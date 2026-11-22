# SPDX-License-Identifier: MPL-2.0
"""Recover the original Linux release and reserve its cache entries for parallel trials."""
import argparse
from dataclasses import asdict
from difflib import unified_diff
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
EXPERIMENTS = HERE.parents[1]
TOOLS = ROOT / "build/logs/linux_cpu"
OLD = TOOLS / "original"
CMAKE = TOOLS / "original-gcc"
OUT = ROOT / "build/truncated_pass2/original-reserved"
LOG = ROOT / "logs/truncated_pass2/historical-linux"
CONTROL = json.loads((EXPERIMENTS / "search_pass2/control_inputs.json").read_bytes())
COMMIT = CONTROL["original"]["commit"]
CORRECTION = "60f9355"
LEVELS = ("1T", "2T", "3T")
EXT = Path(CONTROL["original"]["linux"]["path"]).name
ROLLOUT = Path("cpp/src/rollout.cpp")
WINDOWS = EXPERIMENTS / "truncated_16t/full_outputs.zip"
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import TRUNCATED_LEVELS, file_identity, load_engine, write_json
from compare_truncated_benchmark import compare


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", ROOT, *args])


def environment():
    env = os.environ.copy()
    sysroot = TOOLS / "toolchain/usr"
    env.update(TMPDIR=str(TOOLS / "tmp"), PATH=f"{sysroot}/bin:/usr/local/bin:/usr/bin:/bin",
               LD_LIBRARY_PATH=str(sysroot / "lib/x86_64-linux-gnu"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1")
    return env


def source_tree():
    """Verify the export, take exactly the reservation hunk and confirm it is the sole delta."""
    tree = git("ls-tree", "-r", "-z", COMMIT, "cpp", "python")
    files = {}
    for row in tree.rstrip(b"\0").split(b"\0"):
        identity, name = row.split(b"\t", 1)
        path = Path(name.decode())
        data = (OLD / path).read_bytes().replace(b"\r\n", b"\n")
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        require(blob == identity.split()[2].decode(), f"Original Git blob mismatch for {path}")
        files[path] = data
    require(len(files) == 73, "Original source inventory changed")
    # The archived Linux build selected the CPU-only CMake file through a small wrapper.
    OUT.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(OLD / "CMakeLists.txt", OUT / "CMakeLists.txt")
    corrected = git("show", f"{CORRECTION}:{ROLLOUT.as_posix()}").decode()
    start = corrected.index("        // Reserve every scheduled entry before any worker can start a trial.\n")
    boundary = "        // Use persistent thread pool — same rationale as cubeless path.\n"
    insertion = corrected[start:corrected.index(boundary, start)]
    require(len(insertion.splitlines()) == 10 and insertion.count(".store(1,") == 2,
            "Unexpected published cache reservation")
    original = files[ROLLOUT].decode()
    anchor = "        const Strategy* m0_strat = checker_strat_.get();\n\n" + boundary
    require(original.count(anchor) == 1, "Missing or ambiguous insertion point")
    patched = original.replace(anchor, anchor.removesuffix(boundary) + insertion + boundary)
    for path, data in files.items():
        target = OUT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(patched.encode() if path == ROLLOUT else data)
    for path, data in files.items():
        actual = (OUT / path).read_bytes()
        require(actual == (patched.encode() if path == ROLLOUT else data), f"Export mismatch in {path}")
    for folder in ("models", "data"):
        target = OUT / folder
        if not target.exists():
            target.symlink_to(ROOT / folder, target_is_directory=True)
        require(target.resolve() == ROOT / folder, f"Wrong assets at {target}")
    patch = "".join(unified_diff(original.splitlines(True), patched.splitlines(True), "a/" + ROLLOUT.as_posix(),
                                 "b/" + ROLLOUT.as_posix()))
    (HERE / "prefill.patch").write_text(patch, encoding="utf-8")
    return dict(verified_git_blobs=len(files), source=ROLLOUT.as_posix(), inserted_lines=10,
                deleted_lines=0, original_linux_cmake_wrapper=file_identity(OUT / "CMakeLists.txt"),
                original_source_sha256=digest(original.encode()),
                corrected_source_sha256=digest(patched.encode()), patch_sha256=digest(patch.encode()),
                correction_commit=git("rev-parse", CORRECTION).decode().strip())


def verify_assets():
    engine, _ = load_engine(OUT / EXT, OUT / "python")
    weights = engine.WeightConfig.from_model("stage11")
    weights.validate()
    slots = [file_identity(Path(path)) for path in weights.weight_paths_list]
    with ZipFile(WINDOWS) as z:
        windows = json.loads(z.read("1T-baseline/metadata.json"))
    names = [(Path(w["path"]).name, w["sha256"]) for w in slots]
    expected = [(Path(w["path"]).name, w["sha256"]) for w in windows["weights"]]
    require(names == expected and len(names) == 24, "Stage 11 model slots differ from original Windows run")
    inputs = CONTROL["inputs"]
    require(digest(json.dumps(names, separators=(",", ":")).encode()) == inputs["ordered_name_and_sha256_digest"],
            "Stage 11 identity")
    for fixture in [inputs["bearoff"], inputs["rollout_truth"], *inputs["positions"].values()]:
        require(file_identity(ROOT / fixture["path"]) == fixture, f"Fixture changed: {fixture['path']}")
    return dict(inputs, weights=slots)


def build():
    os.sched_setaffinity(0, {8, 9})
    patch = source_tree()
    env = environment()
    require(file_identity(CMAKE / EXT) == CONTROL["original"]["linux"], "Original extension identity")
    original_archive = CMAKE / "libbgbot_lib.a"
    archive = OUT / original_archive.name
    obj = OUT / "rollout.cpp.o"
    entry, = [e for e in json.loads((CMAKE / "compile_commands.json").read_bytes())
              if Path(e["file"]).name == "rollout.cpp"]
    args = [a.replace(str(OLD) + "/", str(OUT) + "/") for a in shlex.split(entry["command"])]
    args[args.index("-o") + 1] = str(obj)
    subprocess.run(args, env=env, check=True, cwd=OUT)
    shutil.copyfile(original_archive, archive)
    subprocess.run(["ar", "rs", archive, obj], env=env, check=True, cwd=OUT)
    def members(path):
        return subprocess.check_output(["ar", "t", path], env=env).decode().splitlines()
    require(members(original_archive) == members(archive), "Archive member set or ordering changed")
    identities = {}
    for member in members(archive):
        def member_hash(path):
            return digest(subprocess.check_output(["ar", "p", path, member], env=env))
        a, b = member_hash(original_archive), member_hash(archive)
        require((a != b) == (member == obj.name), f"Unexpected object identity for {member}")
        identities[member] = dict(original=a, corrected=b)
    original_link = (CMAKE / "CMakeFiles/bgbot_cpp.dir/link.txt").read_text()
    bindings = Path("CMakeFiles/bgbot_cpp.dir/cpp/pybind/bindings.cpp.o")
    replacements = {"-Wl,--dependency-file=CMakeFiles/bgbot_cpp.dir/link.d": "-Wl,--dependency-file=link.d",
                    str(bindings): str(CMAKE / bindings), "libbgbot_cuda_stub.a": str(CMAKE / "libbgbot_cuda_stub.a")}
    link = [replacements.get(a, a) for a in shlex.split(original_link)]
    subprocess.run(link, env=env, check=True, cwd=OUT)
    # Match the original pybind11 post-build step.
    recipe = (CMAKE / "CMakeFiles/bgbot_cpp.dir/build.make").read_text()
    strip = [line.strip() for line in recipe.splitlines() if line.startswith("\t") and "strip " in line]
    if strip:
        require(len(strip) == 1 and Path(shlex.split(strip[0])[-1]) == CMAKE / EXT, "Unexpected strip recipe")
        strip = [*shlex.split(strip[0])[:-1], EXT]
        subprocess.run(strip, env=env, check=True, cwd=OUT)
    provenance = dict(label="original + cache race fix", commit=COMMIT, source=patch,
                      build_source=OUT.relative_to(ROOT).as_posix(), compiler=subprocess.check_output(
                          [args[0], "--version"], env=env, text=True).splitlines()[0],
                      compile_command=args, link_command=link, strip_command=strip, archive_members=identities,
                      original_library=file_identity(original_archive), original_binding=file_identity(CMAKE / bindings),
                      original_cuda_stub=file_identity(CMAKE / "libbgbot_cuda_stub.a"), original_extension=file_identity(CMAKE / EXT),
                      extension=file_identity(OUT / EXT), inputs=verify_assets())
    write_json(HERE / "provenance.json", provenance)
    print(provenance["compiler"], provenance["extension"], flush=True)


def validate(level):
    directory = LOG / "original" / level
    meta = json.loads((directory / "metadata.json").read_bytes())
    with ZipFile(WINDOWS) as z:
        original = json.loads(z.read(f"{level}-baseline/metadata.json"))
    for key in ("model", "strategy_type", "hidden_sizes", "apis", "threads_requested", "levels",
                "seed_policy", "truncated_presets", "analyzer_threads_by_level", "cache_policy",
                "cube_details", "cubeful", "repeats", "timing_scope", "expected_calls", "completed_calls", "status"):
        require(meta[key] == original[key], f"{level} differs from archived Windows / {key}")
    require(meta["completed_calls"] == 300 and meta["status"] == "complete" and meta["threads_requested"] == 16,
            f"{level} incomplete")
    require(meta["extension"] == file_identity(OUT / EXT), f"{level} extension")
    provenance = json.loads((HERE / "provenance.json").read_bytes())
    require(meta["weights"] == provenance["inputs"]["weights"] and meta["bearoff_database"] == CONTROL["inputs"]["bearoff"]
            and meta["positions"] == CONTROL["inputs"]["positions"], f"{level} runtime assets")


def run():
    require(len(os.sched_getaffinity(0)) == 16, "Historical run requires all 16 logical CPUs")
    provenance = json.loads((HERE / "provenance.json").read_bytes())
    require(file_identity(OUT / EXT) == provenance["extension"], "Historical binary changed")
    with open("/proc/cpuinfo") as cpuinfo:
        cpu = next(line.split(":", 1)[1].strip() for line in cpuinfo if line.startswith("model name"))
    provenance["runtime"] = dict(cpu=cpu, affinity=sorted(os.sched_getaffinity(0)), platform=platform.platform(),
                                 python=sys.version, interpreter=sys.executable,
                                 label="One fresh process per level, one public call at a time, 16 rollout threads")
    write_json(HERE / "provenance.json", provenance)
    for level in LEVELS:
        dest = LOG / "original" / level
        if not dest.exists():
            subprocess.run([sys.executable, "-B", ROOT / "scripts/benchmark_multi_ply.py", "--levels", level,
                            "--threads", "16", "--model", "stage11", "--extension", OUT / EXT,
                            "--python-dir", OUT / "python", "--native-source", OUT, "--output", dest],
                           env=environment(), cwd=ROOT, check=True)
        validate(level)
        if level == "2T":
            subprocess.run([sys.executable, "-B", __file__, "serial"], env=environment(), cwd=ROOT, check=True)
    verify_serial()
    package()


def serial_check():
    """Replay the three published cache regression positions on unmodified, single-threaded Linux."""
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    engine, _ = load_engine(CMAKE / EXT, OUT / "python")
    weights = engine.WeightConfig.from_model("stage11")
    weights.validate()
    cases = (("1T", "cube_action", "cube-money-anchoring-07"),
             ("2T", "checker_play", "checker-money-priming-10"),
             ("3T", "checker_play", "checker-money-priming-07"))
    evidence = []
    for level, api, pid in cases:
        with ZipFile(WINDOWS) as z, z.open(f"{level}-baseline/results.jsonl") as handle:
            case, = (r for r in map(json.loads, handle) if r["api"] == api and r["position_id"] == pid)
        analyzer = engine.BgBotAnalyzer(weights=weights, eval_level=TRUNCATED_LEVELS[level], cubeful=True,
                                        parallel_threads=1, bearoff_db=str(ROOT / CONTROL["inputs"]["bearoff"]["path"]))
        analyzer.set_seed(case["seed"])
        value = asdict(getattr(analyzer, api)(**case["inputs"]))
        evidence.append(dict(level=level, api=api, position_id=pid, seed=case["seed"],
                             inputs=case["inputs"], original_serial=value))
    LOG.mkdir(exist_ok=True, parents=True)
    write_json(LOG / "original_serial.json", dict(original_extension=file_identity(CMAKE / EXT), comparisons=evidence))
    print("Unmodified serial controls complete", flush=True)


def verify_serial():
    evidence = json.loads((LOG / "original_serial.json").read_bytes())
    for row in evidence["comparisons"]:
        with (LOG / "original" / row["level"] / "results.jsonl").open() as handle:
            timed, = (r for r in map(json.loads, handle)
                      if r["api"] == row["api"] and r["position_id"] == row["position_id"])
        require(row["inputs"] == timed["inputs"] and row["seed"] == timed["seed"], "Serial inputs differ")
        row["exact"] = row["original_serial"] == timed["analytics"]
    evidence["corrected_extension"] = file_identity(OUT / EXT)
    write_json(HERE / "serial_check.json", evidence)
    require(all(r["exact"] for r in evidence["comparisons"]),
            "Historical serial and corrected parallel outputs differ; see serial_check.json")


def package():
    for level in LEVELS:
        validate(level)
    with ZipFile(HERE / "full_outputs.zip", "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
        for level in LEVELS:
            for name in ("metadata.json", "results.jsonl", "summary.json"):
                path = LOG / "original" / level / name
                z.write(path, path.relative_to(LOG).as_posix())
    windows = {"baseline": [], "candidate": []}
    with ZipFile(WINDOWS) as z:
        for arm, paths in windows.items():
            for level in LEVELS:
                path = LOG / "windows" / arm / level / "results.jsonl"
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_bytes(z.read(f"{level}-{arm}/results.jsonl"))
                paths.append(path)
    linux = [LOG / "original" / level / "results.jsonl" for level in LEVELS]
    truth = ROOT / CONTROL["inputs"]["rollout_truth"]["path"]
    report = compare(windows["baseline"], linux, truth, LEVELS)
    report["comparison"] = "Windows original 139427d to Linux original + cache race fix; platforms and compiler differ"
    # The archived Windows release also included the ownership correction. Its only two changed public
    # outputs are documented in the original study; separate those from the cross-platform comparison.
    corrected = compare(windows["candidate"], linux, truth, LEVELS)
    detail = {}
    for level, data in corrected["levels"].items():
        detail[level] = {}
        for api, stats in data["by_api"].items():
            changes = stats["material_changes_in_rollout_error"]
            detail[level][api] = dict(exact=stats["exact_analytics"], same_best=stats.get("same_best_action"),
                                      same_checker_order=stats.get("same_full_move_order"),
                                      material_closer=sum(c["change_in_absolute_error"] < 0 for c in changes),
                                      material_farther=sum(c["change_in_absolute_error"] > 0 for c in changes))
    report["corrected_windows_release_crosscheck"] = dict(archive=file_identity(WINDOWS), by_level=detail)
    write_json(HERE / "windows_comparison.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "run", "package", "serial"))
    {"build": build, "run": run, "package": package, "serial": serial_check}[parser.parse_args().action]()
