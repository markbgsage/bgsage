# SPDX-License-Identifier: MPL-2.0
"""Archive the exact native changes and binary identities of all measured source exports."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = "5de8023cc5e548ef5ab4400488508fd1bbe8aa7b"
NATIVE_SUFFIXES = {".h", ".cpp", ".inc", ".cu", ".cuh", ".txt"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sources(root):
    result = {"CMakeLists.txt": (root / "CMakeLists.txt").read_bytes().replace(b"\r\n", b"\n")}
    for folder, suffixes in (("cpp", NATIVE_SUFFIXES),
                             ("python/bgsage", {".py"})):
        for p in (root/folder).rglob("*"):
            if p.is_file() and p.suffix in suffixes:
                result[p.relative_to(root).as_posix()] = p.read_bytes().replace(b"\r\n", b"\n")
    return result


def make_patch(old, new):
    changes = {}
    patch = []
    for name in sorted(old.keys() | new.keys()):
        a, b = old.get(name, b""), new.get(name, b"")
        if a == b:
            continue
        changes[name] = {"before_sha256": digest(a) if name in old else None,
                         "after_sha256": digest(b) if name in new else None}
        patch.extend(difflib.diff_bytes(difflib.unified_diff, a.splitlines(keepends=True), b.splitlines(keepends=True),
                                        fromfile=(("a/"+name) if name in old else "/dev/null").encode(),
                                        tofile=(("b/"+name) if name in new else "/dev/null").encode()))
    return changes, b"".join(patch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT/"build/truncated_pass2/shipped")
    parser.add_argument("--study", nargs="+", required=True, type=Path,
                        help="Each directory contains a verified full_outputs.zip")
    parser.add_argument("--extra", nargs="+", default=[], type=Path,
                        help="Source exports used for independent native checks")
    parser.add_argument("--selected-build", type=Path,
                        help="Require that every native/Python source and CMake manifest in the checkout matches this export")
    parser.add_argument("--output", type=Path, default=HERE/"variants.zip")
    args = parser.parse_args()
    if len({path.resolve() for path in args.study}) != len(args.study):
        parser.error("List each benchmark study once")
    baseline = sources(args.baseline.resolve())
    git_files = subprocess.check_output(["git", "ls-tree", "-r", BASELINE, "--", "CMakeLists.txt", "cpp", "python/bgsage"], cwd=ROOT)
    git_hashes = {line.split(b"\t",1)[1].decode():line.split()[2].decode() for line in git_files.splitlines()}
    expected_names = {name for name in git_hashes if name == "CMakeLists.txt"
                      or (name.startswith("cpp/") and Path(name).suffix in NATIVE_SUFFIXES)
                      or (name.startswith("python/bgsage/") and name.endswith(".py"))}
    if expected_names != baseline.keys():
        raise ValueError(f"Control export file set differs from Git: {sorted(expected_names ^ baseline.keys())}")
    for name, content in baseline.items():
        oid = hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
        if git_hashes.get(name) != oid:
            raise ValueError(f"Control source differs from Git at {name}")
    manifests = {"baseline_commit":BASELINE, "input_sha256_line_endings":"LF", "builds":{},"comparisons":[]}
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Select a destination in the engine checkout")
    with ZipFile(output.with_suffix(".tmp"),"w",ZIP_DEFLATED,compresslevel=9) as result:
        def add_build(binary, identity):
            if not binary.is_relative_to(ROOT) or digest(binary.read_bytes()) != identity["sha256"]:
                raise ValueError(f"Unexpected binary at {binary}")
            name = binary.parent.name
            builds = manifests["builds"]
            if name not in builds:
                changes, patch = make_patch(baseline, sources(binary.parent))
                builds[name] = dict(extension=identity, changed_sources=changes)
                if patch:
                    member = "patches/"+name+".patch"
                    result.writestr(member,patch)
                    builds[name].update(patch=member, patch_sha256=digest(patch))
            if builds[name]["extension"] != identity:
                raise ValueError(f"Build identity changed in place for {name}")
            return name

        for study in args.study:
            with ZipFile(study/"full_outputs.zip") as inputs:
                cfg = json.loads(inputs.read("settings.json"))
            record = {"study":study.resolve().relative_to(HERE).as_posix(),"arms":{}}
            for label, spec in cfg["arms"].items():
                identity = spec["binary"]
                binary = (ROOT/identity["path"]).resolve()
                name = add_build(binary, identity)
                record["arms"][label] = {"build":name, "environment":spec["experiment_environment"]}
            manifests["comparisons"].append(record)
        for directory in args.extra:
            binary, = directory.resolve().glob("bgbot_cpp*.so")
            name = add_build(binary, {"path":binary.relative_to(ROOT).as_posix(),
                "bytes":binary.stat().st_size, "sha256":digest(binary.read_bytes())})
            manifests.setdefault("additional_focused_checks", []).append(name)
        if args.selected_build:
            export = args.selected_build.resolve()
            if not export.is_relative_to(ROOT/"build"):
                raise ValueError("Expected a selected export under the engine build directory")
            actual = sources(ROOT)
            chosen = sources(export)
            if actual != chosen:
                different = sorted(k for k in actual.keys() | chosen.keys() if actual.get(k) != chosen.get(k))
                raise ValueError(f"Measured export differs from production files: {different}")
            by_file = {k:digest(v) for k,v in sorted(actual.items())}
            selected = dict(source_export=export.relative_to(ROOT).as_posix(), extension=manifests["builds"][export.name]["extension"],
                files_compared=len(by_file), normalized_newlines="LF", sha256_by_file=by_file,
                manifest_sha256=digest(json.dumps(by_file, sort_keys=True, separators=(",", ":")).encode()))
            manifests["qualified_checkout"] = selected
        result.writestr("manifest.json", json.dumps(manifests,indent=2,allow_nan=False)+"\n")
    output.with_suffix(".tmp").replace(output)
    print(len(manifests["builds"]), "distinct builds represented by", len(args.study), "qualified studies")


if __name__ == "__main__":
    main()
