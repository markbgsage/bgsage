# SPDX-License-Identifier: MPL-2.0
"""Check published control binaries, source exports and benchmark assets without evaluating a position."""
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import DATA, file_identity, write_json
from measure_multi_ply_pair import verify_control_sources


def metadata(folder, prefix):
    with ZipFile(HERE.parent / folder / "full_outputs.zip") as archive:
        return json.loads(archive.read(prefix + "/metadata.json"))


def binary(path, expected):
    identity = file_identity(path)
    if identity["sha256"] != expected:
        raise ValueError(f"Binary differs from archived qualification at {path}")
    return identity


def main():
    original = metadata("truncated_16t", "2T-baseline")
    checkpoint = metadata("truncated_search", "2T-candidate")
    linux_checkpoint = metadata("truncated_search/evaluator/linux_final", "2T-candidate")
    linux_original = json.loads((HERE.parent / "linux_x86_64/provenance.json").read_bytes())
    specs = [
        ("original", "139427d826dd67798226e7d23c003c90315306ba", "build/logs/linux_cpu/original",
         "logs/multi_ply_rollout_truth/frozen", original, "build/logs/linux_cpu/original-gcc",
         linux_original["module_sha256"]["original"]),
        ("checkpoint", "33c4aa1f6c28d12b846e66e745eef6c2692c5052", "build/pass2_checkpoint",
         "build/pass2_checkpoint", checkpoint, "build/trunc_eval_study/linux_final/candidate",
         linux_checkpoint["extension"]["sha256"]),
    ]
    evidence = {}
    ordered_weights = []
    for arm, revision, source, runtime, previous, linux_build, linux_hash in specs:
        python_dir = ROOT / runtime / "python"
        check = verify_control_sources(revision, ROOT / source, python_dir)
        so, = (ROOT / linux_build).glob("bgbot_cpp.*.so")
        linux_source = linux_build if arm == "checkpoint" else source
        linux_check = verify_control_sources(revision, ROOT / linux_source, python_dir)
        checks = []
        for w in previous["weights"]:
            name = Path(w["path"]).name
            digest = file_identity(ROOT / runtime / "models" / name)["sha256"]
            if digest != w["sha256"]:
                raise ValueError(f"Weight mismatch at {arm}/{name}")
            checks.append([name, digest])
        ordered_weights.append(checks)
        evidence[arm] = dict(**check, native_source=source, python_source=python_dir.relative_to(ROOT).as_posix(),
            windows=binary(ROOT / previous["extension"]["path"], previous["extension"]["sha256"]),
            linux=binary(so, linux_hash), linux_native_source=linux_source,
            linux_verified_native_and_python_files=linux_check["verified_native_and_python_files"])
    if ordered_weights[0] != ordered_weights[1] or len(ordered_weights[0]) != 24:
        raise ValueError("Mismatched Stage 11 weights or slot order")
    for shared in [original["bearoff_database"], *original["positions"].values()]:
        if file_identity(ROOT / shared["path"])["sha256"] != shared["sha256"]:
            raise ValueError(f"Shared benchmark asset changed: {shared['path']}")
    evidence["inputs"] = dict(model="stage11", ordered_weight_slots=24,
        ordered_name_and_sha256_digest=hashlib.sha256(json.dumps(ordered_weights[0], separators=(",", ":")).encode()).hexdigest(),
        bearoff=original["bearoff_database"], positions=original["positions"],
        rollout_truth=file_identity(DATA / "rollout_positions/results.jsonl"))
    write_json(HERE / "control_inputs.json", evidence)


if __name__ == "__main__":
    main()
