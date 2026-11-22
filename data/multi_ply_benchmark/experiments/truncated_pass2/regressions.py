# SPDX-License-Identifier: MPL-2.0
"""Run the CPU search, rollout, progress, cancellation and move-generator suites."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
import json
import os
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import file_identity, load_engine, write_json


def verify_engine(extension, python_dir):
    engine, _ = load_engine(extension, python_dir)
    import bgbot_cpp
    from bgsage.weights import bearoff_db_path

    package = Path(engine.__file__).resolve()
    python_dir = python_dir.resolve()
    if (Path(bgbot_cpp.__file__).resolve() != extension.resolve()
            or package != python_dir / "bgsage/__init__.py"):
        raise ValueError("The suite did not import the requested engine")
    manifest = json.loads((ROOT / "data/multi_ply_benchmark/rollout_3p_vr1296/manifest.json").read_bytes())
    expected_slots = [a["sha256"] for a in manifest["assets"] if a["path"].startswith("models/")]
    assets = {}
    model_slots = {}
    for name in ("stage5", "stage9", "stage11"):
        model = engine.WeightConfig.from_model(name)
        model.validate()
        slots = []
        for raw in model.weight_paths_list:
            path = Path(raw).resolve()
            if not path.is_relative_to(python_dir.parent):
                raise ValueError(f"Model slot outside source export: {path}")
            if path.name not in assets:
                actual = file_identity(path)
                if actual["sha256"] != file_identity(ROOT / "models" / path.name)["sha256"]:
                    raise ValueError(f"Regression weights differ from repository: {path}")
                assets[path.name] = actual
            slots.append(path.name)
        model_slots[name] = slots
    if (engine.default_weights().weight_paths_list != engine.WeightConfig.from_model("stage11").weight_paths_list
            or [assets[n]["sha256"] for n in model_slots["stage11"]] != expected_slots):
        raise ValueError("Default model does not reproduce the reference's Stage 11 slots")
    database = bearoff_db_path()
    if not database or not Path(database).resolve().is_relative_to(python_dir.parent):
        raise ValueError("Default bearoff database missing from selected source export")
    bearoff = file_identity(Path(database))
    pinned = next(a for a in manifest["assets"] if a["path"] == "data/bearoff_1sided.db")
    if bearoff["sha256"] != pinned["sha256"]:
        raise ValueError("Default bearoff database differs from the benchmark reference")
    return dict(extension=file_identity(extension), package=file_identity(package),
                model_slots=model_slots, weights=assets, bearoff_database=bearoff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    parser.add_argument("python_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = spec_from_file_location("search_regression_list", HERE.parent / "search_pass2/regressions.py")
    suite = module_from_spec(spec)
    spec.loader.exec_module(suite)
    names = [*suite.MODULES, "test_move_dice", "test_move_notation", "test_rollout_progress_callback"]
    os.environ["BGBOT_MULTIPLY_THREADS"] = "4"
    os.environ["BGSAGE_MOVE_ARRAY_EXTENSION"] = str(args.extension.resolve())
    identity = verify_engine(args.extension, args.python_dir)
    with StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        rc = pytest.main(["-q", "-rs", "-p", "no:cacheprovider", *[str(ROOT / "tests" / (n + ".py")) for n in names]])
        lines = buf.getvalue().strip().splitlines()
    imported = sorted(Path(module.__file__).resolve() for name, module in sys.modules.items()
                      if name.startswith("bgsage.") and getattr(module, "__file__", None))
    if any(not path.is_relative_to(args.python_dir.resolve() / "bgsage") for path in imported):
        raise RuntimeError("Regression suite imported a module from a different engine package")
    if not rc:
        array_test = next(m for m in sys.modules.values()
                          if getattr(m, "__name__", "").split(".")[-1] == "test_move_arrays")
        if (Path(array_test.probe.__file__).resolve() != args.extension.resolve()
                or Path(array_test.arrays.__file__).resolve() != args.python_dir.resolve() / "bgsage/move_arrays.py"):
            raise RuntimeError("Array tests used a different binary or Python module")
    write_json(args.output, dict(modules=names, exit_code=int(rc), summary=lines[-1],
                                 threads=4, **identity, python_modules=[file_identity(p) for p in imported],
                                 skips=[line for line in lines if line.startswith("SKIPPED")],
                                 diagnostics=lines if rc else []))
    print("\n".join(lines[-40:]))
    raise SystemExit(rc)
