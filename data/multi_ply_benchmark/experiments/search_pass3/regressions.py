# SPDX-License-Identifier: MPL-2.0
"""Check search, rollout and boundary behavior with an explicitly loaded native build."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
from io import StringIO
import os
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import file_identity, load_engine, write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("previous_suite", HERE.parent / "search_pass2/regressions.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    modules = [*suite.MODULES, "test_move_dice", "test_move_notation"]
    os.environ["BGBOT_MULTIPLY_THREADS"] = "4"
    load_engine(args.extension)
    with StringIO() as output, redirect_stdout(output), redirect_stderr(output):
        status = pytest.main(["-q", "-p", "no:cacheprovider", *(str(ROOT / "tests" / (m + ".py")) for m in modules)])
        lines = output.getvalue().strip().splitlines()
    print("\n".join(lines[-40:]))
    write_json(args.output, dict(extension=file_identity(args.extension), modules=modules,
                                 exit_code=int(status), summary=lines[-1], diagnostics=lines if status else []))
    raise SystemExit(status)
