# SPDX-License-Identifier: MPL-2.0
"""Run the search regression suite against an explicitly selected extension."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import os
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import file_identity, load_engine, write_json

MODULES = [
    "test_rollout_prefill", "test_cube_candidate_batch_reuse", "test_multi_ply_bearoff_cache",
    "test_2ply_details", "test_cube_points", "test_vr_and_dead_cube", "test_backgame_category",
    "test_extended_inputs", "test_extended_inputs_v2", "test_checker_cube_symmetry", "test_move_arrays",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.environ["BGBOT_MULTIPLY_THREADS"] = "4"
    load_engine(args.extension)
    files = [str(ROOT / "tests" / (name + ".py")) for name in MODULES]
    buffer = StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        status = pytest.main(["-q", "-p", "no:cacheprovider", *files])
    lines = buffer.getvalue().strip().splitlines()
    print("\n".join(lines[-40:]))
    write_json(args.output, dict(extension=file_identity(args.extension), modules=MODULES,
                                 threads=4, exit_code=int(status), summary=lines[-1] if lines else ""))
    raise SystemExit(status)
