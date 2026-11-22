# SPDX-License-Identifier: MPL-2.0
"""Run existing specialist routing and encoder tests against the selected extension."""
import sys
import psutil
import pytest
from screen import ROOT, binary, load_engine

if __name__ == "__main__":
    psutil.Process().cpu_affinity([4, 5, 6, 7])
    load_engine(binary(sys.argv[1]), ROOT / "build/pass2_checkpoint/python")
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", str(ROOT / "tests/test_backgame_category.py"),
                                 str(ROOT / "tests/test_extended_inputs.py"), str(ROOT / "tests/test_extended_inputs_v2.py")]))
