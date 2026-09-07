"""Compare the standalone C ABI with the Python analyzer built from this tree.

Build mobile/CMakeLists.txt with BGSAGE_MOBILE_REFERENCE_PYTHON=ON and run with
the matching Python version. No app/backend dependencies or network access.
"""
from __future__ import annotations
import argparse
import ctypes as C
import json
import random
import sys
import time
from pathlib import Path


class Request(C.Structure):
    _fields_ = [("board", C.c_int * 26)] + [(name, C.c_int) for name in (
        "die1", "die2", "ply", "cube_value", "cube_owner", "away1", "away2",
        "is_crawford", "jacoby", "beaver", "budget_ms")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = args.build.resolve()
    sys.path[:0] = [str(build), str(root / "python")]
    import bgbot_cpp as cpp
    from bgsage import BgBotAnalyzer, STARTING_BOARD
    from bgsage.weights import default_weights, bearoff_db_path
    assert Path(cpp.__file__).parent.resolve() == build, "Reference extension came from a different build"
    binary = build / ("bgsage_mobile.dll" if sys.platform == "win32" else "libbgsage_mobile.dylib" if sys.platform == "darwin" else "libbgsage_mobile.so")
    lib = C.CDLL(str(binary))
    lib.bgsage_mobile_create.argtypes = [C.c_char_p, C.POINTER(C.c_char_p), C.POINTER(C.c_int), C.c_int, C.c_char_p]
    lib.bgsage_mobile_create.restype = C.c_void_p
    lib.bgsage_mobile_analyze.argtypes = [C.c_void_p, C.c_int, C.POINTER(Request)]
    lib.bgsage_mobile_analyze.restype = C.c_void_p
    lib.bgsage_mobile_free.argtypes = [C.c_void_p]
    lib.bgsage_mobile_destroy.argtypes = [C.c_void_p]
    lib.bgsage_mobile_last_error.restype = C.c_char_p
    weights = default_weights()
    paths = (C.c_char_p * len(weights.weight_paths_list))(*[str(p).encode() for p in weights.weight_paths_list])
    hidden = (C.c_int * len(weights.hidden_sizes_list))(*weights.hidden_sizes_list)
    initialization_started = time.perf_counter()
    handle = lib.bgsage_mobile_create(weights.strategy_type.encode(), paths, hidden, len(paths), str(bearoff_db_path()).encode())
    initialization_ms = (time.perf_counter() - initialization_started) * 1000
    assert handle, lib.bgsage_mobile_last_error()
    boards = [list(STARTING_BOARD)]
    rng = random.Random(471)
    board = boards[0]
    for turn in range(32):
        candidates = list(cpp.possible_moves(board, rng.randint(1, 6), rng.randint(1, 6)))
        board = list(cpp.flip_board(rng.choice(candidates) if candidates else board))
        if turn % 4 == 3: boards.append(board)
    bearoff = [0] * 26
    bearoff[1:5] = [4, 3, 4, 4]
    bearoff[22:25] = [-4, -5, -6]
    boards.append(bearoff)
    cases = 0
    max_error = 0.0
    timings = []
    try:
        for ply in (1, 2):
            reference = BgBotAnalyzer(eval_level=f"{ply}ply", parallel_threads=2)
            for index, board in enumerate(boards):
                match = index % 3 == 1
                settings = dict(cube_value=2 if index % 2 else 1, cube_owner=("centered", "player", "opponent")[index % 3],
                    away1=3 if match else 0, away2=5 if match else 0, is_crawford=False,
                    jacoby=not match and index % 2 == 0, beaver=not match and index % 2 == 0)
                if index == 1: settings.update(cube_value=1,away1=1,away2=3,is_crawford=True)
                if index == 4: settings.update(away1=5,away2=1)
                request = Request((C.c_int * 26)(*board), 3, 1, ply, settings["cube_value"], index % 3,
                    settings["away1"], settings["away2"], int(settings["is_crawford"]), int(settings["jacoby"]), int(settings["beaver"]), 1000)
                for operation in (0, 1):
                    started = time.perf_counter()
                    value = lib.bgsage_mobile_analyze(handle, operation, C.byref(request))
                    timings.append((time.perf_counter() - started) * 1000)
                    assert value, (index, ply, operation, lib.bgsage_mobile_last_error())
                    try: actual = json.loads(C.string_at(value))
                    finally: lib.bgsage_mobile_free(value)
                    if operation == 0:
                        expected = reference._analyzer.checker_play_analytics(board, 3, 1, **settings)
                        actual_moves = {tuple(m["board"]): m for m in actual["moves"]}
                        assert set(actual_moves) == {tuple(m["board"]) for m in expected}
                        for m in expected:
                            got = actual_moves[tuple(m["board"])]
                            errors = [abs(got[key]-m[key]) for key in ("equity", "cubeless_equity", "equity_diff")]
                            errors += [abs(a-b) for a, b in zip(got["probs"], m["probs"])]
                            max_error = max(max_error, *errors)
                            assert max(errors) < 1e-5, (index, ply, operation, errors)
                            assert got["eval_level"] == m["eval_level"]
                        if expected:
                            assert abs(actual["moves"][0]["equity"] - expected[0]["equity"]) < 1e-5
                    else:
                        expected = reference._analyzer.cube_action_analytics(board, **settings)
                        for key in ("should_double", "should_take", "is_beaver", "eval_level"):
                            assert actual[key] == expected[key], (index, ply, key, actual[key], expected[key])
                        errors = [abs(actual[key]-expected[key]) for key in ("equity_nd", "equity_dt", "equity_dp", "optimal_equity", "cubeless_equity")]
                        errors += [abs(a-b) for a,b in zip(actual["probs"], expected["probs"])]
                        max_error = max(max_error, *errors)
                        assert max(errors) < 1e-5, (index, ply, operation, errors)
                    cases += 1
        request.ply = 3
        assert not lib.bgsage_mobile_analyze(handle, 0, C.byref(request))
        request.ply = 1; request.board[1] = 500
        assert not lib.bgsage_mobile_analyze(handle, 0, C.byref(request))
    finally:
        lib.bgsage_mobile_destroy(handle)
    print(json.dumps({"cases": cases, "max_absolute_error": max_error, "max_native_ms": max(timings),
        "median_native_ms": sorted(timings)[len(timings)//2], "model_initialization_ms": initialization_ms,
        "reference": str(cpp.__file__), "binary": str(binary)}, indent=2))


if __name__ == "__main__": main()
