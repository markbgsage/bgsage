# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""A Python exception raised inside a rollout progress callback must not kill the process.

``BgBotAnalyzer.cube_action(progress_callback=...)`` reports trial progress from
the rollout's WORKER threads: the pybind layer releases the GIL for the rollout,
and each reporting thread re-acquires it (``py::gil_scoped_acquire``) to call
back into Python. When that Python call raises, pybind translates it into a C++
``py::error_already_set`` thrown inside the worker thread's lambda -- and it
escapes into the thread runner, which leaves the interpreter in a broken state.

Observed on the pre-fix engine with the small config used below (36 trials,
truncation 5, 1-ply, 4 threads), three consecutive runs:

    PROPAGATED ValueError: boom      -> then SIGSEGV at teardown (exit 139)
    <died before reporting>          -> exit 127
    PROPAGATED ValueError: boom      -> then SIGSEGV at teardown (exit 139)

So the exception usually DOES reach the caller; the damage shows up afterwards,
when the process crashes on the way out. With more threads or a longer rollout
the same fault also appears as a hard hang -- every other worker blocks forever
in ``gil_scoped_acquire`` waiting on a GIL the faulted worker never released.

Two callbacks are exercised, because the failure is about an exception escaping
the callback and not about which exception it is:

  * one that raises explicitly, and
  * one with the WRONG ARITY. The documented signature is
    ``callback(completed, total, partial)``; a two-argument callback is an easy
    mistake to make and produced exactly the same process-level death.

Both run in a SUBPROCESS on purpose: the failure mode is a crash or a hang, and
neither can be asserted on from inside the process that suffers it. Each child
is bounded by a timeout so the hang variant fails the test instead of wedging
the suite. A healthy run of either child takes ~0.2 s.
"""

from pathlib import Path
import subprocess
import sys

import pytest

#: Generous relative to the ~0.2 s a healthy run takes; this only has to be
#: short enough that the deadlock variant fails rather than hangs the suite.
_CHILD_TIMEOUT_S = 90

# A contact position with a live cube decision, so the rollout actually runs
# trials and therefore actually reports progress.
_BOARD = [1, -2, 0, 0, 2, 2, 3, 0, 3, 0, 0, 2, -3, 2, -1, 0, 0, 0,
          -2, -2, -2, -2, 0, 1, 0, 0]

_PREAMBLE = """
import os, sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
if sys.platform == "win32":
    _cuda = r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64"
    if os.path.isdir(_cuda):
        _cuda_handle = os.add_dll_directory(_cuda)

spec = spec_from_file_location("bgbot_cpp", {extension!r})
cpp = module_from_spec(spec)
sys.modules["bgbot_cpp"] = cpp
spec.loader.exec_module(cpp)
sys.path.insert(0, str(Path({package!r}).parent.parent))
import bgsage
from bgsage import BgBotAnalyzer
from bgsage.weights import bearoff_db_path
assert Path(cpp.__file__).resolve() == Path({extension!r})
assert Path(bgsage.__file__).resolve() == Path({package!r})
weights = bgsage.default_weights()
weights.validate()
assert [str(Path(p).resolve()) for p in weights.weight_paths_list] == {weights!r}
database = bearoff_db_path()
assert (str(Path(database).resolve()) if database else None) == {database!r}
print("VERIFIED_ENGINE_FILES")

BOARD = {board!r}

# Small and truncated on purpose: enough trials to fire several progress
# reports from multiple threads, fast enough to keep the test quick.
az = BgBotAnalyzer(
    eval_level="rollout", cubeful=True, parallel_threads=4,
    n_trials=36, truncation_depth=5, decision_ply=1,
    seed=42, target_se=0.0, max_batches=1,
)

{callback}

try:
    az.cube_action(BOARD, cube_value=1, cube_owner="centered",
                   progress_callback=cb)
except {expected} as exc:
    print("PROPAGATED:" + type(exc).__name__)
except BaseException as exc:  # noqa: BLE001 - report whatever else escaped
    print("UNEXPECTED:" + type(exc).__name__)
else:
    print("NOT_RAISED")
print("REACHED_END")
"""


def _run_child(callback_src: str, expected: str) -> subprocess.CompletedProcess:
    import bgbot_cpp
    import bgsage
    from bgsage.weights import bearoff_db_path

    weights = bgsage.default_weights()
    weights.validate()
    database = bearoff_db_path()
    src = _PREAMBLE.format(
        extension=str(Path(bgbot_cpp.__file__).resolve()),
        package=str(Path(bgsage.__file__).resolve()),
        weights=[str(Path(p).resolve()) for p in weights.weight_paths_list],
        database=str(Path(database).resolve()) if database else None,
        board=_BOARD, callback=callback_src, expected=expected,
    )
    try:
        return subprocess.run(
            [sys.executable, "-u", "-c", src],
            capture_output=True, text=True, timeout=_CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"rollout hung for >{_CHILD_TIMEOUT_S}s after the progress callback "
            f"raised - a worker thread faulted while holding the GIL, so every "
            f"other reporting thread is blocked in gil_scoped_acquire forever"
        )


def _assert_clean(proc: subprocess.CompletedProcess, expected_exc: str) -> None:
    out = proc.stdout or ""
    err = proc.stderr or ""
    detail = f"\n--- exit={proc.returncode} ---\nstdout:\n{out}\nstderr:\n{err}"

    # The process-level assertion is the point of this test. A negative return
    # code (POSIX signal) or 139/134/127 (shell-reported SIGSEGV/abort) all mean
    # the interpreter died rather than returned.
    assert proc.returncode == 0, (
        "the rollout process died after a progress callback raised; it must "
        "unwind cleanly instead" + detail
    )
    assert "VERIFIED_ENGINE_FILES" in out, detail
    assert "REACHED_END" in out, (
        "the child never reached the end of the script" + detail
    )
    assert f"PROPAGATED:{expected_exc}" in out, (
        f"expected the callback's {expected_exc} to propagate to the "
        f"cube_action caller" + detail
    )


def test_raising_progress_callback_does_not_kill_the_process():
    """A callback that raises must surface the error, not crash the interpreter."""
    proc = _run_child(
        'def cb(completed, total, partial):\n'
        '    raise ValueError("boom from progress callback")\n',
        expected="ValueError",
    )
    _assert_clean(proc, "ValueError")


def test_wrong_arity_progress_callback_does_not_kill_the_process():
    """A two-argument callback is a TypeError, not a segfault.

    The documented signature is ``callback(completed, total, partial)``.
    """
    proc = _run_child(
        'def cb(completed, total):\n'
        '    pass\n',
        expected="TypeError",
    )
    _assert_clean(proc, "TypeError")


def test_checker_progress_is_opt_in():
    from bgsage import BgBotAnalyzer

    analyzer = BgBotAnalyzer(
        eval_level="rollout", parallel_threads=4, n_trials=36,
        truncation_depth=2, decision_ply=1, seed=42,
    )
    inner = analyzer._analyzer._inner
    native = inner._rollout_strategy
    bridges = []

    class Monitor:
        def cubeful_evaluate_board(self, *args, **kwargs):
            bridges.append(kwargs["progress"])
            return native.cubeful_evaluate_board(*args, **kwargs)

    inner._rollout_strategy = Monitor()
    without = analyzer.checker_play(_BOARD, 4, 2)
    assert bridges and all(callback is None for callback in bridges)

    bridges.clear()
    assert analyzer.checker_play(_BOARD, 4, 2, progress_callback=False) == without
    assert bridges and all(callback is None for callback in bridges)

    bridges.clear()
    reports = []
    with_progress = analyzer.checker_play(
        _BOARD, 4, 2,
        progress_callback=lambda done, total, partial: reports.append((done, total)),
    )
    assert with_progress == without
    assert any(callable(callback) for callback in bridges)
    assert reports and any(done == total > 0 for done, total in reports)


def test_checker_native_cancellation_without_progress():
    from bgsage import BgBotAnalyzer
    from bgsage.analyzer import RolloutCancelled

    analyzer = BgBotAnalyzer(
        eval_level="rollout", parallel_threads=4, n_trials=36,
        truncation_depth=2, decision_ply=1, seed=42,
    )
    inner = analyzer._analyzer._inner
    native = inner._rollout_strategy

    class CancelAtTrialEntry:
        def cubeful_evaluate_board(self, *args, **kwargs):
            assert kwargs["progress"] is None
            native.cancel()
            return native.cubeful_evaluate_board(*args, **kwargs)

    inner._rollout_strategy = CancelAtTrialEntry()
    with pytest.raises(RolloutCancelled):
        analyzer.checker_play(_BOARD, 4, 2)
    inner._rollout_strategy = native
    analyzer.reset_cancel()
    assert analyzer.checker_play(_BOARD, 4, 2).moves


@pytest.mark.parametrize("threads,reports_per_batch", [(1, 37), (4, 5)])
@pytest.mark.parametrize("target_se,batches", [(0.0, 1), (1e-12, 2)])
def test_full_game_cube_progress_cadence(threads, reports_per_batch, target_se, batches):
    from bgsage import BgBotAnalyzer

    # An odd batch size exercises the final partial notification; the tight
    # target exercises the same reporting policy in a second full-game batch.
    board = [0, 1, 0, 3, 3, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0,
             -1, -2, -4, -1, -3, -1, -2, 0]
    analyzer = BgBotAnalyzer(eval_level="rollout", n_trials=37, truncation_depth=0,
                             parallel_threads=threads, target_se=target_se, max_batches=2, seed=42)
    reports = []
    result = analyzer.cube_action(board, progress_callback=lambda done, total, partial: reports.append((done, total)))
    assert result.equity_nd_se > target_se
    assert len(reports) == reports_per_batch * batches
    assert reports.count((37, 37)) == batches
    assert all(1 <= done <= total == 37 for done, total in reports)
