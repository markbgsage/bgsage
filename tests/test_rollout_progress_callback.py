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

import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.abspath(os.path.join(_HERE, '..', '..', 'build'))
_PKG_DIR = os.path.abspath(os.path.join(_HERE, '..', 'python'))

#: Generous relative to the ~0.2 s a healthy run takes; this only has to be
#: short enough that the deadlock variant fails rather than hangs the suite.
_CHILD_TIMEOUT_S = 90

# A contact position with a live cube decision, so the rollout actually runs
# trials and therefore actually reports progress.
_BOARD = [1, -2, 0, 0, 2, 2, 3, 0, 3, 0, 0, 2, -3, 2, -1, 0, 0, 0,
          -2, -2, -2, -2, 0, 1, 0, 0]

_PREAMBLE = """
import os, sys
sys.path.insert(0, {build!r})
sys.path.insert(0, {pkg!r})
if sys.platform == "win32":
    _cuda = r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)

from bgsage import BgBotAnalyzer

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
    src = _PREAMBLE.format(build=_BUILD_DIR, pkg=_PKG_DIR, board=_BOARD,
                           callback=callback_src, expected=expected)
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
