# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Tests for scripts/xg_batch_common.py cache-record construction.

Focus: the perspective convention. XG's checker-move rollout Result1 is from
the post-move (opponent-on-roll) perspective, so checker_cache_record must
negate the equity and invert the probabilities to match Sage's mover-perspective
benchmark. XG's stored N-ply eval, and cube rollouts, are already in the same
perspective as Sage and must NOT be flipped. These were verified empirically
against a real XG batch rollout (mean|-XG - Sage| = 0.008); the tests below pin
the behaviour with synthetic records so it can't silently regress.
"""

import os
import struct
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "build"))
sys.path.insert(0, str(_ROOT / "scripts"))
if sys.platform == "win32":
    _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    if (_ROOT / "build").is_dir():
        os.add_dll_directory(str(_ROOT / "build"))

from bgsage import xg_file  # noqa: E402
import xg_batch_common as xbc  # noqa: E402


def _make_xgr(result_nd, result_dt=None, rolled=1296, rolled2=0, min_roll=1296):
    """Build a one-context temp.xgr buffer with the given ND/DT TResults."""
    buf = bytearray(xg_file.ROLLOUT_CONTEXT_SIZE)
    struct.pack_into('<i', buf, 8, min_roll)          # min_roll
    struct.pack_into('<i', buf, 84, rolled)           # rolled (ND)
    struct.pack_into('<7f', buf, 2028, *result_nd)    # Result1
    if result_dt is not None:
        struct.pack_into('<7f', buf, 2056, *result_dt)  # Result2
    struct.pack_into('<i', buf, 2140, rolled2)        # rolled2 (D/T)
    return bytes(buf)


def _xg_result(win, gw, bw, gl, bl, equity):
    """Assemble an XG 7-float TResult [bg_loss,g_loss,loss,win,g_win,bg_win,eq]."""
    return [bl, gl, 1 - win, win, gw, bw, equity]


def test_checker_rollout_equity_is_negated():
    # XG stores the post-move (opponent) equity +0.30; Sage's mover-perspective
    # value is therefore -0.30. probs are opponent-frame win=0.60 -> mover win 0.40.
    result_nd = _xg_result(win=0.60, gw=0.18, bw=0.01, gl=0.12, bl=0.005, equity=0.30)
    move_rec = {
        "n_moves": 1,
        "analyze_m": 100,
        "rollout_indices": [0] + [-1] * 31,
        "moves": [{"board": [0] * 26, "level": 100, "eval": result_nd}],
    }
    decision = {"key": "k", "board": [0] * 26, "dice": [3, 1]}
    rec = xbc.checker_cache_record(decision, move_rec, _make_xgr(result_nd),
                                   "xg_rollout", {"file": "t"})
    m = rec["moves"][0]
    assert m["eval_level"] == "rollout"
    assert m["equity"] == pytest.approx(-0.30)          # negated
    # probs are inverted: mover win = opponent loss = 0.40
    assert m["probs"][0] == pytest.approx(0.40)         # win
    assert m["probs"][3] == pytest.approx(0.18)         # gammon_loss = opp gammon_win
    assert m["trials"] == 1296


def test_checker_non_rollout_eval_not_negated():
    # No rollout context -> keep XG's stored (mover-perspective) eval as-is.
    ev = _xg_result(win=0.55, gw=0.15, bw=0.01, gl=0.14, bl=0.006, equity=0.20)
    move_rec = {
        "n_moves": 1, "analyze_m": 3, "rollout_indices": [-1] * 32,
        "moves": [{"board": [0] * 26, "level": 3, "eval": ev}],
    }
    rec = xbc.checker_cache_record({"key": "k", "board": [0] * 26, "dice": [3, 1]},
                                   move_rec, None, "xg_rollerpp", {"file": "t"})
    m = rec["moves"][0]
    assert m["eval_level"] == "4ply"
    assert m["equity"] == pytest.approx(0.20)           # NOT negated
    assert m["probs"][0] == pytest.approx(0.55)         # win unchanged


def test_cube_rollout_not_negated():
    # Cube rollout is pre-roll (doubler on roll): ND/DT stay doubler-perspective.
    nd = _xg_result(win=0.62, gw=0.19, bw=0.008, gl=0.08, bl=0.003, equity=0.53)
    dt = _xg_result(win=0.62, gw=0.18, bw=0.008, gl=0.07, bl=0.002, equity=0.48)
    cube_rec = {
        "level": 100, "equity_nd": 0.53, "equity_dt": 0.48, "equity_dp": 1.0,
        "eval_nd": nd, "eval_dt": dt, "flag_double": 0, "is_beaver": 0,
        "rollout_index": 0,
    }
    rec = xbc.cube_cache_record({"key": "k", "board": [0] * 26}, cube_rec,
                                _make_xgr(nd, dt, rolled=1296, rolled2=1296),
                                "xg_rollout", {"file": "t"})
    assert rec["eval_level"] == "rollout"
    assert rec["equity_nd"] == pytest.approx(0.53)      # NOT negated
    assert rec["equity_dt"] == pytest.approx(0.48)
    assert rec["probs"][0] == pytest.approx(0.62)       # win unchanged
    assert rec["should_double"] is False                # min(dt,dp)=0.48 < nd=0.53
    assert rec["trials_nd"] == 1296 and rec["trials_dt"] == 1296


def _decision(equities):
    """A decision dict with moves sorted best-first at the given equities."""
    return {"key": "k", "board": [0] * 26, "dice": [3, 1],
            "moves": [{"board": [i] + [0] * 25, "equity": e}
                      for i, e in enumerate(equities)]}


def _move_rec(boards):
    return {"n_moves": len(boards),
            "moves": [{"board": list(b)} for b in boards]}


def test_flag_policy_threshold_within():
    # best 0.00; moves at -0.01, -0.015 within 0.02; -0.05 outside -> 3 flagged.
    d = _decision([0.0, -0.01, -0.015, -0.05, -0.2])
    rec = _move_rec([[i] + [0] * 25 for i in range(5)])
    plan = xbc.plan_checker_flags(d, rec, threshold=0.02, min_moves=2, max_moves=4)
    assert plan.indices == [0, 1, 2]


def test_flag_policy_min_moves_floor():
    # only best within 0.02 -> still flag min_moves (2).
    d = _decision([0.0, -0.10, -0.20])
    rec = _move_rec([[i] + [0] * 25 for i in range(3)])
    plan = xbc.plan_checker_flags(d, rec, threshold=0.02, min_moves=2, max_moves=4)
    assert plan.indices == [0, 1]


def test_flag_policy_max_moves_cap():
    # 6 moves all within 0.02 -> capped at max_moves (4).
    d = _decision([0.0, -0.005, -0.008, -0.011, -0.014, -0.018])
    rec = _move_rec([[i] + [0] * 25 for i in range(6)])
    plan = xbc.plan_checker_flags(d, rec, threshold=0.02, min_moves=2, max_moves=4)
    assert plan.indices == [0, 1, 2, 3]


def test_flag_policy_missing_from_xg_stored():
    # Sage's #2 move isn't in XG's stored list -> tracked, not flagged.
    d = _decision([0.0, -0.01, -0.015])
    rec = _move_rec([[0] + [0] * 25, [2] + [0] * 25])  # missing board [1]
    plan = xbc.plan_checker_flags(d, rec, threshold=0.02, min_moves=2, max_moves=4)
    assert plan.indices == [0, 1]          # boards [0] and [2] -> XG indices 0,1
    assert plan.sage_missing == 1          # board [1] not stored
    assert not plan.sage_best_missing


def test_cube_rollout_nd_only_not_cached_as_rollout():
    # rolled2 == 0 (D/T line never rolled) -> do not upgrade to rollout level.
    nd = _xg_result(win=0.62, gw=0.19, bw=0.008, gl=0.08, bl=0.003, equity=0.53)
    cube_rec = {
        "level": 3, "equity_nd": 0.50, "equity_dt": 0.45, "equity_dp": 1.0,
        "eval_nd": nd, "eval_dt": nd, "flag_double": 0, "is_beaver": 0,
        "rollout_index": 0,
    }
    rec = xbc.cube_cache_record({"key": "k", "board": [0] * 26}, cube_rec,
                                _make_xgr(nd, None, rolled=1296, rolled2=0),
                                "xg_rollout", {"file": "t"})
    assert rec["eval_level"] == "4ply"                  # stayed at stored level
    assert rec["equity_nd"] == pytest.approx(0.50)      # stored eval, not rollout
    assert "trials_nd" not in rec


# ---------------------------------------------------------------------------
# XG's move ranking: moves[0] is its pick, NOT argmax of the stored equities.
# ---------------------------------------------------------------------------

_XG_DIR = _ROOT / "data" / "money_benchmark" / "xg"


@pytest.mark.skipif(not (_XG_DIR / "seed_1_pp.xg").exists(),
                    reason="money benchmark XG data not present")
def test_xg_move_order_is_the_pick_not_the_argmax():
    """XG evaluates leading candidates deeply and the tail shallowly, so the
    biggest stored equity is often a move XG would never play.

    xg_level_picks must take moves[0]. Over the full money benchmark the argmax
    disagrees on ~3% of records; this pins that the two genuinely differ (so the
    choice matters) and that the disagreement is always the shallow-eval trap.
    """
    n_rec = n_div = n_mixed = 0
    shallower = 0
    for path in sorted(_XG_DIR.glob("seed_*.xg"))[:40]:
        tempxg = xg_file.XgArchive.load(path).get("temp.xg")
        for off, rt in xg_file.iter_records(tempxg):
            if rt != xg_file.TS_MOVE:
                continue
            moves = xg_file.parse_move_record(tempxg, off)["moves"]
            if len(moves) < 2:
                continue
            n_rec += 1
            if len({m["level"] for m in moves}) > 1:
                n_mixed += 1
            eqs = [m["eval"][6] for m in moves]
            am = max(range(len(moves)), key=lambda i: eqs[i])
            if am != 0:
                n_div += 1
                # The usurper is always evaluated no deeper than XG's own pick.
                if moves[am]["level"] <= moves[0]["level"]:
                    shallower += 1

    assert n_rec > 100, f"too few move records to be meaningful ({n_rec})"
    assert n_mixed > n_rec // 2, "expected most records to carry mixed eval levels"
    assert n_div > 0, "argmax never diverged -- the fix would be untestable here"
    assert shallower == n_div, (
        f"{n_div - shallower} argmax winners were evaluated DEEPER than moves[0]; "
        "the shallow-eval explanation for the divergence no longer holds")
