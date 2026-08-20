# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Tests for bgsage.xg_file — XG archive container + record access.

These validate against real batch-analyzed .xg files in data/money_benchmark/xg
(local build artifacts). All tests are skipped when that data is not present.
"""

import json
import os
import struct
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "build"))
if sys.platform == "win32":
    _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    if (_ROOT / "build").is_dir():
        os.add_dll_directory(str(_ROOT / "build"))

from bgsage import xg_file  # noqa: E402

_XG_DIR = _ROOT / "data" / "money_benchmark" / "xg"
_STAGE1 = _ROOT / "data" / "money_benchmark" / "build" / "stage1"
_SAMPLE = _XG_DIR / "seed_1_pp.xg"

pytestmark = pytest.mark.skipif(not _SAMPLE.exists(),
                                reason="money benchmark XG data not present")


def test_archive_roundtrip_unmodified(tmp_path):
    arch = xg_file.XgArchive.load(_SAMPLE)
    out = tmp_path / "copy.xg"
    arch.save(out)
    arch2 = xg_file.XgArchive.load(out)  # implies trailer CRC verified
    assert [e.name for e in arch2.entries] == [e.name for e in arch.entries]
    for e1, e2 in zip(arch.entries, arch2.entries):
        assert e1.data() == e2.data()
        assert e1.crc == e2.crc
    assert arch2.prefix == arch.prefix


def test_archive_replace_entry_roundtrip(tmp_path):
    arch = xg_file.XgArchive.load(_SAMPLE)
    tempxg = arch.get("temp.xg")
    arch.set("temp.xg", tempxg)  # re-set identical content -> recompressed
    out = tmp_path / "reset.xg"
    arch.save(out)
    arch2 = xg_file.XgArchive.load(out)
    assert arch2.get("temp.xg") == tempxg
    assert arch2.get("temp.xgi") == arch.get("temp.xgi")


def test_header_and_xgi():
    arch = xg_file.XgArchive.load(_SAMPLE)
    tempxg = arch.get("temp.xg")
    hdr = xg_file.parse_header(tempxg, 0)
    assert hdr["version"] == 30
    assert hdr["match_length"] == 99999  # money session
    assert hdr["jacoby"] and hdr["beaver"]
    assert hdr["tot_timedelay_move"] == 0
    assert xg_file.rebuild_xgi(tempxg) == arch.get("temp.xgi")


def test_all_stage1_decisions_locate():
    """Every captured decision of the game must be findable by content."""
    sys.path.insert(0, str(_ROOT / "scripts"))
    import xg_batch_common as xbc

    stage1 = json.loads((_STAGE1 / "seed_1.json").read_text(encoding="utf-8"))
    arch = xg_file.XgArchive.load(_SAMPLE)
    index = xbc.GameFileIndex(arch.get("temp.xg"))
    missing = []
    for d in stage1["decisions"]:
        off = index.find(d["kind"], d["board"], d.get("dice") or ())
        if off is None:
            missing.append((d["kind"], d.get("turn")))
    assert not missing, f"unlocated decisions: {missing}"


def test_move_record_fields_match_xg_compare():
    """Record-level equities must agree with the xg_compare turn parser."""
    from bgsage.xg_compare import parse_xg_game

    raw = _SAMPLE.read_bytes()
    turns = [t for t in parse_xg_game(raw) if t.get("checker_analysis")]
    arch = xg_file.XgArchive.load(_SAMPLE)
    tempxg = arch.get("temp.xg")
    move_recs = [xg_file.parse_move_record(tempxg, off)
                 for off, t in xg_file.iter_records(tempxg) if t == xg_file.TS_MOVE]
    with_analysis = [r for r in move_recs if r["n_moves"] > 0]
    assert len(with_analysis) == len(turns)
    for rec, turn in zip(with_analysis, turns):
        ca = turn["checker_analysis"]
        assert len(ca) == rec["n_moves"]
        for m_rec, m_turn in zip(rec["moves"], ca):
            assert m_rec["eval"][6] == pytest.approx(m_turn["equity"], abs=1e-6)


def test_mark_and_reload(tmp_path):
    arch = xg_file.XgArchive.load(_SAMPLE)
    tempxg = bytearray(arch.get("temp.xg"))

    move_offs = [off for off, t in xg_file.iter_records(tempxg) if t == xg_file.TS_MOVE]
    cube_offs = [off for off, t in xg_file.iter_records(tempxg) if t == xg_file.TS_CUBE]
    baseline_move = xg_file.parse_move_record(tempxg, move_offs[0])
    assert baseline_move["timedelay"] == 0

    xg_file.set_move_timedelay(tempxg, move_offs[0], 0b101)   # moves 0 and 2
    xg_file.set_cube_timedelay(tempxg, cube_offs[0], marked=True)
    xg_file.set_header_timedelay_totals(tempxg, 0, 1, 1)
    arch.set("temp.xg", bytes(tempxg))
    arch.set("temp.xgi", xg_file.rebuild_xgi(tempxg))
    out = tmp_path / "marked.xg"
    arch.save(out)

    arch2 = xg_file.XgArchive.load(out)
    t2 = arch2.get("temp.xg")
    rec = xg_file.parse_move_record(t2, move_offs[0])
    assert rec["timedelay"] == 0b101 and rec["timedelay_done"] == 0
    crec = xg_file.parse_cube_record(t2, cube_offs[0])
    assert crec["timedelay"] and not crec["timedelay_done"]
    hdr = xg_file.parse_header(t2, 0)
    assert hdr["tot_timedelay_move"] == 1 and hdr["tot_timedelay_cube"] == 1

    # nothing else in the record changed
    rec_before = bytes(tempxg[move_offs[0]:move_offs[0] + xg_file.TSAVEREC_SIZE])
    rec_after = t2[move_offs[0]:move_offs[0] + xg_file.TSAVEREC_SIZE]
    assert rec_before == rec_after  # tempxg already carries the marks
    # and untouched records are byte-identical to the original file
    orig = xg_file.XgArchive.load(_SAMPLE).get("temp.xg")
    assert t2[move_offs[1]:move_offs[1] + xg_file.TSAVEREC_SIZE] == \
        orig[move_offs[1]:move_offs[1] + xg_file.TSAVEREC_SIZE]


def test_cube_record_sanity():
    arch = xg_file.XgArchive.load(_SAMPLE)
    tempxg = arch.get("temp.xg")
    recs = [xg_file.parse_cube_record(tempxg, off)
            for off, t in xg_file.iter_records(tempxg) if t == xg_file.TS_CUBE]
    analyzed = [r for r in recs if r["analyze_c"] >= 0 and any(r["eval_nd"])]
    assert analyzed, "expected at least one analyzed cube record"
    for r in analyzed:
        assert r["equity_dp"] == pytest.approx(1.0)
        probs = xg_file.xg_eval_to_probs(r["eval_nd"])
        assert 0.0 <= probs[0] <= 1.0
        assert r["rollout_index"] == -1  # no rollouts in batch-analyze output


def test_rollout_context_layout_constants():
    """TRolloutContext layout: last field group must land exactly at 2184 bytes."""
    # Synthetic buffer: place recognizable values at computed offsets and read back.
    buf = bytearray(xg_file.ROLLOUT_CONTEXT_SIZE)
    struct.pack_into('<i', buf, 84, 1296)          # rolled
    struct.pack_into('<7f', buf, 2028, *([0.1] * 6 + [0.25]))  # result_nd
    struct.pack_into('<f', buf, 2132, 123.5)       # duration
    struct.pack_into('<i', buf, 2140, 648)         # rolled2
    struct.pack_into('<HH', buf, 2170, 2, 10)      # ver 2.10
    ctx = xg_file.parse_rollout_context(bytes(buf), 0)
    assert ctx["rolled"] == 1296
    assert ctx["result_nd"][6] == pytest.approx(0.25)
    assert ctx["duration"] == pytest.approx(123.5)
    assert ctx["rolled2"] == 648
    assert (ctx["ver_maj"], ctx["ver_min"]) == (2, 10)


def _cube_template():
    """A real tsCube record to clone (these carry Double = -2 in practice)."""
    tempxg = xg_file.XgArchive.load(_SAMPLE).get("temp.xg")
    for off, rt in xg_file.iter_records(tempxg):
        if rt == xg_file.TS_CUBE:
            return tempxg[off:off + xg_file.TSAVEREC_SIZE]
    pytest.skip("no tsCube record in sample")


def test_cube_double_field_gates_whether_xg_analyzes():
    """Double = -2 ("no cube access") means XG never analyzes the record.

    Measured over data/money_benchmark/xg: every -2 record comes back
    unanalyzed, every 0/1 record comes back analyzed. This is why
    build_cube_record must WRITE the field rather than inherit it.
    """
    seen = {}
    for path in sorted(_XG_DIR.glob("seed_*.xg"))[:20]:
        tempxg = xg_file.XgArchive.load(path).get("temp.xg")
        for off, rt in xg_file.iter_records(tempxg):
            if rt != xg_file.TS_CUBE:
                continue
            r = xg_file.parse_cube_record(tempxg, off)
            analyzed = r["flag_double"] != -100
            seen.setdefault(r["double"], set()).add(analyzed)
    assert -2 in seen and seen[-2] == {False}, seen
    assert 0 in seen and seen[0] == {True}, seen


def test_build_cube_record_writes_double_take_cube_b():
    """The template's Double/Take/CubeB must never leak into a built record."""
    template = _cube_template()
    assert xg_file.parse_cube_record(template, 0)["double"] == -2, \
        "expected a template carrying the -2 that used to be inherited"
    board = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5,
             5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]

    default = xg_file.parse_cube_record(
        xg_file.build_cube_record(template, board), 0)
    assert (default["double"], default["take"], default["cube_b"]) == (0, -1, 0)
    assert tuple(default["mover_board"]) == tuple(board)

    doubled = xg_file.parse_cube_record(
        xg_file.build_cube_record(template, board, double=1, take=1, cube_b=1), 0)
    assert (doubled["double"], doubled["take"], doubled["cube_b"]) == (1, 1, 1)

    filler = xg_file.parse_cube_record(
        xg_file.build_cube_record(template, board, double=-2), 0)
    assert filler["double"] == -2
