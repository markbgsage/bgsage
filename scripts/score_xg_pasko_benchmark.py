# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Score eXtreme Gammon (XG) on the Paskogammon position-eval benchmark.

The benchmark scores ``|eval(B) - rollout(B)|`` where ``eval`` is the *post-move*
static equity (``evaluate_board(B, B)``: positive just moved into B, opponent to
roll) and ``rollout(B)`` is the Sage rolled-out equity in ``pasko-benchmark-rollout``.
XG has no batch position-eval, only game/decision analysis -- and it only evaluates
a board as a *played-move candidate*, which is also post-move. So each position B is
presented as a played move:

    generate  -- for each B find a predecessor P (positive on roll) + dice with
                 ``B in possible_moves(P, dice)``, and write a 1-turn game
                 ``P --dice--> B``. B is the played move, so XG always evaluates it
                 (played moves are never pruned). Many 1-turn games are packed per
                 .xg shard, each shard with a .sidecar.jsonl mapping to rollout targets.
    (you)     -- XG Batch Analyze the shard folder at the desired ply, "Save games
                 after analyze" ON.
    score     -- harvest XG's eval of each B (== eval(B) in the benchmark convention)
                 and report XG's ER/PR vs the rollouts, alongside Sage models.

~6.5% of positions have a checker on the bar (no legal "just moved" state -> no
predecessor) and are dropped; the remainder reproduce the models' known scores, so
the round-trip is self-validating (Stage 9 ~27.2 on the packable subset vs 27.5 full).

Needs a real XG .xg file as a record template (default: an example game you saved
from XG). Usage:

    python bgsage/scripts/score_xg_pasko_benchmark.py generate \
        --template logs/pasko_xg/example.xg --out-dir logs/pasko_xg/benchmark_full
    #  ... XG Batch Analyze that folder at 1-ply, save ...
    python bgsage/scripts/score_xg_pasko_benchmark.py score \
        logs/pasko_xg/benchmark_full --models
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BGSAGE_ROOT = os.path.dirname(_SCRIPT_DIR)
_PARENT_ROOT = os.path.dirname(_BGSAGE_ROOT)

if sys.platform == 'win32':
    _cuda = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    for _d in (os.path.join(_PARENT_ROOT, 'build'), os.path.join(_BGSAGE_ROOT, 'build')):
        if os.path.isdir(_d):
            os.add_dll_directory(_d)
sys.path.insert(0, os.path.join(_PARENT_ROOT, 'build'))
sys.path.insert(0, os.path.join(_BGSAGE_ROOT, 'build'))
sys.path.insert(0, os.path.join(_BGSAGE_ROOT, 'python'))

import bgbot_cpp
from bgsage import xg_file as xf

DATA_DIR = os.path.join(_BGSAGE_ROOT, 'data')
MODELS_DIR = os.path.join(_BGSAGE_ROOT, 'models')


# ---------------------------------------------------------------------------
# Predecessor construction: positive plays P --dice--> B
# ---------------------------------------------------------------------------


def _on_board(b):
    return sum(x for i, x in enumerate(b) if 1 <= i <= 25 and x > 0)


def _reverse_half(board, die, t):
    """Undo a forward positive move (single ``die``) that landed a checker on ``t``."""
    b = list(board)
    if t == 0:                      # undo a bear-off: checker came from point ``die``
        if not (1 <= die <= 24):
            return None
        b[die] += 1
        return b
    if not (1 <= t <= 24) or b[t] <= 0:
        return None
    b[t] -= 1
    src = t + die
    if src <= 24:
        b[src] += 1
    elif t == 25 - die:             # entered from the bar
        b[25] += 1
    else:
        return None
    return b


def find_predecessor(B, max_moves=16):
    """A position P (positive on roll) + dice with ``B in possible_moves(P, dice)``,
    keeping the legal-move count <= ``max_moves``. Tries 2- then 1-half-move undos.
    Returns ``(P, d1, d2)`` or ``None`` (e.g. B has a checker on the bar)."""
    B = list(B)
    on = _on_board(B)
    pairs = ([(a, b) for a in range(1, 7) for b in range(a + 1, 7)]
             + [(a, a) for a in range(1, 7)])
    for d1, d2 in pairs:            # two half-moves undone (the common case)
        for t1 in range(25):
            if t1 == 0 and on >= 15:
                continue
            mid = _reverse_half(B, d1, t1)
            if mid is None:
                continue
            mon = _on_board(mid)
            for t2 in range(25):
                if t2 == 0 and mon >= 15:
                    continue
                P = _reverse_half(mid, d2, t2)
                if P is None or _on_board(P) > 15:
                    continue
                res = bgbot_cpp.possible_moves(P, d1, d2)
                if list(B) in res and 1 <= len(set(map(tuple, res))) <= max_moves:
                    return P, d1, d2
    for d1, d2 in pairs:            # one half-move undone (bar-entry / blocked die)
        for t1 in range(25):
            if t1 == 0 and on >= 15:
                continue
            P = _reverse_half(B, d1, t1)
            if P is None or _on_board(P) > 15:
                continue
            res = bgbot_cpp.possible_moves(P, d1, d2)
            if list(B) in res and 1 <= len(set(map(tuple, res))) <= max_moves:
                return P, d1, d2
    return None


def derive_half_moves(pre, post, d1, d2):
    """Exact per-die (from, to) half-move sequence (bgsage points) for pre -> post.
    XG replays half-moves, so each must use a single die."""
    target = tuple(post)
    orders = [[d1] * k for k in (4, 3, 2, 1)] if d1 == d2 else [[d1, d2], [d2, d1]]

    def dfs(board, dice_left):
        if tuple(board) == target:
            return []
        if not dice_left:
            return None
        for mv in bgbot_cpp.possible_single_die_moves(board, dice_left[0]):
            sub = dfs(mv["board"], dice_left[1:])
            if sub is not None:
                return [(mv["from"], mv["to"])] + sub
        return dfs(board, dice_left[1:])

    best = None
    for order in orders:
        res = dfs(pre, order)
        if res is not None and (best is None or len(res) > len(best)):
            best = res
    return best or []


# ---------------------------------------------------------------------------
# .xg assembly (multi-game: each B a 1-turn game) + IO
# ---------------------------------------------------------------------------


def _templates(template_path):
    """One record of each type from a real .xg, to clone."""
    tempxg = xf.XgArchive.load(template_path).get("temp.xg")
    tpl = {}
    for off, rt in xf.iter_records(tempxg):
        tpl.setdefault(rt, tempxg[off:off + xf.TSAVEREC_SIZE])
    return tpl


def write_shard(games, out_path, template_path, tpl):
    """games: list of {pre, dice, post, half_moves} (positive on roll, plays pre->post=B).
    Reuses the template archive's prefix (RichGameHeader + thumbnail) so the .xg opens."""
    stream = bytearray(tpl[xf.TS_HEADER_MATCH])
    for gi, g in enumerate(games):
        P, (d1, d2), B, hmoves = g["pre"], g["dice"], g["post"], g["half_moves"]
        stream += xf.build_game_header(tpl[xf.TS_HEADER_GAME], P, gi + 1, 0, gi)
        stream += xf.build_cube_record(tpl[xf.TS_CUBE], P, actif=1)
        stream += xf.build_move_record(tpl[xf.TS_MOVE], P, B, d1, d2, hmoves, actif=1)
        if xf.TS_FOOTER_GAME in tpl:   # each 1-turn game "ends" with P1 resigning a single
            stream += xf.build_game_footer(tpl[xf.TS_FOOTER_GAME], 0, gi + 1, -1, 1, 101)
    xf.set_header_timedelay_totals(stream, 0, 0, 0)
    arch = xf.XgArchive.load(template_path)
    arch.set("temp.xg", bytes(stream))
    arch.set("temp.xgi", xf.rebuild_xgi(bytes(stream)))
    arch.save(out_path)


def load_rollout(path):
    """(board, equity) per finished line (26 ints + 5 rolled-out probs)."""
    out = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 31:
                continue
            b = [int(x) for x in p[:26]]
            pr = [float(x) for x in p[26:31]]
            out.append((b, 2 * pr[0] - 1 + pr[1] - pr[3] + pr[2] - pr[4]))
    return out


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_generate(args):
    template = args.template if os.path.isabs(args.template) \
        else os.path.join(_BGSAGE_ROOT, args.template)
    tpl = _templates(template)
    rollout_path = args.rollout if os.path.isabs(args.rollout) \
        else os.path.join(DATA_DIR, args.rollout)
    positions = load_rollout(rollout_path)
    if args.limit:
        positions = positions[:args.limit]

    packed, dropped = [], 0
    for b, eq in positions:
        pred = find_predecessor(b)
        if pred is None:
            dropped += 1
            continue
        P, d1, d2 = pred
        packed.append(({"pre": P, "dice": (d1, d2), "post": list(b),
                        "half_moves": derive_half_moves(P, list(b), d1, d2)},
                       {"board": b, "rollout_eq": eq}))

    os.makedirs(args.out_dir, exist_ok=True)
    n_shards = (len(packed) + args.per_file - 1) // args.per_file
    for s in range(n_shards):
        chunk = packed[s * args.per_file:(s + 1) * args.per_file]
        out = os.path.join(args.out_dir, f"bench_shard_{s:03d}.xg")
        write_shard([g for g, _ in chunk], out, template, tpl)
        with open(out + ".sidecar.jsonl", "w") as f:
            for gi, (_, sc) in enumerate(chunk):
                f.write(json.dumps({"game_index": gi, **sc}) + "\n")

    print(f"{len(positions)} positions: {len(packed)} packed, {dropped} dropped "
          f"({100 * len(packed) / max(1, len(positions)):.1f}%)")
    print(f"{n_shards} shards (<= {args.per_file} games each) in {args.out_dir}")
    print("\nNext: XG Batch Analyze that folder at the target ply ('Save games "
          "after analyze' ON), then:")
    print(f"  python bgsage/scripts/score_xg_pasko_benchmark.py score {args.out_dir} --models")


def _harvest_file(path, rows, levels):
    sidecar = [json.loads(l) for l in open(str(path) + ".sidecar.jsonl")]
    tx = xf.XgArchive.load(path).get("temp.xg")
    moves = [off for off, rt in xf.iter_records(tx) if rt == xf.TS_MOVE]
    if len(moves) != len(sidecar):
        raise ValueError(f"{path}: {len(moves)} move recs != {len(sidecar)} sidecar rows")
    unanalyzed = missing = 0
    for off, sc in zip(moves, sidecar):
        rec = xf.parse_move_record(tx, off)
        B = tuple(sc["board"])
        if rec["n_moves"] <= 0:
            unanalyzed += 1
            continue
        cand = next((m for m in rec["moves"] if tuple(m["board"]) == B), None)
        if cand is None:
            missing += 1
            continue
        levels[xf.player_level_label(cand["level"])] += 1
        rows.append({"board": list(B), "rollout_eq": sc["rollout_eq"],
                     "xg_eq": float(cand["eval"][6])})
    return unanalyzed, missing


def cmd_score(args):
    files = []
    if os.path.isdir(args.path):
        files = sorted(os.path.join(args.path, f) for f in os.listdir(args.path)
                       if f.endswith(".xg")
                       and os.path.exists(os.path.join(args.path, f) + ".sidecar.jsonl"))
    else:
        files = [args.path]

    rows, levels = [], Counter()
    tot_un = tot_miss = 0
    for f in files:
        un, miss = _harvest_file(f, rows, levels)
        tot_un += un
        tot_miss += miss
    n = len(rows)
    print(f"{len(files)} shard(s): {n} harvested, {tot_un} unanalyzed, "
          f"{tot_miss} candidate-missing.  XG eval levels: {dict(levels)}")
    if not n:
        print("Nothing analyzed yet -- run XG Batch Analyze (save games) first.")
        return

    er = statistics.mean(abs(r["xg_eq"] - r["rollout_eq"]) for r in rows) * 1000
    print(f"\n{'Engine':<28s} {'ER (mpips)':>11s} {'PR':>8s}")
    print(f"{'-'*28} {'-'*11} {'-'*8}")
    print(f"{'XG (harvested)':<28s} {er:>11.2f} {er/2:>8.2f}")

    if args.models:
        from bgsage.weights import WeightConfigPair
        w9 = WeightConfigPair.from_model("stage9")
        w9.validate()
        engines = [("Stage 9 (production)", bgbot_cpp.BackgameAwarePairStrategy(w9.paths, w9.hiddens))]
        for name, rel in [("Pasko TD", "models/td_pasko.weights.best"),
                          ("Pasko SL", "models/sl_pasko.weights.best")]:
            p = os.path.join(_BGSAGE_ROOT, rel)
            if os.path.exists(p):
                engines.append((name, bgbot_cpp.NNStrategy(p, 400, 244)))
        for name, strat in engines:
            e = statistics.mean(
                abs(strat.evaluate_board(r["board"], r["board"])["equity"] - r["rollout_eq"])
                for r in rows) * 1000
            print(f"{name:<28s} {e:>11.2f} {e/2:>8.2f}")
    print(f"\n(same {n} positions for every engine)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="build .xg shards for XG Batch Analyze")
    g.add_argument("--rollout", default="pasko-benchmark-rollout")
    g.add_argument("--template", default="logs/pasko_xg/example.xg",
                   help="a real .xg file to clone record templates from")
    g.add_argument("--out-dir", default=os.path.join(_BGSAGE_ROOT, "logs", "pasko_xg", "benchmark_full"))
    g.add_argument("--per-file", type=int, default=1000)
    g.add_argument("--limit", type=int, default=0, help="cap positions (0 = all)")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("score", help="harvest analyzed shards + report PR")
    s.add_argument("path", help="a shard .xg file OR a directory of them")
    s.add_argument("--models", action="store_true", help="also score Sage models")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
