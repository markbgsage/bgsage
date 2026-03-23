"""Benchmark 4-ply checker play: compare old (single-pass) vs new (iterative deepening).

Shows which moves survive each filter step and the final best move.
"""

import os
import sys
import time

sys.path.insert(0, "build")
sys.path.insert(0, "bgsage/python")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64")

import bgbot_cpp
from bgsage.weights import WeightConfig

BOARD = [0, 0, 0, 0, 2, -3, 4, 2, 2, 0, 0, 0, -4, 2, -2, 0, -1, 2, 0, -4, 0, -1, 0, 0, 0, 0]
DIE1, DIE2 = 6, 5


def move_label(before, after):
    """Derive move string from board diff."""
    dice = sorted([DIE1, DIE2], reverse=True)
    froms, tos = [], []
    for i in range(1, 26):
        d = after[i] - before[i]
        if d < 0:
            froms.extend([i] * (-d))
        elif d > 0:
            tos.extend([i] * d)

    if len(froms) == 2 and len(tos) == 2:
        f_s = sorted(froms, reverse=True)
        t_s = sorted(tos, reverse=True)
        for pairing in [(0, 1), (1, 0)]:
            dists = [f_s[i] - t_s[pairing[i]] for i in range(2)]
            if sorted(dists, reverse=True) == dice:
                parts = []
                for i in range(2):
                    f, t = f_s[i], t_s[pairing[i]]
                    hit = "*" if before[t] < 0 else ""
                    parts.append(f"{f}/{t}{hit}")
                return " ".join(sorted(parts, key=lambda s: -int(s.split("/")[0])))
        return f"{f_s[0]}/{t_s[0]} {f_s[1]}/{t_s[1]}"
    if len(froms) == 1 and len(tos) == 1:
        return f"{froms[0]}/{tos[0]}{'*' if before[tos[0]] < 0 else ''}"
    if len(froms) == 1 and len(tos) == 0:
        return f"{froms[0]}/off"
    if len(froms) == 2 and len(tos) == 1:
        f_s = sorted(froms, reverse=True)
        return f"{f_s[0]}/off {f_s[1]}/{tos[0]}"
    if len(froms) == 2 and len(tos) == 0:
        f_s = sorted(froms, reverse=True)
        return f"{f_s[0]}/off {f_s[1]}/off"
    return "??"


def run_old_style(model_name):
    """Simulate the OLD single-pass filter: 1-ply filter, then 4-ply on all survivors."""
    w = WeightConfig.from_model(model_name)
    strat_1ply = bgbot_cpp.GamePlanStrategy(*w.weight_args)
    multipy = bgbot_cpp.create_multipy_5nn(
        *w.weight_args, n_plies=4,
        filter_max_moves=5, filter_threshold=0.08,
        parallel_evaluate=False, parallel_threads=1)
    multipy.set_cache_enabled(False)

    candidates = bgbot_cpp.possible_moves(BOARD, DIE1, DIE2)

    # 1-ply scoring
    scored = []
    for b in candidates:
        bl = list(b)
        r = strat_1ply.evaluate_board(bl, BOARD)
        scored.append((r["equity"], bl, move_label(BOARD, bl)))
    scored.sort(key=lambda x: -x[0])

    best_eq = scored[0][0]
    survivors_1ply = [(eq, b, lab) for eq, b, lab in scored
                      if best_eq - eq <= 0.08][:5]

    # 4-ply on all 1-ply survivors (OLD behavior)
    t0 = time.perf_counter()
    results = []
    for eq_1ply, b, lab in survivors_1ply:
        r = multipy.evaluate_board(b, BOARD)
        results.append((r["equity"], lab, eq_1ply))
    elapsed = time.perf_counter() - t0
    results.sort(key=lambda x: -x[0])

    return {
        "elapsed": elapsed,
        "survivors_1ply": [(lab, eq) for eq, b, lab in survivors_1ply],
        "results_4ply": results,
    }


def run_new_style(model_name):
    """Run with iterative deepening filter chain."""
    w = WeightConfig.from_model(model_name)
    strat_1ply = bgbot_cpp.GamePlanStrategy(*w.weight_args)

    # For intermediate 3-ply scoring
    multipy_3 = bgbot_cpp.create_multipy_5nn(
        *w.weight_args, n_plies=3,
        filter_max_moves=5, filter_threshold=0.08,
        parallel_evaluate=False, parallel_threads=1)
    multipy_3.set_cache_enabled(False)

    multipy_4 = bgbot_cpp.create_multipy_5nn(
        *w.weight_args, n_plies=4,
        filter_max_moves=5, filter_threshold=0.08,
        parallel_evaluate=False, parallel_threads=1)
    multipy_4.set_cache_enabled(False)

    candidates = bgbot_cpp.possible_moves(BOARD, DIE1, DIE2)

    # 1-ply scoring
    scored = []
    for b in candidates:
        bl = list(b)
        r = strat_1ply.evaluate_board(bl, BOARD)
        scored.append((r["equity"], bl, move_label(BOARD, bl)))
    scored.sort(key=lambda x: -x[0])

    best_eq = scored[0][0]
    survivors_1ply = [(eq, b, lab) for eq, b, lab in scored
                      if best_eq - eq <= 0.08][:5]

    # 3-ply intermediate scoring
    scored_3ply = []
    for eq_1ply, b, lab in survivors_1ply:
        r = multipy_3.evaluate_board(b, BOARD)
        scored_3ply.append((r["equity"], b, lab, eq_1ply))
    scored_3ply.sort(key=lambda x: -x[0])

    best_3ply = scored_3ply[0][0]
    survivors_3ply = [(eq, b, lab, eq1) for eq, b, lab, eq1 in scored_3ply
                      if best_3ply - eq <= 0.02][:2]

    # 4-ply final on 3-ply survivors
    t0 = time.perf_counter()
    results = []
    for eq_3ply, b, lab, eq_1ply in survivors_3ply:
        r = multipy_4.evaluate_board(b, BOARD)
        results.append((r["equity"], lab, eq_1ply, eq_3ply))
    elapsed = time.perf_counter() - t0
    results.sort(key=lambda x: -x[0])

    return {
        "elapsed_4ply_only": elapsed,
        "survivors_1ply": [(lab, eq) for eq, b, lab in survivors_1ply],
        "scored_3ply": [(lab, eq, eq1) for eq, b, lab, eq1 in scored_3ply],
        "survivors_3ply": [(lab, eq) for eq, b, lab, eq1 in survivors_3ply],
        "results_4ply": results,
    }


for model_name in ["stage5", "stage5small"]:
    print(f"\n{'=' * 80}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 80}")

    old = run_old_style(model_name)
    new = run_new_style(model_name)

    print(f"\n--- 1-ply survivors (same for both) ---")
    for lab, eq in new["survivors_1ply"]:
        print(f"  {lab:<22} {eq:+.4f}")

    print(f"\n--- 3-ply scores (new: intermediate filter step) ---")
    best_3 = new["scored_3ply"][0][1]
    for lab, eq, eq1 in new["scored_3ply"]:
        diff = eq - best_3
        survives = lab in [l for l, _ in new["survivors_3ply"]]
        print(f"  {lab:<22} 3ply={eq:+.4f} diff={diff:+.4f} {'-> SURVIVES' if survives else '-> pruned'}")

    print(f"\n--- 4-ply results ---")
    print(f"  {'Move':<22} {'OLD 4ply':>9} {'NEW 4ply':>9}")
    print(f"  {'-'*44}")
    old_by_lab = {lab: eq for eq, lab, _ in old["results_4ply"]}
    new_by_lab = {lab: eq for eq, lab, _, _ in new["results_4ply"]}
    all_labs = list(dict.fromkeys(
        [lab for _, lab, _ in old["results_4ply"]] +
        [lab for _, lab, _, _ in new["results_4ply"]]
    ))
    for lab in all_labs:
        o = f"{old_by_lab[lab]:+.4f}" if lab in old_by_lab else "     --"
        n = f"{new_by_lab[lab]:+.4f}" if lab in new_by_lab else "     --"
        print(f"  {lab:<22} {o:>9} {n:>9}")

    print(f"\n  Old best: {old['results_4ply'][0][1]} = {old['results_4ply'][0][0]:+.4f}")
    print(f"  New best: {new['results_4ply'][0][1]} = {new['results_4ply'][0][0]:+.4f}")
    same = old["results_4ply"][0][1] == new["results_4ply"][0][1]
    print(f"  Same best move: {'YES' if same else 'NO'}")
