# SPDX-License-Identifier: MPL-2.0
"""Build-only evaluator experiments against the published truncated-rollout control."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
REVISION = "60f93556137a7a5c4be671502f4b3f521998ff94"

def source(file):
    return subprocess.check_output(["git", "show", REVISION + ":cpp/src/" + file], cwd=ROOT).decode().replace("\r\n", "\n")


ROUTE = """    const auto* pair = dynamic_cast<const BackgameAwarePairStrategy*>(this);
    const int network = pair ? pair->nn_index_for(pre_move_board) : -1;
    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        if (result != GameResult::NOT_OVER)
            probs_per[i] = terminal_probs(result);
        else if (pair)
            probs_per[i] = pair->probs_with_nn(candidates[i], network);
        else
            probs_per[i] = evaluate_probs(candidates[i], pre_move_board);
    }
"""
BATCH = """    if (const auto* pair = dynamic_cast<const BackgameAwarePairStrategy*>(this)) {
        thread_local std::vector<double> equities;
        equities.resize(n);
        pair->batch_evaluate_candidates_equity_probs(
            candidates, pre_move_board, equities.data(), probs_per.data());
    } else {
        for (int i = 0; i < n; ++i) {
            const GameResult result = check_game_over(candidates[i]);
            probs_per[i] = result == GameResult::NOT_OVER
                ? evaluate_probs(candidates[i], pre_move_board) : terminal_probs(result);
        }
    }
"""
FUSE = """        // The pre-move board chooses one network for every legal endpoint.
        // Keep full evaluation order; both winner and VR probabilities come
        // from the same clamped results. Preserve wrapped DB handling at the
        // boundary, where selection and reported probabilities may differ.
        const Strategy* model = &strat;
        const auto* wrapped = dynamic_cast<const BearoffStrategy*>(model);
        if (wrapped) model = &wrapped->base();
        if (const auto* pair = dynamic_cast<const BackgameAwarePairStrategy*>(model)) {
            if (!wrapped || std::none_of(candidates.begin(), candidates.end(),
                    [&](const Board& b) { return wrapped->db().is_bearoff(b); })) {
                const int n = static_cast<int>(candidates.size());
                thread_local std::vector<std::array<float, NUM_OUTPUTS>> probs;
                thread_local std::vector<float> cf;
                probs.resize(n);
                cf.resize(n);
                const int network = pair->nn_index_for(board);
                for (int i = 0; i < n; ++i) {
                    const GameResult result = check_game_over(candidates[i]);
                    probs[i] = result == GameResult::NOT_OVER
                        ? pair->probs_with_nn(candidates[i], network) : terminal_probs(result);
                    clamp_probs_to_board(probs[i], candidates[i]);
                    cf[i] = cl2cf(probs[i], sel_cubes[0], sel_cube_x);
                }
                const int idx = pick_best_with_tiebreak(cf.data(), probs.data(), n);
                if (best_index) *best_index = idx;
                return probs[idx];
            }
        }
"""

def prepare(name):
    work = ROOT / "build/trunc_eval_study" / name
    work.mkdir(exist_ok=True, parents=True)
    strategy = source("strategy.cpp").replace('#include "bgbot/encoding.h"', '#include "bgbot/encoding.h"\n#include "bgbot/neural_net.h"')
    start = strategy.index("    // NOTE: tried batch_evaluate_candidates_equity_probs")
    end = strategy.index("    for (int i = 0; i < n; ++i) {\n        clamp_probs", start)
    if name in ("route", "fuse"):
        strategy = strategy[:start] + ROUTE + strategy[end:]
    if name == "batch":
        strategy = strategy[:start] + BATCH + strategy[end:]
    (work / "strategy.cpp").write_text(strategy, encoding="utf-8")
    if name == "fuse":
        rollout = source("rollout.cpp").replace('#include "bgbot/rollout.h"', '#include "bgbot/rollout.h"\n#include "bgbot/neural_net.h"')
        pat = "    if (sel_cubes && n_sel_cubes > 0 && candidates.size() > 1) {\n"
        assert rollout.count(pat) == 1
        rollout = rollout.replace(pat, pat + FUSE)
        (work / "rollout.cpp").write_text(rollout, encoding="utf-8")

if __name__ == "__main__":
    for name in sys.argv[1:] or ["route", "batch", "fuse"]:
        if name not in {"route", "batch", "fuse"}:
            raise ValueError(name)
        prepare(name)
