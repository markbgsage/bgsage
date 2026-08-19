// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/strategy.h"
#include "bgbot/board.h"
#include "bgbot/cube.h"        // for CubeInfo + cl2cf (cube-aware defaults)
#include "bgbot/encoding.h"
#include <limits>
#include <algorithm>
#include <vector>

namespace bgbot {

std::array<float, NUM_OUTPUTS> Strategy::evaluate_probs(
    const Board& board, bool pre_move_is_race) const
{
    // Default: approximate from equity. Non-NN strategies (e.g. PubEval)
    // only produce a single equity value; we map it to a win probability.
    double eq = evaluate(board, pre_move_is_race);
    float p_win = static_cast<float>(std::clamp((eq + 1.0) / 2.0, 0.0, 1.0));
    return {p_win, 0.0f, 0.0f, 0.0f, 0.0f};
}

std::array<float, NUM_OUTPUTS> Strategy::evaluate_probs(
    const Board& board, const Board& pre_move_board) const
{
    return evaluate_probs(board, is_race(pre_move_board));
}

int Strategy::best_move_index(const std::vector<Board>& candidates,
                              bool pre_move_is_race) const {
    int best_idx = 0;
    double best_val = -1e30;

    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        double val = evaluate(candidates[i], pre_move_is_race);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    return best_idx;
}

int Strategy::best_move_index(const std::vector<Board>& candidates,
                              const Board& pre_move_board) const {
    return best_move_index(candidates, is_race(pre_move_board));
}

// ----- Cube-aware move selection (default implementations) -----
//
// Both defaults dispatch through batch_evaluate_candidates_equity_probs so
// subclasses with optimized batch NN forward passes (e.g. GamePlanStrategy)
// pick up the optimization automatically — no per-subclass override needed
// for cubeful selection unless the subclass wants something fancier (e.g.
// MultiPlyStrategy delegating to cubeful_equity_nply_multi).
//
// batch_evaluate_candidates_equity_probs already substitutes terminal_probs
// for game-over candidates. We clamp afterward to enforce position invariants
// before Janowski (cl2cf) sees them.

int Strategy::best_move_index_cubeful(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    const CubeInfo& ci,
    float cube_x) const
{
    const int n = static_cast<int>(candidates.size());
    std::vector<std::array<float, NUM_OUTPUTS>> probs_per(n);
    batch_evaluate_candidates_equity_probs(candidates, pre_move_board,
                                           nullptr, probs_per.data());

    // Same ranking rule as best_move_index_cubeful_multi below -- cubeful
    // equity, ties broken on money cubeless equity. These two must never
    // diverge: they are the cube-aware selection entry points, and a rollout's
    // VR mean and its played move have to agree on the rule or luck stops
    // having zero mean.
    std::vector<float> cf_vals(n);
    for (int i = 0; i < n; ++i) {
        clamp_probs_to_board(probs_per[i], candidates[i]);
        cf_vals[i] = cl2cf(probs_per[i], ci, cube_x);
    }
    return pick_best_with_tiebreak(cf_vals.data(), probs_per.data(), n);
}

void Strategy::best_move_index_cubeful_multi(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    const CubeInfo* cubes,
    int n_cubes,
    float cube_x,
    int* out_indices) const
{
    // Evaluate cubeless probs once per candidate (expensive part — NN batch),
    // then loop cl2cf per cube state (cheap).
    const int n = static_cast<int>(candidates.size());
    std::vector<std::array<float, NUM_OUTPUTS>> probs_per(n);
    // NOTE: tried batch_evaluate_candidates_equity_probs (Idea 5) and it
    // segfaults under parallel trial dispatch. The batch implementation in
    // GamePlanStrategy is not safe for concurrent use across many threads
    // running this from inside rollout trials. Loop evaluate_probs instead.
    for (int i = 0; i < n; ++i) {
        GameResult r = check_game_over(candidates[i]);
        if (r != GameResult::NOT_OVER) {
            probs_per[i] = terminal_probs(r);
        } else {
            probs_per[i] = evaluate_probs(candidates[i], pre_move_board);
        }
    }
    for (int i = 0; i < n; ++i) {
        clamp_probs_to_board(probs_per[i], candidates[i]);
    }

    // Rank by cubeful equity, breaking exact ties on money cubeless equity
    // (see pick_best_with_tiebreak in cube.h). probs_per is already in hand,
    // so the tie-break costs a handful of flops per candidate.
    std::vector<float> cf_vals(n);
    for (int c = 0; c < n_cubes; ++c) {
        for (int i = 0; i < n; ++i) {
            cf_vals[i] = cl2cf(probs_per[i], cubes[c], cube_x);
        }
        out_indices[c] = pick_best_with_tiebreak(cf_vals.data(), probs_per.data(), n);
    }
}

// ----- Default batch evaluation implementations -----
// These loop over candidates individually. Concrete strategies override
// with optimized batch encoding + forward pass implementations.

int Strategy::evaluate_candidates_equity(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities) const
{
    int best_idx = 0;
    double best_eq = -1e30;
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        GameResult r = check_game_over(candidates[i]);
        double eq;
        if (r != GameResult::NOT_OVER) {
            eq = compute_equity(terminal_probs(r));
        } else {
            eq = compute_equity(evaluate_probs(candidates[i], pre_move_board));
        }
        if (equities) equities[i] = eq;
        if (eq > best_eq) {
            best_eq = eq;
            best_idx = i;
        }
    }
    return best_idx;
}

int Strategy::batch_evaluate_candidates_equity(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities) const
{
    return evaluate_candidates_equity(candidates, pre_move_board, equities);
}

int Strategy::batch_evaluate_candidates_equity_probs(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities,
    std::array<float, NUM_OUTPUTS>* probs_out) const
{
    int best_idx = 0;
    double best_eq = -1e30;
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        GameResult r = check_game_over(candidates[i]);
        std::array<float, NUM_OUTPUTS> probs;
        double eq;
        if (r != GameResult::NOT_OVER) {
            probs = terminal_probs(r);
            eq = compute_equity(probs);
        } else {
            probs = evaluate_probs(candidates[i], pre_move_board);
            eq = compute_equity(probs);
        }
        if (equities) equities[i] = eq;
        if (probs_out) probs_out[i] = probs;
        if (eq > best_eq) {
            best_eq = eq;
            best_idx = i;
        }
    }
    return best_idx;
}

int Strategy::batch_evaluate_candidates_best_prob(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities,
    std::array<float, NUM_OUTPUTS>* best_probs_out) const
{
    int best_idx = 0;
    double best_eq = -1e30;
    std::array<float, NUM_OUTPUTS> best_probs{};
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        GameResult r = check_game_over(candidates[i]);
        std::array<float, NUM_OUTPUTS> probs;
        double eq;
        if (r != GameResult::NOT_OVER) {
            probs = terminal_probs(r);
            eq = compute_equity(probs);
        } else {
            probs = evaluate_probs(candidates[i], pre_move_board);
            eq = compute_equity(probs);
        }
        if (equities) equities[i] = eq;
        if (eq > best_eq) {
            best_eq = eq;
            best_idx = i;
            best_probs = probs;
        }
    }
    if (best_probs_out) *best_probs_out = best_probs;
    return best_idx;
}

} // namespace bgbot
