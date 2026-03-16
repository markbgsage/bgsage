// Profile 3T rollout: measure time breakdown per trial and per move.
//
// Runs a single-threaded 3T rollout (36 trials) with timing instrumentation
// to identify where time is spent inside each trial.

#include "bgbot/rollout.h"
#include "bgbot/cube.h"
#include "bgbot/encoding.h"
#include "bgbot/board.h"
#include "bgbot/neural_net.h"
#include "bgbot/multipy.h"
#include "bgbot/moves.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace bgbot;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
    std::filesystem::path model_dir;
    if (argc > 1) {
        model_dir = std::filesystem::path(argv[1]);
    } else {
        model_dir = std::filesystem::absolute(
            std::filesystem::path(argv[0])).parent_path().parent_path() / "models";
    }

    Board board = {
        0, 0, 0, 2, 2, -2, 3, 2, 2, 0, 0, 0, -3, 4, 0, 0, 0, -3, 0, -3, -2, -2, 0, 0, 0, 0
    };

    const auto pr = (model_dir / "sl_s5_purerace.weights.best").string();
    const auto rc = (model_dir / "sl_s5_racing.weights.best").string();
    const auto at = (model_dir / "sl_s5_attacking.weights.best").string();
    const auto pm = (model_dir / "sl_s5_priming.weights.best").string();
    const auto an = (model_dir / "sl_s5_anchoring.weights.best").string();

    init_escape_tables();

    auto base = std::make_shared<GamePlanStrategy>(pr, rc, at, pm, an, 200, 400, 400, 400, 400);

    CubeInfo cube;
    cube.cube_value = 1;
    cube.owner = CubeOwner::CENTERED;
    cube.jacoby = true;
    cube.beaver = true;

    // Clear caches
    {
        auto tmp = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilters::TINY);
        tmp->clear_cache();
    }

    // --- Measure prefill times ---
    printf("=== 3T Profiling ===\n\n");

    // Create strategies manually to measure their individual costs
    auto strat_3ply = std::make_shared<MultiPlyStrategy>(base, 3, MoveFilters::TINY);
    auto strat_2ply = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilters::TINY);

    // Profile a single 2-ply best_move_index call
    {
        Board flipped = flip(board);
        std::vector<Board> candidates;
        possible_boards(flipped, 3, 1, candidates);
        printf("Candidates for dice 3-1 from flipped board: %d\n", (int)candidates.size());

        auto t0 = Clock::now();
        int best = strat_2ply->best_move_index(candidates, flipped);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("2-ply best_move_index: %.3f ms\n", ms);

        strat_2ply->clear_cache();
        t0 = Clock::now();
        best = strat_3ply->best_move_index(candidates, flipped);
        t1 = Clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("3-ply best_move_index: %.3f ms\n", ms);
    }

    // Profile evaluate_probs at different plies
    {
        Board flipped = flip(board);
        std::vector<Board> candidates;
        possible_boards(flipped, 3, 1, candidates);
        Board post_move = candidates[0];

        strat_2ply->clear_cache();
        auto t0 = Clock::now();
        auto probs = strat_2ply->evaluate_probs(post_move, flipped);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("2-ply evaluate_probs: %.3f ms\n", ms);

        strat_3ply->clear_cache();
        t0 = Clock::now();
        probs = strat_3ply->evaluate_probs(post_move, flipped);
        t1 = Clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("3-ply evaluate_probs: %.3f ms\n", ms);
    }

    printf("\n--- Cost Analysis ---\n");
    // Profile 1-ply batch evaluation
    {
        Board flipped = flip(board);
        std::vector<Board> candidates;
        possible_boards(flipped, 3, 1, candidates);

        auto t0 = Clock::now();
        for (int i = 0; i < 100; ++i) {
            std::vector<double> eq(candidates.size());
            auto gps = dynamic_cast<GamePlanStrategy*>(base.get());
            gps->batch_evaluate_candidates_equity(candidates, flipped, eq.data());
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("1-ply batch_evaluate (%d cands, 100 iters): %.3f ms total, %.3f ms/call\n",
               (int)candidates.size(), ms, ms / 100.0);
    }

    // Profile move generation
    {
        Board flipped = flip(board);
        auto t0 = Clock::now();
        for (int i = 0; i < 1000; ++i) {
            for (int r = 0; r < 21; ++r) {
                std::vector<Board> cands;
                // Use ALL_ROLLS equivalent
                int d1s[] = {1,2,3,4,5,6, 1,1,1,1,1, 2,2,2,2, 3,3,3, 4,4, 5};
                int d2s[] = {1,2,3,4,5,6, 2,3,4,5,6, 3,4,5,6, 4,5,6, 5,6, 6};
                possible_boards(flipped, d1s[r], d2s[r], cands);
            }
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("Move generation (21 rolls, 1000 iters): %.3f ms total, %.3f ms/call-of-21\n",
               ms, ms / 1000.0);
    }

    // --- Run 1-thread rollout to get overall timing ---
    printf("\n--- Single-thread 36-trial 3T rollout ---\n");
    strat_2ply->clear_cache();
    strat_3ply->clear_cache();

    RolloutConfig config;
    config.n_trials = 36;
    config.truncation_depth = 7;
    config.decision_ply = 3;
    config.enable_vr = true;
    config.filter = MoveFilters::TINY;
    config.n_threads = 1;
    config.seed = 42;
    config.late_ply = 2;
    config.late_threshold = 2;

    auto rollout = std::make_shared<RolloutStrategy>(base, config);

    auto t_start = Clock::now();
    auto cfr = rollout->cubeful_cube_decision(board, cube);
    auto t_end = Clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();
    printf("Total: %.3f s (%.1f ms/trial)\n", total_s, total_s * 1000.0 / 36);
    printf("ND: %+.6f  DT: %+.6f\n", cfr.nd_equity, cfr.dt_equity);

    // Cache stats
    printf("\nCache stats (from 3-ply strategy):\n");
    printf("  Hits:   %zu\n", strat_3ply->cache_hits());
    printf("  Misses: %zu\n", strat_3ply->cache_misses());
    size_t total = strat_3ply->cache_hits() + strat_3ply->cache_misses();
    if (total > 0) {
        printf("  Hit rate: %.1f%%\n", 100.0 * strat_3ply->cache_hits() / total);
    }

    // --- Count how many 2-ply and 3-ply evaluations happen per trial ---
    printf("\n--- Cost model estimate ---\n");
    // With late_threshold=2: moves 0,1 use 3-ply, moves 2-6 use 2-ply, truncation uses 3-ply
    // Move 0: cached (prefilled), zero cost
    // Move 1: cached (prefilled), zero cost
    // Moves 2-6 (5 moves): each is best_move_index at 2-ply
    //   - Score all candidates at 1-ply (batch)
    //   - Filter to ~5 survivors
    //   - Each survivor: evaluate_probs at 2-ply = 21 opponent rolls x 1-ply
    //   Total per move: ~5 * 21 = 105 NN forward passes for decision
    //   + 21 * batch_1ply for VR (21 batch calls of ~N candidates each)
    //   + 1 * 1-ply for VR luck
    // Truncation (move 7): evaluate_probs at 3-ply = 21 * (best 1-ply + recurse 2-ply)
    //   where 2-ply = 21 opponent rolls * 1-ply
    //   Total: 21 * 21 = 441 NN forward passes
    printf("Per trial (estimated):\n");
    printf("  Moves 0-1: cached (prefilled)\n");
    printf("  Moves 2-6: 5 x 2-ply best_move_index\n");
    printf("  Truncation: 1 x 3-ply evaluate_probs\n");

    return 0;
}
