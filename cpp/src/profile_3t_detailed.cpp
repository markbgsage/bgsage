// Detailed profiling of 3T per-trial cost breakdown.
// Runs single-threaded 36-trial rollout and reports per-move timing.

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
#include <vector>

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

    // Test: what does the move count distribution look like?
    // With truncation_depth=7, each trial runs at most 7 half-moves.
    // But games can end early if a player wins.

    // Profile individual components that make up per-trial cost
    printf("=== Component Costs ===\n\n");

    // 1. Move generation for all 21 rolls
    {
        Board test_board = flip(board);  // opponent's perspective
        auto t0 = Clock::now();
        int total_cands = 0;
        for (int iter = 0; iter < 1000; ++iter) {
            for (int r = 0; r < 21; ++r) {
                std::vector<Board> cands;
                int d1s[] = {1,2,3,4,5,6, 1,1,1,1,1, 2,2,2,2, 3,3,3, 4,4, 5};
                int d2s[] = {1,2,3,4,5,6, 2,3,4,5,6, 3,4,5,6, 4,5,6, 5,6, 6};
                possible_boards(test_board, d1s[r], d2s[r], cands);
                if (iter == 0) total_cands += (int)cands.size();
            }
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000;
        printf("Move gen (21 rolls): %.3f ms, total candidates: %d\n", ms, total_cands);
    }

    // 2. VR mean computation: 21 x best_move_probs at 1-ply
    {
        Board test_board = flip(board);
        std::vector<std::vector<Board>> all_cands(21);
        int d1s[] = {1,2,3,4,5,6, 1,1,1,1,1, 2,2,2,2, 3,3,3, 4,4, 5};
        int d2s[] = {1,2,3,4,5,6, 2,3,4,5,6, 3,4,5,6, 4,5,6, 5,6, 6};
        for (int r = 0; r < 21; ++r) {
            possible_boards(test_board, d1s[r], d2s[r], all_cands[r]);
        }

        auto t0 = Clock::now();
        for (int iter = 0; iter < 100; ++iter) {
            for (int r = 0; r < 21; ++r) {
                auto gps = dynamic_cast<GamePlanStrategy*>(base.get());
                int best_idx;
                std::array<float, 5> best_probs;
                gps->batch_evaluate_candidates_best_prob(
                    all_cands[r], test_board, nullptr, &best_probs);
            }
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 100;
        printf("VR mean (21 x batch_1ply): %.3f ms\n", ms);
    }

    // 3. 2-ply best_move_index with different filter sizes
    {
        Board test_board = flip(board);
        std::vector<Board> candidates;
        possible_boards(test_board, 3, 1, candidates);
        printf("Test candidates: %d\n", (int)candidates.size());

        auto strat_2ply_tiny = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilters::TINY);
        auto t0 = Clock::now();
        for (int iter = 0; iter < 100; ++iter) {
            strat_2ply_tiny->best_move_index(candidates, test_board);
        }
        auto t1 = Clock::now();
        double ms_tiny = std::chrono::duration<double, std::milli>(t1 - t0).count() / 100;

        // Test with tighter filter
        MoveFilter tight = {3, 0.04f};
        auto strat_2ply_tight = std::make_shared<MultiPlyStrategy>(base, 2, tight);
        t0 = Clock::now();
        for (int iter = 0; iter < 100; ++iter) {
            strat_2ply_tight->best_move_index(candidates, test_board);
        }
        t1 = Clock::now();
        double ms_tight = std::chrono::duration<double, std::milli>(t1 - t0).count() / 100;

        // Test with very tight filter
        MoveFilter vtight = {2, 0.03f};
        auto strat_2ply_vtight = std::make_shared<MultiPlyStrategy>(base, 2, vtight);
        t0 = Clock::now();
        for (int iter = 0; iter < 100; ++iter) {
            strat_2ply_vtight->best_move_index(candidates, test_board);
        }
        t1 = Clock::now();
        double ms_vtight = std::chrono::duration<double, std::milli>(t1 - t0).count() / 100;

        printf("2-ply bmi TINY(5,0.08):   %.3f ms\n", ms_tiny);
        printf("2-ply bmi TIGHT(3,0.04):  %.3f ms\n", ms_tight);
        printf("2-ply bmi VTIGHT(2,0.03): %.3f ms\n", ms_vtight);
    }

    // 4. 3-ply evaluate_probs (truncation cost)
    {
        Board test_board = flip(board);
        std::vector<Board> candidates;
        possible_boards(test_board, 3, 1, candidates);
        Board post_move = candidates[0];

        auto strat_3ply = std::make_shared<MultiPlyStrategy>(base, 3, MoveFilters::TINY);

        // Cold cache
        strat_3ply->clear_cache();
        auto t0 = Clock::now();
        auto probs = strat_3ply->evaluate_probs(post_move, test_board);
        auto t1 = Clock::now();
        double ms_cold = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Warm cache (repeated)
        t0 = Clock::now();
        probs = strat_3ply->evaluate_probs(post_move, test_board);
        t1 = Clock::now();
        double ms_warm = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Different position (partially warm cache from shared sub-positions)
        Board post_move2 = candidates[1];
        t0 = Clock::now();
        probs = strat_3ply->evaluate_probs(post_move2, test_board);
        t1 = Clock::now();
        double ms_partial = std::chrono::duration<double, std::milli>(t1 - t0).count();

        printf("3-ply eval_probs cold:    %.3f ms\n", ms_cold);
        printf("3-ply eval_probs warm:    %.3f ms (cache hit)\n", ms_warm);
        printf("3-ply eval_probs partial: %.3f ms\n", ms_partial);

        // 2-ply evaluate_probs for comparison
        auto strat_2ply = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilters::TINY);
        strat_2ply->clear_cache();
        t0 = Clock::now();
        probs = strat_2ply->evaluate_probs(post_move, test_board);
        t1 = Clock::now();
        double ms_2ply = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("2-ply eval_probs cold:    %.3f ms\n", ms_2ply);
    }

    // 5. Simulate a single trial's work (manually, not via rollout)
    printf("\n=== Simulated Trial Breakdown ===\n");
    {
        auto strat_2ply = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilters::TINY);
        auto strat_3ply = std::make_shared<MultiPlyStrategy>(base, 3, MoveFilters::TINY);
        strat_2ply->clear_cache();
        strat_3ply->clear_cache();

        Board cur = board;
        double total_vr = 0, total_decision = 0, total_movegen = 0, total_trunc = 0;

        // Simulate 5 moves (moves 2-6, since 0-1 are cached)
        for (int m = 0; m < 5; ++m) {
            // Move generation for VR (21 rolls)
            auto t0 = Clock::now();
            std::vector<std::vector<Board>> all_cands(21);
            int d1s[] = {1,2,3,4,5,6, 1,1,1,1,1, 2,2,2,2, 3,3,3, 4,4, 5};
            int d2s[] = {1,2,3,4,5,6, 2,3,4,5,6, 3,4,5,6, 4,5,6, 5,6, 6};
            for (int r = 0; r < 21; ++r) {
                possible_boards(cur, d1s[r], d2s[r], all_cands[r]);
            }
            auto t1 = Clock::now();
            total_movegen += std::chrono::duration<double, std::milli>(t1 - t0).count();

            // VR mean (21 x 1-ply batch)
            t0 = Clock::now();
            auto gps = dynamic_cast<GamePlanStrategy*>(base.get());
            for (int r = 0; r < 21; ++r) {
                std::array<float, 5> best_probs;
                gps->batch_evaluate_candidates_best_prob(
                    all_cands[r], cur, nullptr, &best_probs);
            }
            t1 = Clock::now();
            total_vr += std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Decision: 2-ply best_move_index for actual roll (use roll 6 = 1-2)
            t0 = Clock::now();
            int best = strat_2ply->best_move_index(all_cands[6], cur);
            t1 = Clock::now();
            total_decision += std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Advance board
            if (!all_cands[6].empty()) {
                cur = flip(all_cands[6][best]);
            }
        }

        // Truncation: 3-ply evaluate_probs
        Board last = flip(cur);
        auto t0 = Clock::now();
        auto probs = strat_3ply->evaluate_probs(last, last);
        auto t1 = Clock::now();
        total_trunc = std::chrono::duration<double, std::milli>(t1 - t0).count();

        printf("Move gen (5 moves):   %.3f ms\n", total_movegen);
        printf("VR mean (5 moves):    %.3f ms\n", total_vr);
        printf("2-ply decisions (5):  %.3f ms\n", total_decision);
        printf("3-ply truncation:     %.3f ms\n", total_trunc);
        printf("Total simulated:      %.3f ms\n",
               total_movegen + total_vr + total_decision + total_trunc);
    }

    return 0;
}
