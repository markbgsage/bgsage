// Profiling benchmark for 3T truncated rollout.
// Measures time spent in: move0 prefill, move1 prefill, trial loop, truncation evals.

#include "bgbot/rollout.h"
#include "bgbot/cube.h"
#include "bgbot/encoding.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/neural_net.h"
#include "bgbot/multipy.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace bgbot;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
    int n_threads = 16;
    if (argc > 1) {
        int t = std::atoi(argv[1]);
        if (t > 0) n_threads = t;
    }

    std::filesystem::path model_dir;
    if (argc > 2) {
        model_dir = std::filesystem::path(argv[2]);
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

    // Profile individual components
    printf("=== Component Profiling ===\n\n");

    // 1. Profile 1-ply evaluation
    {
        auto t0 = Clock::now();
        int N = 10000;
        for (int i = 0; i < N; ++i) {
            auto p = base->evaluate_probs(board, board);
            (void)p;
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("1-ply eval:           %.1f us\n", us);
    }

    // 2. Profile possible_boards
    {
        auto t0 = Clock::now();
        int N = 10000;
        std::vector<Board> candidates;
        for (int i = 0; i < N; ++i) {
            candidates.clear();
            possible_boards(board, 3, 1, candidates);
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("possible_boards(3,1): %.1f us  (%zu candidates)\n", us, candidates.size());
    }

    // 3. Profile batch_evaluate_candidates_equity
    {
        std::vector<Board> candidates;
        possible_boards(board, 3, 1, candidates);
        std::vector<double> equities(candidates.size());
        auto t0 = Clock::now();
        int N = 5000;
        for (int i = 0; i < N; ++i) {
            base->batch_evaluate_candidates_equity(candidates, board, equities.data());
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("batch_eval_equity:    %.1f us  (%zu cands)\n", us, candidates.size());
    }

    // 4. Profile 2-ply eval
    {
        auto strat2 = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilter{2, 0.03f});
        strat2->clear_cache();
        auto t0 = Clock::now();
        int N = 100;
        for (int i = 0; i < N; ++i) {
            strat2->clear_cache();
            auto p = strat2->evaluate_probs(board, board);
            (void)p;
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("2-ply eval (cold):    %.0f us\n", us);

        // Measure cache clear cost
        auto tc0 = Clock::now();
        int Nc = 1000;
        for (int i = 0; i < Nc; ++i) {
            strat2->clear_cache();
        }
        auto tc1 = Clock::now();
        double cc_us = std::chrono::duration<double, std::micro>(tc1 - tc0).count() / Nc;
        printf("cache clear:          %.0f us\n", cc_us);
    }

    // 4b. Profile 2-ply eval warm cache
    {
        auto strat2 = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilter{2, 0.03f});
        strat2->clear_cache();
        strat2->evaluate_probs(board, board); // warm cache
        auto t0 = Clock::now();
        int N = 10000;
        for (int i = 0; i < N; ++i) {
            auto p = strat2->evaluate_probs(board, board);
            (void)p;
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("2-ply eval (warm):    %.2f us\n", us);
    }

    // 5. Profile 3-ply eval
    {
        auto strat3 = std::make_shared<MultiPlyStrategy>(base, 3, MoveFilter{2, 0.03f});
        strat3->clear_cache();
        auto t0 = Clock::now();
        int N = 20;
        for (int i = 0; i < N; ++i) {
            strat3->clear_cache();
            auto p = strat3->evaluate_probs(board, board);
            (void)p;
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("3-ply eval (cold+clr):%.0f us\n", us);
        printf("3-ply eval (actual):  ~%.0f us (minus cache clear)\n", us - 3540);
    }

    // 5b. Profile 3-ply eval of different positions (simulating rollout)
    {
        auto strat3 = std::make_shared<MultiPlyStrategy>(base, 3, MoveFilter{2, 0.03f});
        strat3->clear_cache();
        // Generate several different positions to evaluate (simulating diverse truncation points)
        std::vector<Board> positions;
        positions.push_back(board);  // pos 0
        std::vector<Board> cands;
        for (int d1 = 1; d1 <= 6; ++d1) {
            for (int d2 = d1; d2 <= 6; ++d2) {
                cands.clear();
                possible_boards(board, d1, d2, cands);
                if (!cands.empty()) {
                    Board f = flip(cands[0]);
                    std::vector<Board> cands2;
                    possible_boards(f, d1, d2, cands2);
                    if (!cands2.empty()) {
                        positions.push_back(flip(cands2[0]));
                    }
                }
            }
        }
        printf("Diverse 3-ply (%zu positions):\n", positions.size());
        auto t0 = Clock::now();
        for (size_t i = 0; i < positions.size(); ++i) {
            auto p = strat3->evaluate_probs(positions[i], positions[i]);
            (void)p;
        }
        auto t1 = Clock::now();
        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        printf("  Total: %.0f us (avg %.0f us/pos)\n", total_us, total_us / positions.size());
    }

    // 6. Profile 2-ply BMI
    {
        auto strat2 = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilter{2, 0.03f});
        std::vector<Board> candidates;
        possible_boards(board, 3, 1, candidates);
        auto t0 = Clock::now();
        int N = 100;
        for (int i = 0; i < N; ++i) {
            strat2->clear_cache();
            int idx = strat2->best_move_index(candidates, board);
            (void)idx;
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("2-ply BMI (cold):     %.0f us  (%zu cands)\n", us, candidates.size());
    }

    // 7. Profile GamePlanStrategy::best_move_index (1-ply BMI)
    {
        std::vector<Board> candidates;
        possible_boards(board, 3, 1, candidates);
        auto t0 = Clock::now();
        int N = 5000;
        for (int i = 0; i < N; ++i) {
            int idx = base->best_move_index(candidates, board);
            (void)idx;
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("1-ply BMI:            %.1f us  (%zu cands)\n", us, candidates.size());
    }

    // 7b. Profile batch_evaluate_candidates_best_prob
    {
        Board flipped = flip(board);
        std::vector<Board> candidates;
        possible_boards(flipped, 3, 1, candidates);
        std::array<float, NUM_OUTPUTS> best_probs;
        auto t0 = Clock::now();
        int N = 5000;
        for (int i = 0; i < N; ++i) {
            int idx = base->batch_evaluate_candidates_best_prob(candidates, flipped, nullptr, &best_probs);
            (void)idx;
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("batch_best_prob:      %.1f us  (%zu cands)\n", us, candidates.size());
    }

    // 7c. Profile all 21 rolls candidate counts
    {
        Board flipped = flip(board);
        std::vector<Board> candidates;
        printf("Candidate counts per roll (opp board):\n  ");
        int total_cands = 0;
        for (int d1 = 1; d1 <= 6; ++d1) {
            for (int d2 = d1; d2 <= 6; ++d2) {
                candidates.clear();
                possible_boards(flipped, d1, d2, candidates);
                printf("%d-%d:%zu ", d1, d2, candidates.size());
                total_cands += candidates.size();
            }
        }
        printf("\n  Total candidates across 21 rolls: %d\n", total_cands);
    }

    // 7d. Profile encoding of opponent position
    {
        Board flipped = flip(board);
        auto t0 = Clock::now();
        int N = 100000;
        for (int i = 0; i < N; ++i) {
            auto inp = compute_extended_contact_inputs(flipped);
            (void)inp;
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("encode (opp):         %.2f us\n", us);
    }

    // 7d2. Profile batch of 30 candidates - encoding vs forward pass
    {
        Board flipped = flip(board);
        std::vector<Board> cands;
        possible_boards(flipped, 2, 2, cands);  // 2-2 generates many candidates
        printf("Timing for %zu candidates (doubles):\n", cands.size());

        // Just encoding
        auto t0 = Clock::now();
        int N = 1000;
        for (int iter = 0; iter < N; ++iter) {
            for (size_t i = 0; i < cands.size(); ++i) {
                auto inp = compute_extended_contact_inputs(cands[i]);
                (void)inp;
            }
        }
        auto t1 = Clock::now();
        double enc_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("  Encoding all:       %.1f us (%.2f us/cand)\n", enc_us, enc_us / cands.size());

        // Full batch eval
        auto t2 = Clock::now();
        for (int iter = 0; iter < N; ++iter) {
            std::array<float, NUM_OUTPUTS> bp;
            int idx = base->batch_evaluate_candidates_best_prob(cands, flipped, nullptr, &bp);
            (void)idx;
        }
        auto t3 = Clock::now();
        double batch_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;
        printf("  Full batch eval:    %.1f us\n", batch_us);
        printf("  Batch minus encode: %.1f us\n", batch_us - enc_us);
    }

    // 7e. Profile full 2-ply inner loop: all 21 rolls with batch_best_prob
    {
        Board flipped = flip(board);
        std::vector<Board> candidates;
        auto t0 = Clock::now();
        int N = 100;
        for (int iter = 0; iter < N; ++iter) {
            for (int ri = 0; ri < 21; ++ri) {
                int d1 = (ri < 6) ? ri + 1 : 0;
                int d2 = (ri < 6) ? ri + 1 : 0;
                // Use ALL_ROLLS ordering
                static const int rolls[21][2] = {
                    {1,1},{2,2},{3,3},{4,4},{5,5},{6,6},
                    {1,2},{1,3},{1,4},{1,5},{1,6},
                    {2,3},{2,4},{2,5},{2,6},
                    {3,4},{3,5},{3,6},
                    {4,5},{4,6},
                    {5,6}
                };
                candidates.clear();
                possible_boards(flipped, rolls[ri][0], rolls[ri][1], candidates);
                if (!candidates.empty()) {
                    std::array<float, NUM_OUTPUTS> bp;
                    int idx = base->batch_evaluate_candidates_best_prob(candidates, flipped, nullptr, &bp);
                    (void)idx;
                }
            }
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("2-ply inner (21 rolls):%.0f us\n", us);
    }

    // 8. Profile is_race
    {
        auto t0 = Clock::now();
        int N = 1000000;
        for (int i = 0; i < N; ++i) {
            volatile bool r = is_race(board);
            (void)r;
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
        printf("is_race:              %.0f ns\n", ns);
    }

    // 9. Profile encode (extended contact)
    {
        auto t0 = Clock::now();
        int N = 100000;
        for (int i = 0; i < N; ++i) {
            auto inp = compute_extended_contact_inputs(board);
            (void)inp;
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("encode (contact):     %.2f us\n", us);
    }

    // 10. Full benchmark
    printf("\n=== Full 3T Benchmark ===\n");

    CubeInfo cube;
    cube.cube_value = 1;
    cube.owner = CubeOwner::CENTERED;
    cube.jacoby = true;
    cube.beaver = true;

    {
        auto tmp = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilters::TINY);
        tmp->clear_cache();
    }

    RolloutConfig config;
    config.n_trials = 360;
    config.truncation_depth = 7;
    config.decision_ply = 3;
    config.enable_vr = true;
    config.filter = MoveFilters::TINY;
    config.n_threads = n_threads;
    config.seed = 42;
    config.late_ply = 2;
    config.late_threshold = 2;

    auto rollout = std::make_shared<RolloutStrategy>(base, config);

    printf("Threads: %d\n", n_threads);

    // Run 3 times for stable timing
    for (int run = 0; run < 3; ++run) {
        // Clear caches between runs
        {
            auto tmp = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilters::TINY);
            tmp->clear_cache();
        }
        rollout = std::make_shared<RolloutStrategy>(base, config);

        auto start = Clock::now();
        auto cfr = rollout->cubeful_cube_decision(board, cube);
        auto end = Clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        printf("Run %d: %.3f s  ND=%+.6f DT=%+.6f\n", run+1, elapsed, cfr.nd_equity, cfr.dt_equity);
    }

    return 0;
}
