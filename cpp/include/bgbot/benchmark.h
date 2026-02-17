#pragma once

#include "types.h"
#include "strategy.h"
#include "multipy.h"
#include "encoding.h"
#include <vector>

namespace bgbot {

// A single benchmark scenario: a position + dice roll + ranked candidate moves.
struct BenchmarkScenario {
    Board start_board;          // Starting position (player on roll = player 1)
    int die1;
    int die2;
    // Candidate result boards (flipped to opponent's perspective, as stored in .bm files).
    // The first entry is the best move (error = 0 if we pick it).
    std::vector<Board> ranked_boards;
    // Error values for each ranked board. ranked_errors[0] is ignored (best move = 0 error).
    // ranked_errors[i] for i>0 is the equity loss vs the best move.
    std::vector<double> ranked_errors;
};

// Result of scoring a set of benchmark scenarios.
struct BenchmarkResult {
    double total_error = 0.0;
    int count = 0;
    // Score in millipips (mean error * 1000). Lower is better.
    double score() const { return count > 0 ? (total_error / count) * 1000.0 : 0.0; }

    // Merge another result into this one.
    void merge(const BenchmarkResult& other) {
        total_error += other.total_error;
        count += other.count;
    }
};

// Score a set of benchmark scenarios using the given strategy.
// n_threads: number of threads to use (0 = use hardware concurrency, 1 = serial).
// The scenario vector is split into contiguous chunks, one per thread.
// Each thread scores its chunk independently (no shared state during scoring),
// then results are merged at the end.
BenchmarkResult score_benchmarks(const Strategy& strategy,
                                 const std::vector<BenchmarkScenario>& scenarios,
                                 int n_threads = 0);

// Returns per-scenario error values (not aggregated).
// errors_out must have size >= scenarios.size(). Each entry is the equity error
// for that scenario (0 if best move was chosen, else the ranked error).
void score_benchmarks_per_scenario(const Strategy& strategy,
                                   const std::vector<BenchmarkScenario>& scenarios,
                                   double* errors_out,
                                   int n_threads = 0);

// --- Benchmark PR (Performance Rating) ---

// A single decision from a simulated game, with rollout reference equities.
struct PRDecision {
    Board board;                              // Pre-move board (player on roll)
    int die1, die2;
    std::vector<Board> candidates;            // Filtered candidate post-move boards
    std::vector<double> rollout_equities;     // Rollout equity for each candidate
};

// Per-game-plan accumulators for PR scoring.
struct PRResult {
    // Per game plan: total error, decision count, decisions with nonzero error
    double gp_total_error[5] = {};
    int gp_n_decisions[5] = {};
    int gp_n_with_error[5] = {};
    int n_outside = 0;  // strategy chose a move not in the filtered set
    int n_skipped = 0;

    void merge(const PRResult& other) {
        for (int i = 0; i < 5; ++i) {
            gp_total_error[i] += other.gp_total_error[i];
            gp_n_decisions[i] += other.gp_n_decisions[i];
            gp_n_with_error[i] += other.gp_n_with_error[i];
        }
        n_outside += other.n_outside;
        n_skipped += other.n_skipped;
    }

    int total_decisions() const {
        int n = 0;
        for (int i = 0; i < 5; ++i) n += gp_n_decisions[i];
        return n;
    }

    double total_error() const {
        double e = 0;
        for (int i = 0; i < 5; ++i) e += gp_total_error[i];
        return e;
    }

    // Overall PR = mean(error) * 500
    double overall_pr() const {
        int n = total_decisions();
        return n > 0 ? (total_error() / n) * 500.0 : 0.0;
    }

    // Per game-plan PR (returns 0 if no decisions)
    double gp_pr(int gp) const {
        return gp_n_decisions[gp] > 0
            ? (gp_total_error[gp] / gp_n_decisions[gp]) * 500.0 : 0.0;
    }
};

// Score a strategy against benchmark PR decisions.
// For 0-ply: base_strategy should be nullptr (strategy evaluates all moves directly).
// For N-ply: base_strategy is the 0-ply strategy used for pre-filtering, strategy
//            is the N-ply strategy that evaluates survivors.
// filter: move filter for the 0-ply pre-filter pass.
PRResult score_benchmark_pr(const Strategy& strategy,
                            const Strategy* base_strategy,
                            const std::vector<PRDecision>& decisions,
                            const MoveFilter& filter,
                            int n_threads = 0);

} // namespace bgbot
