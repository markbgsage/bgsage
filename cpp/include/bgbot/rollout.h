#pragma once

#include "strategy.h"
#include "multipy.h"
#include "types.h"
#include <memory>
#include <array>
#include <vector>
#include <cstdint>

namespace bgbot {

struct RolloutConfig {
    int n_trials = 36;           // Number of trial games per candidate
    int truncation_depth = 7;    // Half-moves before truncating (0 = play to completion)
    int decision_ply = 0;        // Ply depth for move selection during trials
    int vr_ply = 0;              // Ply depth for VR equity estimation (-1 = disable VR)
    MoveFilter filter = MoveFilters::TINY;  // Filter for candidate selection at top level
    int n_threads = 0;           // Threads for parallelizing trials (0 = auto)
    uint32_t seed = 42;
    int late_ply = -1;           // Decision ply for late moves (-1 = same as decision_ply)
    int late_threshold = 20;     // Half-move index where we switch to late_ply
};

// Result of rolling out a single position.
struct RolloutResult {
    double equity = 0.0;                             // From per-prob VR corrected probs
    double std_error = 0.0;                          // SE of equity
    std::array<float, NUM_OUTPUTS> mean_probs = {};
    std::array<float, NUM_OUTPUTS> prob_std_errors = {};  // SE per probability component
    double scalar_vr_equity = 0.0;                   // Scalar equity VR (diagnostic)
    double scalar_vr_se = 0.0;                       // SE of scalar equity VR
};

// Monte Carlo rollout strategy with XG-style variance reduction.
//
// Wraps a base strategy and evaluates positions by playing out trial games
// from the given position. At each half-move in a trial:
//   1. Compute mean equity across all 36 dice outcomes (at vr_ply) — the "expected" equity
//   2. Make the actual move with pre-determined dice (at decision_ply)
//   3. Luck = actual equity - expected equity; accumulated from starting player's perspective
//   4. At truncation/game-end: VR result = outcome equity - accumulated luck
//
// Both sides' luck is tracked (full XG-style VR).
//
// Parallelism: trials are distributed across threads (not scenarios).
// The base strategy is used read-only and must be thread-safe.
class RolloutStrategy : public Strategy {
public:
    RolloutStrategy(std::shared_ptr<Strategy> base, RolloutConfig config);

    // Strategy interface
    double evaluate(const Board& board, bool pre_move_is_race) const override;
    std::array<float, NUM_OUTPUTS> evaluate_probs(
        const Board& board, bool pre_move_is_race) const override;
    std::array<float, NUM_OUTPUTS> evaluate_probs(
        const Board& board, const Board& pre_move_board) const override;
    int best_move_index(const std::vector<Board>& candidates,
                        bool pre_move_is_race) const override;
    int best_move_index(const std::vector<Board>& candidates,
                        const Board& pre_move_board) const override;

    // Rollout a single post-move position.
    RolloutResult rollout_position(const Board& board,
                                   const Board& pre_move_board) const;

    const RolloutConfig& config() const { return config_; }

private:
    mutable std::vector<std::vector<std::pair<int, int>>> cached_dice_;
    mutable int cached_max_moves_ = 0;

    std::shared_ptr<Strategy> base_;
    GamePlanStrategy* base_gps_;   // Cached downcast (null if not GPS)
    RolloutConfig config_;

    // Decision-making strategy for move selection during trials.
    // If decision_ply > 0, this wraps base_ in a MultiPlyStrategy.
    // If decision_ply == 0, this is the same as base_.
    std::shared_ptr<Strategy> decision_strat_;

    // Late-game decision strategy (used after late_threshold half-moves).
    // If late_ply < 0, same as decision_strat_.
    std::shared_ptr<Strategy> late_decision_strat_;

    // VR evaluation strategy. If vr_ply > 0, wraps base_ in MultiPlyStrategy.
    // If vr_ply == 0, same as base_. If vr_ply < 0, VR is disabled.
    std::shared_ptr<Strategy> vr_strat_;

    // The 21 unique dice rolls (shared with MultiPlyStrategy).
    struct DiceRoll { int d1, d2, weight; };
    static const std::array<DiceRoll, 21> ALL_ROLLS;

    // Result from a single trial.
    struct TrialResult {
        std::array<float, NUM_OUTPUTS> probs;  // Per-prob VR corrected, SP perspective
        double equity;                         // Final equity from `probs`
        double scalar_vr_equity;               // Scalar equity VR corrected, SP perspective
    };

    // Run a single trial from a post-move position.
    // dice_seq has pairs (d1,d2) for each half-move.
    TrialResult run_trial(const Board& start_board,
                          const Board& pre_move_board,
                          const std::pair<int,int>* dice_seq,
                          int max_moves) const;

    // Run N trials in parallel for a position, return mean + std error.
    RolloutResult run_trials_parallel(const Board& board,
                                      const Board& pre_move_board) const;

    // GNUbg-style hierarchical permutation array for quasi-random dice.
    // 6 levels × 128 turns × 36 permutations.
    // For 36^N trials, the first N rolls are jointly stratified.
    struct PerArray {
        uint8_t perm[6][128][36];
        int seed = -1;
        void init(uint32_t s);
    };

    // Generate quasi-random dice for all trials using hierarchical permutations.
    static void generate_stratified_dice(
        int n_trials, int max_moves, uint32_t seed,
        std::vector<std::vector<std::pair<int,int>>>& dice_out);

    // Compute probs of the best move for a given roll (for VR computation).
    // Uses vr_strat_ for both move selection and probability evaluation.
    // Returns probs from mover's perspective.
    std::array<float, NUM_OUTPUTS> best_move_probs(
        const Board& board, int d1, int d2) const;

    // Compute equity of the best move for a given roll (scalar VR).
    double best_move_equity(const Board& board, int d1, int d2) const;

    // Internal helper: evaluate the best move among pre-generated candidate
    // boards (used by both VR mean computation and internal move loops).
    std::array<float, NUM_OUTPUTS> best_move_probs_for_candidates(
        const Board& board, const std::vector<Board>& candidates,
        int* best_index = nullptr) const;
};

} // namespace bgbot
