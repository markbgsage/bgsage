#pragma once

#include "strategy.h"
#include "multipy.h"
#include "cube.h"
#include "types.h"
#include <memory>
#include <array>
#include <vector>
#include <cstdint>
#include <atomic>

namespace bgbot {

struct RolloutConfig {
    int n_trials = 36;           // Number of trial games per candidate
    int truncation_depth = 7;    // Half-moves before truncating (0 = play to completion)
    int decision_ply = 1;        // Ply depth for move selection during trials (1 = raw NN)
    bool enable_vr = true;       // Enable variance reduction (VR uses same ply as decision)
    bool parallelize_trials = false;  // Allow parallel trial dispatch for truncated N-ply rollouts
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
//   1. VR mean: evaluate best move for all 21 dice outcomes at 1-ply
//   2. Move selection: pick best move for actual roll using N-ply decision strategy
//   3. VR luck: evaluate chosen move at 1-ply, luck = actual(1-ply) - mean(1-ply)
//   4. Accumulate luck from starting player's perspective
//   5. At truncation/game-end: VR result = outcome - accumulated luck
//
// VR is decoupled from the decision strategy: VR always uses base_ (1-ply)
// regardless of decision ply. Move selection uses decision_strat_ (N-ply)
// before late_threshold, late_decision_strat_ after, base_ for race positions.
// Since VR tracks luck = (actual - mean) with both at 1-ply, biases cancel.
//
// Truncation evaluation always uses decision_strat_ (highest ply) for best
// accuracy, regardless of late_threshold. Race positions use base_ at truncation.
//
// Move-0 caching: all trials share the same starting position, so there are
// only 21 possible first-roll decisions. These are computed once and shared
// via Move0Cache, eliminating (n_trials - 21) redundant N-ply evaluations.
//
// Parallelism: trials are distributed across threads (not scenarios).
// N-ply strategies inside trials use serial evaluation (parallel_evaluate=false).
// The base strategy is used read-only and must be thread-safe.
//
// Unified trial function: run_trial_unified handles both cubeless (n_branches=0)
// and cubeful (n_branches>0) rollout modes. When all branches have dead cubes
// (cube_is_dead), all cubeful overhead is skipped — zero performance cost
// compared to a dedicated cubeless function.
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
    RolloutResult rollout_position(const Board& board) const;

    // Result of a cubeful cube decision rollout.
    struct CubefulRolloutResult {
        double nd_equity = 0.0;     // ND cubeful equity (basis cube units)
        double nd_se = 0.0;         // Standard error of ND
        double dt_equity = 0.0;     // DT cubeful equity (basis cube units)
        double dt_se = 0.0;         // Standard error of DT

        // Cubeless pre-roll rollout (from player-on-roll's perspective)
        RolloutResult cubeless;
    };

    // Cubeful rollout for cube decisions. Rolls out two branches (ND and DT)
    // simultaneously with the same dice sequences. Cube decisions (double/take/pass)
    // are simulated at each half-move using 1-ply Janowski evaluation.
    // `pre_roll_board` is from the player-on-roll's perspective (before rolling).
    CubefulRolloutResult cubeful_cube_decision(
        const Board& pre_roll_board,
        const CubeInfo& cube) const;

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

    // Whether VR is enabled (from config_.enable_vr).
    bool vr_enabled_;

    // The 21 unique dice rolls (shared with MultiPlyStrategy).
    struct DiceRoll { int d1, d2, weight; };
    static const std::array<DiceRoll, 21> ALL_ROLLS;

    // Result from a single trial.
    struct TrialResult {
        std::array<float, NUM_OUTPUTS> probs;  // Per-prob VR corrected, SP perspective
        double equity;                         // Final equity from `probs`
        double scalar_vr_equity;               // Scalar equity VR corrected, SP perspective
    };

    // --- Move-0 shared cache ---
    //
    // All trials in a rollout share the same starting position. There are only
    // 21 possible first rolls, so the move-0 N-ply decision can be computed
    // once and shared across all trials with the same first roll. This avoids
    // (n_trials - 21) redundant N-ply evaluations at move 0.
    //
    // Thread-safe: the first trial to encounter each dice combo computes the
    // result (CAS state 0→1); others spin-wait briefly then read the cache.
    struct Move0Cache {
        static constexpr int N_ROLLS = 21;
        std::atomic<int> state[N_ROLLS];  // 0=empty, 1=computing, 2=ready
        Board chosen[N_ROLLS];            // The best post-move board for each roll

        Move0Cache() {
            for (int i = 0; i < N_ROLLS; ++i)
                state[i].store(0, std::memory_order_relaxed);
        }
    };

    // Precomputed move-1 data for cubeful pre-roll rollouts.
    // After move 0 there are only 21 possible boards (one per first roll), so
    // we can share the entire move-1 VR table and actual-roll decision across
    // all trials that hit the same first roll.
    struct Move1Cache {
        struct Entry {
            bool race = false;
            float cube_x = 0.0f;
            std::array<float, NUM_OUTPUTS> mover_probs = {};
            std::array<std::array<float, NUM_OUTPUTS>, Move0Cache::N_ROLLS> roll_best_probs = {};
            std::array<int, Move0Cache::N_ROLLS> best_candidate_idx = {};
            std::array<double, NUM_OUTPUTS> cl_mean_probs = {0, 0, 0, 0, 0};
            double cl_mean_eq = 0.0;
            std::array<Board, Move0Cache::N_ROLLS> chosen = {};
            std::array<std::array<float, NUM_OUTPUTS>, Move0Cache::N_ROLLS> actual_probs = {};
        };

        std::atomic<int> state[Move0Cache::N_ROLLS];  // 0=empty, 1=computing, 2=ready
        std::array<Entry, Move0Cache::N_ROLLS> entries = {};

        Move1Cache() {
            for (int i = 0; i < Move0Cache::N_ROLLS; ++i)
                state[i].store(0, std::memory_order_relaxed);
        }
    };

    // --- Cubeful rollout internals ---

    // Per-branch state during a cubeful trial.
    struct CubefulBranch {
        CubeInfo cube;           // Current cube state (mover's perspective)
        int basis_cube;          // For normalization (same for all branches)
        double vr_luck;          // Accumulated VR luck (basis cube units, SP perspective)
        bool finished;
        double final_equity;     // Result (basis cube units, SP perspective)
    };

    // Unified trial function for both cubeless and cubeful rollout.
    //
    // When start_post_move=true: evaluates a post-move position (opponent first).
    //   Board is flipped at start. SP parity: is_sp = (move_num % 2 == 1).
    //   Used by: run_trials_parallel → rollout_position → evaluate_probs, best_move_index.
    //
    // When start_post_move=false: evaluates a pre-roll position (SP first).
    //   No flip at start. SP parity: is_sp = (move_num % 2 == 0).
    //   Used by: cubeful_cube_decision.
    //
    // When n_branches=0 (or all branches have dead cubes), all cubeful overhead
    // is skipped — zero performance cost vs a dedicated cubeless function.
    //
    // Returns: TrialResult with cubeless VR-corrected probs and equity.
    // Side effect: sets branches[b].final_equity for each active branch.
    TrialResult run_trial_unified(
        const Board& start_board,
        bool start_post_move,
        CubefulBranch branches[], int n_branches,
        const std::pair<int,int>* dice_seq,
        int max_moves,
        Move0Cache* move0_cache = nullptr,
        Move1Cache* move1_cache = nullptr) const;

    // Run N trials in parallel for a position, return mean + std error.
    RolloutResult run_trials_parallel(const Board& board) const;

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
    // Uses the provided strategy for both move selection and evaluation.
    // Returns probs from mover's perspective.
    std::array<float, NUM_OUTPUTS> best_move_probs(
        const Board& board, int d1, int d2,
        const Strategy& strat) const;

    // Resolve the worker count for trial parallelism. When n_threads=0, we
    // choose a conservative default that preserves cache locality for
    // truncated N-ply rollouts.
    int rollout_thread_count(int n_trials) const;

    // Internal helper: evaluate the best move among pre-generated candidate
    // boards (used by both VR mean computation and internal move loops).
    std::array<float, NUM_OUTPUTS> best_move_probs_for_candidates(
        const Board& board, const std::vector<Board>& candidates,
        const Strategy& strat,
        int* best_index = nullptr) const;

    // Precompute the move-0 choice for each opening roll.
    void prefill_move0_cache(const Board& start_board, Move0Cache& cache,
                             int n_threads = 1,
                             SharedPosCache* shared = nullptr) const;

    // Compute the move-1 cache entry for a specific first roll.
    void populate_move1_cache_entry(const Move0Cache& move0_cache,
                                    int first_roll_idx,
                                    Move1Cache::Entry& entry) const;

    // Precompute all move-1 cache entries. This is especially important for
    // cubeful rollouts, where move 1 is the first expensive opponent turn.
    void prefill_move1_cache(const Move0Cache& move0_cache, Move1Cache& cache,
                             int n_threads,
                             SharedPosCache* shared = nullptr) const;
};

} // namespace bgbot
