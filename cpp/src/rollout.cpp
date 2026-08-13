// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/rollout.h"
#include "bgbot/multipy.h"
#include "bgbot/cube.h"
#include "bgbot/match_equity.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/encoding.h"
#include "bgbot/bearoff.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include <thread>
#include <atomic>

// Define ROLLOUT_PROFILE to enable lightweight per-phase timing counters.
// #define ROLLOUT_PROFILE
#ifdef ROLLOUT_PROFILE
#include <chrono>
namespace rollout_profile {
    static std::atomic<int64_t> vr_time_ns{0};
    static std::atomic<int64_t> trunc_time_ns{0};
    static std::atomic<int64_t> cube_time_ns{0};
    static std::atomic<int64_t> movegen_time_ns{0};
    static std::atomic<int64_t> bmi_time_ns{0};
    static std::atomic<int64_t> trial_count{0};

    void reset() {
        vr_time_ns = 0; trunc_time_ns = 0; cube_time_ns = 0;
        movegen_time_ns = 0; bmi_time_ns = 0; trial_count = 0;
    }
    void print() {
        int64_t n = trial_count.load();
        printf("  Profile: vr=%.1fms bmi=%.1fms trunc=%.1fms cube=%.1fms movegen=%.1fms trials=%lld\n",
               vr_time_ns / 1e6, bmi_time_ns / 1e6, trunc_time_ns / 1e6, cube_time_ns / 1e6,
               movegen_time_ns / 1e6, (long long)n);
    }
}
#define ROLLOUT_TIMER_START auto _rp_timer = std::chrono::high_resolution_clock::now()
#define ROLLOUT_TIMER_ADD(counter) rollout_profile::counter.fetch_add( \
    std::chrono::duration_cast<std::chrono::nanoseconds>( \
        std::chrono::high_resolution_clock::now() - _rp_timer).count(), \
    std::memory_order_relaxed)
#else
namespace rollout_profile {
    inline void reset() {}
    inline void print() {}
}
#define ROLLOUT_TIMER_START (void)0
#define ROLLOUT_TIMER_ADD(counter) (void)0
#endif

namespace bgbot {

// ======================== Static Data ========================

const std::array<RolloutStrategy::DiceRoll, 21> RolloutStrategy::ALL_ROLLS = {{
    {1,1,1}, {2,2,1}, {3,3,1}, {4,4,1}, {5,5,1}, {6,6,1},
    {1,2,2}, {1,3,2}, {1,4,2}, {1,5,2}, {1,6,2},
    {2,3,2}, {2,4,2}, {2,5,2}, {2,6,2},
    {3,4,2}, {3,5,2}, {3,6,2},
    {4,5,2}, {4,6,2},
    {5,6,2}
}};

namespace {
constexpr std::array<std::array<int, 7>, 7> kOrderedRollToIndex = {{
    {-1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  6,  7,  8,  9, 10},
    {-1,  6,  1, 11, 12, 13, 14},
    {-1,  7, 11,  2, 15, 16, 17},
    {-1,  8, 12, 15,  3, 18, 19},
    {-1,  9, 13, 16, 18,  4, 20},
    {-1, 10, 14, 17, 19, 20,  5},
}};

constexpr int kTrialChunkSize = 8;

} // namespace

// ======================== Cache Management ========================

void RolloutStrategy::clear_internal_caches() const {
    // Clear thread-local PosCache (shared by all MultiPlyStrategy instances
    // on this thread). This prevents state accumulation across independent
    // positions that could lead to memory corruption with deep decision plies.
    if (auto* mps = dynamic_cast<MultiPlyStrategy*>(checker_strat_.get())) {
        mps->clear_cache();
    }
    // Other strategies share the same thread_local cache, but clearing
    // it separately is a no-op since clear_cache() memsets the shared cache.

    // Also clear the cross-thread shared position cache.
    if (shared_pos_cache_) {
        shared_pos_cache_->clear();
    }

    // Clear this thread's cubeful evaluation cache (pool threads clear
    // their own at the start of each parallel rollout dispatch).
    clear_cubeful_eval_cache();
}

// Helper: propagate bearoff DB to a Strategy that may be MultiPly or Rollout.
static void propagate_bearoff_db(Strategy* strat, const BearoffDB* db) {
    if (!strat) return;
    if (auto* mps = dynamic_cast<MultiPlyStrategy*>(strat)) {
        mps->set_bearoff_db(db);
    } else if (auto* rs = dynamic_cast<RolloutStrategy*>(strat)) {
        rs->set_bearoff_db(db);
    }
}

void RolloutStrategy::set_bearoff_db(const BearoffDB* db) {
    bearoff_db_ = db;
    // Create bearoff-wrapped version of base_ for cubeful evaluations
    // (cubeful_equity_nply needs a 1-ply strategy with bearoff support).
    if (db && db->is_loaded()) {
        base_bearoff_ = std::make_shared<BearoffStrategy>(base_, db);
    } else {
        base_bearoff_ = nullptr;
    }
    // Propagate to all internal strategies so N-ply evaluations use exact
    // bearoff probs at their 1-ply leaf nodes.
    propagate_bearoff_db(checker_strat_.get(), db);
    propagate_bearoff_db(checker_late_strat_.get(), db);
    propagate_bearoff_db(truncation_strat_.get(), db);
    if (cube_inner_rollout_) cube_inner_rollout_->set_bearoff_db(db);
    if (cube_late_inner_rollout_) cube_late_inner_rollout_->set_bearoff_db(db);
}

void RolloutStrategy::set_move_filter(std::shared_ptr<Strategy> filter) {
    move_filter_ = filter;
    // Propagate to MultiPlyStrategy instances used for checker play
    auto propagate = [&](Strategy* strat) {
        if (auto* mps = dynamic_cast<MultiPlyStrategy*>(strat)) {
            mps->set_move_prefilter(filter);
        }
    };
    propagate(checker_strat_.get());
    propagate(checker_late_strat_.get());
    propagate(truncation_strat_.get());
}

// ======================== Cancellation ========================

void RolloutStrategy::cancel() {
    owned_cancel_flag_.store(true, std::memory_order_relaxed);
    config_.cancel_flag = &owned_cancel_flag_;
}

void RolloutStrategy::reset_cancel() {
    owned_cancel_flag_.store(false, std::memory_order_relaxed);
    config_.cancel_flag = &owned_cancel_flag_;
}

bool RolloutStrategy::is_cancelled() const {
    return config_.cancel_flag &&
           config_.cancel_flag->load(std::memory_order_relaxed);
}

// ======================== Constructor ========================

// Helper: build a Strategy from a TrialEvalConfig.
// Returns base for 1-ply, MultiPlyStrategy for N-ply, or a child RolloutStrategy
// for truncated rollout (single-threaded, lightweight).
static std::shared_ptr<Strategy> build_eval_strategy(
    const TrialEvalConfig& eval,
    int effective_ply,
    const std::shared_ptr<Strategy>& base,
    const std::shared_ptr<Strategy>& filter_base,
    const MoveFilter& internal_filter)
{
    if (eval.is_rollout()) {
        // Truncated rollout: create child RolloutStrategy with n_threads=1
        RolloutConfig inner;
        inner.n_trials = eval.rollout_trials;
        inner.truncation_depth = eval.rollout_depth;
        inner.decision_ply = eval.rollout_ply;
        inner.n_threads = 1;
        inner.enable_vr = true;
        inner.seed = 42;
        if (filter_base) {
            return std::make_shared<RolloutStrategy>(base, filter_base, inner);
        }
        return std::make_shared<RolloutStrategy>(base, inner);
    }

    int ply = eval.is_set() ? eval.ply : effective_ply;
    if (ply > 1) {
        if (filter_base) {
            return std::make_shared<MultiPlyStrategy>(
                base, filter_base, ply, internal_filter);
        }
        return std::make_shared<MultiPlyStrategy>(
            base, ply, internal_filter);
    }
    return base;
}

// Resolve a TrialEvalConfig, applying defaults from legacy fields.
static TrialEvalConfig resolve_cube_eval_config(
    const TrialEvalConfig& cfg, int default_ply)
{
    if (cfg.is_set()) return cfg;
    TrialEvalConfig resolved;
    resolved.ply = default_ply;
    return resolved;
}

// Helper: build all per-purpose strategies for rollout internals.
// If filter_base is provided, creates hybrid MultiPlyStrategy (fast filter + accurate leaf).
static void build_rollout_strategies(
    const std::shared_ptr<Strategy>& base,
    const std::shared_ptr<Strategy>& filter_base,
    const RolloutConfig& config,
    std::shared_ptr<Strategy>& checker_strat,
    std::shared_ptr<Strategy>& checker_late_strat,
    TrialEvalConfig& cube_eval_config,
    TrialEvalConfig& cube_late_eval_config,
    std::shared_ptr<RolloutStrategy>& cube_inner_rollout,
    std::shared_ptr<RolloutStrategy>& cube_late_inner_rollout,
    std::shared_ptr<Strategy>& truncation_strat)
{
    MoveFilter internal_filter = {2, 0.03f};

    // Resolve effective ply values from legacy fields
    int checker_ply = config.decision_ply;
    int checker_late_ply_eff = (config.late_ply >= 1) ? config.late_ply : config.decision_ply;

    // Build checker strategies
    checker_strat = build_eval_strategy(
        config.checker, checker_ply, base, filter_base, internal_filter);

    // Build late checker strategy (share if same config)
    if (!config.checker_late.is_set() && !config.checker.is_set()
        && checker_late_ply_eff == checker_ply) {
        checker_late_strat = checker_strat;
    } else {
        checker_late_strat = build_eval_strategy(
            config.checker_late.is_set() ? config.checker_late : config.checker,
            checker_late_ply_eff, base, filter_base, internal_filter);
    }

    // Resolve cube eval configs (cube defaults to decision_ply to match checker play)
    cube_eval_config = resolve_cube_eval_config(config.cube, checker_ply);
    cube_late_eval_config = resolve_cube_eval_config(
        config.cube_late.is_set() ? config.cube_late : config.cube,
        checker_late_ply_eff);

    // Build inner rollout strategies for truncated rollout cube decisions
    auto make_inner_rollout = [&](const TrialEvalConfig& cfg)
            -> std::shared_ptr<RolloutStrategy> {
        if (!cfg.is_rollout()) return nullptr;
        RolloutConfig inner;
        inner.n_trials = cfg.rollout_trials;
        inner.truncation_depth = cfg.rollout_depth;
        inner.decision_ply = cfg.rollout_ply;
        inner.n_threads = 1;
        inner.enable_vr = true;
        inner.seed = 42;
        if (filter_base) {
            return std::make_shared<RolloutStrategy>(base, filter_base, inner);
        }
        return std::make_shared<RolloutStrategy>(base, inner);
    };
    cube_inner_rollout = make_inner_rollout(cube_eval_config);
    cube_late_inner_rollout = make_inner_rollout(cube_late_eval_config);

    // Truncation evaluation strategy. N-ply truncation evaluations go
    // through the cubeful evaluation engine (cube_eval.cpp) inside
    // run_trial_unified — the fused cubeful walk and the dead-cube cubeless
    // walk — so the only direct evaluate_probs use left is the 1-ply case,
    // which is just the base strategy.
    truncation_strat = base;
}

// Common post-construction init for both RolloutStrategy constructors.
void RolloutStrategy_common_init(RolloutStrategy& self,
                                  const RolloutConfig& config,
                                  int cached_max_moves);

RolloutStrategy::RolloutStrategy(std::shared_ptr<Strategy> base, RolloutConfig config)
    : base_(std::move(base))
    , config_(config)
    , cached_max_moves_((config.truncation_depth > 0)
                       ? config.truncation_depth + 10
                       : 200)
{
    build_rollout_strategies(base_, nullptr, config_,
                             checker_strat_, checker_late_strat_,
                             cube_eval_config_, cube_late_eval_config_,
                             cube_inner_rollout_, cube_late_inner_rollout_,
                             truncation_strat_);

    // Effective truncation ply (for N-ply cubeful evaluation at truncation)
    truncation_ply_ = (config_.truncation_ply >= 1) ? config_.truncation_ply : config_.decision_ply;

    // VR enabled flag
    vr_enabled_ = config_.enable_vr;

    if (config_.n_trials > 0 && cached_max_moves_ > 0) {
        generate_stratified_dice(
            config_.n_trials, cached_max_moves_, config_.seed, cached_dice_);
    }

    if (rollout_thread_count(config_.n_trials) > 1) {
        shared_pos_cache_ = std::make_unique<SharedPosCache>();
    }
}

RolloutStrategy::RolloutStrategy(std::shared_ptr<Strategy> base,
                                 std::shared_ptr<Strategy> filter_base,
                                 RolloutConfig config)
    : base_(std::move(base))
    , config_(config)
    , cached_max_moves_((config.truncation_depth > 0)
                       ? config.truncation_depth + 10
                       : 200)
{
    build_rollout_strategies(base_, filter_base, config_,
                             checker_strat_, checker_late_strat_,
                             cube_eval_config_, cube_late_eval_config_,
                             cube_inner_rollout_, cube_late_inner_rollout_,
                             truncation_strat_);

    // Effective truncation ply (for N-ply cubeful evaluation at truncation)
    truncation_ply_ = (config_.truncation_ply >= 1) ? config_.truncation_ply : config_.decision_ply;

    // VR enabled flag
    vr_enabled_ = config_.enable_vr;

    if (config_.n_trials > 0 && cached_max_moves_ > 0) {
        generate_stratified_dice(
            config_.n_trials, cached_max_moves_, config_.seed, cached_dice_);
    }

    if (rollout_thread_count(config_.n_trials) > 1) {
        shared_pos_cache_ = std::make_unique<SharedPosCache>();
    }
}

// ======================== Stratified Dice (GNUbg-style) ========================

void RolloutStrategy::PerArray::init(uint32_t s) {
    if (seed == static_cast<int>(s)) return;

    // Use a simple PRNG to generate permutations (equivalent to GNUbg's ISAAC)
    std::mt19937 rng(s);

    for (int i = 0; i < 6; ++i) {
        // j starts at i (no need for permutations below the diagonal)
        for (int j = i; j < 128; ++j) {
            // Initialize identity permutation
            for (uint8_t k = 0; k < 36; ++k) {
                perm[i][j][k] = k;
            }
            // Fisher-Yates shuffle
            for (int k = 0; k < 35; ++k) {
                int r = rng() % (36 - k);
                std::swap(perm[i][j][k], perm[i][j][k + r]);
            }
        }
    }
    seed = static_cast<int>(s);
}

void RolloutStrategy::generate_stratified_dice(
    int n_trials, int max_moves, uint32_t seed,
    std::vector<std::vector<std::pair<int,int>>>& dice_out)
{
    dice_out.resize(n_trials);

    // Initialize hierarchical permutation array
    PerArray pa;
    pa.seed = -1;
    pa.init(seed);

    for (int t = 0; t < n_trials; ++t) {
        dice_out[t].resize(max_moves);

        // Per-trial RNG for rolls beyond the stratified range
        std::mt19937 trial_rng(seed + static_cast<uint32_t>(t) * 1000003u + 7u);
        std::uniform_int_distribution<int> die(1, 6);

        for (int m = 0; m < max_moves; ++m) {
            if (m < 128) {
                // Quasi-random: compose hierarchical permutations.
                // For 36^N trials, the first N rolls are jointly stratified.
                // Level i uses turn=m and composes from the previous level's output.
                unsigned int j = 0;
                unsigned int k = 1;  // 36^i
                int max_level = std::min(5, m);
                for (int i = 0; i <= max_level; ++i) {
                    j = pa.perm[i][m][((t / k) + j) % 36];
                    k *= 36;
                }
                dice_out[t][m] = {static_cast<int>(j / 6) + 1,
                                  static_cast<int>(j % 6) + 1};
            } else {
                // Beyond 128 turns: truly random
                dice_out[t][m] = {die(trial_rng), die(trial_rng)};
            }
        }
    }
}

// ======================== Cubeless N-Ply Selection ========================

// A dead money cube: cube_is_dead() is true, so the cubeful evaluation
// engine bypasses Janowski everywhere and its tree reduces exactly to a
// cubeless N-ply evaluation.
static const CubeInfo& dead_cube_info()
{
    static const CubeInfo dead = [] {
        CubeInfo c;
        c.cube_value = 1;
        c.max_cube_value = 1;
        return c;
    }();
    return dead;
}

// Cubeless N-ply move selection for trial moves. When the strategy is a
// MultiPlyStrategy, route through best_move_index_cubeful_multi with a
// single dead cube — with the cube dead, its 1-ply filter scores by plain
// cubeless equity and its rescore runs the batched cubeful evaluation
// engine (cube_eval.cpp) as a cubeless N-ply tree. Other strategies
// (1-ply base, child truncated-rollout evaluators) keep their own
// best_move_index.
static int cubeless_best_move_index(const Strategy& strat,
                                    const std::vector<Board>& candidates,
                                    const Board& pre_move_board)
{
    if (dynamic_cast<const MultiPlyStrategy*>(&strat)) {
        int pick = 0;
        strat.best_move_index_cubeful_multi(candidates, pre_move_board,
                                            &dead_cube_info(), 1,
                                            0.0f, &pick);
        return pick;
    }
    return strat.best_move_index(candidates, pre_move_board);
}

// ======================== VR Helper ========================

// Evaluate probs of the best move for a given roll, from the MOVER's perspective.
// Returns evaluate_probs(chosen, board) for the best chosen move.
// Uses the provided strategy for both move selection and evaluation.
std::array<float, NUM_OUTPUTS> RolloutStrategy::best_move_probs(
    const Board& board, int d1, int d2,
    const Strategy& strat) const
{
    thread_local std::vector<Board> candidates;
    possible_boards(board, d1, d2, candidates);
    return best_move_probs_for_candidates(board, candidates, strat);
}

int RolloutStrategy::rollout_thread_count(int n_trials) const
{
    int n_threads = config_.n_threads;
    if (n_threads <= 0) {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;

        // Truncated rollouts with N-ply move selection share many early-tree
        // subproblems. Keeping them on one worker preserves thread-local
        // MultiPly caches and is often faster than splitting the trials.
        // Allow opt-in via config_.parallelize_trials; default stays serial
        // to preserve cache locality and historical behavior.
        if (!config_.parallelize_trials && config_.truncation_depth > 0 && config_.decision_ply > 1) {
            n_threads = 1;
        }
    }

    n_threads = std::min(n_threads, n_trials);
    return std::max(1, n_threads);
}

std::array<float, NUM_OUTPUTS> RolloutStrategy::best_move_probs_for_candidates(
    const Board& board, const std::vector<Board>& candidates,
    const Strategy& strat, int* best_index) const
{
    if (candidates.empty()) {
        if (best_index) *best_index = -1;
        // No legal moves: pass. Probs describe `board` itself (the player
        // skipped their move), so clamp against `board`.
        auto probs = strat.evaluate_probs(board, board);
        clamp_probs_to_board(probs, board);
        return probs;
    }

    if (candidates.size() == 1) {
        GameResult r = check_game_over(candidates[0]);
        if (best_index) *best_index = 0;
        if (r != GameResult::NOT_OVER) {
            return terminal_probs(r);
        }
        auto probs = strat.evaluate_probs(candidates[0], board);
        clamp_probs_to_board(probs, candidates[0]);
        return probs;
    }

    // Fast batch path when using base_ (1-ply) or base_bearoff_ (1-ply
    // wrapped with the bearoff DB) directly.
    //
    // VR DB consistency (critical): when the bearoff DB is engaged, the actual
    // trial move is selected by the configured checker strategy. At N-ply that
    // strategy short-circuits bearoff leaves to the exact DB, so it plays the
    // DB-OPTIMAL move. The VR mean must use the SAME move, or luck =
    // (actual − mean) is biased. Selecting the best candidate with the raw NN
    // here (batch_evaluate_candidates_best_prob) and merely DB-correcting that
    // candidate's probs can pick a DB-suboptimal move whose DB value is < the
    // DB-optimal move actually played, so luck skews positive and the
    // VR-corrected result skews low (observed: a bearoff cube rollout reporting
    // P(win)=0.72 vs the exact 0.78). So when any candidate is in DB range,
    // rank candidates by exact DB equity (NN equity for any non-bearoff
    // boundary candidate) to match actual play. With no bearoff candidate, the
    // raw-NN fast path is already correct for VR (the trial plays the NN/N-ply
    // move and the mean uses the same), so it is kept unchanged.
    const bool is_base = (&strat == base_.get());
    const bool is_base_bearoff = (base_bearoff_ && &strat == base_bearoff_.get());
    if (is_base || is_base_bearoff) {
        if (is_base_bearoff && bearoff_db_) {
            bool all_bearoff = true, any_bearoff = false;
            for (const auto& c : candidates) {
                if (bearoff_db_->is_bearoff(c)) any_bearoff = true;
                else all_bearoff = false;
            }
            if (any_bearoff) {
                // NN equities are only needed for non-bearoff boundary
                // candidates; pure-bearoff positions skip the NN entirely.
                thread_local std::vector<double> nn_eq;
                if (!all_bearoff) {
                    nn_eq.resize(candidates.size());
                    base_->batch_evaluate_candidates_equity(
                        candidates, board, nn_eq.data());
                }
                double best_eq = -1e30;
                int best_db_idx = 0;
                for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
                    double eq;
                    GameResult gr = check_game_over(candidates[i]);
                    if (gr != GameResult::NOT_OVER) {
                        eq = static_cast<double>(static_cast<int>(gr));
                    } else if (bearoff_db_->is_bearoff(candidates[i])) {
                        auto p = bearoff_db_->lookup_probs(
                            candidates[i], /*post_move=*/true);
                        clamp_probs_to_board(p, candidates[i]);
                        eq = compute_equity(p);
                    } else {
                        eq = nn_eq[i];
                    }
                    if (eq > best_eq) { best_eq = eq; best_db_idx = i; }
                }
                std::array<float, NUM_OUTPUTS> best_probs{};
                GameResult gr = check_game_over(candidates[best_db_idx]);
                if (gr != GameResult::NOT_OVER) {
                    best_probs = terminal_probs(gr);
                } else if (bearoff_db_->is_bearoff(candidates[best_db_idx])) {
                    best_probs = bearoff_db_->lookup_probs(
                        candidates[best_db_idx], /*post_move=*/true);
                    clamp_probs_to_board(best_probs, candidates[best_db_idx]);
                } else {
                    best_probs = base_->evaluate_probs(
                        candidates[best_db_idx], board);
                    clamp_probs_to_board(best_probs, candidates[best_db_idx]);
                }
                if (best_index) *best_index = best_db_idx;
                return best_probs;
            }
        }
        std::array<float, NUM_OUTPUTS> best_probs{};
        int idx = base_->batch_evaluate_candidates_best_prob(
            candidates, board, nullptr, &best_probs);
        if (idx >= 0 && idx < static_cast<int>(candidates.size())) {
            clamp_probs_to_board(best_probs, candidates[idx]);
        }
        if (best_index) *best_index = idx;
        return best_probs;
    }

    // Non-base strategy (e.g. MultiPly N-ply): generous 1-ply pre-filter
    // to avoid evaluating clearly terrible candidates at expensive N-ply depth.
    // Threshold is 2x wider than TINY (0.08) to virtually never drop a good move.
    constexpr double VR_FILTER_THRESHOLD = 0.12;
    constexpr int VR_FILTER_MAX = 8;

    thread_local std::vector<double> eq_buf;
    eq_buf.resize(candidates.size());
    base_->batch_evaluate_candidates_equity(candidates, board, eq_buf.data());

    double best_1ply = -1e30;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (eq_buf[i] > best_1ply) best_1ply = eq_buf[i];
    }

    // Collect survivors within threshold
    thread_local std::vector<std::pair<double, int>> ranked;
    ranked.clear();
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (eq_buf[i] >= best_1ply - VR_FILTER_THRESHOLD) {
            ranked.push_back({eq_buf[i], static_cast<int>(i)});
        }
    }

    // If too many, sort and keep top VR_FILTER_MAX
    if (static_cast<int>(ranked.size()) > VR_FILTER_MAX) {
        std::partial_sort(ranked.begin(), ranked.begin() + VR_FILTER_MAX,
                          ranked.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        ranked.resize(VR_FILTER_MAX);
    }

    // Evaluate survivors at N-ply
    double best_eq = -1e30;
    std::array<float, NUM_OUTPUTS> best_probs = {};
    int best_original_idx = ranked.empty() ? 0 : ranked[0].second;

    for (const auto& [eq0, idx] : ranked) {
        GameResult r = check_game_over(candidates[idx]);
        std::array<float, NUM_OUTPUTS> probs;
        double eq;
        if (r != GameResult::NOT_OVER) {
            probs = terminal_probs(r);
            eq = static_cast<double>(static_cast<int>(r));
        } else {
            probs = strat.evaluate_probs(candidates[idx], board);
            clamp_probs_to_board(probs, candidates[idx]);
            eq = compute_equity(probs);
        }
        if (eq > best_eq) {
            best_eq = eq;
            best_probs = probs;
            best_original_idx = idx;
        }
    }

    if (best_index) *best_index = best_original_idx;
    return best_probs;
}

void RolloutStrategy::prefill_move0_cache(
    const Board& start_board, Move0Cache& cache, int n_threads,
    SharedPosCache* shared, const CubeInfo* select_cubes, int n_select_cubes) const
{
    // Move0 uses checker strategy for move selection.
    // This also warms the per-thread cubeful evaluation cache that later
    // trial-move selection and truncation evaluation hit.
    const auto& current_strat = *checker_strat_;
    const bool using_base = (&current_strat == base_.get());

    // When cube-aware selection is active, precompute cube_x once for the
    // start board (cube_efficiency's current impl is position-only, so this
    // is exact). cube_x is currently unused at N-ply (the cubeful tree
    // computes leaf cube_x per leaf), but passed for interface consistency.
    const bool cube_aware = (select_cubes != nullptr && n_select_cubes > 0);
    float cube_x_stamp = 0.0f;
    if (cube_aware) {
        std::array<float, NUM_OUTPUTS> dummy{};
        auto [pp, op] = pip_counts(start_board);
        cube_x_stamp = cube_efficiency(dummy, is_race(start_board), pp, op);
    }

    auto compute_roll = [&](int roll_idx) {
        if (shared) MultiPlyStrategy::set_shared_cache(shared);

        thread_local std::vector<Board> candidates;
        candidates.clear();
        possible_boards_unsorted(start_board,
                                 ALL_ROLLS[roll_idx].d1,
                                 ALL_ROLLS[roll_idx].d2,
                                 candidates);

        Board chosen;
        if (candidates.empty()) {
            chosen = start_board;
        } else if (candidates.size() == 1) {
            chosen = candidates[0];
        } else if (cube_aware) {
            // Multi-cube cubeful BMI: matches the trial loop's call exactly
            // so the cache value is byte-identical to per-trial recomputation.
            // Store the result for cubes[0] (the ND/shared-board pick).
            std::array<int, 4> picks{0, 0, 0, 0};
            current_strat.best_move_index_cubeful_multi(
                candidates, start_board,
                select_cubes, n_select_cubes,
                cube_x_stamp, picks.data());
            chosen = candidates[picks[0]];
        } else if (using_base) {
            chosen = candidates[base_->best_move_index(candidates, start_board)];
        } else {
            // Matches the trial loop's cubeless fallback exactly so the
            // cache value is byte-identical to per-trial recomputation.
            chosen = candidates[cubeless_best_move_index(
                current_strat, candidates, start_board)];
        }

        cache.chosen[roll_idx] = chosen;
        cache.state[roll_idx].store(2, std::memory_order_release);

        if (shared) MultiPlyStrategy::set_shared_cache(nullptr);
    };

    const int workers = std::min<int>(Move0Cache::N_ROLLS, std::max(1, n_threads));
    if (workers <= 1) {
        for (int i = 0; i < Move0Cache::N_ROLLS; ++i) {
            compute_roll(i);
        }
    } else {
        multipy_parallel_for(Move0Cache::N_ROLLS, workers, compute_roll);
    }
}

void RolloutStrategy::populate_move1_cache_entry(
    const Move0Cache& move0_cache, int first_roll_idx, Move1Cache::Entry& entry,
    const CubeInfo* select_cubes, int n_select_cubes) const
{
    thread_local std::vector<Board> candidates;

    const Board& move0_chosen = move0_cache.chosen[first_roll_idx];
    const Board move1_board = flip(move0_chosen);
    entry.race = is_race(move1_board);

    // For cube-aware selection at move 1, flip every cube state's perspective
    // to match move1_board (current mover at move 1 is the opponent-of-move-0-
    // mover). The flipped cubes remain valid for the cache's lifetime
    // (per-rollout). Multi-cube path is used so the BMI call matches the
    // trial loop's multi call exactly.
    const bool cube_aware = (select_cubes != nullptr && n_select_cubes > 0);
    std::array<CubeInfo, 4> m1_cubes;
    if (cube_aware) {
        for (int i = 0; i < n_select_cubes && i < 4; ++i) {
            m1_cubes[i] = flip_cube_perspective(select_cubes[i]);
        }
    }

    // Move1 uses 1-ply for move selection. The VR averaging over many trials
    // makes higher-ply move selection unnecessary here. When the bearoff DB
    // is set, use the DB-wrapped 1-ply (base_bearoff_) so move-1 evaluations
    // are DB-exact at bearoff positions.
    const auto& current_strat = base_bearoff_ ? *base_bearoff_ : *base_;
    const bool using_base = true;

    const Board opp_board = flip(move1_board);
    // mover_probs are always 1-ply (used for 1-ply Janowski cube decisions and
    // as fallback). When cube strategy is N-ply or rollout, the trial code
    // bypasses these cached probs and calls cube_decision_nply or
    // cubeful_cube_decision directly.
    // Use bearoff DB for exact probs when available. For non-bearoff,
    // use base_bearoff_ if set so any 1-ply leaves that happen to be
    // bearoff are evaluated exactly.
    if (bearoff_db_ && bearoff_db_->is_bearoff(move1_board)) {
        entry.mover_probs = bearoff_db_->lookup_probs(move1_board);
    } else {
        const Strategy& mover_strat = base_bearoff_ ? *base_bearoff_ : *base_;
        entry.mover_probs = invert_probs(mover_strat.evaluate_probs(opp_board, opp_board));
    }
    // mover_probs is the pre-roll cubeless prob from the mover's POV at
    // move1_board. Clamp against move1_board so impossible outcomes are
    // exactly zero (matters when the mover already has bearoff progress
    // or contact has broken).
    clamp_probs_to_board(entry.mover_probs, move1_board);
    {
        auto [pp, op] = pip_counts(move1_board);
        entry.cube_x = cube_efficiency(entry.mover_probs, entry.race, pp, op);
    }

    for (size_t second_roll = 0; second_roll < ALL_ROLLS.size(); ++second_roll) {
        candidates.clear();
        possible_boards_unsorted(move1_board,
                                 ALL_ROLLS[second_roll].d1,
                                 ALL_ROLLS[second_roll].d2,
                                 candidates);

        int best_idx = -1;
        entry.roll_best_probs[second_roll] = best_move_probs_for_candidates(
            move1_board, candidates, current_strat, &best_idx);
        entry.best_candidate_idx[second_roll] = best_idx;

        Board chosen;
        if (candidates.empty()) {
            chosen = move1_board;
        } else if (candidates.size() == 1) {
            chosen = candidates[0];
        } else if (cube_aware) {
            // Multi-cube cubeful BMI matches the trial loop's call exactly.
            // Uses base (1-ply) per ROLLOUT.md §10's move-1 convention.
            std::array<int, 4> picks{0, 0, 0, 0};
            current_strat.best_move_index_cubeful_multi(
                candidates, move1_board,
                m1_cubes.data(), n_select_cubes,
                entry.cube_x, picks.data());
            chosen = candidates[picks[0]];
        } else if (using_base) {
            if (best_idx >= 0 && best_idx < static_cast<int>(candidates.size())) {
                chosen = candidates[best_idx];
            } else {
                chosen = candidates[current_strat.best_move_index(candidates, move1_board)];
            }
        } else {
            chosen = candidates[current_strat.best_move_index(candidates, move1_board)];
        }
        entry.chosen[second_roll] = chosen;

        GameResult r = check_game_over(chosen);
        if (r != GameResult::NOT_OVER) {
            entry.actual_probs[second_roll] = terminal_probs(r);
        } else if (best_idx >= 0 &&
                   best_idx < static_cast<int>(candidates.size()) &&
                   chosen == candidates[best_idx]) {
            // Chosen matches the 1-ply cubeless-best (the common path when
            // cube_aware is off, or when cubeful and cubeless picks happen
            // to agree). Reuse roll_best_probs — already clamped inside
            // best_move_probs_for_candidates.
            entry.actual_probs[second_roll] = entry.roll_best_probs[second_roll];
        } else {
            // Chosen differs from cubeless-best (cube_aware path picked the
            // cubeful-best instead). Evaluate the actual chosen move at
            // 1-ply so VR's "actual" matches the move the trial will play.
            entry.actual_probs[second_roll] = current_strat.evaluate_probs(chosen, move1_board);
            clamp_probs_to_board(entry.actual_probs[second_roll], chosen);
        }
    }

    entry.cl_mean_probs = {0, 0, 0, 0, 0};
    for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
        for (int k = 0; k < NUM_OUTPUTS; ++k) {
            entry.cl_mean_probs[k] += ALL_ROLLS[i].weight * entry.roll_best_probs[i][k];
        }
    }
    for (int k = 0; k < NUM_OUTPUTS; ++k) {
        entry.cl_mean_probs[k] /= 36.0;
    }
    entry.cl_mean_eq =
        2.0 * entry.cl_mean_probs[0] - 1.0 +
        entry.cl_mean_probs[1] - entry.cl_mean_probs[3] +
        entry.cl_mean_probs[2] - entry.cl_mean_probs[4];
}

void RolloutStrategy::prefill_move1_cache(
    const Move0Cache& move0_cache, Move1Cache& cache, int n_threads,
    SharedPosCache* shared, const CubeInfo* select_cubes, int n_select_cubes) const
{
    auto populate_entry = [&](int roll_idx) {
        if (shared) MultiPlyStrategy::set_shared_cache(shared);
        populate_move1_cache_entry(move0_cache, roll_idx, cache.entries[roll_idx],
                                   select_cubes, n_select_cubes);
        cache.state[roll_idx].store(2, std::memory_order_release);
        if (shared) MultiPlyStrategy::set_shared_cache(nullptr);
    };

    const int workers = std::min<int>(Move0Cache::N_ROLLS, std::max(1, n_threads));
    if (workers <= 1) {
        for (int i = 0; i < Move0Cache::N_ROLLS; ++i) {
            populate_entry(i);
        }
    } else {
        multipy_parallel_for(Move0Cache::N_ROLLS, workers, populate_entry);
    }
}

// End-of-game test gating the dead-cube retry of the cube-decision screen
// (see run_trial_unified, Phase 1). True only when the game is close enough to
// over that the cube is effectively dead — both sides down to at most two
// checkers (bar included) and at least one side within 24 pips. That is the
// regime where cube_efficiency's race floor (x >= 0.6) misprices holding the
// cube; everywhere else the live-cube model is calibrated and the cheap
// cube_x-only screen stands.
static bool is_endgame_cube_position(const Board& board) {
    // Bars are stored as non-negative counts: index 25 = mover, index 0 = opp.
    int mover = board[25];
    int opp   = board[0];
    for (int i = 1; i <= 24; ++i) {
        if (board[i] > 0)      mover += board[i];
        else if (board[i] < 0) opp   -= board[i];
        if (mover > 2 || opp > 2) return false;
    }
    auto [player_pips, opp_pips] = pip_counts(board);
    return player_pips <= 24 || opp_pips <= 24;
}

// ======================== Unified Trial Function ========================
//
// Single function for both cubeless (n_branches=0) and cubeful (n_branches>0)
// rollout modes. When all branches have dead cubes (cube_is_dead), all cubeful
// overhead is skipped — zero performance cost vs a dedicated cubeless function.
//
// STARTING CONVENTIONS:
//   start_post_move=true: post-move position (opponent moves first).
//     Board flipped at start. SP parity: is_sp = (move_num % 2 == 1).
//     Used by: run_trials_parallel → rollout_position → evaluate_probs.
//   start_post_move=false: pre-roll position (SP moves first).
//     No flip. SP parity: is_sp = (move_num % 2 == 0).
//     Used by: cubeful_cube_decision.
//
// VR OPTIMIZATION: VR always uses 1-ply (base_) regardless of decision ply.
// Luck = (actual - mean), both at 1-ply, so biases cancel. Eliminates ~90%
// of N-ply evaluations (21 rolls × N-ply → 21 rolls × 1-ply batch).
//
// STRATIFICATION OPTIMIZATION: When n_trials % 36 == 0, the first roll is
// fully stratified, so VR luck at move 0 sums to zero — skip VR on move 0.
//
// Returns: TrialResult with cubeless VR-corrected probs and equity (always).
// Side effect: sets branches[b].final_equity for each active branch.
RolloutStrategy::TrialResult RolloutStrategy::run_trial_unified(
    const Board& start_board,
    bool start_post_move,
    CubefulBranch branches[], int n_branches,
    const std::pair<int,int>* dice_seq,
    int max_moves,
    Move0Cache* move0_cache,
    Move1Cache* move1_cache) const
{
    thread_local std::array<std::vector<Board>, ALL_ROLLS.size()> move_candidates;
    thread_local bool candidates_initialized = false;
    if (!candidates_initialized) {
        for (auto& c : move_candidates) c.reserve(24);
        candidates_initialized = true;
    }

    // Determine if ANY branch has an active (non-dead) cube.
    // When cube_active=false, ALL cubeful overhead is skipped.
    bool cube_active = false;
    for (int b = 0; b < n_branches; ++b) {
        if (!cube_is_dead(branches[b].cube)) { cube_active = true; break; }
    }
    const bool is_match = cube_active && n_branches > 0 && !branches[0].cube.is_money();

    // Trial-scope flag for cube-aware checker selection. When on, move-0 /
    // move-1 caches and best_candidate_idx reuse are bypassed because they
    // store cubeless-best moves; we always re-pick via cubeful BMI against
    // the active branches' cube states. See Phase 4 below for the full
    // discussion. Computed once per trial: deterministic given config_,
    // cube_active, and n_branches.
    const bool use_cubeful_select =
        (config_.cubeful_trial_moves && cube_active && n_branches > 0);

    // Cubeful-late threshold: stop using cube-aware selection at half-moves
    // >= this value, falling back to cubeless BMI. Defaults to
    // ultra_late_threshold (no separate fallback); set lower for full
    // rollouts where ultra_late=9999 keeps cubeful active for ~50 half-moves.
    const int cubeful_late_threshold =
        (config_.cubeful_late_threshold > 0)
            ? config_.cubeful_late_threshold
            : config_.ultra_late_threshold;

    // Starting convention
    Board board;
    int sp_parity_offset;
    if (start_post_move) {
        board = flip(start_board);
        sp_parity_offset = 1;
        // Flip each branch's cube perspective: when start_post_move=true, the
        // input cube is from the post-move mover's (SP's) perspective, but
        // the trial loop maintains branches[b].cube in the *current mover's*
        // perspective. After the board flip above, the current mover (at move 0)
        // is the opponent, so each branch's cube must be flipped to match.
        // Phase 6's per-move flip then keeps it in sync for subsequent moves.
        for (int b = 0; b < n_branches; ++b) {
            branches[b].cube.owner = flip_owner(branches[b].cube.owner);
            if (is_match) {
                std::swap(branches[b].cube.match.away1,
                          branches[b].cube.match.away2);
            }
        }
    } else {
        board = start_board;
        sp_parity_offset = 0;
    }

    // Cubeless VR luck tracking (per-prob component, SP perspective)
    std::array<double, NUM_OUTPUTS> cl_accumulated_luck = {0, 0, 0, 0, 0};
    double cl_scalar_eq_luck = 0.0;
    bool vr_enabled = vr_enabled_;
    int truncation = (config_.truncation_depth > 0) ? config_.truncation_depth : 9999;
    int move0_roll_idx = -1;

    for (int move_num = 0; move_num < truncation && move_num < max_moves; ++move_num) {
        const Move1Cache::Entry* move1_entry = nullptr;
        if (move1_cache && move_num == 1 && move0_roll_idx >= 0 &&
            move0_roll_idx < Move0Cache::N_ROLLS) {
            int s = move1_cache->state[move0_roll_idx].load(std::memory_order_acquire);
            if (s == 2) {
                move1_entry = &move1_cache->entries[move0_roll_idx];
            } else {
                int expected = 0;
                if (move1_cache->state[move0_roll_idx].compare_exchange_strong(
                        expected, 1, std::memory_order_acq_rel)) {
                    // On-demand populate. When use_cubeful_select is on, pass
                    // ALL active-branch cubes in MOVE-0 mover's perspective so
                    // the BMI call matches the trial loop's multi call.
                    // At move_num==1, Phase 6 has already flipped each branch's
                    // cube to MOVE-1 mover perspective; flip back here.
                    // populate_move1_cache_entry re-flips internally to align
                    // with move 1's candidates (two flips = identity).
                    std::array<CubeInfo, 4> sel_storage;
                    const CubeInfo* sel = nullptr;
                    int n_sel = 0;
                    if (use_cubeful_select && n_branches > 0) {
                        for (int b = 0; b < n_branches && b < 4; ++b) {
                            if (!branches[b].finished) {
                                sel_storage[n_sel++] = flip_cube_perspective(branches[b].cube);
                            }
                        }
                        sel = (n_sel > 0) ? sel_storage.data() : nullptr;
                    }
                    populate_move1_cache_entry(*move0_cache, move0_roll_idx,
                                               move1_cache->entries[move0_roll_idx],
                                               sel, n_sel);
                    move1_cache->state[move0_roll_idx].store(2, std::memory_order_release);
                    move1_entry = &move1_cache->entries[move0_roll_idx];
                } else {
                    while (move1_cache->state[move0_roll_idx].load(std::memory_order_acquire) != 2) {
                        std::this_thread::yield();
                    }
                    move1_entry = &move1_cache->entries[move0_roll_idx];
                }
            }
        }

        bool is_sp_turn = (move_num % 2 == sp_parity_offset);
        bool race = move1_entry ? move1_entry->race : is_race(board);
        bool is_late = (move_num >= config_.late_threshold);
        float cube_x = 0.0f;
        bool cube_x_ready = false;

        // Phase 1: Cube check (cubeful only, skip on move 0)
        if (cube_active && move_num > 0) {
            // Determine cube evaluation mode for this move.
            //
            // Cube take/pass decisions are evaluated at the configured cube
            // strategy (decision_ply) for the ENTIRE trial. Unlike checker-play
            // move selection, they are NOT dropped to 1-ply at the late /
            // ultra-late thresholds. Rationale: the trial's outcome is scored
            // by the N-ply truncation at decision_ply, so deciding take/pass at
            // a shallower ply creates a decision-vs-evaluation mismatch — the
            // opponent takes doubles that a consistent-depth evaluation would
            // pass, and the deeper continuation then over-credits the doubler.
            // Under Jacoby with a centered cube this is visible as a No-Double
            // equity above the +1.0 cash ceiling (impossible: while the cube is
            // centered the doubler can realize at most the cash). Checker play
            // can still drop to 1-ply late (its quality is diluted by trial
            // averaging); cube decisions cannot.
            //
            // To keep this affordable, the N-ply cube path below runs a cheap
            // 1-ply screen and only escalates branches the screen flags as
            // doubles to the deep decision_ply recursion (see there) — so the
            // common no-double moves never pay for the deep eval.
            const TrialEvalConfig* cube_cfg = &cube_eval_config_;
            bool use_cube_1ply = (cube_cfg->ply <= 1 && !cube_cfg->is_rollout());

            if (use_cube_1ply) {
                // 1-ply Janowski path (original behavior)
                std::array<float, NUM_OUTPUTS> mover_probs;
                if (move1_entry) {
                    mover_probs = move1_entry->mover_probs;
                    cube_x = move1_entry->cube_x;
                    cube_x_ready = true;
                } else if (bearoff_db_ && bearoff_db_->is_bearoff(board)) {
                    mover_probs = bearoff_db_->lookup_probs(board);
                    auto [pp, op] = pip_counts(board);
                    cube_x = cube_efficiency(mover_probs, race, pp, op);
                    cube_x_ready = true;
                } else {
                    Board opp_board = flip(board);
                    auto opp_probs = base_->evaluate_probs(opp_board, opp_board);
                    mover_probs = invert_probs(opp_probs);
                    auto [pp, op] = pip_counts(board);
                    cube_x = cube_efficiency(mover_probs, race, pp, op);
                    cube_x_ready = true;
                }
                // mover_probs are pre-roll probs from the current mover's POV
                // at `board`. Clamp before applying Janowski.
                clamp_probs_to_board(mover_probs, board);

                for (int b = 0; b < n_branches; ++b) {
                    if (branches[b].finished) continue;
                    if (!can_double(branches[b].cube)) continue;
                    CubeDecision cd = cube_decision_1ply(mover_probs, branches[b].cube, cube_x);
                    if (cd.should_double) {
                        if (cd.is_beaver) {
                            branches[b].cube.cube_value *= 4;
                            branches[b].cube.owner = CubeOwner::OPPONENT;
                        } else if (cd.should_take) {
                            branches[b].cube.cube_value *= 2;
                            branches[b].cube.owner = CubeOwner::OPPONENT;
                        } else {
                            double sp_val;
                            if (is_match) {
                                float mwc = dp_mwc(
                                    branches[b].cube.match.away1,
                                    branches[b].cube.match.away2,
                                    branches[b].cube.cube_value,
                                    branches[b].cube.match.is_crawford);
                                sp_val = is_sp_turn ? static_cast<double>(mwc)
                                                    : (1.0 - static_cast<double>(mwc));
                            } else {
                                sp_val = static_cast<double>(branches[b].cube.cube_value)
                                         / branches[b].basis_cube;
                                if (!is_sp_turn) sp_val = -sp_val;
                            }
                            branches[b].final_equity = sp_val - branches[b].vr_luck;
                            branches[b].finished = true;
                        }
                    }
                }
            } else {
                // N-ply or rollout cube decision (proper cubeful evaluation)
                MoveFilter cube_filter = {2, 0.03f};
                const RolloutStrategy* cube_rollout = nullptr;
                if (cube_cfg->is_rollout()) {
                    // cube_cfg is always the (non-late) cube config now, so use
                    // the matching non-late inner rollout for consistency.
                    cube_rollout = cube_inner_rollout_.get();
                }

                // Also compute 1-ply cube_x for the cubeful VR mean later
                if (!cube_x_ready) {
                    if (move1_entry) {
                        cube_x = move1_entry->cube_x;
                    } else {
                        // Probs not computed in this branch; current cube_efficiency
                        // impl ignores them. When the ML-based formula lands, compute
                        // probs here or plumb them through.
                        std::array<float, NUM_OUTPUTS> dummy_probs{};
                        auto [pp, op] = pip_counts(board);
                        cube_x = cube_efficiency(dummy_probs, race, pp, op);
                    }
                    cube_x_ready = true;
                }

                // --- Batch-friendly path for N-ply cubeful cube decisions ---
                //
                // When cube_rollout is null (i.e. the cube evaluator is plain
                // N-ply cubeful, not a truncated rollout), all branches with
                // can_double=true share the same board and N-ply evaluator,
                // differing only in cube state.  Collect them and evaluate in
                // a single cube_decision_nply_multi call that shares move
                // selection and NN evaluations across branches.
                //
                // For the truncated-rollout cube evaluator (cube_rollout != null)
                // we keep the per-branch loop because each inner rollout has
                // its own dice sequence and state; batching is not trivial.
                CubeDecision cds[8];                // n_branches is small (2)
                int cd_branch[8];
                int n_cd = 0;

                if (cube_rollout == nullptr) {
                    // 1-ply screen + escalate. A branch is sent to the deep
                    // decision_ply cube recursion only when a cheap 1-ply
                    // decision already wants to double; branches the screen
                    // clears keep their cube unchanged (no double). This keeps
                    // the deep recursion off the common no-double moves (it
                    // dominates the cost on contact positions where the cube
                    // stays live for many moves).
                    //
                    // The screen is a COST filter, not a decision: a false
                    // negative silently overrides the configured cube evaluator
                    // and freezes that branch's cube for the rest of the trial.
                    // That is NOT a conservative under-count in any direction —
                    // a missed double belongs to whichever side is on roll, so
                    // it can inflate or deflate the reported equity.
                    //
                    // Screening at the position's cube_x alone produces exactly
                    // such false negatives, most sharply in near-terminal races:
                    // cube_efficiency floors a race at x = 0.6, but a last-roll
                    // bearoff has an effectively DEAD cube, and Janowski's
                    // live-cube term then badly overvalues HOLDING it. Worked
                    // example — a 7-vs-6-pip bearoff, cube owned by the mover at
                    // 63.9% wins: 1-ply says ND +0.47 / DT +0.34 (no double)
                    // while the true cubeful answer is ND +0.278 / DT +0.556 (a
                    // clear double). The gap is 0.13, so no epsilon tolerance on
                    // the screen would catch it; the redouble was simply never
                    // evaluated, and the cube froze for the rest of the trial.
                    //
                    // So in the endgame a branch the cube_x screen clears is
                    // retried at a dead cube (x = 0), which MAXIMIZES the
                    // doubling advantage min(DT, DP) − ND over x:
                    // DT = 2·cl2cf(OPPONENT, x) falls as x rises (the taker's
                    // cube gets livelier) while ND = cl2cf(owner, x) rises when
                    // the mover owns the cube. The two screens can therefore
                    // only disagree one way — x = 0 wanting a double the
                    // position's x does not — so a branch both clear wants no
                    // double at ANY cube efficiency in [0, 1].
                    //
                    // The retry is closed-form on probs already in hand (no
                    // extra NN evaluation), but the escalations it ADMITS are
                    // not free: a dead cube says "double whenever ahead", so
                    // running it unconditionally puts the deep recursion on
                    // roughly every move the mover leads — measured at ~1.5-1.7x
                    // wall-clock across the three cubeful rollout benchmarks.
                    // is_endgame_cube_position() confines it to the positions
                    // whose cube is genuinely near-dead (<= 2 checkers a side,
                    // one side within 24 pips), which is where the mispricing
                    // lives; mid-game positions keep the cube_x-only screen,
                    // where the 0.68 live-cube model is calibrated and the cube
                    // really does stay live. cube_x is resolved above for the
                    // VR mean.
                    std::array<float, NUM_OUTPUTS> screen_probs;
                    bool screen_ready = false;
                    // Endgame test for the dead-cube retry: board-only, so it
                    // is computed at most once per move, not per branch.
                    bool endgame = false, endgame_ready = false;
                    CubeInfo cubes_in[8];
                    for (int b = 0; b < n_branches; ++b) {
                        if (branches[b].finished) continue;
                        if (!can_double(branches[b].cube)) continue;
                        if (!screen_ready) {
                            if (move1_entry) {
                                screen_probs = move1_entry->mover_probs;
                            } else if (bearoff_db_ && bearoff_db_->is_bearoff(board)) {
                                screen_probs = bearoff_db_->lookup_probs(board);
                            } else {
                                Board opp_board = flip(board);
                                auto opp_probs =
                                    base_->evaluate_probs(opp_board, opp_board);
                                screen_probs = invert_probs(opp_probs);
                            }
                            clamp_probs_to_board(screen_probs, board);
                            screen_ready = true;
                        }
                        CubeDecision cd1 =
                            cube_decision_1ply(screen_probs, branches[b].cube, cube_x);
                        if (!cd1.should_double) {
                            if (!endgame_ready) {
                                endgame = is_endgame_cube_position(board);
                                endgame_ready = true;
                            }
                            if (!endgame) continue;  // screen: no double
                            // Retry at a dead cube (x = 0), which maximizes the
                            // doubling advantage — see above.
                            CubeDecision cd0 = cube_decision_1ply(
                                screen_probs, branches[b].cube, 0.0f);
                            if (!cd0.should_double) continue;  // screen: no double
                        }
                        cd_branch[n_cd] = b;
                        cubes_in[n_cd]  = branches[b].cube;
                        ++n_cd;
                    }
                    if (n_cd > 0) {
                        const Strategy& cf_base = base_bearoff_ ? *base_bearoff_ : *base_;
                        {
                            // Shared per-first-roll decision cache at move 1
                            // (board and branch cube states are deterministic
                            // across trials there — see Move1Cache::cd*).
                            bool served = false;
                            if (move_num == 1 && move1_cache &&
                                move0_roll_idx >= 0 &&
                                move0_roll_idx < Move0Cache::N_ROLLS &&
                                n_branches <= 2) {
                                auto& cdst = move1_cache->cd_state[move0_roll_idx];
                                int st = cdst.load(std::memory_order_acquire);
                                if (st != 2) {
                                    int expected = 0;
                                    if (cdst.compare_exchange_strong(
                                            expected, 1,
                                            std::memory_order_acq_rel)) {
                                        CubeDecision tmp[8];
                                        cube_decision_nply_multi(
                                            board, cubes_in, n_cd, cf_base,
                                            cube_cfg->ply, tmp, cube_filter,
                                            1 /*serial*/, move_filter_.get(),
                                            /*deep_prefilter=*/true);
                                        uint8_t mask = 0;
                                        for (int k = 0; k < n_cd; ++k) {
                                            int b = cd_branch[k];
                                            move1_cache->cd[move0_roll_idx][b] = tmp[k];
                                            move1_cache->cd_fp[move0_roll_idx][b] =
                                                cube_state_fingerprint(&cubes_in[k], 1);
                                            mask |= static_cast<uint8_t>(1u << b);
                                        }
                                        move1_cache->cd_mask[move0_roll_idx] = mask;
                                        cdst.store(2, std::memory_order_release);
                                        for (int k = 0; k < n_cd; ++k) cds[k] = tmp[k];
                                        served = true;
                                    } else {
                                        while (cdst.load(std::memory_order_acquire) != 2) {
                                            std::this_thread::yield();
                                        }
                                        st = 2;
                                    }
                                }
                                if (!served && st == 2) {
                                    bool ok = true;
                                    for (int k = 0; k < n_cd; ++k) {
                                        int b = cd_branch[k];
                                        if (!(move1_cache->cd_mask[move0_roll_idx] &
                                              (1u << b)) ||
                                            move1_cache->cd_fp[move0_roll_idx][b] !=
                                                cube_state_fingerprint(&cubes_in[k], 1)) {
                                            ok = false;
                                            break;
                                        }
                                    }
                                    if (ok) {
                                        for (int k = 0; k < n_cd; ++k) {
                                            cds[k] = move1_cache->cd[move0_roll_idx][cd_branch[k]];
                                        }
                                        served = true;
                                    }
                                }
                            }
                            if (!served) {
                                cube_decision_nply_multi(
                                    board, cubes_in, n_cd, cf_base,
                                    cube_cfg->ply, cds, cube_filter,
                                    1 /*serial*/, move_filter_.get(),
                                    /*deep_prefilter=*/true);
                            }
                        }
                    }
                } else {
                    // Truncated-rollout cube decisions: still per-branch.
                    for (int b = 0; b < n_branches; ++b) {
                        if (branches[b].finished) continue;
                        if (!can_double(branches[b].cube)) continue;

                        CubeDecision cd = {};
                        auto cfr = cube_rollout->cubeful_cube_decision(
                            board, branches[b].cube);
                        if (branches[b].cube.is_money()) {
                            float nd_eq = static_cast<float>(cfr.nd_equity);
                            float dt_eq = static_cast<float>(cfr.dt_equity);
                            cd.equity_nd = nd_eq;
                            cd.equity_dp = 1.0f;
                            if (branches[b].cube.beaver && dt_eq < 0.0f) {
                                cd.equity_dt = 2.0f * dt_eq;
                                cd.is_beaver = true;
                            } else {
                                cd.equity_dt = dt_eq;
                            }
                            float best = std::min(cd.equity_dt, cd.equity_dp);
                            cd.should_double = (best > cd.equity_nd);
                            cd.should_take = (cd.equity_dt <= cd.equity_dp);
                        } else {
                            float nd_m = static_cast<float>(cfr.nd_equity);
                            float dt_m = static_cast<float>(cfr.dt_equity);
                            float dp_m = dp_mwc(
                                branches[b].cube.match.away1,
                                branches[b].cube.match.away2,
                                branches[b].cube.cube_value,
                                branches[b].cube.match.is_crawford);
                            float best = std::min(dt_m, dp_m);
                            cd.should_double = (best > nd_m);
                            cd.should_take = (dt_m <= dp_m);
                            cd.equity_nd = nd_m;
                            cd.equity_dt = dt_m;
                            cd.equity_dp = dp_m;
                        }
                        cd_branch[n_cd] = b;
                        cds[n_cd]       = cd;
                        ++n_cd;
                    }
                }

                // Apply decisions.  Each branch's decision depends only on its
                // own cube state + board, so order of application doesn't
                // matter.
                for (int k = 0; k < n_cd; ++k) {
                    int b = cd_branch[k];
                    const CubeDecision& cd = cds[k];
                    if (!cd.should_double) continue;
                    if (cd.is_beaver) {
                        branches[b].cube.cube_value *= 4;
                        branches[b].cube.owner = CubeOwner::OPPONENT;
                    } else if (cd.should_take) {
                        branches[b].cube.cube_value *= 2;
                        branches[b].cube.owner = CubeOwner::OPPONENT;
                    } else {
                        double sp_val;
                        if (is_match) {
                            float mwc = dp_mwc(
                                branches[b].cube.match.away1,
                                branches[b].cube.match.away2,
                                branches[b].cube.cube_value,
                                branches[b].cube.match.is_crawford);
                            sp_val = is_sp_turn ? static_cast<double>(mwc)
                                                : (1.0 - static_cast<double>(mwc));
                        } else {
                            sp_val = static_cast<double>(branches[b].cube.cube_value)
                                     / branches[b].basis_cube;
                            if (!is_sp_turn) sp_val = -sp_val;
                        }
                        branches[b].final_equity = sp_val - branches[b].vr_luck;
                        branches[b].finished = true;
                    }
                }
            } // end N-ply / rollout cube path

            // Check if all branches finished (all D/P'd)
            bool all_done = true;
            for (int b = 0; b < n_branches; ++b) {
                if (!branches[b].finished) { all_done = false; break; }
            }
            if (all_done) {
                // All branches D/P'd — use 1-ply pre-roll cubeless probs
                Board opp_board_early = flip(board);
                const Strategy& early_strat =
                    base_bearoff_ ? *base_bearoff_ : *base_;
                auto opp_probs_early = early_strat.evaluate_probs(opp_board_early, opp_board_early);
                auto early_probs = invert_probs(opp_probs_early);
                // early_probs are pre-roll probs from the current mover's POV
                // at `board`. Clamp against board before perspective conversion.
                clamp_probs_to_board(early_probs, board);
                std::array<float, NUM_OUTPUTS> sp_probs;
                if (is_sp_turn) {
                    sp_probs = early_probs;
                } else {
                    sp_probs = invert_probs(early_probs);
                }
                double raw_eq = compute_equity(sp_probs);
                std::array<float, NUM_OUTPUTS> vr_probs;
                for (int k = 0; k < NUM_OUTPUTS; ++k)
                    vr_probs[k] = static_cast<float>(sp_probs[k] - cl_accumulated_luck[k]);
                return {vr_probs, raw_eq - cl_scalar_eq_luck, raw_eq - cl_scalar_eq_luck};
            }
        }

        // Phase 2: Generate moves + compute actual dice index
        int d1 = dice_seq[move_num].first;
        int d2 = dice_seq[move_num].second;
        int a_die = d1, b_die = d2;
        if (a_die > b_die) std::swap(a_die, b_die);
        int actual_idx = kOrderedRollToIndex[a_die][b_die];
        if (actual_idx < 0) actual_idx = 0;

        // Ultra-late: for moves deep in the trial, drop to 1-ply move
        // selection AND skip VR. Rollout averaging over many trials dilutes
        // both move-selection quality and per-move VR contribution at depth.
        // 1-ply lets us reuse the VR best-candidate pick (zero extra cost).
        const int ultra_late_threshold = config_.ultra_late_threshold;
        // Skip VR at move 0 (stratified) and at ultra-late moves that aren't
        // multiples of 2 (thinned VR). Compute VR at moves 1,2,4,6 only (skip 3,5).
        // Since E[luck] = 0, skipping moves doesn't bias the estimate, just increases
        // variance — but that increase is only NEGLIGIBLE when the trial count is
        // high. At low trial counts (e.g. 1T's 72-path truncated rollouts) the extra
        // variance dominates the estimate; because PR is one-sided (error >= 0),
        // higher variance directly INFLATES the measured error (most visibly in match
        // play, where decisions are sharper). So only thin VR when there are enough
        // trials for the "negligible variance" assumption to actually hold. (2T/3T
        // and full rollouts use ultra_late_threshold=9999 and never reach this path.)
        constexpr int kVrThinningMinTrials = 288;
        bool skip_vr_this_move = (move_num == 0 && config_.n_trials % 36 == 0)
                               || (move_num >= ultra_late_threshold && (move_num % 2 == 1)
                                   && config_.n_trials >= kVrThinningMinTrials);
        bool do_vr = vr_enabled && !skip_vr_this_move;

        ROLLOUT_TIMER_START;
        if (move1_entry) {
            // Fully precomputed for move 1.
            // Exception: cube-aware checker selection bypasses the move1
            // cache's `chosen[]` (cubeless) and needs candidates for the
            // actual roll to re-pick via cubeful BMI.
            if (use_cubeful_select) {
                move_candidates[actual_idx].clear();
                possible_boards_unsorted(board, d1, d2,
                                         move_candidates[actual_idx]);
            }
        } else if (do_vr) {
            for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
                move_candidates[i].clear();
                possible_boards_unsorted(board, ALL_ROLLS[i].d1, ALL_ROLLS[i].d2,
                                         move_candidates[i]);
            }
            // No VR candidate prefiltering — evaluating all candidates at 1-ply
            // is fast and avoids the bias that prefiltering introduces (selecting
            // the best from a filtered subset systematically underestimates the
            // VR mean, biasing luck positive and the VR-corrected result negative).
        } else {
            move_candidates[actual_idx].clear();
            possible_boards_unsorted(board, d1, d2,
                                     move_candidates[actual_idx]);
        }


        // Phase 3: VR mean — always use base_ (1-ply) for efficiency.
        // Checker play strategy for move selection (N-ply or truncated rollout).
        const auto& current_strat =
            (move_num >= config_.ultra_late_threshold ? *base_
               : (is_late ? *checker_late_strat_ : *checker_strat_));
        bool using_base = (&current_strat == base_.get());
        bool can_reuse_vr_idx = do_vr;

        std::array<std::array<float, NUM_OUTPUTS>, 21> roll_best_probs;
        std::array<int, 21> best_candidate_idx{};
        if (cube_active && !cube_x_ready) {
            if (move1_entry) {
                cube_x = move1_entry->cube_x;
            } else {
                // Probs not yet computed in this branch; current cube_efficiency
                // impl ignores them. Revisit when ML-based formula lands.
                std::array<float, NUM_OUTPUTS> dummy_probs{};
                auto [pp, op] = pip_counts(board);
                cube_x = cube_efficiency(dummy_probs, race, pp, op);
            }
            cube_x_ready = true;
        }

        // Per-branch cubeful VR mean (only when cube_active)
        double mean_cf_branch[2] = {0.0, 0.0};
        // Cubeless VR mean
        std::array<double, NUM_OUTPUTS> cl_mean_probs = {0, 0, 0, 0, 0};
        double cl_mean_eq = 0.0;

        if (do_vr) {
            if (move1_entry) {
                roll_best_probs = move1_entry->roll_best_probs;
                best_candidate_idx = move1_entry->best_candidate_idx;
                cl_mean_probs = move1_entry->cl_mean_probs;
                cl_mean_eq = move1_entry->cl_mean_eq;
            } else {
                // Evaluate all 21 rolls at 1-ply. Use base_bearoff_ when the
                // bearoff DB is set so the VR mean is DB-exact for bearoff
                // candidates. This is essential at low trial counts: with
                // unwrapped NN the VR luck = (NN_actual − NN_mean) is finite
                // even though the DB outcome is deterministic, and the move-1+
                // luck doesn't fully sum to zero unless the trial count
                // stratifies all the way to that move.
                const Strategy& vr_base = base_bearoff_ ? *base_bearoff_ : *base_;
                for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
                    int idx = -1;
                    roll_best_probs[i] = best_move_probs_for_candidates(
                        board, move_candidates[i], vr_base, &idx);
                    best_candidate_idx[i] = idx;
                }

                // Cubeless VR mean (always computed)
                for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
                    for (int k = 0; k < NUM_OUTPUTS; ++k)
                        cl_mean_probs[k] += ALL_ROLLS[i].weight * roll_best_probs[i][k];
                }
                for (int k = 0; k < NUM_OUTPUTS; ++k) cl_mean_probs[k] /= 36.0;
                cl_mean_eq = 2.0*cl_mean_probs[0]-1.0 + cl_mean_probs[1]-cl_mean_probs[3]
                             + cl_mean_probs[2]-cl_mean_probs[4];
            }

            // Per-branch cubeful VR mean (only when cube_active)
            if (cube_active) {
                for (int b = 0; b < n_branches; ++b) {
                    if (branches[b].finished) continue;
                    double mean_cf = 0.0;
                    for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
                        double val;
                        if (is_match) {
                            val = cl2cf_match(roll_best_probs[i], branches[b].cube, cube_x);
                        } else {
                            float cf = cl2cf_money(roll_best_probs[i], branches[b].cube.owner, cube_x,
                                                    branches[b].cube.jacoby_active());
                            val = cf * branches[b].cube.cube_value
                                     / branches[b].basis_cube;
                        }
                        mean_cf += ALL_ROLLS[i].weight * val;
                    }
                    mean_cf_branch[b] = mean_cf / 36.0;
                }
            }
        }

        ROLLOUT_TIMER_ADD(vr_time_ns);
#ifdef ROLLOUT_PROFILE
        auto _rp_bmi_timer = std::chrono::high_resolution_clock::now();
#endif

        // Phase 4: Pick best move for actual roll.
        // Move selection uses the full decision strategy (N-ply when applicable).
        //
        // CUBE-AWARE SELECTION: when config_.cubeful_trial_moves is true AND
        // any cube branch is active, the chosen move is the one that maximizes
        // CUBEFUL equity (cl2cf) for the branch cube state, not cubeless
        // equity. This is the trial-level cube-awareness gated by the flag.
        // Single-branch case: chosen move is correctly optimal for that branch.
        // Multi-branch case (cube_decision: ND + DT): branches still share the
        // single trial board, so we use branches[0] (ND) as the selection
        // driver — a shared-board approximation; per-branch trial boards are
        // a possible future extension (see ROLLOUT.md "Cube-Aware Selection").
        // The cubeful-multi BMI call
        // computes per-branch best indices internally, but only picks[0] is
        // applied to the shared board for now.
        //
        // When cube-aware selection is on we bypass the move0 / move1 caches
        // and best_candidate_idx reuse — those store cubeless-best moves.
        //
        // MOVE-0 CACHE: At move 0, all trials share the same starting position
        // and there are only 21 possible first rolls. The first trial to encounter
        // each dice combo computes the N-ply best move; subsequent trials reuse
        // the cached result via CAS.
        // (use_cubeful_select is trial-scope, defined above near cube_active.)

        // Per-move gate: drop to cubeless selection at and beyond the
        // cubeful-late threshold (Idea 3). Cache hits at moves 0/1 are
        // still cube-stamped, so the cubeful path is taken for those moves
        // regardless of this threshold.
        const bool use_cubeful_select_now =
            use_cubeful_select && (move_num < cubeful_late_threshold);

        Board chosen;
        bool used_move0_cache = false;
        if (use_cubeful_select_now) {
            // Cube-aware selection. The Move0Cache and Move1Cache are now
            // cube-stamped (populated with cubeful-best chosen[] under
            // branches[0]'s cube state at prefill time), so we can reuse them
            // here exactly like the cubeless path. The cache stamping
            // assumes ND-branch drives selection, matching the shared-board
            // MVP that applies picks[0] to the trial.
            const auto& candidates = move_candidates[actual_idx];
            if (move1_entry) {
                // Move 1: cache stores cubeful-best chosen[]; use directly.
                chosen = move1_entry->chosen[actual_idx];
            } else if (candidates.empty()) {
                chosen = board;
            } else if (candidates.size() == 1) {
                chosen = candidates[0];
            } else if (move0_cache && move_num == 0) {
                // Move 0: try cube-stamped cache. CAS to claim/populate. The
                // fallback BMI call matches the prefill: multi-cube with all
                // active branches' cubes (in move-0 mover perspective).
                int s = move0_cache->state[actual_idx].load(std::memory_order_acquire);
                if (s == 2) {
                    chosen = move0_cache->chosen[actual_idx];
                    used_move0_cache = true;
                } else {
                    int expected = 0;
                    if (move0_cache->state[actual_idx].compare_exchange_strong(
                            expected, 1, std::memory_order_acq_rel)) {
                        std::array<CubeInfo, 4> active_cubes;
                        int n_active = 0;
                        for (int b = 0; b < n_branches && b < 4; ++b) {
                            if (!branches[b].finished) {
                                active_cubes[n_active++] = branches[b].cube;
                            }
                        }
                        if (n_active == 0) {
                            chosen = candidates[cubeless_best_move_index(
                                current_strat, candidates, board)];
                        } else {
                            std::array<int, 4> picks{0, 0, 0, 0};
                            current_strat.best_move_index_cubeful_multi(
                                candidates, board,
                                active_cubes.data(), n_active, cube_x,
                                picks.data());
                            chosen = candidates[picks[0]];
                        }
                        move0_cache->chosen[actual_idx] = chosen;
                        move0_cache->state[actual_idx].store(2, std::memory_order_release);
                        used_move0_cache = true;
                    } else {
                        while (move0_cache->state[actual_idx].load(std::memory_order_acquire) != 2) {
                            std::this_thread::yield();
                        }
                        chosen = move0_cache->chosen[actual_idx];
                        used_move0_cache = true;
                    }
                }
            } else {
                // No cache or move > 1: fresh cubeful BMI per the original
                // shared-board MVP — gather active cubes and dispatch.
                std::array<CubeInfo, 2> active_cubes;
                int n_active = 0;
                for (int b = 0; b < n_branches; ++b) {
                    if (!branches[b].finished) {
                        active_cubes[n_active++] = branches[b].cube;
                    }
                }
                if (n_active == 0) {
                    chosen = candidates[cubeless_best_move_index(
                        current_strat, candidates, board)];
                } else {
                    std::array<int, 2> picks{0, 0};
                    current_strat.best_move_index_cubeful_multi(
                        candidates, board,
                        active_cubes.data(), n_active, cube_x,
                        picks.data());
                    chosen = candidates[picks[0]];
                }
            }
        } else if (move1_entry) {
            chosen = move1_entry->chosen[actual_idx];
        } else {
            const auto& candidates = move_candidates[actual_idx];
            if (move0_cache && move_num == 0 && !using_base) {
            int s = move0_cache->state[actual_idx].load(std::memory_order_acquire);
            if (s == 2) {
                chosen = move0_cache->chosen[actual_idx];
                used_move0_cache = true;
            } else {
                int expected = 0;
                if (move0_cache->state[actual_idx].compare_exchange_strong(
                        expected, 1, std::memory_order_acq_rel)) {
                    if (candidates.empty()) {
                        chosen = board;
                    } else if (candidates.size() == 1) {
                        chosen = candidates[0];
                    } else {
                        chosen = candidates[cubeless_best_move_index(
                            current_strat, candidates, board)];
                    }
                    move0_cache->chosen[actual_idx] = chosen;
                    move0_cache->state[actual_idx].store(2, std::memory_order_release);
                    used_move0_cache = true;
                } else {
                    while (move0_cache->state[actual_idx].load(std::memory_order_acquire) != 2) {
                        std::this_thread::yield();
                    }
                    chosen = move0_cache->chosen[actual_idx];
                    used_move0_cache = true;
                }
            }
        }

            if (!used_move0_cache) {
                if (candidates.empty()) {
                    chosen = board;
                } else if (candidates.size() == 1) {
                    chosen = candidates[0];
                } else if (using_base) {
                    if (can_reuse_vr_idx && best_candidate_idx[actual_idx] >= 0 &&
                        best_candidate_idx[actual_idx] < static_cast<int>(candidates.size())) {
                        chosen = candidates[best_candidate_idx[actual_idx]];
                    } else {
                        chosen = candidates[base_->best_move_index(candidates, board)];
                    }
                } else {
                    chosen = candidates[cubeless_best_move_index(
                        current_strat, candidates, board)];
                }
            }
        }
#ifdef ROLLOUT_PROFILE
        rollout_profile::bmi_time_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - _rp_bmi_timer).count(),
            std::memory_order_relaxed);
#endif
        if (move_num == 0) move0_roll_idx = actual_idx;

        // Phase 4b: VR luck computation
        if (do_vr) {
            std::array<float, NUM_OUTPUTS> actual_probs;
            if (move1_entry) {
                actual_probs = move1_entry->actual_probs[actual_idx];
            } else if (using_base &&
                       best_candidate_idx[actual_idx] >= 0 &&
                       best_candidate_idx[actual_idx] <
                           static_cast<int>(move_candidates[actual_idx].size()) &&
                       chosen == move_candidates[actual_idx]
                                     [best_candidate_idx[actual_idx]]) {
                // Decision used 1-ply AND chosen == 1-ply-cubeless-best.
                // Reuse VR's stored probs (already clamped inside
                // best_move_probs_for_candidates). When cubeful_trial_moves
                // is on, the cubeful pick can diverge from the cubeless
                // best — the candidate-match guard above ensures we only
                // take this shortcut when the move actually played matches.
                actual_probs = roll_best_probs[actual_idx];
            } else {
                // Decision used N-ply, OR using_base with cube-aware
                // selection picking a non-cubeless-best move. Evaluate
                // chosen at 1-ply for VR.
                // Use base_bearoff_ when the bearoff DB is set so the VR
                // actual is DB-exact at bearoff (matching the VR mean above).
                GameResult r = check_game_over(chosen);
                if (r != GameResult::NOT_OVER) {
                    actual_probs = terminal_probs(r);
                } else {
                    const Strategy& vr_base =
                        base_bearoff_ ? *base_bearoff_ : *base_;
                    actual_probs = vr_base.evaluate_probs(chosen, board);
                    // Clamp against `chosen` (the post-move board these
                    // probs describe) so VR actual matches the clamped VR
                    // mean from best_move_probs_for_candidates.
                    clamp_probs_to_board(actual_probs, chosen);
                }
            }

            // Per-branch cubeful VR luck (only when cube_active)
            if (cube_active) {
                for (int b = 0; b < n_branches; ++b) {
                    if (branches[b].finished) continue;
                    double actual_val;
                    if (is_match) {
                        actual_val = cl2cf_match(actual_probs, branches[b].cube, cube_x);
                    } else {
                        float cf = cl2cf_money(actual_probs, branches[b].cube.owner, cube_x,
                                                branches[b].cube.jacoby_active());
                        actual_val = cf * branches[b].cube.cube_value
                                        / branches[b].basis_cube;
                    }
                    double luck = actual_val - mean_cf_branch[b];
                    if (is_sp_turn) {
                        branches[b].vr_luck += luck;
                    } else {
                        branches[b].vr_luck -= luck;
                    }
                }
            }

            // Cubeless VR luck (always computed)
            std::array<double, NUM_OUTPUTS> luck_mover;
            for (int k = 0; k < NUM_OUTPUTS; ++k)
                luck_mover[k] = actual_probs[k] - cl_mean_probs[k];
            double actual_eq = compute_equity(actual_probs);
            double luck_eq = actual_eq - cl_mean_eq;

            // Cross-map luck to SP perspective
            if (is_sp_turn) {
                for (int k = 0; k < NUM_OUTPUTS; ++k)
                    cl_accumulated_luck[k] += luck_mover[k];
                cl_scalar_eq_luck += luck_eq;
            } else {
                cl_accumulated_luck[0] -= luck_mover[0];
                cl_accumulated_luck[1] += luck_mover[3];
                cl_accumulated_luck[2] += luck_mover[4];
                cl_accumulated_luck[3] += luck_mover[1];
                cl_accumulated_luck[4] += luck_mover[2];
                cl_scalar_eq_luck -= luck_eq;
            }
        }

        // Phase 5: Terminal check
        GameResult result = check_game_over(chosen);
        if (result != GameResult::NOT_OVER) {
            auto t_probs = terminal_probs(result);

            // Cubeful branch terminal (only when cube_active)
            if (cube_active) {
                double mover_eq_full = compute_equity(t_probs);
                for (int b = 0; b < n_branches; ++b) {
                    if (branches[b].finished) continue;
                    double sp_val;
                    if (is_match) {
                        float mwc = cubeless_mwc(
                            t_probs,
                            branches[b].cube.match.away1,
                            branches[b].cube.match.away2,
                            branches[b].cube.cube_value,
                            branches[b].cube.match.is_crawford);
                        sp_val = is_sp_turn ? static_cast<double>(mwc)
                                            : (1.0 - static_cast<double>(mwc));
                    } else {
                        double mover_eq = branches[b].cube.jacoby_active()
                            ? (2.0 * t_probs[0] - 1.0) : mover_eq_full;
                        double points = mover_eq * branches[b].cube.cube_value;
                        sp_val = points / branches[b].basis_cube;
                        if (!is_sp_turn) sp_val = -sp_val;
                    }
                    branches[b].final_equity = sp_val - branches[b].vr_luck;
                    branches[b].finished = true;
                }
            } else if (n_branches > 0) {
                // Dead-cube branches: cubeful = cubeless * scaling
                std::array<float, NUM_OUTPUTS> sp_probs_t;
                if (is_sp_turn) { sp_probs_t = t_probs; }
                else { sp_probs_t = invert_probs(t_probs); }
                double raw_sp = compute_equity(sp_probs_t);
                double vr_sp = raw_sp - cl_scalar_eq_luck;
                for (int b = 0; b < n_branches; ++b) {
                    if (branches[b].finished) continue;
                    branches[b].final_equity = vr_sp
                        * branches[b].cube.cube_value / branches[b].basis_cube;
                    branches[b].finished = true;
                }
            }

            // Cubeless terminal: convert to SP probs, VR correct
            std::array<float, NUM_OUTPUTS> sp_probs;
            if (is_sp_turn) { sp_probs = t_probs; }
            else { sp_probs = invert_probs(t_probs); }
            double raw_eq = compute_equity(sp_probs);
            std::array<float, NUM_OUTPUTS> vr_probs;
            for (int k = 0; k < NUM_OUTPUTS; ++k)
                vr_probs[k] = static_cast<float>(sp_probs[k] - cl_accumulated_luck[k]);
            return {vr_probs, raw_eq - cl_scalar_eq_luck, raw_eq - cl_scalar_eq_luck};
        }

        // Phase 6: Flip board + cube ownership
        board = flip(chosen);
        if (cube_active) {
            for (int b = 0; b < n_branches; ++b) {
                if (!branches[b].finished) {
                    branches[b].cube.owner = flip_owner(branches[b].cube.owner);
                    if (is_match) {
                        std::swap(branches[b].cube.match.away1,
                                  branches[b].cube.match.away2);
                    }
                }
            }
        }
    }

    // Truncation: evaluate the position at the truncation point.
    // `board` is from the next mover's perspective (after Phase 6 flip).
    // `flip(board)` = last mover's post-move board.
    ROLLOUT_TIMER_START;
    Board last_mover_board = flip(board);
    int trunc_move = std::min(truncation, max_moves);
    bool trunc_race = is_race(last_mover_board);
    const auto& trunc_strat = *truncation_strat_;

    // SP parity at truncation: last mover at trunc_move-1.
    bool last_mover_is_sp = ((trunc_move - 1) % 2 == sp_parity_offset);

    // Fused N-ply truncation: when any cube branch is live and the
    // truncation ply is > 1, ONE cubeful tree walk produces both the
    // per-branch cubeful equities and the trial's cubeless probs (the probs
    // are accumulated through the same traversal — see cube_eval.cpp). The
    // 21-roll tree is evaluated serially here; rollout parallelism is across
    // trials. All active branches share the truncation board, so they are
    // batched into a single call and differ only in cube state.
    std::array<float, NUM_OUTPUTS> last_mover_probs;
    bool cubeful_done = false;
    if (cube_active && truncation_ply_ > 1) {
        int active_idx[8];
        CubeInfo active_cubes[8];
        int n_active = 0;
        for (int b = 0; b < n_branches; ++b) {
            if (branches[b].finished) continue;
            active_idx[n_active] = b;
            active_cubes[n_active] = branches[b].cube;
            ++n_active;
        }
        if (n_active > 0) {
            float cfs[8];
            std::array<float, NUM_OUTPUTS> tree_probs{};
            const Strategy& cf_strat = base_bearoff_ ? *base_bearoff_ : *base_;
            cubeful_equity_nply_multi(
                board, active_cubes, n_active, cf_strat,
                truncation_ply_, cfs, MoveFilters::TINY,
                1 /*serial*/, move_filter_.get(),
                false, &tree_probs, /*deep_prefilter=*/true);
            // tree_probs are pre-roll probs in the NEXT mover's perspective;
            // invert to the last mover's POV and clamp (the convention of
            // the cubeless truncation output below).
            last_mover_probs = invert_probs(tree_probs);
            clamp_probs_to_board(last_mover_probs, last_mover_board);

            for (int k = 0; k < n_active; ++k) {
                int b = active_idx[k];
                float cf = cfs[k];
                double sp_val;
                if (is_match) {
                    float mwc = eq2mwc(cf,
                        branches[b].cube.match.away1,
                        branches[b].cube.match.away2,
                        branches[b].cube.cube_value,
                        branches[b].cube.match.is_crawford);
                    sp_val = last_mover_is_sp ? (1.0 - static_cast<double>(mwc))
                                              : static_cast<double>(mwc);
                } else {
                    double points = cf * branches[b].cube.cube_value;
                    sp_val = points / branches[b].basis_cube;
                    if (last_mover_is_sp) sp_val = -sp_val;
                }
                branches[b].final_equity = sp_val - branches[b].vr_luck;
                branches[b].finished = true;
            }
            cubeful_done = true;
        }
    }

    if (!cubeful_done) {
        if (bearoff_db_ && bearoff_db_->is_bearoff(last_mover_board)) {
            // Exact DB probs for bearoff truncation positions.
            last_mover_probs =
                bearoff_db_->lookup_probs(last_mover_board, /*post_move=*/true);
        } else if (truncation_ply_ > 1) {
            // Cubeless N-ply truncation through the cubeful evaluation
            // engine: a single dead cube reduces the tree to a cubeless
            // N-ply evaluation, with the probs accumulated through the same
            // batched walk the fused cubeful truncation above uses.
            float cf_unused;
            std::array<float, NUM_OUTPUTS> tree_probs{};
            const Strategy& cf_strat = base_bearoff_ ? *base_bearoff_ : *base_;
            cubeful_equity_nply_multi(
                board, &dead_cube_info(), 1, cf_strat,
                truncation_ply_, &cf_unused, MoveFilters::TINY,
                1 /*serial*/, move_filter_.get(),
                false, &tree_probs, /*deep_prefilter=*/true);
            // tree_probs are pre-roll probs in the next mover's perspective;
            // invert to the last mover's POV (the convention below).
            last_mover_probs = invert_probs(tree_probs);
        } else {
            last_mover_probs =
                trunc_strat.evaluate_probs(last_mover_board, last_mover_board);
        }
        // Sanity-clamp the truncation probabilities against `last_mover_board`
        // so impossible outcomes (gammon/backgammon when the player has borne
        // off, backgammon when contact has been broken and no checker is in
        // the danger zone) are exactly zero. This is the canonical "amend the
        // cubeless probabilities at the truncation point" behavior — the
        // resulting probs feed both the cubeless VR-corrected return and the
        // 1-ply Janowski branch of the cubeful truncation path.
        clamp_probs_to_board(last_mover_probs, last_mover_board);
    }
    ROLLOUT_TIMER_ADD(trunc_time_ns);
#ifdef ROLLOUT_PROFILE
    rollout_profile::trial_count.fetch_add(1, std::memory_order_relaxed);
#endif

    // Cubeful branch truncation at 1-ply (truncation_ply == 1): Janowski on
    // the truncation cubeless probs. (N-ply cubeful truncation was handled
    // above in the fused tree walk.)
    if (!cubeful_done && cube_active) {
        {
            auto [last_pp, last_op] = pip_counts(last_mover_board);
            float trunc_x = cube_efficiency(last_mover_probs, trunc_race, last_pp, last_op);
            for (int b = 0; b < n_branches; ++b) {
                if (branches[b].finished) continue;
                // Cube is from next mover's perspective; flip to last mover's
                CubeInfo last_cube = branches[b].cube;
                last_cube.owner = flip_owner(last_cube.owner);
                if (is_match) {
                    std::swap(last_cube.match.away1, last_cube.match.away2);
                }
                double sp_val;
                if (is_match) {
                    float mwc = cl2cf_match(last_mover_probs, last_cube, trunc_x);
                    sp_val = last_mover_is_sp ? static_cast<double>(mwc)
                                              : (1.0 - static_cast<double>(mwc));
                } else {
                    float cf = cl2cf_money(last_mover_probs, last_cube.owner, trunc_x,
                                            last_cube.jacoby_active());
                    double points = cf * last_cube.cube_value;
                    sp_val = points / branches[b].basis_cube;
                    if (!last_mover_is_sp) sp_val = -sp_val;
                }
                branches[b].final_equity = sp_val - branches[b].vr_luck;
                branches[b].finished = true;
            }
        }
    } else if (!cubeful_done && n_branches > 0) {
        // Dead-cube branches at truncation
        std::array<float, NUM_OUTPUTS> sp_probs_t;
        if (last_mover_is_sp) { sp_probs_t = last_mover_probs; }
        else { sp_probs_t = invert_probs(last_mover_probs); }
        double raw_sp = compute_equity(sp_probs_t);
        double vr_sp = raw_sp - cl_scalar_eq_luck;
        for (int b = 0; b < n_branches; ++b) {
            if (branches[b].finished) continue;
            branches[b].final_equity = vr_sp
                * branches[b].cube.cube_value / branches[b].basis_cube;
            branches[b].finished = true;
        }
    }

    // Cubeless truncation: convert to SP perspective, VR correct
    std::array<float, NUM_OUTPUTS> sp_probs;
    if (last_mover_is_sp) { sp_probs = last_mover_probs; }
    else { sp_probs = invert_probs(last_mover_probs); }
    double raw_eq = compute_equity(sp_probs);
    std::array<float, NUM_OUTPUTS> vr_probs;
    for (int k = 0; k < NUM_OUTPUTS; ++k)
        vr_probs[k] = static_cast<float>(sp_probs[k] - cl_accumulated_luck[k]);
    return {vr_probs, raw_eq - cl_scalar_eq_luck, raw_eq - cl_scalar_eq_luck};
}

// ======================== Parallel Trial Execution ========================

RolloutResult RolloutStrategy::run_trials_parallel(
    const Board& board,
    RolloutProgressCallback progress) const
{
    // Clear thread-local N-ply cache on the calling thread. Pool threads are
    // cleared inside their lambda. This prevents stale entries from other
    // strategies (which share the same persistent thread pool) from persisting
    // across independent rollout evaluations.
    MultiPlyStrategy::get_cache().clear();
    clear_cubeful_eval_cache();

    const int n_trials = config_.n_trials;
    const int max_moves = (config_.truncation_depth > 0)
        ? config_.truncation_depth + 10  // extra buffer for safety
        : 200;  // generous upper bound for full games

    // Use cached stratified dice generated at construction for this strategy.
    // This preserves deterministic output while avoiding repeated allocation
    // and shuffle work in repeated benchmark loops.
    if (cached_dice_.empty() || cached_max_moves_ != max_moves) {
        cached_dice_.clear();
        cached_max_moves_ = max_moves;
        generate_stratified_dice(n_trials, max_moves, config_.seed, cached_dice_);
    }
    const auto& all_dice = cached_dice_;

    // Determine thread count
    int n_threads = rollout_thread_count(n_trials);

    // Move-0 and Move-1 shared caches: all trials share the same starting
    // position, so first two moves can be precomputed and shared.
    Move0Cache move0_cache;
    Move1Cache move1_cache;
    const bool uses_move1_cache =
        (config_.truncation_depth == 0) || (config_.truncation_depth > 1);

    // For the cubeless path, start_post_move=true means the board is flipped
    // at the start of run_trial_unified. Prefill with the flipped board.
    prefill_move0_cache(flip(board), move0_cache, n_threads);
    if (uses_move1_cache) {
        prefill_move1_cache(move0_cache, move1_cache, n_threads);
    }

    // Fast path: n_threads == 1 can accumulate directly without trial buffers.
    if (n_threads == 1) {
        RolloutResult result;
        std::array<double, NUM_OUTPUTS> sum_probs = {0, 0, 0, 0, 0};
        std::array<double, NUM_OUTPUTS> sum_probs_sq = {0, 0, 0, 0, 0};
        double sum_eq = 0.0, sum_eq_sq = 0.0;
        double sum_svr_eq = 0.0, sum_svr_eq_sq = 0.0;

        const int report_interval = progress ? std::max(1, n_trials / 100) : 0;
        for (int t = 0; t < n_trials; ++t) {
            auto r = run_trial_unified(board, true, nullptr, 0, all_dice[t].data(), max_moves,
                                       &move0_cache, &move1_cache);
            double eq = r.equity;
            for (int k = 0; k < NUM_OUTPUTS; ++k) {
                double v = r.probs[k];
                sum_probs[k] += v;
                sum_probs_sq[k] += v * v;
            }
            sum_eq += eq;
            sum_eq_sq += eq * eq;
            sum_svr_eq += r.scalar_vr_equity;
            sum_svr_eq_sq += r.scalar_vr_equity * r.scalar_vr_equity;
            if (report_interval && ((t + 1) % report_interval == 0 || t + 1 == n_trials)) {
                progress(t + 1, n_trials);
            }
        }

        for (int k = 0; k < NUM_OUTPUTS; ++k) {
            result.mean_probs[k] = static_cast<float>(sum_probs[k] / n_trials);
        }

        for (int k = 0; k < NUM_OUTPUTS; ++k) {
            double mean_k = sum_probs[k] / n_trials;
            double var_k = (sum_probs_sq[k] / n_trials) - (mean_k * mean_k);
            if (var_k < 0) var_k = 0;
            result.prob_std_errors[k] = static_cast<float>(std::sqrt(var_k / n_trials));
        }

        result.equity = sum_eq / n_trials;
        double var_eq = (sum_eq_sq / n_trials) - (result.equity * result.equity);
        if (var_eq < 0) var_eq = 0;
        result.std_error = std::sqrt(var_eq / n_trials);

        result.scalar_vr_equity = sum_svr_eq / n_trials;
        double var_svr = (sum_svr_eq_sq / n_trials) - (result.scalar_vr_equity * result.scalar_vr_equity);
        if (var_svr < 0) var_svr = 0;
        result.scalar_vr_se = std::sqrt(var_svr / n_trials);
        return result;
    }

    // Allocate per-trial results
    std::vector<TrialResult> trial_results(n_trials);

    {
        // Unified threading: same threads do combined move0+move1 prefill (already done
        // above for serial) then trials with work-stealing. SharedPosCache enables
        // cross-thread N-ply cache sharing. For dp==1, all strategies are 1-ply so
        // SharedPosCache/unified threading have minimal overhead.
        if (!shared_pos_cache_) {
            shared_pos_cache_ = std::make_unique<SharedPosCache>();
        }
        if (shared_pos_cache_->inserts.load(std::memory_order_relaxed) >=
            (SharedPosCache::CAPACITY * 3) / 4) {
            shared_pos_cache_->clear();
        }
        SharedPosCache* shared_cache = shared_pos_cache_.get();
        std::atomic<int> next_trial{0};
        std::atomic<int> completed_trials{0};
        const int report_interval = progress ? std::max(1, n_trials / 100) : 0;

        // Use persistent thread pool to avoid thread churn. Creating
        // ephemeral threads per rollout exhausts Windows TLS slots and
        // fragments memory after thousands of create/destroy cycles.
        multipy_parallel_run(n_threads, [&]() {
            // Clear thread-local N-ply cache to prevent stale entries from
            // previous evaluations (other strategies sharing the same pool
            // threads) from accumulating and causing memory corruption.
            MultiPlyStrategy::get_cache().clear();
            clear_cubeful_eval_cache();
            MultiPlyStrategy::set_shared_cache(shared_cache);
            int start;
            while ((start = next_trial.fetch_add(kTrialChunkSize, std::memory_order_relaxed))
                   < n_trials) {
                // Check cancellation between trial chunks
                if (config_.cancel_flag &&
                    config_.cancel_flag->load(std::memory_order_relaxed)) {
                    break;
                }
                int end = std::min(start + kTrialChunkSize, n_trials);
                for (int t = start; t < end; ++t) {
                    trial_results[t] = run_trial_unified(
                        board, true, nullptr, 0,
                        all_dice[t].data(), max_moves,
                        &move0_cache, &move1_cache);
                }
                if (report_interval) {
                    int done = completed_trials.fetch_add(end - start, std::memory_order_relaxed) + (end - start);
                    if (done % report_interval < kTrialChunkSize || done >= n_trials) {
                        progress(std::min(done, n_trials), n_trials);
                    }
                }
            }
            MultiPlyStrategy::set_shared_cache(nullptr);
        });

        // Check if cancelled after trial loop completes
        if (is_cancelled()) {
            throw RolloutCancelled();
        }
    }

    RolloutResult result;
    std::array<double, NUM_OUTPUTS> sum_probs = {0, 0, 0, 0, 0};
    std::array<double, NUM_OUTPUTS> sum_probs_sq = {0, 0, 0, 0, 0};
    double sum_eq = 0.0, sum_eq_sq = 0.0;
    double sum_svr_eq = 0.0, sum_svr_eq_sq = 0.0;

    for (int t = 0; t < n_trials; ++t) {
        double eq = trial_results[t].equity;
        for (int k = 0; k < NUM_OUTPUTS; ++k) {
            double v = trial_results[t].probs[k];
            sum_probs[k] += v;
            sum_probs_sq[k] += v * v;
        }
        sum_eq += eq;
        sum_eq_sq += eq * eq;
        sum_svr_eq += trial_results[t].scalar_vr_equity;
        sum_svr_eq_sq += trial_results[t].scalar_vr_equity * trial_results[t].scalar_vr_equity;
    }

    // Mean probabilities (per-prob VR corrected)
    for (int k = 0; k < NUM_OUTPUTS; ++k) {
        result.mean_probs[k] = static_cast<float>(sum_probs[k] / n_trials);
    }

    // Per-probability standard errors
    for (int k = 0; k < NUM_OUTPUTS; ++k) {
        double mean_k = sum_probs[k] / n_trials;
        double var_k = (sum_probs_sq[k] / n_trials) - (mean_k * mean_k);
        if (var_k < 0) var_k = 0;
        result.prob_std_errors[k] = static_cast<float>(std::sqrt(var_k / n_trials));
    }

    // Equity from per-prob VR corrected probs
    result.equity = sum_eq / n_trials;
    double var_eq = (sum_eq_sq / n_trials) - (result.equity * result.equity);
    if (var_eq < 0) var_eq = 0;
    result.std_error = std::sqrt(var_eq / n_trials);

    // Scalar equity VR
    result.scalar_vr_equity = sum_svr_eq / n_trials;
    double var_svr = (sum_svr_eq_sq / n_trials) - (result.scalar_vr_equity * result.scalar_vr_equity);
    if (var_svr < 0) var_svr = 0;
    result.scalar_vr_se = std::sqrt(var_svr / n_trials);

    return result;
}

// ======================== Cubeful Cube Decision ========================

RolloutStrategy::CubefulRolloutResult RolloutStrategy::cubeful_cube_decision(
    const Board& pre_roll_board,
    const CubeInfo& cube,
    RolloutProgressCallback progress) const
{
    // Clear thread-local N-ply cache (same rationale as run_trials_parallel).
    MultiPlyStrategy::get_cache().clear();
    clear_cubeful_eval_cache();

    const int n_trials = config_.n_trials;
    const int max_moves = (config_.truncation_depth > 0)
        ? config_.truncation_depth + 10 : 200;

    // Ensure stratified dice are generated
    if (cached_dice_.empty() || cached_max_moves_ != max_moves) {
        cached_dice_.clear();
        cached_max_moves_ = max_moves;
        generate_stratified_dice(n_trials, max_moves, config_.seed, cached_dice_);
    }
    const auto& all_dice = cached_dice_;

    // Branch templates
    CubefulBranch nd_template{};
    nd_template.cube = cube;
    nd_template.basis_cube = cube.cube_value;

    CubefulBranch dt_template{};
    dt_template.cube = cube;
    dt_template.cube.cube_value = 2 * cube.cube_value;
    dt_template.cube.owner = CubeOwner::OPPONENT;
    dt_template.basis_cube = cube.cube_value;

    // Per-trial results
    struct CubefulTrialResult {
        double nd_equity;
        double dt_equity;
        TrialResult cubeless;  // VR-corrected cubeless probs from the same game
    };
    std::vector<CubefulTrialResult> trial_results(n_trials);

    // Determine thread count
    int n_threads = rollout_thread_count(n_trials);

    Move0Cache move0_cache;
    Move1Cache move1_cache;
    const bool uses_move1_cache =
        (config_.truncation_depth == 0) || (config_.truncation_depth > 1);

    // Cube-aware cache stamping: when cubeful_trial_moves is on, prefill the
    // caches by calling the SAME multi-cube BMI the trial loop calls (both ND
    // and DT cubes) so cache values are byte-identical to per-trial
    // recomputation. chosen[] stores the result for cubes[0] (ND) — that's
    // what the shared-board MVP applies in the trial.
    std::array<CubeInfo, 2> select_cubes_for_cache{nd_template.cube, dt_template.cube};
    const CubeInfo* select_cubes_ptr = nullptr;
    int n_select_cubes = 0;
    if (config_.cubeful_trial_moves) {
        select_cubes_ptr = select_cubes_for_cache.data();
        n_select_cubes = 2;
    }

    const int report_interval = progress ? std::max(1, n_trials / 100) : 0;

    if (n_threads == 1) {
        // Serial: prefill + trials on a single thread (PosCache stays warm)
        prefill_move0_cache(pre_roll_board, move0_cache, 1, nullptr,
                            select_cubes_ptr, n_select_cubes);
        if (uses_move1_cache) {
            for (int i = 0; i < Move0Cache::N_ROLLS; ++i) {
                populate_move1_cache_entry(move0_cache, i, move1_cache.entries[i],
                                           select_cubes_ptr, n_select_cubes);
                move1_cache.state[i].store(2, std::memory_order_release);
            }
        }
        for (int t = 0; t < n_trials; ++t) {
            CubefulBranch branches[2] = {nd_template, dt_template};
            trial_results[t].cubeless = run_trial_unified(
                pre_roll_board, false, branches, 2,
                all_dice[t].data(), max_moves, &move0_cache, &move1_cache);
            trial_results[t].nd_equity = branches[0].final_equity;
            trial_results[t].dt_equity = branches[1].final_equity;
            if (report_interval && ((t + 1) % report_interval == 0 || t + 1 == n_trials)) {
                progress(t + 1, n_trials);
            }
        }
    } else {
        // Unified threading: same threads do combined move0+move1 prefill then
        // trials. This keeps thread-local PosCache warm across all phases.
        // Move0 and move1 for the same roll index are done back-to-back by the
        // same thread (no barrier needed — move1[r] only depends on move0[r]).
        // After all 21 entries are done, threads proceed to trial work-stealing.
        if (!shared_pos_cache_) {
            shared_pos_cache_ = std::make_unique<SharedPosCache>();
        }
        if (shared_pos_cache_->inserts.load(std::memory_order_relaxed) >=
            (SharedPosCache::CAPACITY * 3) / 4) {
            shared_pos_cache_->clear();
        }
        SharedPosCache* shared_cache = shared_pos_cache_.get();
        std::atomic<int> next_roll{0};
        std::atomic<int> next_trial{0};
        std::atomic<int> completed_trials{0};

        // Precompute move0 strategy selection (same for all rolls).
        const Strategy* m0_strat = checker_strat_.get();

        // Use persistent thread pool — same rationale as cubeless path.
        multipy_parallel_run(n_threads, [&]() {
            // Clear thread-local N-ply cache (same as cubeless path).
            MultiPlyStrategy::get_cache().clear();
            clear_cubeful_eval_cache();
            MultiPlyStrategy::set_shared_cache(shared_cache);

            // Phase 1+2: Combined move0 + move1 prefill per roll.
            // When cube-aware selection is on, prefill calls the SAME multi-
            // cube BMI as the trial loop, with ND and DT cubes — guarantees
            // chosen[] is byte-identical to per-trial recomputation.
            float cube_x_stamp = 0.0f;
            if (select_cubes_ptr) {
                std::array<float, NUM_OUTPUTS> dummy{};
                auto [pp, op] = pip_counts(pre_roll_board);
                cube_x_stamp = cube_efficiency(dummy, is_race(pre_roll_board), pp, op);
            }

            int r;
            while ((r = next_roll.fetch_add(1, std::memory_order_relaxed)) < 21) {
                thread_local std::vector<Board> candidates;
                candidates.clear();
                const auto& roll = ALL_ROLLS[r];
                possible_boards_unsorted(pre_roll_board,
                                         roll.d1, roll.d2, candidates);
                Board chosen;
                if (candidates.empty()) {
                    chosen = pre_roll_board;
                } else if (candidates.size() == 1) {
                    chosen = candidates[0];
                } else if (select_cubes_ptr) {
                    std::array<int, 4> picks{0, 0, 0, 0};
                    m0_strat->best_move_index_cubeful_multi(
                        candidates, pre_roll_board,
                        select_cubes_ptr, n_select_cubes,
                        cube_x_stamp, picks.data());
                    chosen = candidates[picks[0]];
                } else if (m0_strat == base_.get()) {
                    chosen = candidates[base_->best_move_index(
                        candidates, pre_roll_board)];
                } else {
                    // Matches the trial loop's cubeless fallback exactly so
                    // the cache value is byte-identical to per-trial
                    // recomputation (the trial-side CAS fallback uses the
                    // same dead-cube selection).
                    chosen = candidates[cubeless_best_move_index(
                        *m0_strat, candidates, pre_roll_board)];
                }
                move0_cache.chosen[r] = chosen;
                move0_cache.state[r].store(2, std::memory_order_release);

                if (uses_move1_cache) {
                    populate_move1_cache_entry(move0_cache, r, move1_cache.entries[r],
                                               select_cubes_ptr, n_select_cubes);
                    move1_cache.state[r].store(2, std::memory_order_release);
                }
            }

            // No barrier: trials start immediately, cache entries computed on demand.

            // Phase 3: Trials (work-stealing)
            int start;
            while ((start = next_trial.fetch_add(kTrialChunkSize, std::memory_order_relaxed))
                   < n_trials) {
                // Check cancellation between trial chunks
                if (config_.cancel_flag &&
                    config_.cancel_flag->load(std::memory_order_relaxed)) {
                    break;
                }
                int end = std::min(start + kTrialChunkSize, n_trials);
                for (int t = start; t < end; ++t) {
                    CubefulBranch branches[2] = {nd_template, dt_template};
                    trial_results[t].cubeless = run_trial_unified(
                        pre_roll_board, false, branches, 2,
                        all_dice[t].data(), max_moves, &move0_cache, &move1_cache);
                    trial_results[t].nd_equity = branches[0].final_equity;
                    trial_results[t].dt_equity = branches[1].final_equity;
                }
                if (report_interval) {
                    int done = completed_trials.fetch_add(end - start, std::memory_order_relaxed) + (end - start);
                    if (done % report_interval < kTrialChunkSize || done >= n_trials) {
                        progress(std::min(done, n_trials), n_trials);
                    }
                }
            }

            MultiPlyStrategy::set_shared_cache(nullptr);
        });

        // Check if cancelled after trial loop completes
        if (is_cancelled()) {
            throw RolloutCancelled();
        }
    }

    // Aggregate results: cubeful equities + cubeless probs (all from same trials)
    double sum_nd = 0, sum_nd_sq = 0;
    double sum_dt = 0, sum_dt_sq = 0;
    std::array<double, NUM_OUTPUTS> sum_probs = {0,0,0,0,0};
    std::array<double, NUM_OUTPUTS> sum_probs_sq = {0,0,0,0,0};
    double sum_cl_eq = 0, sum_cl_eq_sq = 0;

    for (int t = 0; t < n_trials; ++t) {
        double nd = trial_results[t].nd_equity;
        double dt = trial_results[t].dt_equity;
        sum_nd += nd; sum_nd_sq += nd * nd;
        sum_dt += dt; sum_dt_sq += dt * dt;

        for (int k = 0; k < NUM_OUTPUTS; ++k) {
            double v = trial_results[t].cubeless.probs[k];
            sum_probs[k] += v;
            sum_probs_sq[k] += v * v;
        }
        double eq = trial_results[t].cubeless.equity;
        sum_cl_eq += eq;
        sum_cl_eq_sq += eq * eq;
    }

    CubefulRolloutResult result;
    result.nd_equity = sum_nd / n_trials;
    double var_nd = (sum_nd_sq / n_trials) - (result.nd_equity * result.nd_equity);
    if (var_nd < 0) var_nd = 0;
    result.nd_se = std::sqrt(var_nd / n_trials);

    result.dt_equity = sum_dt / n_trials;
    double var_dt = (sum_dt_sq / n_trials) - (result.dt_equity * result.dt_equity);
    if (var_dt < 0) var_dt = 0;
    result.dt_se = std::sqrt(var_dt / n_trials);

    // Cubeless: mean probs and SEs from the same trial games
    for (int k = 0; k < NUM_OUTPUTS; ++k) {
        result.cubeless.mean_probs[k] = static_cast<float>(sum_probs[k] / n_trials);
        double mean_k = sum_probs[k] / n_trials;
        double var_k = (sum_probs_sq[k] / n_trials) - (mean_k * mean_k);
        if (var_k < 0) var_k = 0;
        result.cubeless.prob_std_errors[k] = static_cast<float>(std::sqrt(var_k / n_trials));
    }
    result.cubeless.equity = cubeless_equity(result.cubeless.mean_probs);
    double mean_cl = sum_cl_eq / n_trials;
    double var_cl = (sum_cl_eq_sq / n_trials) - (mean_cl * mean_cl);
    if (var_cl < 0) var_cl = 0;
    result.cubeless.std_error = std::sqrt(var_cl / n_trials);
    result.cubeless.scalar_vr_equity = mean_cl;
    result.cubeless.scalar_vr_se = result.cubeless.std_error;

    return result;
}

RolloutStrategy::CubefulRolloutResult RolloutStrategy::cubeful_cube_decision_batched(
    const Board& pre_roll_board,
    const CubeInfo& cube,
    RolloutProgressCallback progress) const
{
    // Clear thread-local N-ply cache (same rationale as cubeful_cube_decision).
    MultiPlyStrategy::get_cache().clear();
    clear_cubeful_eval_cache();

    const int n_trials = config_.n_trials;
    const int max_moves = (config_.truncation_depth > 0)
        ? config_.truncation_depth + 10 : 200;
    const double target_se = config_.target_se;
    const int max_batches = config_.max_batches > 0 ? config_.max_batches : 1;

    // Branch templates (identical to cubeful_cube_decision).
    CubefulBranch nd_template{};
    nd_template.cube = cube;
    nd_template.basis_cube = cube.cube_value;
    CubefulBranch dt_template{};
    dt_template.cube = cube;
    dt_template.cube.cube_value = 2 * cube.cube_value;
    dt_template.cube.owner = CubeOwner::OPPONENT;
    dt_template.basis_cube = cube.cube_value;

    struct CubefulTrialResult {
        double nd_equity;
        double dt_equity;
        TrialResult cubeless;
    };
    std::vector<CubefulTrialResult> trial_results(n_trials);

    const int n_threads = rollout_thread_count(n_trials);
    const bool uses_move1_cache =
        (config_.truncation_depth == 0) || (config_.truncation_depth > 1);

    std::array<CubeInfo, 2> select_cubes_for_cache{nd_template.cube, dt_template.cube};
    const CubeInfo* select_cubes_ptr = nullptr;
    int n_select_cubes = 0;
    if (config_.cubeful_trial_moves) {
        select_cubes_ptr = select_cubes_for_cache.data();
        n_select_cubes = 2;
    }

    // Move0/Move1 caches are prefilled ONCE and reused across every batch: they
    // are determined by the board + strategy, independent of the trial dice/seed.
    Move0Cache move0_cache;
    Move1Cache move1_cache;

    if (n_threads > 1 && !shared_pos_cache_) {
        shared_pos_cache_ = std::make_unique<SharedPosCache>();
    }
    if (shared_pos_cache_ &&
        shared_pos_cache_->inserts.load(std::memory_order_relaxed) >=
            (SharedPosCache::CAPACITY * 3) / 4) {
        shared_pos_cache_->clear();
    }
    SharedPosCache* shared_cache = shared_pos_cache_.get();

    // ---- Prefill move0 + move1 once (cube-stamped, byte-identical to the
    //      per-trial recomputation, exactly as cubeful_cube_decision does). ----
    if (n_threads == 1) {
        prefill_move0_cache(pre_roll_board, move0_cache, 1, nullptr,
                            select_cubes_ptr, n_select_cubes);
        if (uses_move1_cache) {
            for (int i = 0; i < Move0Cache::N_ROLLS; ++i) {
                populate_move1_cache_entry(move0_cache, i, move1_cache.entries[i],
                                           select_cubes_ptr, n_select_cubes);
                move1_cache.state[i].store(2, std::memory_order_release);
            }
        }
    } else {
        const Strategy* m0_strat = checker_strat_.get();
        std::atomic<int> next_roll{0};
        multipy_parallel_run(n_threads, [&]() {
            MultiPlyStrategy::get_cache().clear();
            clear_cubeful_eval_cache();
            MultiPlyStrategy::set_shared_cache(shared_cache);

            float cube_x_stamp = 0.0f;
            if (select_cubes_ptr) {
                std::array<float, NUM_OUTPUTS> dummy{};
                auto [pp, op] = pip_counts(pre_roll_board);
                cube_x_stamp = cube_efficiency(dummy, is_race(pre_roll_board), pp, op);
            }

            int r;
            while ((r = next_roll.fetch_add(1, std::memory_order_relaxed)) < 21) {
                thread_local std::vector<Board> candidates;
                candidates.clear();
                const auto& roll = ALL_ROLLS[r];
                possible_boards_unsorted(pre_roll_board, roll.d1, roll.d2, candidates);
                Board chosen;
                if (candidates.empty()) {
                    chosen = pre_roll_board;
                } else if (candidates.size() == 1) {
                    chosen = candidates[0];
                } else if (select_cubes_ptr) {
                    std::array<int, 4> picks{0, 0, 0, 0};
                    m0_strat->best_move_index_cubeful_multi(
                        candidates, pre_roll_board,
                        select_cubes_ptr, n_select_cubes,
                        cube_x_stamp, picks.data());
                    chosen = candidates[picks[0]];
                } else if (m0_strat == base_.get()) {
                    chosen = candidates[base_->best_move_index(candidates, pre_roll_board)];
                } else {
                    chosen = candidates[cubeless_best_move_index(
                        *m0_strat, candidates, pre_roll_board)];
                }
                move0_cache.chosen[r] = chosen;
                move0_cache.state[r].store(2, std::memory_order_release);

                if (uses_move1_cache) {
                    populate_move1_cache_entry(move0_cache, r, move1_cache.entries[r],
                                               select_cubes_ptr, n_select_cubes);
                    move1_cache.state[r].store(2, std::memory_order_release);
                }
            }
            MultiPlyStrategy::set_shared_cache(nullptr);
        });
    }

    // ---- Batch loop: accumulate per-trial stats until the ND-equity SE drops
    //      below target_se (or max_batches is reached). Each batch uses a fresh
    //      seed so the trial paths are independent; the caches above are reused. ----
    double sum_nd = 0, sum_nd_sq = 0;
    double sum_dt = 0, sum_dt_sq = 0;
    std::array<double, NUM_OUTPUTS> sum_probs = {0, 0, 0, 0, 0};
    std::array<double, NUM_OUTPUTS> sum_probs_sq = {0, 0, 0, 0, 0};
    double sum_cl_eq = 0, sum_cl_eq_sq = 0;
    long long total_n = 0;
    int batches_done = 0;
    bool converged = false;

    std::vector<std::vector<std::pair<int, int>>> batch_dice;

    for (int b = 0; b < max_batches; ++b) {
        // Batch 0 uses config_.seed exactly, so a single batch reproduces
        // cubeful_cube_decision; later batches are spread far apart in seed
        // space (golden-ratio step) so trial streams don't overlap.
        const uint32_t batch_seed =
            config_.seed + static_cast<uint32_t>(b) * 0x9E3779B9u;
        generate_stratified_dice(n_trials, max_moves, batch_seed, batch_dice);

        const int report_interval = progress ? std::max(1, n_trials / 100) : 0;

        if (n_threads == 1) {
            for (int t = 0; t < n_trials; ++t) {
                CubefulBranch branches[2] = {nd_template, dt_template};
                trial_results[t].cubeless = run_trial_unified(
                    pre_roll_board, false, branches, 2,
                    batch_dice[t].data(), max_moves, &move0_cache, &move1_cache);
                trial_results[t].nd_equity = branches[0].final_equity;
                trial_results[t].dt_equity = branches[1].final_equity;
                if (report_interval && ((t + 1) % report_interval == 0 || t + 1 == n_trials)) {
                    progress(t + 1, n_trials);
                }
            }
        } else {
            std::atomic<int> next_trial{0};
            std::atomic<int> completed_trials{0};
            multipy_parallel_run(n_threads, [&]() {
                MultiPlyStrategy::get_cache().clear();
                clear_cubeful_eval_cache();
                MultiPlyStrategy::set_shared_cache(shared_cache);
                int start;
                while ((start = next_trial.fetch_add(kTrialChunkSize, std::memory_order_relaxed))
                       < n_trials) {
                    if (config_.cancel_flag &&
                        config_.cancel_flag->load(std::memory_order_relaxed)) {
                        break;
                    }
                    int end = std::min(start + kTrialChunkSize, n_trials);
                    for (int t = start; t < end; ++t) {
                        CubefulBranch branches[2] = {nd_template, dt_template};
                        trial_results[t].cubeless = run_trial_unified(
                            pre_roll_board, false, branches, 2,
                            batch_dice[t].data(), max_moves, &move0_cache, &move1_cache);
                        trial_results[t].nd_equity = branches[0].final_equity;
                        trial_results[t].dt_equity = branches[1].final_equity;
                    }
                    if (report_interval) {
                        int done = completed_trials.fetch_add(end - start, std::memory_order_relaxed) + (end - start);
                        if (done % report_interval < kTrialChunkSize || done >= n_trials) {
                            progress(std::min(done, n_trials), n_trials);
                        }
                    }
                }
                MultiPlyStrategy::set_shared_cache(nullptr);
            });
            if (is_cancelled()) {
                throw RolloutCancelled();
            }
        }

        for (int t = 0; t < n_trials; ++t) {
            double nd = trial_results[t].nd_equity;
            double dt = trial_results[t].dt_equity;
            sum_nd += nd; sum_nd_sq += nd * nd;
            sum_dt += dt; sum_dt_sq += dt * dt;
            for (int k = 0; k < NUM_OUTPUTS; ++k) {
                double v = trial_results[t].cubeless.probs[k];
                sum_probs[k] += v;
                sum_probs_sq[k] += v * v;
            }
            double eq = trial_results[t].cubeless.equity;
            sum_cl_eq += eq;
            sum_cl_eq_sq += eq * eq;
        }
        total_n += n_trials;
        batches_done = b + 1;

        // Gating quantity: ND-equity standard error over all trials so far.
        double nd_mean = sum_nd / total_n;
        double var_nd = (sum_nd_sq / total_n) - (nd_mean * nd_mean);
        if (var_nd < 0) var_nd = 0;
        double nd_se = std::sqrt(var_nd / total_n);
        if (target_se <= 0.0 || nd_se <= target_se) {
            converged = (target_se <= 0.0) || (nd_se <= target_se);
            break;
        }
    }

    // ---- Finalize from accumulated sums / total_n (same formulas as the
    //      single-batch path, with total_n in place of n_trials). ----
    CubefulRolloutResult result;
    result.nd_equity = sum_nd / total_n;
    double var_nd = (sum_nd_sq / total_n) - (result.nd_equity * result.nd_equity);
    if (var_nd < 0) var_nd = 0;
    result.nd_se = std::sqrt(var_nd / total_n);

    result.dt_equity = sum_dt / total_n;
    double var_dt = (sum_dt_sq / total_n) - (result.dt_equity * result.dt_equity);
    if (var_dt < 0) var_dt = 0;
    result.dt_se = std::sqrt(var_dt / total_n);

    for (int k = 0; k < NUM_OUTPUTS; ++k) {
        double mean_k = sum_probs[k] / total_n;
        result.cubeless.mean_probs[k] = static_cast<float>(mean_k);
        double var_k = (sum_probs_sq[k] / total_n) - (mean_k * mean_k);
        if (var_k < 0) var_k = 0;
        result.cubeless.prob_std_errors[k] = static_cast<float>(std::sqrt(var_k / total_n));
    }
    result.cubeless.equity = cubeless_equity(result.cubeless.mean_probs);
    double mean_cl = sum_cl_eq / total_n;
    double var_cl = (sum_cl_eq_sq / total_n) - (mean_cl * mean_cl);
    if (var_cl < 0) var_cl = 0;
    result.cubeless.std_error = std::sqrt(var_cl / total_n);
    result.cubeless.scalar_vr_equity = mean_cl;
    result.cubeless.scalar_vr_se = result.cubeless.std_error;

    result.n_batches = batches_done;
    result.total_trials = total_n;
    result.se_converged = converged;
    return result;
}

// ======================== Cubeful Position Rollout (post-move) =============
//
// Returns the rollout-level cubeful equity of a post-move position, including
// the opponent's optimal cube action at the start of their turn. Used by the
// checker-play analyzer to score candidate moves consistently with the
// cube-action analyzer at the same eval level (checker_play_cubeful(M) ==
// -cube_action_optimal_equity(opp_perspective_after_M)).
//
// Implemented as a thin wrapper that flips perspective and delegates to
// cubeful_cube_decision, then collapses ND/DT/DP into opp's optimal action
// (the same logic the cube_decision pybind binding uses). All rollout work
// happens in cubeful_cube_decision; this function only does perspective
// inversion and the cube-action collapse.

namespace {
// Scale a standard error in MWC space to equity space by the local slope of
// mwc2eq around mean_mwc. mwc2eq is piecewise linear, so a small numerical
// derivative is exact within the active segment.
inline float mwc_se_to_eq_se(float mwc_se, float mean_mwc,
                             int away1, int away2, int cube_value,
                             bool is_crawford) {
    float eps = std::max(1e-3f, mwc_se);
    float eq_plus = mwc2eq(mean_mwc + eps, away1, away2, cube_value, is_crawford);
    float eq_minus = mwc2eq(mean_mwc - eps, away1, away2, cube_value, is_crawford);
    float slope = (eq_plus - eq_minus) / (2.0f * eps);
    return std::abs(slope) * mwc_se;
}
} // namespace

RolloutStrategy::CubefulPositionResult RolloutStrategy::cubeful_rollout_position(
    const Board& post_move_board,
    const CubeInfo& cube,
    RolloutProgressCallback progress) const
{
    // Cubeful equity of a post-move position must include the opponent's
    // optimal cube action at the start of their turn — otherwise moves that
    // leave a drop position are scored as if the opponent will not double.
    // This is the rollout analog of cubeful_equity_nply on the flipped opp
    // position used by the multi-ply checker-play path.
    //
    // Implementation: delegate to cubeful_cube_decision on the opp-perspective
    // board, then collapse ND/DT/DP into opp's optimal cube action using the
    // same min-of-double-vs-no-double rule the cube_decision binding applies.
    // SP's cubeful equity is -opp_optimal (perspective flip); cubeless probs
    // come from the same trials, inverted to SP's perspective.

    // Terminal short-circuit: if the post-move board is already decided (the
    // just-moved player has borne off all 15 checkers), the game is over and the
    // cube is irrelevant. Mirror the N-ply engine (cube_eval.cpp ~486), which
    // checks check_game_over at the top of the recursion. Without this guard the
    // code below flips to the opponent and "plays out" the already-decided game,
    // so a WON post-move board is scored as -1.0 (the loser bears its checkers off
    // and "wins" the game back) — a 2.0-equity blunder concentrated in race/bearoff
    // endings. post_move_board is in SP's perspective and terminal_probs /
    // cubeless_equity are SP-relative, so NO inversion is needed here.
    {
        GameResult gr_term = check_game_over(post_move_board);
        if (gr_term != GameResult::NOT_OVER) {
            auto t_probs = terminal_probs(gr_term);
            CubefulPositionResult tr;
            if (cube.is_money()) {
                tr.cubeful_equity = cube.jacoby_active()
                    ? (2.0 * static_cast<double>(t_probs[0]) - 1.0)
                    : static_cast<double>(cubeless_equity(t_probs));
            } else {
                float mwc = cubeless_mwc(t_probs, cube.match.away1, cube.match.away2,
                                         cube.cube_value, cube.match.is_crawford);
                tr.cubeful_equity = static_cast<double>(
                    mwc2eq(mwc, cube.match.away1, cube.match.away2,
                           cube.cube_value, cube.match.is_crawford));
            }
            tr.cubeful_se = 0.0;
            tr.cubeless.mean_probs = t_probs;
            tr.cubeless.equity = static_cast<double>(cubeless_equity(t_probs));
            tr.cubeless.std_error = 0.0;
            tr.cubeless.scalar_vr_equity = tr.cubeless.equity;
            tr.cubeless.scalar_vr_se = 0.0;
            return tr;
        }
    }

    // Flip board and cube to opp's perspective. Cube ownership flips because
    // "owner" is relative to the player on roll; in match play the away-score
    // pair also swaps.
    Board opp_board = flip(post_move_board);
    CubeInfo opp_cube = cube;
    opp_cube.owner = flip_owner(opp_cube.owner);
    if (!opp_cube.is_money()) {
        std::swap(opp_cube.match.away1, opp_cube.match.away2);
    }

    // Run the two-branch (ND/DT) cube decision rollout from opp's perspective.
    CubefulRolloutResult cfr = cubeful_cube_decision(opp_board, opp_cube, progress);

    // Compute opp's optimal cube action equity. Mirrors the logic in the
    // pybind cube_decision binding so callers see identical optimal_equity
    // values from cubeful_evaluate_board and cube_decision on the same board.
    //
    // When opp can't legally double (cube is owned by SP or already at
    // max_cube_value, or match-play rules forbid it), only the ND branch is
    // a valid action — opp must just play. The cubeful_cube_decision call
    // above still ran a DT trial, but that DT result represents an illegal
    // cube turn and must NOT be considered. Without this guard, post-move
    // positions where the original SP owns the cube spuriously collapse to
    // -DP = -1.0 because the rollout's noisy dt_equity > 1 makes the DP
    // branch "optimal" even though opp can't actually offer the cube.
    const bool opp_can_double = can_double(opp_cube);
    float optimal_equity;
    double opt_se;
    bool should_double, should_take;
    if (opp_cube.is_money()) {
        float equity_nd = static_cast<float>(cfr.nd_equity);
        float actual_dt = static_cast<float>(cfr.dt_equity);
        float equity_dp = 1.0f;
        float equity_dt = (opp_cube.beaver && actual_dt < 0.0f)
            ? 2.0f * actual_dt  // Double/Beaver
            : actual_dt;
        if (opp_can_double) {
            float best_double = std::min(equity_dt, equity_dp);
            should_double = (best_double > equity_nd);
            should_take = (equity_dt <= equity_dp);
            optimal_equity = should_double ? best_double : equity_nd;
        } else {
            should_double = false;
            should_take = true;   // not applicable; pick a neutral default
            optimal_equity = equity_nd;
        }

        // SE of the chosen optimal action. DP equity is exactly +1.0 in money
        // (no rollout variance), so D/P contributes zero SE.
        if (should_double && !should_take) {
            opt_se = 0.0;
        } else if (should_double) {
            opt_se = cfr.dt_se;
        } else {
            opt_se = cfr.nd_se;
        }
    } else {
        // Match play: work in MWC, convert to equity at the end.
        int a1 = opp_cube.match.away1, a2 = opp_cube.match.away2;
        int cv = opp_cube.cube_value;
        bool craw = opp_cube.match.is_crawford;
        float nd_m = static_cast<float>(cfr.nd_equity);
        float dt_m = static_cast<float>(cfr.dt_equity);
        float dp_m = dp_mwc(a1, a2, cv, craw);

        float optimal_mwc;
        if (opp_can_double) {
            bool auto_double = (!craw && a1 > 1 && a2 == 1);
            float best_mwc = std::min(dt_m, dp_m);
            should_double = auto_double || (best_mwc > nd_m);
            should_take = (dt_m <= dp_m);
            optimal_mwc = should_double ? best_mwc : nd_m;
        } else {
            should_double = false;
            should_take = true;
            optimal_mwc = nd_m;
        }
        optimal_equity = mwc2eq(optimal_mwc, a1, a2, cv, craw);

        float opt_mwc_se;
        if (should_double && !should_take) {
            opt_mwc_se = 0.0f;
        } else if (should_double) {
            opt_mwc_se = static_cast<float>(cfr.dt_se);
        } else {
            opt_mwc_se = static_cast<float>(cfr.nd_se);
        }
        opt_se = mwc_se_to_eq_se(opt_mwc_se, optimal_mwc, a1, a2, cv, craw);
    }

    // Assemble result: SP's cubeful is -opp_optimal; cubeless probs invert.
    CubefulPositionResult result;
    result.cubeful_equity = -static_cast<double>(optimal_equity);
    result.cubeful_se = opt_se;

    // invert_probs swaps gammon/backgammon indices 1<->3, 2<->4 and reflects
    // P(win); the per-prob SEs use the same swap (var(1-X)=var(X) so index 0
    // SE is unchanged).
    result.cubeless.mean_probs = invert_probs(cfr.cubeless.mean_probs);
    result.cubeless.prob_std_errors[0] = cfr.cubeless.prob_std_errors[0];
    result.cubeless.prob_std_errors[1] = cfr.cubeless.prob_std_errors[3];
    result.cubeless.prob_std_errors[2] = cfr.cubeless.prob_std_errors[4];
    result.cubeless.prob_std_errors[3] = cfr.cubeless.prob_std_errors[1];
    result.cubeless.prob_std_errors[4] = cfr.cubeless.prob_std_errors[2];
    result.cubeless.equity = -cfr.cubeless.equity;
    result.cubeless.std_error = cfr.cubeless.std_error;
    result.cubeless.scalar_vr_equity = -cfr.cubeless.scalar_vr_equity;
    result.cubeless.scalar_vr_se = cfr.cubeless.scalar_vr_se;

    return result;
}

// ======================== Public Interface ========================

double RolloutStrategy::evaluate(const Board& board, bool pre_move_is_race) const {
    auto r = rollout_position(board);
    return r.equity;
}

std::array<float, NUM_OUTPUTS> RolloutStrategy::evaluate_probs(
    const Board& board, bool pre_move_is_race) const
{
    auto r = rollout_position(board);
    return r.mean_probs;
}

std::array<float, NUM_OUTPUTS> RolloutStrategy::evaluate_probs(
    const Board& board, const Board& pre_move_board) const
{
    auto r = rollout_position(board);
    return r.mean_probs;
}

RolloutResult RolloutStrategy::rollout_position(
    const Board& board,
    RolloutProgressCallback progress) const
{
    // Terminal short-circuit (cubeless analog of the guard in
    // cubeful_rollout_position): an already-decided board has exact terminal
    // probs. Without this the trial loop plays out an already-lost game from the
    // wrong perspective. Probs are SP-relative; no inversion needed.
    GameResult gr_term = check_game_over(board);
    if (gr_term != GameResult::NOT_OVER) {
        RolloutResult tr;
        tr.mean_probs = terminal_probs(gr_term);
        tr.equity = static_cast<double>(cubeless_equity(tr.mean_probs));
        tr.scalar_vr_equity = tr.equity;
        return tr;
    }
    rollout_profile::reset();
    auto result = run_trials_parallel(board, std::move(progress));
    rollout_profile::print();
    return result;
}

int RolloutStrategy::best_move_index(const std::vector<Board>& candidates,
                                      bool pre_move_is_race) const
{
    // Delegate to the Board overload using the first candidate as proxy
    if (candidates.empty()) return 0;
    // No real pre-move board available — use base for filtering
    return base_->best_move_index(candidates, pre_move_is_race);
}

int RolloutStrategy::best_move_index(const std::vector<Board>& candidates,
                                      const Board& pre_move_board) const
{
    const int n = static_cast<int>(candidates.size());
    if (n <= 1) return 0;

    // Clear thread-local N-ply caches between positions.
    // Accumulated cache state from other strategies sharing the same thread
    // pool can cause memory corruption. Always clear — the cost is negligible
    // relative to rollout computation.
    clear_internal_caches();

    // Step 1: Score all candidates at 1-ply for filtering
    std::vector<double> equities(n);
    double best_1ply = -1e30;

    base_->batch_evaluate_candidates_equity(
        candidates, pre_move_board, equities.data());
    for (int i = 0; i < n; ++i) {
        if (equities[i] > best_1ply) best_1ply = equities[i];
    }

    // Step 2: Filter candidates
    std::vector<int> sorted_indices(n);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](int a, int b) { return equities[a] > equities[b]; });

    std::vector<int> survivors;
    survivors.reserve(std::min(n, config_.filter.max_moves));
    for (int idx : sorted_indices) {
        if (static_cast<int>(survivors.size()) >= config_.filter.max_moves) break;
        if (best_1ply - equities[idx] > config_.filter.threshold) break;
        survivors.push_back(idx);
    }

    if (survivors.size() == 1) return survivors[0];

    // Step 3: Rollout each surviving candidate
    double best_rollout = -1e30;
    int best_idx = survivors[0];

    for (int idx : survivors) {
        auto r = rollout_position(candidates[idx]);
        if (r.equity > best_rollout) {
            best_rollout = r.equity;
            best_idx = idx;
        }
    }

    return best_idx;
}

// ========================== Cube-aware best_move_index ==========================
//
// Mirrors the cubeless best_move_index above: cubeless 1-ply filter, then
// per-survivor evaluation — but each evaluation is a cubeful rollout per cube
// state via cubeful_rollout_position (which already incorporates the
// opponent's optimal cube action). For each cube state, pick the survivor
// with the highest cubeful equity.

int RolloutStrategy::best_move_index_cubeful(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    const CubeInfo& ci,
    float cube_x) const
{
    int out = 0;
    best_move_index_cubeful_multi(candidates, pre_move_board, &ci, 1, cube_x, &out);
    return out;
}

void RolloutStrategy::best_move_index_cubeful_multi(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    const CubeInfo* cubes,
    int n_cubes,
    float cube_x,
    int* out_indices) const
{
    const int n = static_cast<int>(candidates.size());
    if (n == 0) return;
    if (n_cubes <= 0) return;
    if (n == 1) {
        for (int c = 0; c < n_cubes; ++c) out_indices[c] = 0;
        return;
    }

    clear_internal_caches();

    // Step 1: 1-ply CUBEFUL filter per cube, unioned across cubes — mirrors
    // the fix in MultiPlyStrategy::best_move_index_cubeful_multi. A pure
    // cubeless filter would drop a candidate whose cubeful equity is best
    // for a particular cube state but whose cubeless equity sits more than
    // `threshold` below the cubeless-best (e.g. a match-defensive move at
    // 1-away with cube=2). Per-cube cubeful filtering guarantees each cube's
    // cubeful-1-ply favorite reaches the per-survivor rollout step.

    std::vector<std::array<float, NUM_OUTPUTS>> probs_per(n);
    for (int i = 0; i < n; ++i) {
        GameResult r = check_game_over(candidates[i]);
        if (r != GameResult::NOT_OVER) {
            probs_per[i] = terminal_probs(r);
        } else {
            probs_per[i] = base_->evaluate_probs(candidates[i], pre_move_board);
        }
        clamp_probs_to_board(probs_per[i], candidates[i]);
    }

    std::vector<bool> in_set(n, false);
    std::vector<int> survivors;
    survivors.reserve(std::min(n, config_.filter.max_moves * n_cubes));
    {
        std::vector<std::pair<float, int>> cube_eqs;
        cube_eqs.reserve(n);
        for (int c = 0; c < n_cubes; ++c) {
            cube_eqs.clear();
            for (int i = 0; i < n; ++i) {
                float cf = cl2cf(probs_per[i], cubes[c], cube_x);
                cube_eqs.emplace_back(cf, i);
            }
            std::sort(cube_eqs.begin(), cube_eqs.end(),
                      [](const std::pair<float, int>& a,
                         const std::pair<float, int>& b) {
                          return a.first > b.first;
                      });
            float best_cf = cube_eqs[0].first;
            int keep = 0;
            for (const auto& p : cube_eqs) {
                if (keep >= config_.filter.max_moves) break;
                if (best_cf - p.first > config_.filter.threshold) break;
                if (!in_set[p.second]) {
                    in_set[p.second] = true;
                    survivors.push_back(p.second);
                }
                ++keep;
            }
        }
    }

    if (survivors.size() == 1) {
        for (int c = 0; c < n_cubes; ++c) out_indices[c] = survivors[0];
        return;
    }

    // Step 2: per surviving candidate, run cubeful_rollout_position once per
    // cube state. (No sharing across cubes — each call has its own dice and
    // trial state. The cubes generally share structure, so the cube_inner_*
    // path handles them efficiently.)
    const int n_surv = static_cast<int>(survivors.size());
    std::vector<std::vector<double>> cf_equities(n_surv, std::vector<double>(n_cubes));
    for (int i = 0; i < n_surv; ++i) {
        for (int c = 0; c < n_cubes; ++c) {
            auto r = cubeful_rollout_position(candidates[survivors[i]], cubes[c]);
            cf_equities[i][c] = r.cubeful_equity;
        }
    }

    for (int c = 0; c < n_cubes; ++c) {
        double best_cf = -1e30;
        int best_surv_i = 0;
        for (int i = 0; i < n_surv; ++i) {
            if (cf_equities[i][c] > best_cf) {
                best_cf = cf_equities[i][c];
                best_surv_i = i;
            }
        }
        out_indices[c] = survivors[best_surv_i];
    }
}

} // namespace bgbot
