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

    // Build truncation evaluation strategy (unchanged logic)
    int effective_trunc_ply = (config.truncation_ply >= 1) ? config.truncation_ply : config.decision_ply;
    if (effective_trunc_ply == checker_ply && !config.checker.is_set()) {
        truncation_strat = checker_strat;
    } else if (effective_trunc_ply > 1) {
        if (filter_base) {
            truncation_strat = std::make_shared<MultiPlyStrategy>(
                base, filter_base, effective_trunc_ply, internal_filter);
        } else {
            truncation_strat = std::make_shared<MultiPlyStrategy>(
                base, effective_trunc_ply, internal_filter);
        }
    } else {
        truncation_strat = base;
    }
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
        return strat.evaluate_probs(board, board);
    }

    if (candidates.size() == 1) {
        GameResult r = check_game_over(candidates[0]);
        if (best_index) *best_index = 0;
        if (r != GameResult::NOT_OVER) {
            return terminal_probs(r);
        }
        return strat.evaluate_probs(candidates[0], board);
    }

    // Fast batch path when using base_ (1-ply) strategy directly.
    if (&strat == base_.get()) {
        std::array<float, NUM_OUTPUTS> best_probs{};
        int idx = base_->batch_evaluate_candidates_best_prob(
            candidates, board, nullptr, &best_probs);
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
    SharedPosCache* shared) const
{
    // Move0 uses checker strategy for move selection.
    // This also warms the thread-local PosCache for truncation evaluation.
    const auto& current_strat = *checker_strat_;
    const bool using_base = (&current_strat == base_.get());

    auto compute_roll = [&](int roll_idx) {
        if (shared) MultiPlyStrategy::set_shared_cache(shared);

        thread_local std::vector<Board> candidates;
        candidates.clear();
        possible_boards(start_board,
                        ALL_ROLLS[roll_idx].d1,
                        ALL_ROLLS[roll_idx].d2,
                        candidates);

        Board chosen;
        if (candidates.empty()) {
            chosen = start_board;
        } else if (candidates.size() == 1) {
            chosen = candidates[0];
        } else if (using_base) {
            chosen = candidates[base_->best_move_index(candidates, start_board)];
        } else {
            chosen = candidates[current_strat.best_move_index(candidates, start_board)];
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
    const Move0Cache& move0_cache, int first_roll_idx, Move1Cache::Entry& entry) const
{
    thread_local std::vector<Board> candidates;

    const Board& move0_chosen = move0_cache.chosen[first_roll_idx];
    const Board move1_board = flip(move0_chosen);
    entry.race = is_race(move1_board);
    entry.cube_x = cube_efficiency(move1_board, entry.race);

    // Move1 uses 1-ply (base_) for move selection. The VR averaging over many
    // trials makes higher-ply move selection unnecessary here.
    const auto& current_strat = *base_;
    const bool using_base = true;

    const Board opp_board = flip(move1_board);
    // mover_probs are always 1-ply (used for 1-ply Janowski cube decisions and
    // as fallback). When cube strategy is N-ply or rollout, the trial code
    // bypasses these cached probs and calls cube_decision_nply or
    // cubeful_cube_decision directly.
    // Use bearoff DB for exact probs when available.
    if (bearoff_db_ && bearoff_db_->is_bearoff(move1_board)) {
        entry.mover_probs = bearoff_db_->lookup_probs(move1_board);
    } else {
        entry.mover_probs = invert_probs(base_->evaluate_probs(opp_board, opp_board));
    }

    for (size_t second_roll = 0; second_roll < ALL_ROLLS.size(); ++second_roll) {
        candidates.clear();
        possible_boards(move1_board,
                        ALL_ROLLS[second_roll].d1,
                        ALL_ROLLS[second_roll].d2,
                        candidates);

        int best_idx = -1;
        entry.roll_best_probs[second_roll] = best_move_probs_for_candidates(
            move1_board, candidates, *base_, &best_idx);
        entry.best_candidate_idx[second_roll] = best_idx;

        Board chosen;
        if (candidates.empty()) {
            chosen = move1_board;
        } else if (candidates.size() == 1) {
            chosen = candidates[0];
        } else if (using_base) {
            if (best_idx >= 0 && best_idx < static_cast<int>(candidates.size())) {
                chosen = candidates[best_idx];
            } else {
                chosen = candidates[base_->best_move_index(candidates, move1_board)];
            }
        } else {
            chosen = candidates[current_strat.best_move_index(candidates, move1_board)];
        }
        entry.chosen[second_roll] = chosen;

        GameResult r = check_game_over(chosen);
        if (r != GameResult::NOT_OVER) {
            entry.actual_probs[second_roll] = terminal_probs(r);
        } else if (using_base) {
            entry.actual_probs[second_roll] = entry.roll_best_probs[second_roll];
        } else if (best_idx >= 0 &&
                   best_idx < static_cast<int>(candidates.size()) &&
                   chosen == candidates[best_idx]) {
            entry.actual_probs[second_roll] = entry.roll_best_probs[second_roll];
        } else {
            entry.actual_probs[second_roll] = base_->evaluate_probs(chosen, move1_board);
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
    SharedPosCache* shared) const
{
    auto populate_entry = [&](int roll_idx) {
        if (shared) MultiPlyStrategy::set_shared_cache(shared);
        populate_move1_cache_entry(move0_cache, roll_idx, cache.entries[roll_idx]);
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

    // Starting convention
    Board board;
    int sp_parity_offset;
    if (start_post_move) {
        board = flip(start_board);
        sp_parity_offset = 1;
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
                    populate_move1_cache_entry(*move0_cache, move0_roll_idx,
                                               move1_cache->entries[move0_roll_idx]);
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
            // Ultra-late positions always use 1-ply Janowski.
            // Otherwise use the configured cube evaluation strategy.
            const int ultra_late_cube = config_.ultra_late_threshold;
            bool use_cube_1ply = move_num >= ultra_late_cube;
            const TrialEvalConfig* cube_cfg = nullptr;
            if (!use_cube_1ply) {
                cube_cfg = is_late ? &cube_late_eval_config_ : &cube_eval_config_;
                use_cube_1ply = (cube_cfg->ply <= 1 && !cube_cfg->is_rollout());
            }

            if (use_cube_1ply) {
                // 1-ply Janowski path (original behavior)
                std::array<float, NUM_OUTPUTS> mover_probs;
                if (move1_entry) {
                    mover_probs = move1_entry->mover_probs;
                    cube_x = move1_entry->cube_x;
                    cube_x_ready = true;
                } else if (bearoff_db_ && bearoff_db_->is_bearoff(board)) {
                    mover_probs = bearoff_db_->lookup_probs(board);
                    cube_x = cube_efficiency(board, race);
                    cube_x_ready = true;
                } else {
                    Board opp_board = flip(board);
                    auto opp_probs = base_->evaluate_probs(opp_board, opp_board);
                    mover_probs = invert_probs(opp_probs);
                    cube_x = cube_efficiency(board, race);
                    cube_x_ready = true;
                }

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
                    cube_rollout = is_late ? cube_late_inner_rollout_.get()
                                           : cube_inner_rollout_.get();
                }

                // Also compute 1-ply cube_x for the cubeful VR mean later
                if (!cube_x_ready) {
                    cube_x = move1_entry ? move1_entry->cube_x
                                         : cube_efficiency(board, race);
                    cube_x_ready = true;
                }

                for (int b = 0; b < n_branches; ++b) {
                    if (branches[b].finished) continue;
                    if (!can_double(branches[b].cube)) continue;

                    CubeDecision cd;
                    if (cube_rollout) {
                        // Cubeful rollout: run inner truncated rollout
                        auto cfr = cube_rollout->cubeful_cube_decision(
                            board, branches[b].cube);
                        // Extract decision from rollout result
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
                            // Convert to equity for consistency (not strictly needed)
                            cd.equity_nd = nd_m;
                            cd.equity_dt = dt_m;
                            cd.equity_dp = dp_m;
                        }
                    } else {
                        // N-ply cubeful recursion
                        const Strategy& cf_base = base_bearoff_ ? *base_bearoff_ : *base_;
                        cd = cube_decision_nply(
                            board, branches[b].cube, cf_base,
                            cube_cfg->ply, cube_filter, 1 /*serial*/,
                            move_filter_.get());
                    }

                    // Same double/take/pass handling as 1-ply path
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
            } // end N-ply / rollout cube path

            // Check if all branches finished (all D/P'd)
            bool all_done = true;
            for (int b = 0; b < n_branches; ++b) {
                if (!branches[b].finished) { all_done = false; break; }
            }
            if (all_done) {
                // All branches D/P'd — use 1-ply pre-roll cubeless probs
                Board opp_board_early = flip(board);
                auto opp_probs_early = base_->evaluate_probs(opp_board_early, opp_board_early);
                auto early_probs = invert_probs(opp_probs_early);
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
        // variance slightly.
        bool skip_vr_this_move = (move_num == 0 && config_.n_trials % 36 == 0)
                               || (move_num >= ultra_late_threshold && (move_num % 2 == 1));
        bool do_vr = vr_enabled && !skip_vr_this_move;

        ROLLOUT_TIMER_START;
        if (move1_entry) {
            // Fully precomputed for move 1.
        } else if (do_vr) {
            for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
                move_candidates[i].clear();
                possible_boards(board, ALL_ROLLS[i].d1, ALL_ROLLS[i].d2, move_candidates[i]);
            }
            // No VR candidate prefiltering — evaluating all candidates at 1-ply
            // is fast and avoids the bias that prefiltering introduces (selecting
            // the best from a filtered subset systematically underestimates the
            // VR mean, biasing luck positive and the VR-corrected result negative).
        } else {
            move_candidates[actual_idx].clear();
            possible_boards(board, d1, d2, move_candidates[actual_idx]);
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
            cube_x = move1_entry ? move1_entry->cube_x : cube_efficiency(board, race);
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
                // Evaluate all 21 rolls at 1-ply (fast batch path via base_)
                for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
                    int idx = -1;
                    roll_best_probs[i] = best_move_probs_for_candidates(
                        board, move_candidates[i], *base_, &idx);
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
        // MOVE-0 CACHE: At move 0, all trials share the same starting position
        // and there are only 21 possible first rolls. The first trial to encounter
        // each dice combo computes the N-ply best move; subsequent trials reuse
        // the cached result via CAS.
        Board chosen;
        bool used_move0_cache = false;
        if (move1_entry) {
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
                        chosen = candidates[current_strat.best_move_index(candidates, board)];
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
                    chosen = candidates[current_strat.best_move_index(candidates, board)];
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
            } else if (using_base) {
                // Decision also used 1-ply — reuse VR's stored probs
                actual_probs = roll_best_probs[actual_idx];
            } else {
                // Decision used N-ply — evaluate chosen at 1-ply for VR
                GameResult r = check_game_over(chosen);
                if (r != GameResult::NOT_OVER) {
                    actual_probs = terminal_probs(r);
                } else {
                    actual_probs = base_->evaluate_probs(chosen, board);
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
    auto last_mover_probs = trunc_strat.evaluate_probs(last_mover_board, last_mover_board);
    ROLLOUT_TIMER_ADD(trunc_time_ns);
#ifdef ROLLOUT_PROFILE
    rollout_profile::trial_count.fetch_add(1, std::memory_order_relaxed);
#endif

    // SP parity at truncation: last mover at trunc_move-1.
    bool last_mover_is_sp = ((trunc_move - 1) % 2 == sp_parity_offset);

    // Cubeful branch truncation (only when cube_active)
    if (cube_active) {
        if (truncation_ply_ > 1) {
            // N-ply cubeful evaluation at truncation point.
            // `board` is the next mover's pre-roll position; branches[b].cube
            // is from the next mover's perspective.
            MoveFilter trunc_filter = {2, 0.03f};
            for (int b = 0; b < n_branches; ++b) {
                if (branches[b].finished) continue;
                const Strategy& cf_strat = base_bearoff_ ? *base_bearoff_ : *base_;
                float cf = cubeful_equity_nply(
                    board, branches[b].cube, cf_strat,
                    truncation_ply_, trunc_filter, 1 /*serial*/,
                    move_filter_.get());
                double sp_val;
                if (is_match) {
                    // cubeful_equity_nply returns equity; convert back to MWC
                    // (branch final_equity uses MWC space for match play).
                    float mwc = eq2mwc(cf,
                        branches[b].cube.match.away1,
                        branches[b].cube.match.away2,
                        branches[b].cube.cube_value,
                        branches[b].cube.match.is_crawford);
                    // mwc is next mover's match winning chance
                    sp_val = last_mover_is_sp ? (1.0 - static_cast<double>(mwc))
                                              : static_cast<double>(mwc);
                } else {
                    // cf is per-cube-unit equity from next mover's perspective
                    double points = cf * branches[b].cube.cube_value;
                    sp_val = points / branches[b].basis_cube;
                    if (last_mover_is_sp) sp_val = -sp_val;
                }
                branches[b].final_equity = sp_val - branches[b].vr_luck;
                branches[b].finished = true;
            }
        } else {
            // 1-ply: Janowski on cubeless probs (original behavior)
            float trunc_x = cube_efficiency(last_mover_board, trunc_race);
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
    } else if (n_branches > 0) {
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

    const int report_interval = progress ? std::max(1, n_trials / 100) : 0;

    if (n_threads == 1) {
        // Serial: prefill + trials on a single thread (PosCache stays warm)
        prefill_move0_cache(pre_roll_board, move0_cache, 1, nullptr);
        if (uses_move1_cache) {
            for (int i = 0; i < Move0Cache::N_ROLLS; ++i) {
                populate_move1_cache_entry(move0_cache, i, move1_cache.entries[i]);
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
            MultiPlyStrategy::set_shared_cache(shared_cache);

            // Phase 1+2: Combined move0 + move1 prefill per roll
            int r;
            while ((r = next_roll.fetch_add(1, std::memory_order_relaxed)) < 21) {
                thread_local std::vector<Board> candidates;
                candidates.clear();
                const auto& roll = ALL_ROLLS[r];
                possible_boards(pre_roll_board, roll.d1, roll.d2, candidates);
                Board chosen;
                if (candidates.empty()) {
                    chosen = pre_roll_board;
                } else if (candidates.size() == 1) {
                    chosen = candidates[0];
                } else {
                    chosen = candidates[m0_strat->best_move_index(
                        candidates, pre_roll_board)];
                }
                move0_cache.chosen[r] = chosen;
                move0_cache.state[r].store(2, std::memory_order_release);

                if (uses_move1_cache) {
                    populate_move1_cache_entry(move0_cache, r, move1_cache.entries[r]);
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

} // namespace bgbot
