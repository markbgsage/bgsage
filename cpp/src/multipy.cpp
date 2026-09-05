// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/multipy.h"
#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/cube.h"       // for cubeful_equity_nply_multi (cubeful BMI)
#include "bgbot/moves.h"
#include "bgbot/encoding.h"
#include <algorithm>
#include <numeric>
#include <limits>
#include <functional>
#include <cmath>
#include <thread>
#include <array>
#include <condition_variable>
#include <deque>
#include <mutex>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace bgbot {

// ======================== Static Data ========================

// 21 unique dice rolls: 6 doubles (weight 1) + 15 non-doubles (weight 2).
// Total weight = 6*1 + 15*2 = 36.
const std::array<MultiPlyStrategy::DiceRoll, 21> MultiPlyStrategy::ALL_ROLLS = {{
    {1,1,1}, {2,2,1}, {3,3,1}, {4,4,1}, {5,5,1}, {6,6,1},
    {1,2,2}, {1,3,2}, {1,4,2}, {1,5,2}, {1,6,2},
    {2,3,2}, {2,4,2}, {2,5,2}, {2,6,2},
    {3,4,2}, {3,5,2}, {3,6,2},
    {4,5,2}, {4,6,2},
    {5,6,2}
}};

namespace {

class MultiPlyThreadPool {
public:
    explicit MultiPlyThreadPool(int n_threads) {
        constexpr int kMinThreads = 1;
        n_threads_ = std::max(kMinThreads, n_threads);
#ifdef _WIN32
        // Use 8 MB stacks on Windows to handle deep N-ply recursion on
        // crashed positions (90+ legal moves, 3+ ply depth).
        handles_.reserve(n_threads_);
        for (int i = 0; i < n_threads_; ++i) {
            HANDLE h = CreateThread(
                nullptr, 8 * 1024 * 1024,
                [](LPVOID param) -> DWORD {
                    static_cast<MultiPlyThreadPool*>(param)->worker_loop();
                    return 0;
                },
                this, 0, nullptr);
            handles_.push_back(h);
        }
#else
        threads_.reserve(n_threads_);
        for (int i = 0; i < n_threads_; ++i) {
            threads_.emplace_back([this]() { worker_loop(); });
        }
#endif
    }

    ~MultiPlyThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
#ifdef _WIN32
        if (!handles_.empty()) {
            WaitForMultipleObjects(
                static_cast<DWORD>(handles_.size()),
                handles_.data(), TRUE, INFINITE);
            for (HANDLE h : handles_) CloseHandle(h);
        }
#else
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
#endif
    }

    template<typename F>
    void parallel_for(int n_items, int n_threads, F&& fn) const {
        if (n_items <= 1 || n_threads <= 1) {
            for (int i = 0; i < n_items; ++i) {
                fn(i);
            }
            return;
        }

        const int workers = std::min(n_threads, n_items);
        const auto task_fn = std::forward<F>(fn);

        struct JobState {
            std::atomic<int> completed_workers{0};
            std::mutex done_mutex;
            std::condition_variable done_cv;
        };

        JobState state;

        // Barrier-exit synchronization must go through done_mutex, NOT the
        // atomic alone: state is on the caller's stack and gets destroyed
        // when this function returns. If the last worker increments the
        // atomic without holding the lock, the caller can observe count ==
        // workers, return, and destroy state while that worker is still
        // about to enter its lock_guard — use-after-free on done_mutex.
        auto run_chunk = [&](int start) {
            for (int item = start; item < n_items; item += workers) {
                task_fn(item);
            }
            std::lock_guard<std::mutex> lock(state.done_mutex);
            if (state.completed_workers.fetch_add(1, std::memory_order_acq_rel) + 1 == workers) {
                state.done_cv.notify_one();
            }
        };

        for (int t = 1; t < workers; ++t) {
            enqueue([&state, run_chunk, t]() mutable { run_chunk(t); });
        }

        run_chunk(0);

        {
            std::unique_lock<std::mutex> lock(state.done_mutex);
            state.done_cv.wait(lock, [&state, workers]() {
                return state.completed_workers.load(std::memory_order_acquire) == workers;
            });
        }
    }

    // Dispatch n_workers copies of fn() — caller runs one, pool runs the rest.
    // Each worker runs the same function; callers use shared atomics for
    // work-stealing coordination.
    void parallel_run(int n_workers, const std::function<void()>& fn) const {
        if (n_workers <= 1) {
            fn();
            return;
        }

        struct JobState {
            std::atomic<int> completed_workers{0};
            std::mutex done_mutex;
            std::condition_variable done_cv;
        };

        JobState state;

        // See barrier-exit comment in parallel_for above.
        auto run_worker = [&]() {
            fn();
            std::lock_guard<std::mutex> lock(state.done_mutex);
            if (state.completed_workers.fetch_add(1, std::memory_order_acq_rel) + 1 == n_workers) {
                state.done_cv.notify_one();
            }
        };

        for (int t = 1; t < n_workers; ++t) {
            enqueue([run_worker]() mutable { run_worker(); });
        }

        run_worker();  // caller is worker 0

        {
            std::unique_lock<std::mutex> lock(state.done_mutex);
            state.done_cv.wait(lock, [&state, n_workers]() {
                return state.completed_workers.load(std::memory_order_acquire) == n_workers;
            });
        }
    }

private:
    struct WorkItem {
        std::function<void()> fn;
    };

    void worker_loop() {
        while (true) {
            WorkItem item;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this]() {
                    return stopping_ || !queue_.empty();
                });
                if (stopping_ && queue_.empty()) {
                    return;
                }
                item = std::move(queue_.front());
                queue_.pop_front();
            }
            item.fn();
        }
    }

    void enqueue(std::function<void()> fn) const {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_.push_back(WorkItem{std::move(fn)});
        }
        queue_cv_.notify_one();
    }

    int n_threads_;
#ifdef _WIN32
    std::vector<HANDLE> handles_;
#else
    mutable std::vector<std::thread> threads_;
#endif
    mutable std::mutex queue_mutex_;
    mutable std::condition_variable queue_cv_;
    mutable std::deque<WorkItem> queue_;
    bool stopping_ = false;
};

MultiPlyThreadPool& get_multipy_executor() {
    static MultiPlyThreadPool executor(
        std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
    return executor;
}

} // namespace

// ======================== Public Thread Pool Access ========================

void multipy_parallel_for(int n_items, int n_threads,
                          const std::function<void(int)>& fn) {
    get_multipy_executor().parallel_for(n_items, n_threads, fn);
}

void multipy_parallel_run(int n_workers, const std::function<void()>& fn) {
    get_multipy_executor().parallel_run(n_workers, fn);
}

// ======================== Static Members ========================

std::atomic<std::size_t> MultiPlyStrategy::PosCache::global_hits{0};
std::atomic<std::size_t> MultiPlyStrategy::PosCache::global_misses{0};
std::atomic<std::uint64_t> MultiPlyStrategy::next_cache_salt_{1};

// ======================== Shared Cache ========================

static thread_local SharedPosCache* tl_shared_cache = nullptr;

void MultiPlyStrategy::set_shared_cache(SharedPosCache* cache) {
    tl_shared_cache = cache;
}

SharedPosCache* MultiPlyStrategy::get_shared_cache() {
    return tl_shared_cache;
}

// ======================== Helpers ========================

MultiPlyStrategy::PosCache& MultiPlyStrategy::get_cache() {
    thread_local PosCache cache;
    return cache;
}

std::size_t MultiPlyStrategy::hash_board(const Board& b) {
    std::size_t h = 0;
    for (int i = 0; i < 26; ++i) {
        h ^= std::hash<int>()(b[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
}

// ======================== Constructor ========================

MultiPlyStrategy::MultiPlyStrategy(std::shared_ptr<Strategy> base_strategy,
                                   int n_plies,
                                   MoveFilter filter,
                                   bool full_depth_opponent,
                                   bool parallel_evaluate,
                                   int parallel_threads)
    : base_(std::move(base_strategy))
    , n_plies_(std::max(1, n_plies))
    , move_filter_(filter)
    , full_depth_opponent_(full_depth_opponent)
    , parallel_evaluate_(parallel_evaluate)
    , parallel_threads_(parallel_threads)
    , cache_salt_(std::hash<const void*>{}(static_cast<const void*>(base_.get())))
{
    if (move_filter_.max_moves < 1) move_filter_.max_moves = 1;
    if (move_filter_.threshold < 0.0f) move_filter_.threshold = 0.0f;
    filter_chain_ = build_filter_chain(move_filter_, n_plies_);
}

MultiPlyStrategy::MultiPlyStrategy(std::shared_ptr<Strategy> base_strategy,
                                   std::shared_ptr<Strategy> filter_strategy,
                                   int n_plies,
                                   MoveFilter filter,
                                   bool full_depth_opponent,
                                   bool parallel_evaluate,
                                   int parallel_threads)
    : base_(std::move(base_strategy))
    , filter_strat_(std::move(filter_strategy))
    , n_plies_(std::max(1, n_plies))
    , move_filter_(filter)
    , full_depth_opponent_(full_depth_opponent)
    , parallel_evaluate_(parallel_evaluate)
    , parallel_threads_(parallel_threads)
    , cache_salt_(std::hash<const void*>{}(static_cast<const void*>(base_.get())))
{
    if (move_filter_.max_moves < 1) move_filter_.max_moves = 1;
    if (move_filter_.threshold < 0.0f) move_filter_.threshold = 0.0f;
    filter_chain_ = build_filter_chain(move_filter_, n_plies_);
}

// ======================== Cache Management ========================

// ======================== Bearoff DB ========================
//
// Setting the bearoff DB wraps base_ (and filter_strat_, when present) in a
// BearoffStrategy so that any leaf evaluation through these strategies —
// including the plies==2 fast path which calls
// base_->batch_evaluate_candidates_best_prob directly and the 1-ply leaf in
// evaluate_probs_nply_impl — short-circuits to the exact DB lookup whenever
// the candidate position is in the bearoff range. Without the wrap, the
// MultiPlyStrategy member bearoff_db_ alone is not sufficient: only
// evaluate_probs_nply_impl's entry-level check would consult it, and the
// plies==2 fast path bypasses that entry check on its leaves.

void MultiPlyStrategy::set_bearoff_db(const BearoffDB* db) {
    bearoff_db_ = db;
    if (db) {
        if (!std::dynamic_pointer_cast<BearoffStrategy>(base_)) {
            base_ = std::make_shared<BearoffStrategy>(base_, db);
        }
        if (filter_strat_ &&
            !std::dynamic_pointer_cast<BearoffStrategy>(filter_strat_)) {
            filter_strat_ = std::make_shared<BearoffStrategy>(filter_strat_, db);
        }
    }
}

void MultiPlyStrategy::clear_cache() const {
    get_cache().clear();
    // The cubeful evaluation engine's per-thread cache backs the delegated
    // cubeless N-ply paths — clear it too so the documented "call
    // clear_cache() between strategy comparisons" idiom keeps working.
    // (Engine keys also carry a process-unique evaluator id, so this is
    // belt-and-braces for the calling thread; pool threads are protected
    // by the id alone.)
    clear_cubeful_eval_cache();
    // Salt is derived from base strategy pointer — no need to bump.
    // Clearing the cache (memset to 0) is sufficient to prevent stale lookups.
    // Also reset global atomic counters
    PosCache::global_hits.store(0, std::memory_order_relaxed);
    PosCache::global_misses.store(0, std::memory_order_relaxed);
}

size_t MultiPlyStrategy::cache_size() const {
    return get_cache().size();
}

size_t MultiPlyStrategy::cache_hits() const {
    return PosCache::global_hits.load(std::memory_order_relaxed);
}

size_t MultiPlyStrategy::cache_misses() const {
    return PosCache::global_misses.load(std::memory_order_relaxed);
}

std::size_t MultiPlyStrategy::cache_key_for(const Board& b, int plies) const {
    std::size_t h = hash_board(b);
    const std::uint64_t cache_salt = cache_salt_.load(std::memory_order_relaxed);
    h ^= (static_cast<std::size_t>(cache_salt) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    h ^= (static_cast<std::size_t>(plies) + 0x9e3779b9 + (h << 6) + (h >> 2));
    h ^= (full_depth_opponent_ ? 0x85ebca6bU : 0x27d4eb2fU);
    return h;
}

int MultiPlyStrategy::parallel_thread_count(int plies) const {
    if (!parallel_evaluate_) return 1;

    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 1;

    int n_threads = parallel_threads_;
    if (n_threads <= 0) {
        n_threads = hw;
    } else if (n_threads < 1) {
        n_threads = 1;
    }
    // Never request more workers than the shared pool has threads: a
    // parallel_for caller blocks waiting for its queued chunks, so
    // oversubscribed nested dispatch (4-ply trees nest a second level of
    // blocking parallel_for) can otherwise fill the pool with waiting
    // chunks and deadlock.
    n_threads = std::min(n_threads, hw);

    if (plies > 1) {
        const int target_cores = 4 * std::max(1, plies - 1);
        n_threads = std::min(n_threads, target_cores);
    }

    return n_threads;
}

// ======================== Core: N-Ply Probability Evaluation ========================
//
// PERSPECTIVE SEMANTICS:
// The NN outputs mean: "probabilities assuming the player just moved to this position,
// and it's about to switch to the other player's turn (before their roll)."
//
// Therefore: evaluate(board) != invert(evaluate(flip(board)))
// These differ by one tempo — who gets to move next.
//
// In evaluate_probs_nply(board, pre_move_board, plies):
//   - `board` is a post-move position from the "current player's" perspective
//   - Returns probabilities for the current player (the one who just moved to `board`)
//   - When recurring after opponent moves, we call evaluate_probs_nply on the
//     opponent's post-move board (from opponent's perspective) and INVERT the result
//   - We do NOT flip the board back — that would change the semantic meaning
//
// GAME PLAN CLASSIFICATION:
//   - The hybrid recursion (evaluate_probs_nply_impl) selects the NN by the
//     PRE-MOVE board — the `pre_move_board` parameter throughout.
//   - The cubeful evaluation engine's candidate kernel (the non-hybrid path)
//     classifies each candidate by its own board instead; on plan-boundary
//     candidates the two conventions pick different NNs (the per-candidate
//     convention scores better on the benchmark ER). `pre_move_board` still
//     drives the 1-ply base case.

// Cubeless N-ply evaluation through the cubeful evaluation engine
// (cube_eval.cpp): a single dead cube (cube_value=1, max_cube_value=1)
// makes cl2cf bypass Janowski everywhere and interior picks reduce to
// cubeless 1-ply equity, so the tree is exactly a cubeless N-ply
// evaluation of flip(board); its accumulated probs, inverted, are the
// post-move probs of `board`'s mover. Mirrors the dead-cube paths the
// rollout trial loop uses for cubeless move selection and truncation.
std::array<float, NUM_OUTPUTS> MultiPlyStrategy::cubeless_tree_probs(
    const Board& board, const Board& pre_move_board,
    int plies, int n_threads, bool deep_prefilter) const
{
    // Exact-DB short-circuit and 1-ply base case mirror the recursion's.
    if (bearoff_db_ && bearoff_db_->is_bearoff(board)) {
        auto probs = bearoff_db_->lookup_probs(board, /*post_move=*/true);
        clamp_probs_to_board(probs, board);
        return probs;
    }
    if (plies <= 1) {
        auto probs = base_->evaluate_probs(board, pre_move_board);
        clamp_probs_to_board(probs, board);
        return probs;
    }
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_probs(result);
    }

    static const CubeInfo dead = [] {
        CubeInfo c;
        c.cube_value = 1;
        c.max_cube_value = 1;
        return c;
    }();
    float cf_unused;
    std::array<float, NUM_OUTPUTS> tree_probs{};
    cubeful_equity_nply_multi(
        flip(board), &dead, 1, *base_, plies, &cf_unused,
        MoveFilters::TINY, n_threads,
        move_prefilter_ ? move_prefilter_.get() : nullptr,
        /*fTop=*/false, &tree_probs, deep_prefilter, &pre_move_board);
    auto probs = invert_probs(tree_probs);
    clamp_probs_to_board(probs, board);
    return probs;
}

std::array<float, NUM_OUTPUTS> MultiPlyStrategy::evaluate_probs_nply(
    const Board& board, const Board& pre_move_board, int plies) const
{
    if (!filter_strat_ && !full_depth_opponent_) {
        // Standalone N-ply evaluation: conservative (no deep prune), like
        // the standalone cubeful entry points. BMI rescoring below uses the
        // deep prune (its band-level shifts were validated on the checker
        // benchmark and the benchmark ER).
        int n_threads = (parallel_evaluate_ && plies == n_plies_ && plies > 2)
            ? parallel_thread_count(plies) : 1;
        return cubeless_tree_probs(board, pre_move_board, plies, n_threads,
                                   /*deep_prefilter=*/false);
    }
    // Hybrid evaluator (separate filter strategy for opponent picks) and
    // the full-depth-opponent reference mode: the engine has no hook for
    // either, so keep the recursion.
    return evaluate_probs_nply_impl(
        board, pre_move_board, plies,
        parallel_evaluate_ && plies == n_plies_ && plies > 2);
}

std::array<float, NUM_OUTPUTS> MultiPlyStrategy::evaluate_probs_nply_impl(
    const Board& board, const Board& pre_move_board, int plies,
    bool allow_parallel) const
{
    // Bearoff DB short-circuit at any depth: positions in the DB get exact
    // cubeless probs without recursion. Mirrors the BearoffStrategy wrapping
    // used in the cube_action path, but works whether or not base_ has been
    // wrapped — the caller only sets the DB on the MultiPlyStrategy itself.
    if (bearoff_db_ && bearoff_db_->is_bearoff(board)) {
        auto probs = bearoff_db_->lookup_probs(board, /*post_move=*/true);
        clamp_probs_to_board(probs, board);
        return probs;
    }

    // Base case: 1-ply -> delegate to base strategy with proper pre-move context
    if (plies <= 1) {
        auto probs = base_->evaluate_probs(board, pre_move_board);
        clamp_probs_to_board(probs, board);
        return probs;
    }

    // Terminal position check
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_probs(result);
    }

    // Cache lookup
    std::size_t bh = cache_enabled_ ? cache_key_for(board, plies) : 0;
    auto& cache = get_cache();
    if (cache_enabled_) {
        const auto* cached = cache.lookup(bh, /*plies folded into key*/ 0, board);
        if (cached) {
            return *cached;
        }
    } else {
        // Still count misses for profiling even when cache is disabled
        PosCache::global_misses.fetch_add(1, std::memory_order_relaxed);
    }

    SharedPosCache::Entry* shared_reservation = nullptr;

    // Check shared cross-thread cache (active during parallel rollouts)
    if (cache_enabled_ && tl_shared_cache) {
        auto shared_result = tl_shared_cache->lookup_or_reserve(bh, 0, board);
        if (shared_result.probs) {
            cache.insert(bh, 0, board, *shared_result.probs);  // promote to thread-local
            return *shared_result.probs;
        }
        shared_reservation = shared_result.reservation;
    }

    // Flip to opponent's perspective. The opponent is now "player 1" on this board.
    // opp_board is the opponent's PRE-MOVE board (before they roll/move).
    Board opp_board = flip(board);

    // Accumulate weighted probabilities from the CURRENT PLAYER's perspective
    // (the one who moved to `board`). After the opponent responds, we get the
    // opponent's probs and invert them back to the current player's perspective.
    std::array<double, NUM_OUTPUTS> sum_probs = {0, 0, 0, 0, 0};

    // PubEval pre-filter: configurable via set_prefilter_params().
    const int PREFILTER_THRESHOLD = prefilter_threshold_;
    const int PREFILTER_KEEP = prefilter_keep_;

    auto evaluate_roll = [&](const DiceRoll& roll) -> std::array<float, NUM_OUTPUTS> {
        thread_local std::vector<Board> opp_candidates;
        thread_local std::vector<Board> filtered_candidates;
        thread_local std::vector<double> opp_equities;

        opp_candidates.clear();
        if (opp_candidates.capacity() < 32) opp_candidates.reserve(32);

        // Generate opponent's legal moves
        possible_boards(opp_board, roll.d1, roll.d2, opp_candidates);

        // Pre-filter with cheap evaluator (PubEval) if available, otherwise skip.
        // Replaces old pip-based heuristic with same threshold.
        if (move_prefilter_ && static_cast<int>(opp_candidates.size()) > PREFILTER_THRESHOLD) {
            const int n_cand = static_cast<int>(opp_candidates.size());
            thread_local std::vector<std::pair<double, int>> filter_scores;
            filter_scores.clear();
            filter_scores.reserve(n_cand);
            bool opp_race = is_race(opp_board);
            for (int ci = 0; ci < n_cand; ++ci) {
                GameResult gr = check_game_over(opp_candidates[ci]);
                double eq = (gr != GameResult::NOT_OVER)
                    ? 1e30  // terminals always survive
                    : move_prefilter_->evaluate(opp_candidates[ci], opp_race);
                filter_scores.push_back({-eq, ci});
            }
            int keep = std::min(PREFILTER_KEEP, n_cand);
            std::partial_sort(filter_scores.begin(), filter_scores.begin() + keep,
                              filter_scores.end());
            filtered_candidates.clear();
            filtered_candidates.reserve(keep);
            for (int k = 0; k < keep; ++k) {
                filtered_candidates.push_back(opp_candidates[filter_scores[k].second]);
            }
            opp_candidates.swap(filtered_candidates);
        }

        // Find opponent's best move and get the current player's probabilities
        std::array<float, NUM_OUTPUTS> p1_probs;

        if (opp_candidates.size() == 1) {
            // Only one legal move (or no moves — forced pass)
            Board opp_best = opp_candidates[0];
            GameResult opp_result = check_game_over(opp_best);
            if (opp_result != GameResult::NOT_OVER) {
                // Terminal from opponent's perspective -> invert for current player
                p1_probs = invert_probs(terminal_probs(opp_result));
            } else {
                // Evaluate from opponent's perspective at (plies-1), invert for current player.
                // opp_best is opponent's post-move board, opp_board is their pre-move board.
                auto opp_probs = evaluate_probs_nply_impl(opp_best, opp_board, plies - 1, false);
                p1_probs = invert_probs(opp_probs);
            }
        } else if (full_depth_opponent_ && plies > 2) {
            // Full-depth mode: opponent evaluates all candidates at (plies-1) depth.
            // Opponent picks the move that maximizes THEIR equity (minimizes current player's).
            const int n_opp = static_cast<int>(opp_candidates.size());
            int full_depth_threads = std::max(1, parallel_thread_count(plies - 1));
            std::vector<std::array<float, NUM_OUTPUTS>> opp_full_depth_probs(
                n_opp);
            std::vector<double> opp_full_depth_eq(n_opp);

            if (n_opp > 1 && full_depth_threads > 1) {
                full_depth_threads = std::min(full_depth_threads, n_opp);
                get_multipy_executor().parallel_for(
                    n_opp, full_depth_threads,
                    [&](int i) {
                        GameResult opp_result = check_game_over(opp_candidates[i]);
                        std::array<float, NUM_OUTPUTS> candidate_p1_probs;

                        if (opp_result != GameResult::NOT_OVER) {
                            candidate_p1_probs = invert_probs(terminal_probs(opp_result));
                        } else {
                            auto opp_probs = evaluate_probs_nply_impl(
                                opp_candidates[i], opp_board, plies - 1, false);
                            candidate_p1_probs = invert_probs(opp_probs);
                        }
                        opp_full_depth_probs[i] = candidate_p1_probs;
                        opp_full_depth_eq[i] = compute_equity(candidate_p1_probs);
                    });
            } else {
                for (int i = 0; i < n_opp; ++i) {
                    GameResult opp_result = check_game_over(opp_candidates[i]);
                    std::array<float, NUM_OUTPUTS> candidate_p1_probs;

                    if (opp_result != GameResult::NOT_OVER) {
                        candidate_p1_probs = invert_probs(terminal_probs(opp_result));
                    } else {
                        auto opp_probs = evaluate_probs_nply_impl(
                            opp_candidates[i], opp_board, plies - 1, false);
                        candidate_p1_probs = invert_probs(opp_probs);
                    }
                    opp_full_depth_probs[i] = candidate_p1_probs;
                    opp_full_depth_eq[i] = compute_equity(candidate_p1_probs);
                }
            }

            double worst_p1_eq = 1e30;
            std::array<float, NUM_OUTPUTS> best_p1_probs_for_opp{};
            for (int i = 0; i < n_opp; ++i) {
                if (opp_full_depth_eq[i] < worst_p1_eq) {
                    worst_p1_eq = opp_full_depth_eq[i];
                    best_p1_probs_for_opp = opp_full_depth_probs[i];
                }
            }

            p1_probs = best_p1_probs_for_opp;
        } else {
            // Fast mode (default): opponent picks best move at 1-ply (raw NN).
            // In hybrid mode, filter_strat_ does the 1-ply scoring (fast model)
            // while base_ is used for leaf evaluation (accurate model).
            int best_opp_idx;
            Strategy* filter_s = filter_strat_ ? filter_strat_.get() : base_.get();

            // Optimization: when plies==2 and NOT in hybrid mode, the recursion
            // would evaluate the best move at 1-ply (re-encoding + re-evaluating).
            // Instead, use batch_evaluate_candidates_best_prob to get equity + probs
            // in one pass. In hybrid mode, we can't reuse filter probs as leaf probs.
            if (plies == 2 && !filter_strat_) {
                std::array<float, NUM_OUTPUTS> best_probs{};
                best_opp_idx = base_->batch_evaluate_candidates_best_prob(
                    opp_candidates, opp_board, nullptr, &best_probs);
                // At plies==2, evaluate_probs_nply(best, board, 1) would just
                // return base_->evaluate_probs(best, board). We already have those
                // probs from the batch evaluation. Use them directly.
                // Clamp against the opponent's chosen post-move board so the
                // sanity invariants are enforced before inversion.
                if (best_opp_idx >= 0 && best_opp_idx < static_cast<int>(opp_candidates.size())) {
                    clamp_probs_to_board(best_probs, opp_candidates[best_opp_idx]);
                }
                p1_probs = invert_probs(best_probs);
            } else {
                // Batch evaluation with filter strategy (or base_ if no filter)
                opp_equities.resize(opp_candidates.size());
                best_opp_idx = filter_s->batch_evaluate_candidates_equity(
                    opp_candidates, opp_board, opp_equities.data());

                Board opp_best = opp_candidates[best_opp_idx];
                GameResult opp_result = check_game_over(opp_best);
                if (opp_result != GameResult::NOT_OVER) {
                    p1_probs = invert_probs(terminal_probs(opp_result));
                } else {
                    auto opp_probs = evaluate_probs_nply_impl(opp_best, opp_board, plies - 1, false);
                    p1_probs = invert_probs(opp_probs);
                }
            }
        }

        return p1_probs;
    };

    if (!allow_parallel) {
        for (const auto& roll : ALL_ROLLS) {
            auto p1_probs = evaluate_roll(roll);
            for (int k = 0; k < NUM_OUTPUTS; ++k) {
                sum_probs[k] += roll.weight * p1_probs[k];
            }
        }
    } else {
        std::array<std::array<float, NUM_OUTPUTS>, ALL_ROLLS.size()> roll_results{};
        int n_threads = std::min(parallel_thread_count(plies),
                                 static_cast<int>(roll_results.size()));

        get_multipy_executor().parallel_for(
            static_cast<int>(roll_results.size()), n_threads,
            [&](int idx) {
                roll_results[idx] = evaluate_roll(ALL_ROLLS[idx]);
            });

        for (int i = 0; i < static_cast<int>(roll_results.size()); ++i) {
            for (int k = 0; k < NUM_OUTPUTS; ++k) {
                sum_probs[k] += ALL_ROLLS[i].weight * roll_results[i][k];
            }
        }
    }

    // Average over 36 total weight
    std::array<float, NUM_OUTPUTS> avg;
    for (int k = 0; k < NUM_OUTPUTS; ++k) {
        avg[k] = static_cast<float>(sum_probs[k] / 36.0);
    }

    // Sanity-clamp the averaged result against `board`. Each summand is
    // already clamped (recursive calls clamp their returns, and the leaf
    // path clamps too), but float accumulation can leak tiny non-zero
    // values into impossible-outcome slots. This guarantees the returned
    // probs satisfy the board-state invariants exactly.
    clamp_probs_to_board(avg, board);

    // Store in cache (auto-clears at 75% load)
    if (cache_enabled_) {
        cache.insert(bh, /*plies folded into key*/ 0, board, avg);
    }

    // Also insert into shared cross-thread cache if active
    if (cache_enabled_ && tl_shared_cache) {
        if (shared_reservation) {
            tl_shared_cache->publish(shared_reservation, avg);
        } else {
            tl_shared_cache->insert(bh, 0, board, avg);
        }
    }

    return avg;
}


// ======================== Public Interface ========================

double MultiPlyStrategy::evaluate(const Board& board, bool pre_move_is_race) const {
    if (n_plies_ == 1) {
        return base_->evaluate(board, pre_move_is_race);
    }
    // For the bool overload, we don't have a real pre-move board.
    // Use the post-move board itself as a reasonable proxy for game plan classification.
    auto probs = evaluate_probs_nply(board, board, n_plies_);
    return compute_equity(probs);
}

std::array<float, NUM_OUTPUTS> MultiPlyStrategy::evaluate_probs(
    const Board& board, bool pre_move_is_race) const
{
    if (n_plies_ == 1) {
        return base_->evaluate_probs(board, pre_move_is_race);
    }
    // Same proxy as evaluate(): use board itself when no pre-move board available.
    return evaluate_probs_nply(board, board, n_plies_);
}

std::array<float, NUM_OUTPUTS> MultiPlyStrategy::evaluate_probs(
    const Board& board, const Board& pre_move_board) const
{
    if (n_plies_ == 1) {
        return base_->evaluate_probs(board, pre_move_board);
    }
    return evaluate_probs_nply(board, pre_move_board, n_plies_);
}

// ======================== best_move_index with Filtering ========================

int MultiPlyStrategy::best_move_index_impl(
    const std::vector<Board>& candidates, const Board& pre_move_board) const
{
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;

    // At 1-ply, just delegate to base strategy with proper pre-move board
    if (n_plies_ == 1) {
        return base_->best_move_index(candidates, pre_move_board);
    }

    // Iterative deepening: walk through filter_chain_ steps, progressively
    // narrowing candidates at increasing ply depths before the final evaluation.
    // Each step scores all current survivors at the step's ply level, then
    // filters to the top moves within threshold.

    // Start with all candidate indices
    std::vector<int> survivors(n);
    std::iota(survivors.begin(), survivors.end(), 0);

    std::vector<double> equities(n);

    for (const auto& step : filter_chain_) {
        if (static_cast<int>(survivors.size()) <= 1) break;

        // Score survivors at this step's ply level
        if (step.ply == 1) {
            // Batch 1-ply evaluation via virtual dispatch
            Strategy* filter_s = filter_strat_ ? filter_strat_.get() : base_.get();
            filter_s->batch_evaluate_candidates_equity(
                candidates, pre_move_board, equities.data());
        } else {
            // N-ply evaluation for each survivor
            for (int idx : survivors) {
                auto probs = (filter_strat_ || full_depth_opponent_)
                    ? evaluate_probs_nply_impl(
                          candidates[idx], pre_move_board, step.ply, false)
                    : cubeless_tree_probs(
                          candidates[idx], pre_move_board, step.ply, 1,
                          /*deep_prefilter=*/true);
                equities[idx] = compute_equity(probs);
            }
        }

        // Sort survivors by equity (descending)
        std::sort(survivors.begin(), survivors.end(),
                  [&](int a, int b) { return equities[a] > equities[b]; });

        // Filter: keep top max_moves within threshold of best
        double best_eq = equities[survivors[0]];
        int keep = 0;
        for (int idx : survivors) {
            if (keep >= step.max_moves) break;
            if (best_eq - equities[idx] > step.threshold) break;
            ++keep;
        }
        if (keep < 1) keep = 1;
        survivors.resize(keep);
    }

    // If only 1 survivor after filtering, return it
    if (survivors.size() == 1) return survivors[0];

    // Final evaluation: score all remaining survivors at full N-ply depth.
    double best_nply = -1e30;
    int best_idx = survivors[0];

    bool use_parallel = parallel_evaluate_ && n_plies_ > 2;
    if (use_parallel) {
        int n_threads = std::min<int>(parallel_thread_count(n_plies_),
                                      static_cast<int>(survivors.size()));
        if (n_threads > 1) {
            std::vector<double> survivor_eq(survivors.size(), -1e30);
            get_multipy_executor().parallel_for(
                static_cast<int>(survivors.size()), n_threads,
                [&](int local_i) {
                    int idx = survivors[local_i];
                    auto probs = (filter_strat_ || full_depth_opponent_)
                        ? evaluate_probs_nply_impl(
                              candidates[idx], pre_move_board, n_plies_, false)
                        : cubeless_tree_probs(
                              candidates[idx], pre_move_board, n_plies_, 1,
                              /*deep_prefilter=*/true);
                    survivor_eq[local_i] = compute_equity(probs);
                });

            for (int i = 0; i < static_cast<int>(survivors.size()); ++i) {
                if (survivor_eq[i] > best_nply) {
                    best_nply = survivor_eq[i];
                    best_idx = survivors[i];
                }
            }

            return best_idx;
        }
    }

    for (int idx : survivors) {
        auto probs = (filter_strat_ || full_depth_opponent_)
            ? evaluate_probs_nply_impl(candidates[idx], pre_move_board,
                                       n_plies_, false)
            : cubeless_tree_probs(candidates[idx], pre_move_board,
                                  n_plies_, 1, /*deep_prefilter=*/true);
        double eq = compute_equity(probs);
        if (eq > best_nply) {
            best_nply = eq;
            best_idx = idx;
        }
    }

    return best_idx;
}

// ========================== Cube-aware best_move_index ==========================
//
// Picks the best candidate(s) by CUBEFUL equity at full N-ply depth, against
// one or more cube states. Strategy:
//   1. 1-ply cubeful filter per cube, UNIONED across cubes — each cube's
//      cubeful-1-ply favorites reach the N-ply rescore. (See below for the
//      correctness motivation; the old cubeless filter dropped match-defensive
//      moves before the cubeful tree could see them.)
//   2. Subsequent ply > 1 filter steps stay cubeless for now (only fire at
//      4-ply target).
//   3. Final N-ply cubeful evaluation per survivor via cubeful_equity_nply_multi
//      (one call per survivor; shares its NN tree across all cube states).
//      For each cube state, return the survivor with the highest cubeful equity.
//
// cube_x is used at the 1-ply filter (passed to cl2cf). At N-ply leaves,
// cubeful_equity_nply_multi computes its own per-leaf cube efficiency.

void MultiPlyStrategy::best_move_index_cubeful_multi(
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

    // 1-ply shortcut: delegate to base strategy's cubeful selection (it knows
    // how to batch evaluate + apply cl2cf per cube).
    if (n_plies_ == 1) {
        base_->best_move_index_cubeful_multi(candidates, pre_move_board,
                                              cubes, n_cubes, cube_x, out_indices);
        return;
    }

    // ----- 1-ply cubeful filter (per cube, unioned across cubes) -----
    //
    // Why per-cube: a pure cubeless 1-ply filter drops candidates whose
    // cubeless equity sits more than `threshold` below the cubeless-best —
    // but in extreme match scores (e.g. 1-away with cube=2) the CUBEFUL
    // winner can be a defensive move whose cubeless equity is well below
    // the cubeless-best. Same applies when ND and DT branches have different
    // cube ownership/value: their cubeful preferences can diverge. Without
    // per-cube filtering the cubeful N-ply rescore never sees the defensive
    // move and picks the same aggressive move under all cubes — a real
    // correctness bug at extreme match scores.
    //
    // We compute 1-ply probs once per candidate, then for each cube state
    // apply cl2cf and union the top max_moves within threshold into the
    // survivor set. Each cube's cubeful-1-ply best is guaranteed in the
    // set that reaches the N-ply rescore.
    //
    Strategy* filter_s = filter_strat_ ? filter_strat_.get() : base_.get();

    // Batched scoring (delta evaluation); classification follows the
    // pre-move board. Terminal candidates get terminal_probs inside the
    // batch call.
    std::vector<std::array<float, NUM_OUTPUTS>> probs_per(n);
    thread_local std::vector<double> eq_tmp;
    eq_tmp.resize(n);
    filter_s->batch_evaluate_candidates_equity_probs(
        candidates, pre_move_board, eq_tmp.data(), probs_per.data());
    for (int i = 0; i < n; ++i) {
        clamp_probs_to_board(probs_per[i], candidates[i]);
    }

    // First filter step from the chain (always ply=1 in the current default
    // chains). Use its max_moves and threshold for the cubeful filter.
    const MoveFilterStep& first_step = filter_chain_[0];
    std::vector<bool> in_set(n, false);
    std::vector<int> survivors;
    survivors.reserve(std::min(n, first_step.max_moves * n_cubes));

    {
        // Per-cube top-K-within-threshold; union into in_set/survivors.
        std::vector<std::pair<float, int>> cube_eqs;
        cube_eqs.reserve(n);
        for (int c = 0; c < n_cubes; ++c) {
            cube_eqs.clear();
            for (int i = 0; i < n; ++i) {
                float cf = cl2cf(probs_per[i], cubes[c], cube_x);
                cube_eqs.emplace_back(cf, i);
            }
            // Ties broken on money cubeless equity. This decides WHICH
            // candidates survive to the cubeful rescore, and a move dropped
            // here can never be recovered -- at a score where every cubeful
            // key is bit-identical, survival was otherwise decided by
            // introsort partitioning order.
            //
            // The tie test is exact equality, NOT MOVE_TIE_EPSILON: an
            // epsilon comparator is not a strict weak ordering (a~b, b~c,
            // a!~c), which is undefined behaviour in std::sort. Exact
            // equality gives a plain lexicographic order and covers the
            // bit-identical case that actually occurs.
            std::sort(cube_eqs.begin(), cube_eqs.end(),
                      [&](const std::pair<float, int>& a,
                          const std::pair<float, int>& b) {
                          if (a.first != b.first) return a.first > b.first;
                          return cubeless_equity(probs_per[a.second])
                               > cubeless_equity(probs_per[b.second]);
                      });
            float best_cf = cube_eqs[0].first;
            int keep = 0;
            for (const auto& p : cube_eqs) {
                if (keep >= first_step.max_moves) break;
                if (best_cf - p.first > first_step.threshold) break;
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

    // ----- Subsequent filter steps (currently cubeless; only fires at 4-ply) -----
    //
    // TODO: For full correctness at 4-ply in extreme match positions, this
    // step should also become per-cube cubeful (call cubeful_equity_nply_multi
    // at step.ply for each survivor across all cubes, then union top-K per
    // cube). For now it stays cubeless — the 1-ply cubeful union above already
    // ensures each cube's cubeful-1-ply favorite reaches this step.

    std::vector<double> equities(n);
    for (size_t step_idx = 1; step_idx < filter_chain_.size(); ++step_idx) {
        if (static_cast<int>(survivors.size()) <= 1) break;
        const auto& step = filter_chain_[step_idx];

        for (int idx : survivors) {
            auto probs = (filter_strat_ || full_depth_opponent_)
                ? evaluate_probs_nply_impl(
                      candidates[idx], pre_move_board, step.ply, false)
                : cubeless_tree_probs(
                      candidates[idx], pre_move_board, step.ply, 1,
                      /*deep_prefilter=*/true);
            equities[idx] = compute_equity(probs);
        }
        std::sort(survivors.begin(), survivors.end(),
                  [&](int a, int b) { return equities[a] > equities[b]; });

        double best_eq = equities[survivors[0]];
        int keep = 0;
        for (int idx : survivors) {
            if (keep >= step.max_moves) break;
            if (best_eq - equities[idx] > step.threshold) break;
            ++keep;
        }
        if (keep < 1) keep = 1;
        survivors.resize(keep);
    }

    if (survivors.size() == 1) {
        for (int c = 0; c < n_cubes; ++c) out_indices[c] = survivors[0];
        return;
    }

    // ----- Final cubeful evaluation per survivor -----
    //
    // cubeful_equity_nply_multi(board, cubes, ...) expects `board` to be a
    // PRE-ROLL position from the perspective of the player about to roll, and
    // returns equity from that player's POV. For our candidates (mover's
    // post-move boards in mover's POV), the player about to roll is the
    // OPPONENT — so we must flip both the board and the cubes to the
    // opponent's perspective before the call. The returned equity is then in
    // the opponent's POV; argmin picks the worst-for-opp = best-for-mover.
    //
    // (Matches the convention in BgBotAnalyzer._cubeful_equity, which flips
    // the post-move board and negates the result.)
    const int n_surv = static_cast<int>(survivors.size());
    // cf_equities[i][c] = cubeful equity (in OPPONENT's POV) of survivor i under cube c
    std::vector<std::vector<float>> cf_equities(n_surv, std::vector<float>(n_cubes));

    // Pre-flip cube states once: owner toggles, away scores swap (match only),
    // cube_value/jacoby/beaver/etc. unchanged.
    std::vector<CubeInfo> opp_cubes(n_cubes);
    for (int c = 0; c < n_cubes; ++c) {
        opp_cubes[c] = cubes[c];
        opp_cubes[c].owner = flip_owner(opp_cubes[c].owner);
        if (!opp_cubes[c].is_money()) {
            std::swap(opp_cubes[c].match.away1, opp_cubes[c].match.away2);
        }
    }

    auto eval_one = [&](int i) {
        Board opp_board = flip(candidates[survivors[i]]);
        // Serial inside (trial-level parallelism is across trials). The deep
        // pre-filter is appropriate here: cubeful BMI rescoring runs inside
        // rollout trials, where per-call shifts average out over many trials.
        cubeful_equity_nply_multi(
            opp_board, opp_cubes.data(), n_cubes,
            *base_, n_plies_,
            cf_equities[i].data(),
            MoveFilters::TINY, 1,
            move_prefilter_ ? move_prefilter_.get() : nullptr,
            /*fTop=*/false, nullptr, /*deep_prefilter=*/true, &pre_move_board);
    };

    bool use_parallel = parallel_evaluate_ && n_plies_ > 2;
    if (use_parallel) {
        int n_threads = std::min<int>(parallel_thread_count(n_plies_), n_surv);
        if (n_threads > 1) {
            multipy_parallel_for(n_surv, n_threads, eval_one);
        } else {
            for (int i = 0; i < n_surv; ++i) eval_one(i);
        }
    } else {
        for (int i = 0; i < n_surv; ++i) eval_one(i);
    }

    // For each cube state, pick the survivor with the LOWEST cubeful equity.
    //
    // cf_equities was computed by flipping the candidate to the opponent's
    // perspective (since the opp is the next on-roll after our mover's move),
    // so the values are in OPPONENT's POV. The MOVER wants to minimize the
    // opponent's resulting equity → argmin picks the best move for the mover.
    for (int c = 0; c < n_cubes; ++c) {
        float best_cf = std::numeric_limits<float>::infinity();
        int best_surv_i = 0;
        for (int i = 0; i < n_surv; ++i) {
            if (cf_equities[i][c] < best_cf) {
                best_cf = cf_equities[i][c];
                best_surv_i = i;
            }
        }

        // Tie-break, PAID ONLY WHEN A TIE EXISTS. This is the pick the trial
        // actually plays at decision_ply > 1, so a flat score would otherwise
        // send every tie to survivors[0] and undo the tie-break applied deeper
        // in the search.
        //
        // The probs are re-derived at 1-ply here rather than threaded out of
        // cubeful_equity_nply_multi: asking that call for probs makes every
        // survivor miss any cache entry stored without them (see the
        // out_probs note on CubefulEvalCache::get), which measured ~29% slower
        // on the checker benchmark. Ties are rare, so this pays nothing in the
        // common case.
        //
        // Ranked on the MOVER's own post-move probs and MAXIMISED -- unlike
        // cf_equities above, which is opponent-POV and hence argmin'd.
        int n_tied = 0;
        for (int i = 0; i < n_surv; ++i) {
            if (cf_equities[i][c] <= best_cf + MOVE_TIE_EPSILON) ++n_tied;
        }
        if (n_tied > 1) {
            float best_cl = -1e30f;
            for (int i = 0; i < n_surv; ++i) {
                if (cf_equities[i][c] > best_cf + MOVE_TIE_EPSILON) continue;
                const Board& cand = candidates[survivors[i]];
                std::array<float, NUM_OUTPUTS> p;
                GameResult gr = check_game_over(cand);
                if (gr != GameResult::NOT_OVER) {
                    p = terminal_probs(gr);
                } else {
                    p = base_->evaluate_probs(cand, pre_move_board);
                }
                clamp_probs_to_board(p, cand);
                const float cl = cubeless_equity(p);
                if (cl > best_cl) { best_cl = cl; best_surv_i = i; }
            }
        }
        out_indices[c] = survivors[best_surv_i];
    }
}

int MultiPlyStrategy::best_move_index_cubeful(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    const CubeInfo& ci,
    float cube_x) const
{
    int out = 0;
    best_move_index_cubeful_multi(candidates, pre_move_board, &ci, 1, cube_x, &out);
    return out;
}

int MultiPlyStrategy::best_move_index(const std::vector<Board>& candidates,
                                       bool pre_move_is_race) const {
    const int n = static_cast<int>(candidates.size());
    if (n <= 1) return 0;

    if (n_plies_ == 1) {
        return base_->best_move_index(candidates, pre_move_is_race);
    }

    // Bool overload fallback: use the same filter + rescore pattern as the
    // board overload, but with bool-context 1-ply scoring and post-move board
    // proxy for N-ply pre-move context.
    // In hybrid mode, use filter strategy for 1-ply scoring.
    Strategy* filter_s = filter_strat_ ? filter_strat_.get() : base_.get();
    std::vector<double> equities(n);
    std::vector<double> ranked_equities(n);
    double best_1ply = -1e30;
    for (int i = 0; i < n; ++i) {
        equities[i] = filter_s->evaluate(candidates[i], pre_move_is_race);
        ranked_equities[i] = equities[i];
        if (ranked_equities[i] > best_1ply) best_1ply = ranked_equities[i];
    }

    std::vector<int> sorted_indices(n);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](int a, int b) { return ranked_equities[a] > ranked_equities[b]; });

    std::vector<int> survivors;
    survivors.reserve(std::min(n, move_filter_.max_moves));
    for (int idx : sorted_indices) {
        if (static_cast<int>(survivors.size()) >= move_filter_.max_moves) break;
        if (best_1ply - ranked_equities[idx] > move_filter_.threshold) break;
        survivors.push_back(idx);
    }
    if (survivors.empty()) return sorted_indices[0];
    if (survivors.size() == 1) return survivors[0];

    double best_nply = -1e30;
    int best_idx = survivors[0];
    for (int idx : survivors) {
        auto probs = evaluate_probs_nply(candidates[idx], candidates[idx], n_plies_);
        double eq = compute_equity(probs);
        if (eq > best_nply) {
            best_nply = eq;
            best_idx = idx;
        }
    }
    return best_idx;
}

int MultiPlyStrategy::best_move_index(const std::vector<Board>& candidates,
                                       const Board& pre_move_board) const {
    return best_move_index_impl(candidates, pre_move_board);
}

} // namespace bgbot
