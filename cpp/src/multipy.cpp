#include "bgbot/multipy.h"
#include "bgbot/board.h"
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
        threads_.reserve(n_threads_);
        for (int i = 0; i < n_threads_; ++i) {
            threads_.emplace_back([this]() { worker_loop(); });
        }
    }

    ~MultiPlyThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
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
        std::atomic<int> completed_workers{0};
        const auto task_fn = std::forward<F>(fn);

        struct JobState {
            std::atomic<int> completed_workers{0};
            std::mutex done_mutex;
            std::condition_variable done_cv;
        };

        JobState state;

        auto run_chunk = [&](int start) {
            for (int item = start; item < n_items; item += workers) {
                task_fn(item);
            }
            if (state.completed_workers.fetch_add(1, std::memory_order_acq_rel) + 1 == workers) {
                std::lock_guard<std::mutex> lock(state.done_mutex);
                state.done_cv.notify_one();
            }
        };

        for (int t = 1; t < workers; ++t) {
            enqueue([&state, run_chunk, t]() mutable { run_chunk(t); });
        }

        run_chunk(0);

        if (state.completed_workers.load(std::memory_order_acquire) != workers) {
            std::unique_lock<std::mutex> lock(state.done_mutex);
            state.done_cv.wait(lock, [&state, workers]() {
                return state.completed_workers.load(std::memory_order_acquire) == workers;
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
    mutable std::vector<std::thread> threads_;
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

// ======================== Static Members ========================

std::atomic<std::size_t> MultiPlyStrategy::PosCache::global_hits{0};
std::atomic<std::size_t> MultiPlyStrategy::PosCache::global_misses{0};
std::atomic<std::uint64_t> MultiPlyStrategy::next_cache_salt_{1};

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
    , base_gps_(dynamic_cast<GamePlanStrategy*>(base_.get()))
    , n_plies_(std::max(0, n_plies))
    , filter_(filter)
    , full_depth_opponent_(full_depth_opponent)
    , parallel_evaluate_(parallel_evaluate)
    , parallel_threads_(parallel_threads)
    , cache_salt_(next_cache_salt_.fetch_add(1, std::memory_order_relaxed))
{
    if (filter_.max_moves < 1) filter_.max_moves = 1;
    if (filter_.threshold < 0.0f) filter_.threshold = 0.0f;
}

// ======================== Cache Management ========================

void MultiPlyStrategy::clear_cache() const {
    get_cache().clear();
    cache_salt_.store(
        next_cache_salt_.fetch_add(1, std::memory_order_relaxed),
        std::memory_order_relaxed);
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

    int n_threads = parallel_threads_;
    if (n_threads <= 0) {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;
    } else if (n_threads < 1) {
        n_threads = 1;
    }

    if (plies > 0) {
        const int target_cores = 4 * std::max(1, plies);
        const int effective_target = (plies >= 3) ? std::min(8, target_cores) : target_cores;
        n_threads = std::min(n_threads, effective_target);
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
//   - The NN used must be determined by the PRE-MOVE board (before the player moves)
//   - This is the `pre_move_board` parameter throughout

std::array<float, NUM_OUTPUTS> MultiPlyStrategy::evaluate_probs_nply(
    const Board& board, const Board& pre_move_board, int plies) const
{
    return evaluate_probs_nply_impl(
        board, pre_move_board, plies,
        parallel_evaluate_ && plies == n_plies_ && plies > 1);
}

std::array<float, NUM_OUTPUTS> MultiPlyStrategy::evaluate_probs_nply_impl(
    const Board& board, const Board& pre_move_board, int plies,
    bool allow_parallel) const
{
    // Base case: 0-ply -> delegate to base strategy with proper pre-move context
    if (plies <= 0) {
        return base_->evaluate_probs(board, pre_move_board);
    }

    // Terminal position check
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_probs(result);
    }

    // Cache lookup
    std::size_t bh = cache_key_for(board, plies);
    auto& cache = get_cache();
    const auto* cached = cache.lookup(bh, /*plies folded into key*/ 0);
    if (cached) {
        return *cached;
    }

    // Flip to opponent's perspective. The opponent is now "player 1" on this board.
    // opp_board is the opponent's PRE-MOVE board (before they roll/move).
    Board opp_board = flip(board);

    // Accumulate weighted probabilities from the CURRENT PLAYER's perspective
    // (the one who moved to `board`). After the opponent responds, we get the
    // opponent's probs and invert them back to the current player's perspective.
    std::array<double, NUM_OUTPUTS> sum_probs = {0, 0, 0, 0, 0};

    constexpr int PREFILTER_THRESHOLD = 20;
    constexpr int PREFILTER_KEEP = 15;

    auto evaluate_roll = [&](const DiceRoll& roll) -> std::array<float, NUM_OUTPUTS> {
        thread_local std::vector<Board> opp_candidates;
        thread_local std::vector<std::pair<int, int>> pip_ranking;
        thread_local std::vector<Board> filtered_candidates;
        thread_local std::vector<double> opp_equities;

        opp_candidates.clear();
        if (opp_candidates.capacity() < 32) opp_candidates.reserve(32);

        // Generate opponent's legal moves
        possible_boards(opp_board, roll.d1, roll.d2, opp_candidates);

        // Pre-filter if too many candidates
        if (static_cast<int>(opp_candidates.size()) > PREFILTER_THRESHOLD) {
            pip_ranking.clear();
            pip_ranking.reserve(opp_candidates.size());
            for (int ci = 0; ci < static_cast<int>(opp_candidates.size()); ++ci) {
                const auto& cand = opp_candidates[ci];
                int blots = 0;
                int p_pips = 0;
                for (int pt = 1; pt <= 24; ++pt) {
                    const int p = cand[pt];
                    if (p > 0) {
                        p_pips += pt * p;
                        if (p == 1) ++blots;
                    }
                }
                int score = p_pips + 8 * blots + 20 * cand[25];
                pip_ranking.push_back({score, ci});
            }
            std::nth_element(pip_ranking.begin(),
                             pip_ranking.begin() + PREFILTER_KEEP,
                             pip_ranking.end());
            pip_ranking.resize(PREFILTER_KEEP);
            std::sort(pip_ranking.begin(), pip_ranking.end());
            filtered_candidates.clear();
            filtered_candidates.reserve(PREFILTER_KEEP);
            for (int fi = 0; fi < PREFILTER_KEEP; ++fi) {
                filtered_candidates.push_back(opp_candidates[pip_ranking[fi].second]);
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
        } else if (full_depth_opponent_ && plies > 1) {
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
                        opp_full_depth_eq[i] = NeuralNetwork::compute_equity(candidate_p1_probs);
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
                    opp_full_depth_eq[i] = NeuralNetwork::compute_equity(candidate_p1_probs);
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
            // Fast mode (default): opponent picks best move at 0-ply.
            // Use evaluate_candidates_equity when available (classifies game plan
            // once for all candidates instead of per-candidate).
            int best_opp_idx;

            // Optimization: when plies==1, the recursion would evaluate the best
            // move at 0-ply (re-encoding + re-evaluating). Instead, use
            // batch_evaluate_candidates_best_prob to get only best equity and probs.
            // in one pass, avoiding the redundant re-evaluation.
            if (plies == 1 && base_gps_) {
                std::array<float, NUM_OUTPUTS> best_probs{};
                best_opp_idx = base_gps_->batch_evaluate_candidates_best_prob(
                    opp_candidates, opp_board, nullptr, &best_probs);
                // At plies==1, evaluate_probs_nply(best, board, 0) would just
                // return base_->evaluate_probs(best, board). We already have those
                // probs from the batch evaluation. Use them directly.
                p1_probs = invert_probs(best_probs);
            } else if (base_gps_) {
                // Batch evaluation: classify once, encode all, forward_batch
                opp_equities.resize(opp_candidates.size());
                best_opp_idx = base_gps_->batch_evaluate_candidates_equity(
                    opp_candidates, opp_board, opp_equities.data());

                Board opp_best = opp_candidates[best_opp_idx];
                GameResult opp_result = check_game_over(opp_best);
                if (opp_result != GameResult::NOT_OVER) {
                    p1_probs = invert_probs(terminal_probs(opp_result));
                } else {
                    auto opp_probs = evaluate_probs_nply_impl(opp_best, opp_board, plies - 1, false);
                    p1_probs = invert_probs(opp_probs);
                }
            } else {
                // Fallback for non-GamePlanStrategy bases
                double best_opp_eq = -1e30;
                best_opp_idx = 0;
                for (int i = 0; i < static_cast<int>(opp_candidates.size()); ++i) {
                    GameResult r = check_game_over(opp_candidates[i]);
                    double eq;
                    if (r != GameResult::NOT_OVER) {
                        switch (r) {
                            case GameResult::WIN_SINGLE:      eq =  1.0; break;
                            case GameResult::WIN_GAMMON:       eq =  2.0; break;
                            case GameResult::WIN_BACKGAMMON:   eq =  3.0; break;
                            case GameResult::LOSS_SINGLE:      eq = -1.0; break;
                            case GameResult::LOSS_GAMMON:      eq = -2.0; break;
                            case GameResult::LOSS_BACKGAMMON:  eq = -3.0; break;
                            default: eq = 0.0; break;
                        }
                    } else {
                        eq = NeuralNetwork::compute_equity(
                            base_->evaluate_probs(opp_candidates[i], opp_board));
                    }
                    if (eq > best_opp_eq) {
                        best_opp_eq = eq;
                        best_opp_idx = i;
                    }
                }

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

    // Store in cache (auto-clears at 75% load)
    cache.insert(bh, /*plies folded into key*/ 0, avg);

    return avg;
}


// ======================== Public Interface ========================

double MultiPlyStrategy::evaluate(const Board& board, bool pre_move_is_race) const {
    if (n_plies_ == 0) {
        return base_->evaluate(board, pre_move_is_race);
    }
    // For the bool overload, we don't have a real pre-move board.
    // Use the post-move board itself as a reasonable proxy for game plan classification.
    auto probs = evaluate_probs_nply(board, board, n_plies_);
    return NeuralNetwork::compute_equity(probs);
}

std::array<float, NUM_OUTPUTS> MultiPlyStrategy::evaluate_probs(
    const Board& board, bool pre_move_is_race) const
{
    if (n_plies_ == 0) {
        return base_->evaluate_probs(board, pre_move_is_race);
    }
    // Same proxy as evaluate(): use board itself when no pre-move board available.
    return evaluate_probs_nply(board, board, n_plies_);
}

std::array<float, NUM_OUTPUTS> MultiPlyStrategy::evaluate_probs(
    const Board& board, const Board& pre_move_board) const
{
    if (n_plies_ == 0) {
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

    // At 0-ply, just delegate to base strategy with proper pre-move board
    if (n_plies_ == 0) {
        return base_->best_move_index(candidates, pre_move_board);
    }

    // Step 1: Score all candidates at 0-ply using base strategy.
    // Use evaluate_candidates_equity when available (classifies game plan once).
    std::vector<double> equities(n);
    std::vector<double> ranked_equities(n);
    double best_0ply = -1e30;
    if (base_gps_) {
        base_gps_->batch_evaluate_candidates_equity(candidates, pre_move_board, equities.data());
        for (int i = 0; i < n; ++i) {
            ranked_equities[i] = equities[i];
            if (ranked_equities[i] > best_0ply) best_0ply = ranked_equities[i];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            equities[i] = NeuralNetwork::compute_equity(
                base_->evaluate_probs(candidates[i], pre_move_board));
            ranked_equities[i] = equities[i];
            if (ranked_equities[i] > best_0ply) best_0ply = ranked_equities[i];
        }
    }

    // Step 2: Filter candidates — keep top moves within threshold
    std::vector<int> sorted_indices(n);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](int a, int b) { return ranked_equities[a] > ranked_equities[b]; });

    std::vector<int> survivors;
    survivors.reserve(std::min(n, filter_.max_moves));
    for (int idx : sorted_indices) {
        if (static_cast<int>(survivors.size()) >= filter_.max_moves) break;
        if (best_0ply - ranked_equities[idx] > filter_.threshold) break;
        survivors.push_back(idx);
    }
    if (survivors.empty()) return sorted_indices[0];

    // If only 1 survivor, return it without expensive N-ply evaluation
    if (survivors.size() == 1) return survivors[0];

    // Step 3: Re-score survivors at full N-ply depth.
    // Use pre_move_board for proper game plan context.
    double best_nply = -1e30;
    int best_idx = survivors[0];

    bool use_parallel = parallel_evaluate_ && n_plies_ > 1;
    if (use_parallel) {
        int n_threads = std::min<int>(parallel_thread_count(n_plies_),
                                      static_cast<int>(survivors.size()));
        if (n_threads > 1) {
            std::vector<double> survivor_eq(survivors.size(), -1e30);
            get_multipy_executor().parallel_for(
                static_cast<int>(survivors.size()), n_threads,
                [&](int local_i) {
                    int idx = survivors[local_i];
                    auto probs = evaluate_probs_nply_impl(candidates[idx], pre_move_board, n_plies_, false);
                    survivor_eq[local_i] = NeuralNetwork::compute_equity(probs);
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
        auto probs = evaluate_probs_nply_impl(candidates[idx], pre_move_board, n_plies_, false);
        double eq = NeuralNetwork::compute_equity(probs);
        if (eq > best_nply) {
            best_nply = eq;
            best_idx = idx;
        }
    }

    return best_idx;
}

int MultiPlyStrategy::best_move_index(const std::vector<Board>& candidates,
                                       bool pre_move_is_race) const {
    const int n = static_cast<int>(candidates.size());
    if (n <= 1) return 0;

    if (n_plies_ == 0) {
        return base_->best_move_index(candidates, pre_move_is_race);
    }

    // Bool overload fallback: use the same filter + rescore pattern as the
    // board overload, but with bool-context 0-ply scoring and post-move board
    // proxy for N-ply pre-move context.
    std::vector<double> equities(n);
    std::vector<double> ranked_equities(n);
    double best_0ply = -1e30;
    for (int i = 0; i < n; ++i) {
        equities[i] = base_->evaluate(candidates[i], pre_move_is_race);
        ranked_equities[i] = equities[i];
        if (ranked_equities[i] > best_0ply) best_0ply = ranked_equities[i];
    }

    std::vector<int> sorted_indices(n);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](int a, int b) { return ranked_equities[a] > ranked_equities[b]; });

    std::vector<int> survivors;
    survivors.reserve(std::min(n, filter_.max_moves));
    for (int idx : sorted_indices) {
        if (static_cast<int>(survivors.size()) >= filter_.max_moves) break;
        if (best_0ply - ranked_equities[idx] > filter_.threshold) break;
        survivors.push_back(idx);
    }
    if (survivors.empty()) return sorted_indices[0];
    if (survivors.size() == 1) return survivors[0];

    double best_nply = -1e30;
    int best_idx = survivors[0];
    for (int idx : survivors) {
        auto probs = evaluate_probs_nply(candidates[idx], candidates[idx], n_plies_);
        double eq = NeuralNetwork::compute_equity(probs);
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
