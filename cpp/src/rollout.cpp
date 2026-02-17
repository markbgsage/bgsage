#include "bgbot/rollout.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/encoding.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include <thread>
#include <atomic>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
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

} // namespace

// ======================== Constructor ========================

RolloutStrategy::RolloutStrategy(std::shared_ptr<Strategy> base, RolloutConfig config)
    : base_(std::move(base))
    , base_gps_(dynamic_cast<GamePlanStrategy*>(base_.get()))
    , config_(config)
    , cached_max_moves_((config.truncation_depth > 0)
                       ? config.truncation_depth + 10
                       : 200)
{
    // Build decision strategy
    if (config_.decision_ply > 0) {
        decision_strat_ = std::make_shared<MultiPlyStrategy>(
            base_, config_.decision_ply, config_.filter);
    } else {
        decision_strat_ = base_;
    }

    // Build late-game decision strategy
    int effective_late_ply = (config_.late_ply >= 0) ? config_.late_ply : config_.decision_ply;
    if (effective_late_ply == config_.decision_ply) {
        late_decision_strat_ = decision_strat_;
    } else if (effective_late_ply > 0) {
        late_decision_strat_ = std::make_shared<MultiPlyStrategy>(
            base_, effective_late_ply, config_.filter);
    } else {
        late_decision_strat_ = base_;
    }

    // Build VR strategy
    if (config_.vr_ply > 0) {
        vr_strat_ = std::make_shared<MultiPlyStrategy>(
            base_, config_.vr_ply, config_.filter);
    } else if (config_.vr_ply == 0) {
        vr_strat_ = base_;
    }
    // vr_ply < 0: VR disabled, vr_strat_ left as nullptr

    if (config_.n_trials > 0 && cached_max_moves_ > 0) {
        generate_stratified_dice(
            config_.n_trials, cached_max_moves_, config_.seed, cached_dice_);
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
// Uses vr_strat_ for both move selection and evaluation.
std::array<float, NUM_OUTPUTS> RolloutStrategy::best_move_probs(
    const Board& board, int d1, int d2) const
{
    thread_local std::vector<Board> candidates;
    possible_boards(board, d1, d2, candidates);
    return best_move_probs_for_candidates(board, candidates);
}

std::array<float, NUM_OUTPUTS> RolloutStrategy::best_move_probs_for_candidates(
    const Board& board, const std::vector<Board>& candidates, int* best_index) const
{
    if (candidates.empty()) {
        if (best_index) *best_index = -1;
        return vr_strat_->evaluate_probs(board, board);
    }

    if (candidates.size() == 1) {
        GameResult r = check_game_over(candidates[0]);
        if (best_index) *best_index = 0;
        if (r != GameResult::NOT_OVER) {
            return terminal_probs(r);
        }
        return vr_strat_->evaluate_probs(candidates[0], board);
    }

    if (base_gps_ && config_.vr_ply == 0) {
        std::array<float, NUM_OUTPUTS> best_probs{};
        int idx = base_gps_->batch_evaluate_candidates_best_prob(
            candidates, board, nullptr, &best_probs);
        if (best_index) *best_index = idx;
        return best_probs;
    }

    double best_eq = -1e30;
    std::array<float, NUM_OUTPUTS> best_probs = {};
    if (best_index) *best_index = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        GameResult r = check_game_over(candidates[i]);
        std::array<float, NUM_OUTPUTS> probs;
        double eq;
        if (r != GameResult::NOT_OVER) {
            probs = terminal_probs(r);
            eq = static_cast<double>(static_cast<int>(r));
        } else {
            probs = vr_strat_->evaluate_probs(candidates[i], board);
            eq = NeuralNetwork::compute_equity(probs);
        }
        if (eq > best_eq) {
            best_eq = eq;
            best_probs = probs;
            if (best_index) *best_index = static_cast<int>(i);
        }
    }
    return best_probs;
}

double RolloutStrategy::best_move_equity(const Board& board, int d1, int d2) const {
    return NeuralNetwork::compute_equity(best_move_probs(board, d1, d2));
}

// ======================== Single Trial ========================

RolloutStrategy::TrialResult RolloutStrategy::run_trial(
    const Board& start_board,
    const Board& pre_move_board,
    const std::pair<int,int>* dice_seq,
    int max_moves) const
{
    thread_local std::array<std::vector<Board>, ALL_ROLLS.size()> move_candidates;
    thread_local bool candidates_initialized = false;
    if (!candidates_initialized) {
        for (auto& c : move_candidates) c.reserve(24);
        candidates_initialized = true;
    }
    Board board = start_board;
    // VR luck accumulated in starting player's probability space.
    std::array<double, NUM_OUTPUTS> accumulated_luck = {0, 0, 0, 0, 0};
    double scalar_eq_luck = 0.0;
    bool vr_enabled = (vr_strat_ != nullptr);
    int truncation = (config_.truncation_depth > 0) ? config_.truncation_depth : 9999;

    // start_board is POST-move from starting player's perspective.
    // Opponent moves next → flip to opponent's perspective.
    board = flip(board);

    for (int move_num = 0; move_num < truncation && move_num < max_moves; ++move_num) {
        bool is_starting_players_turn = (move_num % 2 == 1);

    // Once contact is broken (pure race), 0-ply is nearly perfect.
    // Use base strategy for both decisions and VR — much cheaper.
    bool use_base = (move_num >= config_.late_threshold) || is_race(board);

    // Step 1: VR mean — average best_move_probs across all 36 rolls.
    // best_move_probs returns probs from MOVER's perspective.
    std::array<double, NUM_OUTPUTS> mean_mover_probs = {0, 0, 0, 0, 0};
    std::array<int, ALL_ROLLS.size()> best_candidate_idx{};
    double mean_mover_equity = 0.0;

        int d1 = dice_seq[move_num].first;
        int d2 = dice_seq[move_num].second;

        for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
            move_candidates[i].clear();
            possible_boards(board, ALL_ROLLS[i].d1, ALL_ROLLS[i].d2, move_candidates[i]);
        }

    if (vr_enabled) {
            bool reuse_vr_idx = (config_.vr_ply == 0) && base_gps_ &&
                                (config_.decision_ply == 0 || use_base);

            for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
                int best_idx = -1;
                auto mover_probs = best_move_probs_for_candidates(
                    board, move_candidates[i],
                    reuse_vr_idx ? &best_idx : nullptr);
                if (reuse_vr_idx) {
                    best_candidate_idx[i] = best_idx;
                }
                for (int k = 0; k < NUM_OUTPUTS; ++k) {
                    mean_mover_probs[k] += ALL_ROLLS[i].weight * mover_probs[k];
                }
            }

            for (int k = 0; k < NUM_OUTPUTS; ++k) {
                mean_mover_probs[k] /= 36.0;
            }
            mean_mover_equity =
                2.0 * mean_mover_probs[0] - 1.0
                + mean_mover_probs[1] - mean_mover_probs[3]
                + mean_mover_probs[2] - mean_mover_probs[4];
        }

    int a = d1;
    int b = d2;
    if (a > b) std::swap(a, b);
    int actual_idx = kOrderedRollToIndex[a][b];
    if (actual_idx < 0) {
        // Should never happen for valid dice, but keep deterministic fallback.
        actual_idx = 0;
    }

    const auto& candidates = move_candidates[actual_idx];
    const bool can_use_cached_actual =
        vr_enabled && (config_.decision_ply == 0 || use_base);
    const int cached_actual_idx = best_candidate_idx[actual_idx];

    Board chosen;
    if (candidates.empty()) {
        chosen = board;
    } else if (candidates.size() == 1) {
        chosen = candidates[0];
    } else {
        int idx = -1;
        if (can_use_cached_actual && cached_actual_idx >= 0 &&
            cached_actual_idx < static_cast<int>(candidates.size())) {
            idx = cached_actual_idx;
        } else {
            const auto& strat = use_base ? *base_ : *decision_strat_;
            idx = strat.best_move_index(candidates, board);
        }
        chosen = candidates[idx];
    }

        // Step 3: VR luck — evaluate actual move from MOVER's perspective,
        // compute luck in mover's space, then cross-map to SP.
        if (vr_enabled) {
            std::array<float, NUM_OUTPUTS> actual_mover_probs;
            GameResult r = check_game_over(chosen);
            if (r != GameResult::NOT_OVER) {
                actual_mover_probs = terminal_probs(r);
            } else {
                actual_mover_probs = vr_strat_->evaluate_probs(chosen, board);
            }
            double actual_mover_eq = NeuralNetwork::compute_equity(actual_mover_probs);

            // Luck in mover's space
            std::array<double, NUM_OUTPUTS> luck_mover;
            for (int k = 0; k < NUM_OUTPUTS; ++k) {
                luck_mover[k] = actual_mover_probs[k] - mean_mover_probs[k];
            }
            double luck_eq = actual_mover_eq - mean_mover_equity;

            // Cross-map luck to starting player's space
            if (is_starting_players_turn) {
                // Mover is SP: luck maps directly
                for (int k = 0; k < NUM_OUTPUTS; ++k) {
                    accumulated_luck[k] += luck_mover[k];
                }
                scalar_eq_luck += luck_eq;
            } else {
                // Mover is opponent: cross-map via invert_probs relationship
                // SP's P(win) luck = -(opp's P(win) luck)
                // SP's P(gw) luck = opp's P(gl) luck (they swap)
                // SP's P(bw) luck = opp's P(bl) luck (they swap)
                // SP's P(gl) luck = opp's P(gw) luck
                // SP's P(bl) luck = opp's P(bw) luck
                accumulated_luck[0] -= luck_mover[0];  // win
                accumulated_luck[1] += luck_mover[3];   // gw ← opp gl
                accumulated_luck[2] += luck_mover[4];   // bw ← opp bl
                accumulated_luck[3] += luck_mover[1];   // gl ← opp gw
                accumulated_luck[4] += luck_mover[2];   // bl ← opp bw
                scalar_eq_luck -= luck_eq;
            }
        }

        // Step 4: Check terminal
        GameResult result = check_game_over(chosen);
        if (result != GameResult::NOT_OVER) {
            // terminal_probs returns mover's perspective
            auto t_probs = terminal_probs(result);
            // Convert to SP perspective
            std::array<float, NUM_OUTPUTS> sp_probs;
            if (is_starting_players_turn) {
                sp_probs = t_probs;
            } else {
                sp_probs = invert_probs(t_probs);
            }
            double raw_eq = NeuralNetwork::compute_equity(sp_probs);
            std::array<float, NUM_OUTPUTS> vr_probs;
            for (int k = 0; k < NUM_OUTPUTS; ++k) {
                vr_probs[k] = static_cast<float>(sp_probs[k] - accumulated_luck[k]);
            }
            return {vr_probs, raw_eq - scalar_eq_luck, raw_eq - scalar_eq_luck};
        }

        board = flip(chosen);
    }

    // Truncation: evaluate from the LAST MOVER's perspective.
    //
    // After the loop, `board` = flip(chosen) = position from the NEXT mover's
    // perspective. But the NN evaluates post-move boards: it assumes the player
    // whose perspective this is JUST MOVED. Since the next mover hasn't moved yet,
    // evaluating `board` directly would be semantically wrong (it would assume an
    // extra move that didn't happen, creating a ~0.12 equity tempo artifact).
    //
    // The correct evaluation: flip(board) = chosen = the LAST mover's post-move
    // board, evaluated from the last mover's perspective. This is exactly what
    // the NN expects — a position the player just moved to.
    Board last_mover_board = flip(board);
    int trunc_move = std::min(truncation, max_moves);
    bool trunc_use_base = (trunc_move > config_.late_threshold) || is_race(last_mover_board);
    const auto& trunc_strat = trunc_use_base ? *base_ : *decision_strat_;
    auto last_mover_probs = trunc_strat.evaluate_probs(last_mover_board, last_mover_board);

    // Convert to SP perspective. The last mover at truncation depth T moved
    // at move_num = T-1. is_sp = ((T-1) % 2 == 1) = (T % 2 == 0).
    bool last_mover_is_sp = (trunc_move % 2 == 0);
    std::array<float, NUM_OUTPUTS> sp_probs;
    if (last_mover_is_sp) {
        sp_probs = last_mover_probs;
    } else {
        sp_probs = invert_probs(last_mover_probs);
    }
    double raw_eq = NeuralNetwork::compute_equity(sp_probs);

    std::array<float, NUM_OUTPUTS> vr_probs;
    for (int k = 0; k < NUM_OUTPUTS; ++k) {
        vr_probs[k] = static_cast<float>(sp_probs[k] - accumulated_luck[k]);
    }
    return {vr_probs, raw_eq - scalar_eq_luck, raw_eq - scalar_eq_luck};
}

// ======================== Parallel Trial Execution ========================

RolloutResult RolloutStrategy::run_trials_parallel(
    const Board& board, const Board& pre_move_board) const
{
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
    int n_threads = config_.n_threads;
    if (n_threads <= 0) {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;
    }
    n_threads = std::min(n_threads, n_trials);
    if (n_threads <= 0) n_threads = 1;

    // Fast path: n_threads == 1 can accumulate directly without trial buffers.
    if (n_threads == 1) {
        RolloutResult result;
        std::array<double, NUM_OUTPUTS> sum_probs = {0, 0, 0, 0, 0};
        std::array<double, NUM_OUTPUTS> sum_probs_sq = {0, 0, 0, 0, 0};
        double sum_eq = 0.0, sum_eq_sq = 0.0;
        double sum_svr_eq = 0.0, sum_svr_eq_sq = 0.0;

        for (int t = 0; t < n_trials; ++t) {
            auto r = run_trial(board, pre_move_board, all_dice[t].data(), max_moves);
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

    if (config_.decision_ply == 0 && config_.vr_ply <= 0) {
        // For fixed-depth shallow rollout (decision_ply/vr_ply == 0) and short
        // batches, static partitioning is often faster than atomic work stealing.
#ifdef _WIN32
        struct ThreadArg {
            const RolloutStrategy* self;
            const Board* board;
            const Board* pre_move_board;
            const std::vector<std::vector<std::pair<int, int>>>* all_dice;
            TrialResult* trial_results;
            int n_trials;
            int max_moves;
            int n_threads;
            int thread_idx;
        };

        std::vector<ThreadArg> args(n_threads);
        for (int th = 0; th < n_threads; ++th) {
            args[th] = {this, &board, &pre_move_board, &all_dice,
                        trial_results.data(), n_trials, max_moves,
                        n_threads, th};
        }

        std::vector<HANDLE> handles(n_threads);
        for (int th = 0; th < n_threads; ++th) {
            handles[th] = CreateThread(
                nullptr, 4 * 1024 * 1024,
                [](LPVOID param) -> DWORD {
                    auto* a = static_cast<ThreadArg*>(param);
                    for (int t = a->thread_idx; t < a->n_trials; t += a->n_threads) {
                        a->trial_results[t] = a->self->run_trial(
                            *a->board, *a->pre_move_board,
                            (*a->all_dice)[t].data(), a->max_moves);
                    }
                    return 0;
                },
                &args[th], 0, nullptr);
        }

        WaitForMultipleObjects(n_threads, handles.data(), TRUE, INFINITE);
        for (int th = 0; th < n_threads; ++th) {
            CloseHandle(handles[th]);
        }
#else
        std::vector<std::thread> threads;
        threads.reserve(n_threads);

        for (int th = 0; th < n_threads; ++th) {
            threads.emplace_back([this, &board, &pre_move_board, &all_dice,
                                  &trial_results, n_threads, n_trials, max_moves, th]() {
                for (int t = th; t < n_trials; t += n_threads) {
                    trial_results[t] = run_trial(
                        board, pre_move_board,
                        all_dice[t].data(), max_moves);
                }
            });
        }

        for (auto& t : threads) t.join();
#endif
    } else {
        // Work-stealing: threads grab trials from a shared atomic counter.
        // This handles variance in trial length (some games are 10 moves,
        // others 100+) much better than static chunking.
        std::atomic<int> next_trial{0};

#ifdef _WIN32
        struct ThreadArg {
            const RolloutStrategy* self;
            const Board* board;
            const Board* pre_move_board;
            const std::vector<std::vector<std::pair<int,int>>>* all_dice;
            TrialResult* trial_results;
            std::atomic<int>* next_trial;
            int n_trials;
            int max_moves;
        };

        std::vector<HANDLE> handles(n_threads);
        ThreadArg arg = {this, &board, &pre_move_board, &all_dice,
                         trial_results.data(), &next_trial, n_trials, max_moves};

        for (int th = 0; th < n_threads; ++th) {
            handles[th] = CreateThread(
                nullptr, 4 * 1024 * 1024,
                [](LPVOID param) -> DWORD {
                    auto* a = static_cast<ThreadArg*>(param);
                    int t;
                    while ((t = a->next_trial->fetch_add(1, std::memory_order_relaxed)) < a->n_trials) {
                        a->trial_results[t] = a->self->run_trial(
                            *a->board, *a->pre_move_board,
                            (*a->all_dice)[t].data(), a->max_moves);
                    }
                    return 0;
                },
                &arg, 0, nullptr);
        }

        WaitForMultipleObjects(n_threads, handles.data(), TRUE, INFINITE);
        for (int th = 0; th < n_threads; ++th) {
            CloseHandle(handles[th]);
        }
#else
        std::vector<std::thread> threads;
        threads.reserve(n_threads);

        for (int th = 0; th < n_threads; ++th) {
            threads.emplace_back([this, &board, &pre_move_board, &all_dice,
                                  &trial_results, &next_trial, n_trials, max_moves]() {
                int t;
                while ((t = next_trial.fetch_add(1, std::memory_order_relaxed)) < n_trials) {
                    trial_results[t] = run_trial(
                        board, pre_move_board,
                        all_dice[t].data(), max_moves);
                }
            });
        }

        for (auto& t : threads) t.join();
#endif
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

// ======================== Public Interface ========================

double RolloutStrategy::evaluate(const Board& board, bool pre_move_is_race) const {
    auto r = rollout_position(board, board);
    return r.equity;
}

std::array<float, NUM_OUTPUTS> RolloutStrategy::evaluate_probs(
    const Board& board, bool pre_move_is_race) const
{
    auto r = rollout_position(board, board);
    return r.mean_probs;
}

std::array<float, NUM_OUTPUTS> RolloutStrategy::evaluate_probs(
    const Board& board, const Board& pre_move_board) const
{
    auto r = rollout_position(board, pre_move_board);
    return r.mean_probs;
}

RolloutResult RolloutStrategy::rollout_position(
    const Board& board, const Board& pre_move_board) const
{
    return run_trials_parallel(board, pre_move_board);
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

    // Step 1: Score all candidates at 0-ply for filtering
    std::vector<double> equities(n);
    double best_0ply = -1e30;

    if (base_gps_) {
        base_gps_->batch_evaluate_candidates_equity(
            candidates, pre_move_board, equities.data());
        for (int i = 0; i < n; ++i) {
            if (equities[i] > best_0ply) best_0ply = equities[i];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            equities[i] = NeuralNetwork::compute_equity(
                base_->evaluate_probs(candidates[i], pre_move_board));
            if (equities[i] > best_0ply) best_0ply = equities[i];
        }
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
        if (best_0ply - equities[idx] > config_.filter.threshold) break;
        survivors.push_back(idx);
    }

    if (survivors.size() == 1) return survivors[0];

    // Step 3: Rollout each surviving candidate
    double best_rollout = -1e30;
    int best_idx = survivors[0];

    for (int idx : survivors) {
        auto r = rollout_position(candidates[idx], pre_move_board);
        if (r.equity > best_rollout) {
            best_rollout = r.equity;
            best_idx = idx;
        }
    }

    return best_idx;
}

} // namespace bgbot
