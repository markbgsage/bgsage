#include "bgbot/benchmark.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/neural_net.h"
#include <thread>
#include <algorithm>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace bgbot {

// Score a contiguous slice of scenarios. Pure function — no shared state.
static BenchmarkResult score_slice(const Strategy& strategy,
                                   const BenchmarkScenario* scenarios,
                                   int count) {
    BenchmarkResult result;
    std::vector<Board> candidates;  // reuse across scenarios
    candidates.reserve(32);

    for (int s = 0; s < count; ++s) {
        const auto& scenario = scenarios[s];

        // Generate all legal post-move boards (reuses vector)
        possible_boards(scenario.start_board, scenario.die1, scenario.die2, candidates);

        // Pick the best move using the strategy
        Board chosen;
        if (candidates.size() == 1) {
            chosen = candidates[0];
        } else {
            int idx = strategy.best_move_index(candidates, scenario.start_board);
            chosen = candidates[idx];
        }

        // Flip the chosen board to opponent's perspective (that's how .bm files store results)
        Board chosen_flipped = flip(chosen);

        // Match against the ranked boards
        double err = 0.0;
        bool found = false;

        for (size_t i = 0; i < scenario.ranked_boards.size(); ++i) {
            if (chosen_flipped == scenario.ranked_boards[i]) {
                if (i == 0) {
                    err = 0.0; // Picked the best move
                } else {
                    err = scenario.ranked_errors[i];
                }
                found = true;
                break;
            }
        }

        if (!found) {
            // Move not in the ranked list — use the worst error as estimate
            if (scenario.ranked_errors.size() > 1) {
                err = scenario.ranked_errors.back();
            } else {
                err = 0.0;
            }
        }

        result.total_error += err;
        result.count++;
    }

    return result;
}

BenchmarkResult score_benchmarks(const Strategy& strategy,
                                 const std::vector<BenchmarkScenario>& scenarios,
                                 int n_threads) {
    const int n = static_cast<int>(scenarios.size());
    if (n == 0) return {};

    // Determine thread count
    if (n_threads <= 0) {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;
        // Use all logical cores for compute-bound workloads.
        // Hyperthreading can still help due to memory latency hiding.
    }

    // Don't use more threads than scenarios, and ensure a minimum chunk size
    // to avoid thread creation overhead dominating for small workloads.
    const int MIN_SCENARIOS_PER_THREAD = 64;
    int max_useful_threads = std::max(1, n / MIN_SCENARIOS_PER_THREAD);
    n_threads = std::min(n_threads, max_useful_threads);

    // Serial path: avoid thread overhead for single-threaded case
    if (n_threads == 1) {
        return score_slice(strategy, scenarios.data(), n);
    }

    // Parallel path: each thread gets its own result on the stack inside
    // score_slice, then writes it out — no false sharing during computation.
    struct alignas(64) PaddedResult {
        BenchmarkResult result;
    };
    std::vector<PaddedResult> thread_results(n_threads);

    const int base_chunk = n / n_threads;
    const int remainder = n % n_threads;
    int offset = 0;

#ifdef _WIN32
    // On Windows, use native threads with explicit 4MB stack size.
    // The default 1MB stack can overflow with deep move generation recursion.
    std::vector<HANDLE> handles(n_threads);

    struct ThreadArg {
        const Strategy* strategy;
        const BenchmarkScenario* scenarios;
        int count;
        BenchmarkResult* result;
    };
    std::vector<ThreadArg> args(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        const int chunk_size = base_chunk + (t < remainder ? 1 : 0);
        args[t] = {&strategy, scenarios.data() + offset, chunk_size,
                   &thread_results[t].result};
        offset += chunk_size;

        handles[t] = CreateThread(
            nullptr, 4 * 1024 * 1024,  // 4 MB stack
            [](LPVOID param) -> DWORD {
                auto* a = static_cast<ThreadArg*>(param);
                *a->result = score_slice(*a->strategy, a->scenarios, a->count);
                return 0;
            },
            &args[t], 0, nullptr);
    }

    WaitForMultipleObjects(n_threads, handles.data(), TRUE, INFINITE);
    for (int t = 0; t < n_threads; ++t) {
        CloseHandle(handles[t]);
    }
#else
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        const int chunk_size = base_chunk + (t < remainder ? 1 : 0);

        threads.emplace_back([&strategy, &scenarios, &thread_results, t, offset, chunk_size]() {
            thread_results[t].result = score_slice(strategy, scenarios.data() + offset, chunk_size);
        });

        offset += chunk_size;
    }

    for (int t = 0; t < n_threads; ++t) {
        threads[t].join();
    }
#endif

    // Merge results
    BenchmarkResult result;
    for (int t = 0; t < n_threads; ++t) {
        result.merge(thread_results[t].result);
    }

    return result;
}

// Score a contiguous slice, writing per-scenario errors to errors_out.
static void score_slice_per_scenario(const Strategy& strategy,
                                     const BenchmarkScenario* scenarios,
                                     int count,
                                     double* errors_out) {
    std::vector<Board> candidates;
    candidates.reserve(32);

    for (int s = 0; s < count; ++s) {
        const auto& scenario = scenarios[s];
        possible_boards(scenario.start_board, scenario.die1, scenario.die2, candidates);

        Board chosen;
        if (candidates.size() == 1) {
            chosen = candidates[0];
        } else {
            int idx = strategy.best_move_index(candidates, scenario.start_board);
            chosen = candidates[idx];
        }

        Board chosen_flipped = flip(chosen);

        double err = 0.0;
        bool found = false;
        for (size_t i = 0; i < scenario.ranked_boards.size(); ++i) {
            if (chosen_flipped == scenario.ranked_boards[i]) {
                err = (i == 0) ? 0.0 : scenario.ranked_errors[i];
                found = true;
                break;
            }
        }
        if (!found && scenario.ranked_errors.size() > 1) {
            err = scenario.ranked_errors.back();
        }

        errors_out[s] = err;
    }
}

void score_benchmarks_per_scenario(const Strategy& strategy,
                                   const std::vector<BenchmarkScenario>& scenarios,
                                   double* errors_out,
                                   int n_threads) {
    const int n = static_cast<int>(scenarios.size());
    if (n == 0) return;

    if (n_threads <= 0) {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;
    }

    const int MIN_SCENARIOS_PER_THREAD = 64;
    int max_useful_threads = std::max(1, n / MIN_SCENARIOS_PER_THREAD);
    n_threads = std::min(n_threads, max_useful_threads);

    if (n_threads == 1) {
        score_slice_per_scenario(strategy, scenarios.data(), n, errors_out);
        return;
    }

    const int base_chunk = n / n_threads;
    const int remainder = n % n_threads;
    int offset = 0;

#ifdef _WIN32
    std::vector<HANDLE> handles(n_threads);

    struct ThreadArg {
        const Strategy* strategy;
        const BenchmarkScenario* scenarios;
        int count;
        double* errors_out;
    };
    std::vector<ThreadArg> args(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        const int chunk_size = base_chunk + (t < remainder ? 1 : 0);
        args[t] = {&strategy, scenarios.data() + offset, chunk_size,
                   errors_out + offset};
        offset += chunk_size;

        handles[t] = CreateThread(
            nullptr, 4 * 1024 * 1024,
            [](LPVOID param) -> DWORD {
                auto* a = static_cast<ThreadArg*>(param);
                score_slice_per_scenario(*a->strategy, a->scenarios,
                                        a->count, a->errors_out);
                return 0;
            },
            &args[t], 0, nullptr);
    }

    WaitForMultipleObjects(n_threads, handles.data(), TRUE, INFINITE);
    for (int t = 0; t < n_threads; ++t) {
        CloseHandle(handles[t]);
    }
#else
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        const int chunk_size = base_chunk + (t < remainder ? 1 : 0);

        threads.emplace_back([&strategy, &scenarios, errors_out, offset, chunk_size]() {
            score_slice_per_scenario(strategy, scenarios.data() + offset,
                                    chunk_size, errors_out + offset);
        });

        offset += chunk_size;
    }

    for (auto& t : threads) t.join();
#endif
}

// --- Benchmark PR scoring ---

// Score a slice of PR decisions. Pure function — no shared state.
static PRResult score_pr_slice(const Strategy& strategy,
                               const Strategy* base_strategy,
                               const PRDecision* decisions,
                               int count,
                               const MoveFilter& filter) {
    PRResult result;
    std::vector<Board> all_moves;
    all_moves.reserve(32);

    for (int s = 0; s < count; ++s) {
        const auto& dec = decisions[s];

        // Skip single-candidate decisions
        if (dec.candidates.size() < 2) {
            result.n_skipped++;
            continue;
        }

        // Classify game plan from pre-roll board
        GamePlan gp = classify_game_plan(dec.board);
        int gp_idx = static_cast<int>(gp);

        // Generate all legal moves
        possible_boards(dec.board, dec.die1, dec.die2, all_moves);

        Board best_move;

        if (base_strategy != nullptr) {
            // N-ply mode: pre-filter at 0-ply, then score survivors at full depth

            // Score all moves at 0-ply
            struct MoveEq { double eq; int idx; };
            std::vector<MoveEq> move_eqs(all_moves.size());
            for (size_t i = 0; i < all_moves.size(); ++i) {
                move_eqs[i].eq = base_strategy->evaluate(all_moves[i],
                    is_race(dec.board));
                move_eqs[i].idx = static_cast<int>(i);
            }

            // Sort descending by equity
            std::sort(move_eqs.begin(), move_eqs.end(),
                [](const MoveEq& a, const MoveEq& b) { return a.eq > b.eq; });

            // Apply TINY filter
            double best_0ply = move_eqs[0].eq;
            std::vector<int> survivor_indices;
            for (const auto& me : move_eqs) {
                if (static_cast<int>(survivor_indices.size()) >= filter.max_moves)
                    break;
                if (best_0ply - me.eq > filter.threshold)
                    break;
                survivor_indices.push_back(me.idx);
            }

            // Full-depth eval on survivors
            double best_eq = -1e9;
            int best_idx = survivor_indices[0];
            for (int idx : survivor_indices) {
                double eq;
                auto probs = strategy.evaluate_probs(all_moves[idx], dec.board);
                eq = NeuralNetwork::compute_equity(probs);
                if (eq > best_eq) {
                    best_eq = eq;
                    best_idx = idx;
                }
            }
            best_move = all_moves[best_idx];
        } else {
            // 0-ply mode: evaluate all moves directly
            int best_idx = strategy.best_move_index(all_moves, dec.board);
            best_move = all_moves[best_idx];
        }

        // Find best and worst rollout equity among candidates
        double best_rollout_eq = -1e9;
        double worst_rollout_eq = 1e9;
        for (size_t i = 0; i < dec.candidates.size(); ++i) {
            double eq = dec.rollout_equities[i];
            if (eq > best_rollout_eq) best_rollout_eq = eq;
            if (eq < worst_rollout_eq) worst_rollout_eq = eq;
        }

        // Find strategy's chosen move in the filtered candidate set
        double chosen_rollout_eq = worst_rollout_eq;
        bool found = false;
        for (size_t i = 0; i < dec.candidates.size(); ++i) {
            if (best_move == dec.candidates[i]) {
                chosen_rollout_eq = dec.rollout_equities[i];
                found = true;
                break;
            }
        }
        if (!found) {
            result.n_outside++;
        }

        double error = best_rollout_eq - chosen_rollout_eq;
        result.gp_total_error[gp_idx] += error;
        result.gp_n_decisions[gp_idx]++;
        if (error > 0) {
            result.gp_n_with_error[gp_idx]++;
        }
    }

    return result;
}

PRResult score_benchmark_pr(const Strategy& strategy,
                            const Strategy* base_strategy,
                            const std::vector<PRDecision>& decisions,
                            const MoveFilter& filter,
                            int n_threads) {
    const int n = static_cast<int>(decisions.size());
    if (n == 0) return {};

    if (n_threads <= 0) {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;
    }

    const int MIN_PER_THREAD = 64;
    int max_useful_threads = std::max(1, n / MIN_PER_THREAD);
    n_threads = std::min(n_threads, max_useful_threads);

    if (n_threads == 1) {
        return score_pr_slice(strategy, base_strategy, decisions.data(), n, filter);
    }

    struct alignas(64) PaddedResult {
        PRResult result;
    };
    std::vector<PaddedResult> thread_results(n_threads);

    const int base_chunk = n / n_threads;
    const int remainder = n % n_threads;
    int offset = 0;

#ifdef _WIN32
    std::vector<HANDLE> handles(n_threads);

    struct ThreadArg {
        const Strategy* strategy;
        const Strategy* base_strategy;
        const PRDecision* decisions;
        int count;
        const MoveFilter* filter;
        PRResult* result;
    };
    std::vector<ThreadArg> args(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        const int chunk_size = base_chunk + (t < remainder ? 1 : 0);
        args[t] = {&strategy, base_strategy, decisions.data() + offset,
                   chunk_size, &filter, &thread_results[t].result};
        offset += chunk_size;

        handles[t] = CreateThread(
            nullptr, 4 * 1024 * 1024,
            [](LPVOID param) -> DWORD {
                auto* a = static_cast<ThreadArg*>(param);
                *a->result = score_pr_slice(*a->strategy, a->base_strategy,
                                            a->decisions, a->count, *a->filter);
                return 0;
            },
            &args[t], 0, nullptr);
    }

    WaitForMultipleObjects(n_threads, handles.data(), TRUE, INFINITE);
    for (int t = 0; t < n_threads; ++t) {
        CloseHandle(handles[t]);
    }
#else
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        const int chunk_size = base_chunk + (t < remainder ? 1 : 0);

        threads.emplace_back([&strategy, base_strategy, &decisions, &filter,
                              &thread_results, t, offset, chunk_size]() {
            thread_results[t].result = score_pr_slice(
                strategy, base_strategy, decisions.data() + offset,
                chunk_size, filter);
        });

        offset += chunk_size;
    }

    for (auto& t : threads) t.join();
#endif

    PRResult result;
    for (int t = 0; t < n_threads; ++t) {
        result.merge(thread_results[t].result);
    }

    return result;
}

} // namespace bgbot
