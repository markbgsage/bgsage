#pragma once

#include "strategy.h"
#include "neural_net.h"
#include "types.h"
#include <memory>
#include <array>
#include <vector>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>

namespace bgbot {

// Lock-free shared position cache for cross-thread sharing during rollouts.
// Multiple threads read/write concurrently using a per-entry state machine:
// 0=empty, 1=writing (CAS-protected), 2=ready (safe to read).
struct SharedPosCache {
    static constexpr std::size_t CAPACITY = 512 * 1024;
    static constexpr std::size_t MASK = CAPACITY - 1;
    static constexpr int MAX_PROBE = 8;
    static constexpr int STATE_EMPTY = 0;
    static constexpr int STATE_CLAIMED = 1;
    static constexpr int STATE_COMPUTING = 2;
    static constexpr int STATE_READY = 3;
    static constexpr int WAIT_SPINS = 256;

    struct Entry {
        std::atomic<int> state{STATE_EMPTY};
        std::size_t hash = 0;
        int plies = 0;
        std::array<float, NUM_OUTPUTS> probs = {};
    };

    struct LookupResult {
        const std::array<float, NUM_OUTPUTS>* probs = nullptr;
        Entry* reservation = nullptr;
    };

    std::vector<Entry> entries;
    mutable std::atomic<std::size_t> hits{0};
    mutable std::atomic<std::size_t> misses{0};
    mutable std::atomic<std::size_t> inserts{0};

    SharedPosCache() : entries(CAPACITY) {}

    void clear() {
        for (auto& e : entries) {
            e.hash = 0;
            e.plies = 0;
            e.state.store(STATE_EMPTY, std::memory_order_relaxed);
        }
        hits.store(0, std::memory_order_relaxed);
        misses.store(0, std::memory_order_relaxed);
        inserts.store(0, std::memory_order_relaxed);
    }

    const std::array<float, NUM_OUTPUTS>* lookup(std::size_t h, int plies) const {
        std::size_t key = h | 1;
        std::size_t idx = key & MASK;
        for (int probe = 0; probe < MAX_PROBE; ++probe) {
            int s = entries[idx].state.load(std::memory_order_acquire);
            if (s == STATE_EMPTY) return nullptr;
            if (s == STATE_READY && entries[idx].hash == key && entries[idx].plies == plies) {
                hits.fetch_add(1, std::memory_order_relaxed);
                return &entries[idx].probs;
            }
            idx = (idx + 1) & MASK;
        }
        misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    LookupResult lookup_or_reserve(std::size_t h, int plies) {
        std::size_t key = h | 1;
        std::size_t idx = key & MASK;
        for (int probe = 0; probe < MAX_PROBE; ++probe) {
            auto& e = entries[idx];
            int s = e.state.load(std::memory_order_acquire);
            if (s == STATE_EMPTY) {
                int expected = STATE_EMPTY;
                if (e.state.compare_exchange_strong(expected, STATE_CLAIMED,
                                                    std::memory_order_acq_rel)) {
                    e.hash = key;
                    e.plies = plies;
                    std::atomic_thread_fence(std::memory_order_release);
                    e.state.store(STATE_COMPUTING, std::memory_order_release);
                    inserts.fetch_add(1, std::memory_order_relaxed);
                    return {nullptr, &e};
                }
                s = expected;
            }

            if ((s == STATE_COMPUTING || s == STATE_READY) &&
                e.hash == key && e.plies == plies) {
                if (s == STATE_READY) {
                    hits.fetch_add(1, std::memory_order_relaxed);
                    return {&e.probs, nullptr};
                }

                for (int spin = 0; spin < WAIT_SPINS; ++spin) {
                    std::this_thread::yield();
                    int ready = e.state.load(std::memory_order_acquire);
                    if (ready == STATE_READY && e.hash == key && e.plies == plies) {
                        hits.fetch_add(1, std::memory_order_relaxed);
                        return {&e.probs, nullptr};
                    }
                    if (ready == STATE_EMPTY) break;
                }
                misses.fetch_add(1, std::memory_order_relaxed);
                return {};
            }

            idx = (idx + 1) & MASK;
        }

        misses.fetch_add(1, std::memory_order_relaxed);
        return {};
    }

    void publish(Entry* entry, const std::array<float, NUM_OUTPUTS>& probs) {
        if (!entry) return;
        entry->probs = probs;
        entry->state.store(STATE_READY, std::memory_order_release);
    }

    void abandon(Entry* entry) {
        if (!entry) return;
        entry->hash = 0;
        entry->plies = 0;
        entry->state.store(STATE_EMPTY, std::memory_order_release);
    }

    void insert(std::size_t h, int plies, const std::array<float, NUM_OUTPUTS>& probs) {
        std::size_t key = h | 1;
        std::size_t idx = key & MASK;
        for (int probe = 0; probe < MAX_PROBE; ++probe) {
            auto& e = entries[idx];
            int s = e.state.load(std::memory_order_relaxed);
            if (s == STATE_EMPTY) {
                int expected = STATE_EMPTY;
                if (e.state.compare_exchange_strong(expected, STATE_CLAIMED,
                                                    std::memory_order_acq_rel)) {
                    e.hash = key;
                    e.plies = plies;
                    e.probs = probs;
                    e.state.store(STATE_READY, std::memory_order_release);
                    inserts.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                s = expected;
            }
            if ((s == STATE_COMPUTING || s == STATE_READY) &&
                e.hash == key && e.plies == plies) {
                return;  // already cached
            }
            idx = (idx + 1) & MASK;
        }
    }
};

// Move filtering parameters for N-ply search.
// After scoring all candidates at 1-ply, keep up to `max_moves` that are
// within `threshold` equity of the best.
struct MoveFilter {
    int max_moves = 8;
    float threshold = 0.16f;
};

// Predefined filter presets (matching GNUbg named settings).
namespace MoveFilters {
    constexpr MoveFilter TINY    = {5,  0.08f};
    constexpr MoveFilter NARROW  = {8,  0.12f};
    constexpr MoveFilter NORMAL  = {8,  0.16f};
    constexpr MoveFilter LARGE   = {16, 0.32f};
    constexpr MoveFilter HUGE_   = {20, 0.44f};  // trailing underscore avoids macOS math.h HUGE macro
}

// N-ply lookahead strategy that wraps any base strategy.
//
// At 1-ply: delegates directly to the base strategy (raw NN evaluation).
// At N-ply (N>=2): for each post-move position, flips the board to the opponent's
// perspective, iterates over all 21 unique dice rolls, finds the opponent's
// best response, then recursively evaluates the resulting position at (N-1)-ply.
//
// Two modes for opponent move selection:
//   fast (default):     opponent picks best move at 1-ply, result evaluated at (N-1)-ply
//   full_depth:         opponent evaluates all candidates at (N-1)-ply, picks the best
//
// Move filtering is applied in best_move_index() only (not in evaluate()):
// candidates are first scored at 1-ply and pruned, then survivors are scored
// at the full N-ply depth.
class MultiPlyStrategy : public Strategy {
public:
    MultiPlyStrategy(std::shared_ptr<Strategy> base_strategy,
                     int n_plies,
                     MoveFilter filter = MoveFilters::TINY,
                     bool full_depth_opponent = false,
                     bool parallel_evaluate = false,
                     int parallel_threads = 0);

    // Evaluate a post-move board at N-ply depth. Returns equity.
    double evaluate(const Board& board, bool pre_move_is_race) const override;

    // Returns the full 5 probabilities at N-ply depth.
    std::array<float, NUM_OUTPUTS> evaluate_probs(
        const Board& board, bool pre_move_is_race) const override;
    std::array<float, NUM_OUTPUTS> evaluate_probs(
        const Board& board, const Board& pre_move_board) const override;

    // Filter candidates at 1-ply, then evaluate survivors at N-ply.
    int best_move_index(const std::vector<Board>& candidates,
                        bool pre_move_is_race) const override;
    int best_move_index(const std::vector<Board>& candidates,
                        const Board& pre_move_board) const override;

    // Cache management.
    void clear_cache() const;
    size_t cache_size() const;
    size_t cache_hits() const;
    size_t cache_misses() const;

    // Set/get a shared cross-thread cache for parallel rollouts.
    // When set, evaluate_probs_nply_impl checks the shared cache on miss
    // and inserts computed results into it.
    static void set_shared_cache(SharedPosCache* cache);
    static SharedPosCache* get_shared_cache();

    // Accessors.
    int n_plies() const { return n_plies_; }
    const MoveFilter& move_filter() const { return filter_; }
    const Strategy& base_strategy() const { return *base_; }

private:
    static std::atomic<std::uint64_t> next_cache_salt_;

    std::shared_ptr<Strategy> base_;
    GamePlanStrategy* base_gps_;  // Cached downcast (null if base isn't GamePlanStrategy)
    int n_plies_;
    MoveFilter filter_;
    bool full_depth_opponent_;
    bool parallel_evaluate_;
    int parallel_threads_;
    mutable std::atomic<std::uint64_t> cache_salt_;

    // The 21 unique dice rolls with probability weights.
    struct DiceRoll { int d1, d2, weight; };
    static const std::array<DiceRoll, 21> ALL_ROLLS;

    // Core recursive N-ply evaluation.
    // Board is a post-move position from player 1's perspective.
    // pre_move_board is the board BEFORE the move (for game plan classification).
    // Returns 5 probabilities from player 1's perspective.
    std::array<float, NUM_OUTPUTS> evaluate_probs_nply(
        const Board& board, const Board& pre_move_board, int plies) const;
    std::array<float, NUM_OUTPUTS> evaluate_probs_nply_impl(
        const Board& board, const Board& pre_move_board, int plies,
        bool allow_parallel) const;
    int parallel_thread_count(int plies = 1) const;

    // Open-addressing position cache with power-of-2 sizing.
    // Each entry stores: hash (0 = empty), plies, and 5 output probabilities.
    // Linear probing with a max probe distance. Much faster than std::unordered_map
    // due to no pointer chasing and better cache locality.
    struct CacheEntry {
        std::size_t hash;  // 0 = empty slot
        int plies;
        std::array<float, NUM_OUTPUTS> probs;
    };

    struct PosCache {
        static constexpr std::size_t CAPACITY = 512 * 1024;  // must be power of 2
        static constexpr std::size_t MASK = CAPACITY - 1;
        static constexpr int MAX_PROBE = 8;

        std::vector<CacheEntry> entries;
        std::size_t count = 0;
        // Per-cache counters (for debugging single-threaded only)
        mutable std::size_t hits = 0;
        mutable std::size_t misses = 0;
        // Global atomic counters (aggregated across all threads)
        static std::atomic<std::size_t> global_hits;
        static std::atomic<std::size_t> global_misses;

        PosCache() : entries(CAPACITY) {
            clear();
        }

        void clear() {
            std::memset(entries.data(), 0, CAPACITY * sizeof(CacheEntry));
            count = 0;
            hits = 0;
            misses = 0;
        }

        // Returns pointer to cached probs, or nullptr if not found.
        const std::array<float, NUM_OUTPUTS>* lookup(std::size_t h, int plies) const {
            // Ensure hash is never 0 (our empty marker)
            std::size_t key = h | 1;
            std::size_t idx = key & MASK;
            for (int probe = 0; probe < MAX_PROBE; ++probe) {
                const auto& e = entries[idx];
                if (e.hash == 0) {
                    global_misses.fetch_add(1, std::memory_order_relaxed);
                    return nullptr;
                }
                if (e.hash == key && e.plies == plies) {
                    global_hits.fetch_add(1, std::memory_order_relaxed);
                    return &e.probs;
                }
                idx = (idx + 1) & MASK;
            }
            global_misses.fetch_add(1, std::memory_order_relaxed);
            return nullptr;  // max probe exceeded
        }

        void insert(std::size_t h, int plies, const std::array<float, NUM_OUTPUTS>& probs) {
            // Clear if too full (>75% load factor)
            if (count >= (CAPACITY * 3) / 4) {
                clear();
            }
            std::size_t key = h | 1;
            std::size_t idx = key & MASK;
            for (int probe = 0; probe < MAX_PROBE; ++probe) {
                auto& e = entries[idx];
                if (e.hash == 0) {
                    e.hash = key;
                    e.plies = plies;
                    e.probs = probs;
                    ++count;
                    return;
                }
                if (e.hash == key && e.plies == plies) {
                    e.probs = probs;  // update existing
                    return;
                }
                idx = (idx + 1) & MASK;
            }
            // Max probe exceeded — don't insert (rare at <75% load)
        }

        std::size_t size() const { return count; }
    };

    // Access thread-local cache (MSVC-compatible: function-local thread_local).
    static PosCache& get_cache();

    static std::size_t hash_board(const Board& b);
    std::size_t cache_key_for(const Board& b, int plies) const;

    // Helper: filter + re-score candidates. Used by both best_move_index overloads.
    int best_move_index_impl(const std::vector<Board>& candidates,
                             const Board& pre_move_board) const;
};

// ---------------------------------------------------------------------------
// Shared thread pool for parallel fan-out (used by multipy and cube).
// ---------------------------------------------------------------------------

// Execute fn(i) for i in [0, n_items) using the shared persistent thread pool.
// Falls back to serial execution when n_threads <= 1 or n_items <= 1.
void multipy_parallel_for(int n_items, int n_threads,
                          const std::function<void(int)>& fn);

} // namespace bgbot
