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

namespace bgbot {

// Move filtering parameters for N-ply search.
// After scoring all candidates at 0-ply, keep up to `max_moves` that are
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
// At 0-ply: delegates directly to the base strategy.
// At N-ply: for each post-move position, flips the board to the opponent's
// perspective, iterates over all 21 unique dice rolls, finds the opponent's
// best response, then recursively evaluates the resulting position at (N-1)-ply.
//
// Two modes for opponent move selection:
//   fast (default):     opponent picks best move at 0-ply, result evaluated at (N-1)-ply
//   full_depth:         opponent evaluates all candidates at (N-1)-ply, picks the best
//
// Move filtering is applied in best_move_index() only (not in evaluate()):
// candidates are first scored at 0-ply and pruned, then survivors are scored
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

    // Filter candidates at 0-ply, then evaluate survivors at N-ply.
    int best_move_index(const std::vector<Board>& candidates,
                        bool pre_move_is_race) const override;
    int best_move_index(const std::vector<Board>& candidates,
                        const Board& pre_move_board) const override;

    // Cache management.
    void clear_cache() const;
    size_t cache_size() const;
    size_t cache_hits() const;
    size_t cache_misses() const;

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
    int parallel_thread_count(int plies = 0) const;

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
        static constexpr std::size_t CAPACITY = 2 * 1024 * 1024;  // must be power of 2
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
