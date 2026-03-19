#pragma once

#include "types.h"
#include "strategy.h"
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace bgbot {

// One-sided bearoff database for 15 checkers on 6 points.
//
// Stores the probability distribution of bearing off all checkers in exactly
// k rolls (k=0..31) for each of 54,264 possible positions (C(21,6)).
// Also stores mean rolls (for EPC) and a gammon distribution for positions
// where all 15 checkers are on board.
//
// Positions are indexed using the combinatorial number system (stars-and-bars),
// giving O(1) bidirectional mapping between checker layouts and indices.
//
// Two-sided cubeless probabilities are computed exactly by combining two
// one-sided distributions. Cubeful equity uses existing Janowski interpolation.
class BearoffDB {
public:
    static constexpr int POINTS = 6;
    static constexpr int CHECKERS = 15;
    static constexpr int MAX_ROLLS = 32;
    static constexpr int N_POSITIONS = 54264;  // C(21,6)

    // File format magic and version
    static constexpr uint32_t MAGIC = 0x42454152;  // "BEAR"
    static constexpr uint32_t VERSION = 1;

    BearoffDB() = default;

    // Load database from binary file. Returns true on success.
    bool load(const std::string& path);

    // Save database to binary file. Returns true on success.
    bool save(const std::string& path) const;

    // Generate the database via backward induction. Computes all 54,264
    // one-sided positions with their bearoff distributions and mean rolls.
    void generate();

    // Check if a full 26-element board position is covered by the bearoff DB.
    // True when both players have all checkers in their home boards (or borne off),
    // no checkers on bar, and no contact.
    bool is_bearoff(const Board& board) const;

    // Check if DB has been loaded/generated.
    bool is_loaded() const { return loaded_; }

    // --- One-sided lookups ---

    // Get bearoff distribution for a one-sided position index.
    // dist[k] = P(all checkers borne off in exactly k rolls), quantized to uint16 (0-65535).
    const std::array<uint16_t, MAX_ROLLS>& get_distribution(unsigned int index) const {
        return distributions_[index];
    }

    // Get expected rolls to bear off all checkers for a one-sided position.
    float get_mean_rolls(unsigned int index) const {
        return mean_rolls_[index];
    }

    // Get gammon distribution for a one-sided position (all 15 checkers on board).
    // Returns nullptr if the position has fewer than 15 checkers on board.
    // gammon_dist[k] = P(0 checkers borne off after k rolls of optimal play).
    const std::array<uint16_t, MAX_ROLLS>* get_gammon_distribution(unsigned int index) const;

    // --- Two-sided lookups (combine one-sided distributions) ---

    // Compute exact cubeless probabilities for a full board position.
    // Returns [P(win), P(gw), P(bw), P(gl), P(bl)] where P(bw)=P(bl)=0.
    // The board must be a valid bearoff position (is_bearoff() == true).
    //
    // post_move=false (default): player on points 1-6 rolls FIRST (pre-roll/on-roll).
    //   Use for standalone queries and EPC-related lookups.
    // post_move=true: OPPONENT rolls first (post-move evaluation).
    //   Use when evaluating a position after the player has moved, matching NN
    //   output semantics ("probs from the perspective of the player who just moved,
    //   before the opponent rolls").
    std::array<float, NUM_OUTPUTS> lookup_probs(const Board& board,
                                                 bool post_move = false) const;

    // Get EPC (Effective Pip Count) for one side.
    // player=0: player on roll (checkers on points 1-6)
    // player=1: opponent (checkers on points 19-24 from player's view)
    // Returns mean_rolls * (49.0 / 6.0).
    float lookup_epc(const Board& board, int player) const;

    // --- Position indexing ---

    // Map a 6-element checker count array to a unique index in [0, N_POSITIONS-1].
    // checkers[i] = number of checkers on point (i+1), i=0..5.
    // Sum of checkers must be <= CHECKERS (15).
    static unsigned int position_index(const int checkers[POINTS]);

    // Reverse: map an index back to a 6-element checker count array.
    static void index_to_position(unsigned int index, int checkers[POINTS]);

    // Extract the player-on-roll's one-sided index from a full board.
    // Reads points 1-6 (positive checker counts).
    static unsigned int board_to_player_index(const Board& board);

    // Extract the opponent's one-sided index from a full board.
    // Maps opponent's checkers on points 19-24 to one-sided points 1-6.
    static unsigned int board_to_opponent_index(const Board& board);

    // Get total number of checkers on board for a one-sided position index.
    int checkers_on_board(unsigned int index) const;

private:
    // Pre-computed binomial coefficients for position indexing.
    // binom_[n][k] = C(n, k) for n=0..21, k=0..6.
    static constexpr int MAX_N = 22;  // CHECKERS + POINTS + 1
    static constexpr int MAX_K = 7;   // POINTS + 1
    static unsigned int binom_table_[MAX_N][MAX_K];
    static bool binom_initialized_;
    static void init_binom();

    static unsigned int binom(int n, int k) {
        if (k < 0 || k > n || n < 0) return 0;
        if (n >= MAX_N || k >= MAX_K) return 0;
        if (!binom_initialized_) init_binom();
        return binom_table_[n][k];
    }

    // Storage
    std::array<std::array<uint16_t, MAX_ROLLS>, N_POSITIONS> distributions_{};
    std::array<float, N_POSITIONS> mean_rolls_{};

    // Gammon distributions: only for positions with all 15 checkers on board.
    // Indexed by a secondary index into gammon_distributions_.
    // gammon_index_[pos_index] = index into gammon_distributions_, or -1 if N/A.
    std::array<int, N_POSITIONS> gammon_index_{};
    std::vector<std::array<uint16_t, MAX_ROLLS>> gammon_distributions_;

    bool loaded_ = false;

    // Generation helpers
    void generate_position(unsigned int index,
                           const int checkers[POINTS],
                           std::vector<Board>& move_buf);
    void generate_gammon_distributions();
};


// Strategy wrapper that intercepts bearoff positions with exact DB lookups,
// falling through to a base strategy for non-bearoff positions.
class BearoffStrategy : public Strategy {
public:
    BearoffStrategy(std::shared_ptr<Strategy> base, const BearoffDB* db);

    double evaluate(const Board& board, bool pre_move_is_race) const override;

    std::array<float, NUM_OUTPUTS> evaluate_probs(
        const Board& board, bool pre_move_is_race) const override;

    std::array<float, NUM_OUTPUTS> evaluate_probs(
        const Board& board, const Board& pre_move_board) const override;

    int best_move_index(const std::vector<Board>& candidates,
                        bool pre_move_is_race) const override;

    int best_move_index(const std::vector<Board>& candidates,
                        const Board& pre_move_board) const override;

    const Strategy& base() const { return *base_; }
    const BearoffDB& db() const { return *db_; }

private:
    std::shared_ptr<Strategy> base_;
    const BearoffDB* db_;
};

} // namespace bgbot
