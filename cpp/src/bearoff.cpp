#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/neural_net.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace bgbot {

// --- Binomial coefficient table ---

unsigned int BearoffDB::binom_table_[MAX_N][MAX_K] = {};
bool BearoffDB::binom_initialized_ = false;

void BearoffDB::init_binom() {
    for (int n = 0; n < MAX_N; ++n) {
        binom_table_[n][0] = 1;
        for (int k = 1; k < MAX_K && k <= n; ++k) {
            binom_table_[n][k] = binom_table_[n-1][k-1] + binom_table_[n-1][k];
        }
    }
    binom_initialized_ = true;
}

// --- Position indexing ---
//
// Uses the combinatorial number system (stars-and-bars). A position with
// checkers[0..5] on points 1-6 (sum <= 15) maps to a unique index.
//
// The encoding treats the position as distributing up to 15 checkers into
// 6 bins (points) plus an implicit "borne off" bin. The index is computed
// as the lexicographic rank of this distribution.
//
// We use the "bars" representation: given checkers c[0..5] with sum S <= 15,
// and borne_off = 15 - S, we have 7 bins. The combinatorial rank in the
// (n+k choose k) space uses partial sums.

unsigned int BearoffDB::position_index(const int checkers[POINTS]) {
    if (!binom_initialized_) init_binom();

    // Compute using GNUbg's fBits approach:
    // Encode as a bit pattern of length (CHECKERS + POINTS) = 21 bits,
    // with POINTS = 6 separator bits and CHECKERS = 15 item bits.
    // The rank is the combinatorial number of this pattern.

    // Equivalent direct computation using partial sums:
    unsigned int index = 0;
    int remaining = CHECKERS; // checkers not yet accounted for
    for (int i = POINTS - 1; i >= 0; --i) {
        remaining -= checkers[i];
        // At bin i, we have 'remaining' checkers still to distribute
        // among bins 0..i-1 plus borne-off.
        // The contribution uses binomial coefficients.
    }

    // Actually, let's use the standard combinatorial approach directly.
    // Position = distributing 15 checkers into 7 slots (6 points + borne-off).
    // Index = sum of C(partial_sum + slot_index, slot_index + 1) terms.

    // Using the "stars and bars" ranking:
    // For slots 0..6 with counts c[0]..c[5] and c[6]=15-sum(c[0..5]):
    // Index = sum_{j=0}^{5} C(c[j] + c[j+1] + ... + c[6] + (6-j) - 1, 6-j)
    //       where the sum starts from the "right" end.

    // GNUbg approach: build fBits word, then rank it.
    // The bit pattern has (CHECKERS + POINTS) bits total.
    // Set bits at positions corresponding to the running sum of checkers + separator.

    // Direct implementation of GNUbg's PositionBearoff:
    // j starts at nPoints - 1 = 5
    int j = POINTS - 1;
    for (int i = 0; i < POINTS; ++i) {
        j += checkers[i];
    }
    // j is now sum(checkers) + POINTS - 1

    // Build the index using combinatorial ranking
    // We enumerate from MSB to LSB of the fBits pattern.
    // Instead of building the bitmask, compute the rank directly.

    // Rank of combination: given n bits total, r bits set, and the
    // bit pattern, the rank is computed by scanning from high to low.
    int n = CHECKERS + POINTS;  // 21 total bits
    int r = POINTS;  // 6 bits are "separator" bits
    index = 0;

    // Position of the highest bit set (j = sum + POINTS - 1)
    // First separator is at position j
    // Then for each point i = 0..POINTS-2, the next separator is at j - checkers[i] - 1

    int pos = j;  // Current bit position (0-indexed from 0)
    for (int i = 0; i < POINTS; ++i) {
        // Separator bit at position 'pos'
        // If this bit is set at position pos, and we need r more set bits
        // out of (pos+1) remaining positions, the rank contribution is:
        // C(pos, r) counts how many patterns have a higher bit set earlier.
        // Actually, we use the standard combinatorial number system:
        // For a combination {c_1, c_2, ..., c_r} with c_1 > c_2 > ... > c_r >= 0,
        // the rank = C(c_1, r) + C(c_2, r-1) + ... + C(c_r, 1).

        // Our separator positions in descending order give us the combination elements.
        // But we need the positions in descending order.

        // Let's just use the simple iterative GNUbg approach.
        // We'll compute fBits and then rank it.
        break;
    }

    // Simpler approach: direct GNUbg-style PositionBearoff
    // Compute the combinatorial index using the standard method.

    // Restart with clean approach:
    // Think of it as choosing 6 positions (for the separators) out of 21 positions.
    // The separators are at positions p[0] > p[1] > ... > p[5] >= 0.
    //
    // p[0] = sum(checkers[0..5]) + 5  (= j from above)
    // p[1] = p[0] - checkers[0] - 1
    // p[2] = p[1] - checkers[1] - 1
    // ...
    // p[i] = p[i-1] - checkers[i-1] - 1

    // Rank = C(p[0], 6) + C(p[1], 5) + C(p[2], 4) + C(p[3], 3) + C(p[4], 2) + C(p[5], 1)

    index = 0;
    pos = POINTS - 1;
    for (int i = 0; i < POINTS; ++i) {
        pos += checkers[i];
    }
    // pos = sum(checkers) + POINTS - 1

    int k = POINTS;  // decreasing from 6 to 1
    for (int i = 0; i < POINTS; ++i) {
        index += binom(pos, k);
        pos -= checkers[i] + 1;
        --k;
    }

    return index;
}

void BearoffDB::index_to_position(unsigned int index, int checkers[POINTS]) {
    if (!binom_initialized_) init_binom();

    // Reverse of position_index. We need to recover the separator positions
    // p[0] > p[1] > ... > p[5] from the rank.
    //
    // Rank = C(p[0], 6) + C(p[1], 5) + ... + C(p[5], 1)
    //
    // To decode: for each k from 6 down to 1, find the largest p such that
    // C(p, k) <= remaining_rank, then subtract C(p, k) from the rank.
    // The checker count for point i = p[i-1] - p[i] - 1 (with p[-1] computed).

    int positions[POINTS];
    unsigned int remaining = index;

    for (int k = POINTS; k >= 1; --k) {
        // Find largest p such that C(p, k) <= remaining
        int p = k - 1;  // minimum valid p for C(p,k) to be nonzero
        while (binom(p + 1, k) <= remaining) {
            ++p;
        }
        positions[POINTS - k] = p;
        remaining -= binom(p, k);
    }

    // Recover checker counts from separator positions
    // checkers[0] = p[0] - p[1] - 1
    // checkers[i] = p[i] - p[i+1] - 1  for i < POINTS-1
    // checkers[POINTS-1] = p[POINTS-1]  (last separator position = remaining checkers on last point)
    for (int i = 0; i < POINTS - 1; ++i) {
        checkers[i] = positions[i] - positions[i + 1] - 1;
    }
    checkers[POINTS - 1] = positions[POINTS - 1];
}

unsigned int BearoffDB::board_to_player_index(const Board& board) {
    int checkers[POINTS];
    for (int i = 0; i < POINTS; ++i) {
        checkers[i] = std::max(0, board[i + 1]);  // points 1-6
    }
    return position_index(checkers);
}

unsigned int BearoffDB::board_to_opponent_index(const Board& board) {
    // Opponent's checkers on points 19-24 (from player's view) map to their points 1-6.
    // Point 24 (player's view) = opponent's point 1
    // Point 23 = opponent's point 2, etc.
    // Point 19 = opponent's point 6
    int checkers[POINTS];
    for (int i = 0; i < POINTS; ++i) {
        checkers[i] = std::max(0, -board[24 - i]);  // board[24]=opp's 1, board[23]=opp's 2, etc.
    }
    return position_index(checkers);
}

int BearoffDB::checkers_on_board(unsigned int index) const {
    int checkers[POINTS];
    index_to_position(index, checkers);
    int sum = 0;
    for (int i = 0; i < POINTS; ++i) sum += checkers[i];
    return sum;
}

// --- is_bearoff ---

bool BearoffDB::is_bearoff(const Board& board) const {
    if (!loaded_) return false;

    // Both bars must be empty
    if (board[0] != 0 || board[25] != 0) return false;

    // Player on roll: no checkers on points 7-24
    for (int i = 7; i <= 24; ++i) {
        if (board[i] > 0) return false;
    }

    // Opponent: no checkers on points 1-18
    for (int i = 1; i <= 18; ++i) {
        if (board[i] < 0) return false;
    }

    return true;
}

// --- Two-sided lookups ---

std::array<float, NUM_OUTPUTS> BearoffDB::lookup_probs(
    const Board& board, bool post_move) const
{
    unsigned int p_idx = board_to_player_index(board);
    unsigned int o_idx = board_to_opponent_index(board);

    const auto& p_dist = distributions_[p_idx];
    const auto& o_dist = distributions_[o_idx];

    constexpr float SCALE = 1.0f / 65535.0f;

    // Build opponent CDF and player CDF
    float opp_cdf[MAX_ROLLS];
    float player_cdf[MAX_ROLLS];
    {
        float cum_o = 0.0f, cum_p = 0.0f;
        for (int k = 0; k < MAX_ROLLS; ++k) {
            cum_o += o_dist[k] * SCALE;
            opp_cdf[k] = cum_o;
            cum_p += p_dist[k] * SCALE;
            player_cdf[k] = cum_p;
        }
    }

    // P(win):
    //   on-roll  (post_move=false): player rolls first.
    //     P(win) = P_p[0] + sum_{i=1} P_p[i] * (1 - CDF_o[i-1])
    //     Player bears off in i rolls, opponent has had i-1 rolls.
    //
    //   post-move (post_move=true): opponent rolls first.
    //     P(win) = sum_{i=1} P_p[i] * (1 - CDF_o[i])
    //     Player bears off in i rolls, opponent has had i rolls.
    //     Ties go to opponent (they completed their i-th roll before player).
    //     p_dist[0] contributes 0: player already off means game was over before this eval.

    float p_win = 0.0f;
    if (!post_move) {
        // On-roll: player advantage on ties
        p_win += p_dist[0] * SCALE;  // already off = instant win
        for (int i = 1; i < MAX_ROLLS; ++i) {
            float p_player_i = p_dist[i] * SCALE;
            if (p_player_i == 0.0f) continue;
            p_win += p_player_i * (1.0f - opp_cdf[i - 1]);
        }
    } else {
        // Post-move: opponent rolls first, wins ties
        for (int i = 1; i < MAX_ROLLS; ++i) {
            float p_player_i = p_dist[i] * SCALE;
            if (p_player_i == 0.0f) continue;
            p_win += p_player_i * (1.0f - opp_cdf[i]);
        }
    }

    // Gammon probabilities
    float p_gammon_win = 0.0f;
    float p_gammon_loss = 0.0f;

    const auto* opp_gammon = get_gammon_distribution(o_idx);
    const auto* player_gammon = get_gammon_distribution(p_idx);

    if (!post_move) {
        // On-roll gammons
        // P(gammon_win): player off in i rolls, opp has 0 off after i-1 rolls
        if (opp_gammon) {
            p_gammon_win += p_dist[0] * SCALE;  // instant win, opp had 0 rolls
            for (int i = 1; i < MAX_ROLLS; ++i) {
                float p_player_i = p_dist[i] * SCALE;
                if (p_player_i == 0.0f) continue;
                float opp_zero = (*opp_gammon)[i - 1] * SCALE;
                p_gammon_win += p_player_i * opp_zero;
            }
        }
        // P(gammon_loss): opp off in j rolls, player has 0 off after j rolls
        if (player_gammon) {
            for (int j = 1; j < MAX_ROLLS; ++j) {
                float p_opp_j = o_dist[j] * SCALE;
                if (p_opp_j == 0.0f) continue;
                float player_zero = (*player_gammon)[j] * SCALE;
                p_gammon_loss += p_opp_j * player_zero;
            }
        }
    } else {
        // Post-move gammons (opponent rolls first)
        // P(gammon_win): player off in i rolls, opp has 0 off after i rolls
        if (opp_gammon) {
            for (int i = 1; i < MAX_ROLLS; ++i) {
                float p_player_i = p_dist[i] * SCALE;
                if (p_player_i == 0.0f) continue;
                float opp_zero = (*opp_gammon)[i] * SCALE;
                p_gammon_win += p_player_i * opp_zero;
            }
        }
        // P(gammon_loss): opp off in j rolls, player has 0 off after j-1 rolls
        if (player_gammon) {
            for (int j = 1; j < MAX_ROLLS; ++j) {
                float p_opp_j = o_dist[j] * SCALE;
                if (p_opp_j == 0.0f) continue;
                float player_zero = (j - 1 >= 0 && j - 1 < MAX_ROLLS)
                    ? (*player_gammon)[j - 1] * SCALE : 0.0f;
                p_gammon_loss += p_opp_j * player_zero;
            }
        }
    }

    return {p_win, p_gammon_win, 0.0f, p_gammon_loss, 0.0f};
}

float BearoffDB::lookup_epc(const Board& board, int player) const {
    unsigned int idx = (player == 0)
        ? board_to_player_index(board)
        : board_to_opponent_index(board);
    return mean_rolls_[idx] * (49.0f / 6.0f);
}

const std::array<uint16_t, BearoffDB::MAX_ROLLS>*
BearoffDB::get_gammon_distribution(unsigned int index) const {
    int gi = gammon_index_[index];
    if (gi < 0) return nullptr;
    return &gammon_distributions_[gi];
}

// --- Generation ---

// Helper: convert one-sided checkers to a full board for move generation.
// Places checkers on points 1-6 as player 1 (positive values).
// All other points, bar slots are 0.
static Board checkers_to_board(const int checkers[BearoffDB::POINTS]) {
    Board b{};
    for (int i = 0; i < BearoffDB::POINTS; ++i) {
        b[i + 1] = checkers[i];
    }
    return b;
}

// Helper: extract one-sided checkers from a board after a move.
static void board_to_checkers(const Board& board, int checkers[BearoffDB::POINTS]) {
    for (int i = 0; i < BearoffDB::POINTS; ++i) {
        checkers[i] = std::max(0, board[i + 1]);
    }
}

// Helper: total pips for one-sided checkers
static int total_pips(const int checkers[BearoffDB::POINTS]) {
    int pips = 0;
    for (int i = 0; i < BearoffDB::POINTS; ++i) {
        pips += checkers[i] * (i + 1);
    }
    return pips;
}

// Helper: total checkers count
static int total_checkers(const int checkers[BearoffDB::POINTS]) {
    int sum = 0;
    for (int i = 0; i < BearoffDB::POINTS; ++i) {
        sum += checkers[i];
    }
    return sum;
}

void BearoffDB::generate() {
    if (!binom_initialized_) init_binom();

    // Build sorted order: positions with fewer pips first (dependencies computed first).
    // Position 0 (all checkers off) has 0 pips and is the terminal.
    struct PosOrder {
        unsigned int index;
        int pips;
    };
    std::vector<PosOrder> order(N_POSITIONS);

    for (unsigned int idx = 0; idx < N_POSITIONS; ++idx) {
        int checkers[POINTS];
        index_to_position(idx, checkers);
        order[idx] = {idx, total_pips(checkers)};
    }

    std::sort(order.begin(), order.end(),
              [](const PosOrder& a, const PosOrder& b) { return a.pips < b.pips; });

    // Initialize terminal position: all checkers off (index for [0,0,0,0,0,0])
    int zero_checkers[POINTS] = {};
    unsigned int terminal_idx = position_index(zero_checkers);
    distributions_[terminal_idx].fill(0);
    distributions_[terminal_idx][0] = 65535;  // 100% borne off in 0 rolls
    mean_rolls_[terminal_idx] = 0.0f;

    // Mark which positions are computed
    std::vector<bool> computed(N_POSITIONS, false);
    computed[terminal_idx] = true;

    // Move generation buffer (reused)
    std::vector<Board> move_buf;

    // 21 unique dice rolls
    struct DiceRoll { int d1, d2, weight; };
    static const DiceRoll rolls[] = {
        {1,1,1}, {2,2,1}, {3,3,1}, {4,4,1}, {5,5,1}, {6,6,1},  // doubles
        {1,2,2}, {1,3,2}, {1,4,2}, {1,5,2}, {1,6,2},
        {2,3,2}, {2,4,2}, {2,5,2}, {2,6,2},
        {3,4,2}, {3,5,2}, {3,6,2},
        {4,5,2}, {4,6,2},
        {5,6,2}
    };

    // Process positions in pip-count order
    for (const auto& po : order) {
        if (computed[po.index]) continue;

        int checkers[POINTS];
        index_to_position(po.index, checkers);

        // Check: is this really reachable? (sum > 0, since terminal already handled)
        int ncheck = total_checkers(checkers);
        if (ncheck == 0) {
            computed[po.index] = true;
            continue;
        }

        // Build full board for move generation
        Board board = checkers_to_board(checkers);

        // Accumulate distribution as doubles (for precision before quantization)
        double acc_dist[MAX_ROLLS] = {};
        double total_weight = 0.0;

        for (const auto& roll : rolls) {
            possible_boards(board, roll.d1, roll.d2, move_buf);

            // Find the best move: minimize mean rolls to bear off
            float best_mean = 1e9f;
            int best_move_idx = -1;

            for (int m = 0; m < static_cast<int>(move_buf.size()); ++m) {
                int next_checkers[POINTS];
                board_to_checkers(move_buf[m], next_checkers);
                unsigned int next_idx = position_index(next_checkers);
                assert(computed[next_idx]);  // Should be computed already (fewer pips)
                float next_mean = mean_rolls_[next_idx];
                if (next_mean < best_mean) {
                    best_mean = next_mean;
                    best_move_idx = m;
                }
            }

            // Get the distribution of the best move's resulting position
            int best_checkers[POINTS];
            board_to_checkers(move_buf[best_move_idx], best_checkers);
            unsigned int best_idx = position_index(best_checkers);
            const auto& next_dist = distributions_[best_idx];

            // Accumulate: shift distribution by 1 roll and weight by dice probability
            for (int k = 0; k < MAX_ROLLS - 1; ++k) {
                acc_dist[k + 1] += static_cast<double>(next_dist[k]) * roll.weight;
            }
            total_weight += roll.weight;
        }

        // Normalize to sum to 65535
        // total_weight should be 36
        double scale = 65535.0 / (total_weight * 65535.0);  // = 1/36
        double sum_check = 0.0;
        for (int k = 0; k < MAX_ROLLS; ++k) {
            double val = acc_dist[k] * scale * 65535.0;
            // acc_dist[k] is sum of (uint16 * weight), total_weight = 36
            // We want: result[k] = sum(dist[k] * weight) / total_weight
            // In uint16 space: result[k] = round(sum(dist[k] * weight) / 36)
            distributions_[po.index][k] = 0;
        }

        // Redo the normalization more carefully
        // acc_dist[k] = sum over rolls of (next_dist_shifted[k] * weight)
        // where next_dist values are in [0, 65535] and weights sum to 36.
        // Result = acc_dist[k] / 36, rounded to uint16, with adjustment for sum=65535.
        double frac_dist[MAX_ROLLS];
        double total_frac = 0.0;
        for (int k = 0; k < MAX_ROLLS; ++k) {
            frac_dist[k] = acc_dist[k] / (total_weight * 65535.0);
            total_frac += frac_dist[k];
        }

        // Quantize to uint16, summing to 65535
        int32_t quant[MAX_ROLLS];
        int32_t quant_sum = 0;
        int max_idx = 0;
        double max_frac = 0.0;
        for (int k = 0; k < MAX_ROLLS; ++k) {
            double val = frac_dist[k] * 65535.0;
            quant[k] = static_cast<int32_t>(val + 0.5);
            quant_sum += quant[k];
            if (frac_dist[k] > max_frac) {
                max_frac = frac_dist[k];
                max_idx = k;
            }
        }
        // Adjust mode to fix rounding error
        quant[max_idx] += (65535 - quant_sum);

        for (int k = 0; k < MAX_ROLLS; ++k) {
            distributions_[po.index][k] = static_cast<uint16_t>(
                std::max(0, std::min(65535, quant[k])));
        }

        // Compute mean rolls
        double mean = 0.0;
        for (int k = 0; k < MAX_ROLLS; ++k) {
            mean += k * (distributions_[po.index][k] / 65535.0);
        }
        mean_rolls_[po.index] = static_cast<float>(mean);

        computed[po.index] = true;
    }

    // Generate gammon distributions
    generate_gammon_distributions();

    loaded_ = true;
}

void BearoffDB::generate_gammon_distributions() {
    // Gammon distribution: for positions with all 15 checkers on board,
    // track P(0 checkers borne off after k rolls of optimal play).
    //
    // zero_off[0] = 1.0 (no rolls taken, no checkers off)
    // For each dice roll, find the optimal move (same move that minimizes total rolls).
    // If the optimal move bears off >= 1 checker: contributes 0 to zero_off[k+1..].
    // If it doesn't bear off any: inherits next_position's zero_off shifted by 1.

    // Build gammon index: map position indices with 15 checkers → gammon array index
    gammon_index_.fill(-1);
    int gammon_count = 0;
    for (unsigned int idx = 0; idx < N_POSITIONS; ++idx) {
        int checkers[POINTS];
        index_to_position(idx, checkers);
        if (total_checkers(checkers) == CHECKERS) {
            gammon_index_[idx] = gammon_count++;
        }
    }
    gammon_distributions_.resize(gammon_count);

    // Sort by pip count ascending for dependency ordering
    struct PosOrder { unsigned int index; int pips; };
    std::vector<PosOrder> order;
    order.reserve(gammon_count);
    for (unsigned int idx = 0; idx < N_POSITIONS; ++idx) {
        if (gammon_index_[idx] >= 0) {
            int checkers[POINTS];
            index_to_position(idx, checkers);
            order.push_back({idx, total_pips(checkers)});
        }
    }
    std::sort(order.begin(), order.end(),
              [](const PosOrder& a, const PosOrder& b) { return a.pips < b.pips; });

    std::vector<Board> move_buf;

    static const struct { int d1, d2, weight; } rolls[] = {
        {1,1,1}, {2,2,1}, {3,3,1}, {4,4,1}, {5,5,1}, {6,6,1},
        {1,2,2}, {1,3,2}, {1,4,2}, {1,5,2}, {1,6,2},
        {2,3,2}, {2,4,2}, {2,5,2}, {2,6,2},
        {3,4,2}, {3,5,2}, {3,6,2},
        {4,5,2}, {4,6,2},
        {5,6,2}
    };

    for (const auto& po : order) {
        int gi = gammon_index_[po.index];
        auto& zero_off = gammon_distributions_[gi];
        zero_off.fill(0);
        zero_off[0] = 65535;  // P(0 off after 0 rolls) = 1.0

        int checkers[POINTS];
        index_to_position(po.index, checkers);
        Board board = checkers_to_board(checkers);

        double acc[MAX_ROLLS] = {};

        for (const auto& roll : rolls) {
            possible_boards(board, roll.d1, roll.d2, move_buf);

            // Find best move (same criterion as main generation: minimize mean rolls)
            float best_mean = 1e9f;
            int best_move_idx = 0;
            for (int m = 0; m < static_cast<int>(move_buf.size()); ++m) {
                int next_checkers[POINTS];
                board_to_checkers(move_buf[m], next_checkers);
                unsigned int next_idx = position_index(next_checkers);
                if (mean_rolls_[next_idx] < best_mean) {
                    best_mean = mean_rolls_[next_idx];
                    best_move_idx = m;
                }
            }

            // Check if best move bears off any checker
            int next_checkers[POINTS];
            board_to_checkers(move_buf[best_move_idx], next_checkers);
            int next_total = total_checkers(next_checkers);

            if (next_total < CHECKERS) {
                // A checker was borne off — zero_off contribution is 0 for all k >= 1
                // (nothing to accumulate)
            } else {
                // No checker borne off — inherit next position's zero_off, shifted by 1
                unsigned int next_idx = position_index(next_checkers);
                int next_gi = gammon_index_[next_idx];
                assert(next_gi >= 0);  // Must be a 15-checker position
                const auto& next_zero = gammon_distributions_[next_gi];
                for (int k = 0; k < MAX_ROLLS - 1; ++k) {
                    acc[k + 1] += static_cast<double>(next_zero[k]) * roll.weight;
                }
            }
        }

        // Normalize and quantize
        double frac[MAX_ROLLS];
        double total_frac = 0.0;
        frac[0] = 1.0;  // zero_off[0] is always 1.0
        total_frac = 1.0;
        for (int k = 1; k < MAX_ROLLS; ++k) {
            frac[k] = acc[k] / (36.0 * 65535.0);
            total_frac += frac[k];
        }

        // Quantize each value independently (no need to sum to 65535,
        // this is not a probability distribution that sums to 1)
        zero_off[0] = 65535;
        for (int k = 1; k < MAX_ROLLS; ++k) {
            double val = frac[k] * 65535.0;
            zero_off[k] = static_cast<uint16_t>(
                std::max(0.0, std::min(65535.0, val + 0.5)));
        }
    }
}

// --- Save/Load ---

bool BearoffDB::save(const std::string& path) const {
    if (!loaded_) return false;

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    // Header
    out.write(reinterpret_cast<const char*>(&MAGIC), 4);
    out.write(reinterpret_cast<const char*>(&VERSION), 4);
    uint32_t points = POINTS;
    uint32_t checkers = CHECKERS;
    uint32_t max_rolls = MAX_ROLLS;
    uint32_t n_positions = N_POSITIONS;
    out.write(reinterpret_cast<const char*>(&points), 4);
    out.write(reinterpret_cast<const char*>(&checkers), 4);
    out.write(reinterpret_cast<const char*>(&max_rolls), 4);
    out.write(reinterpret_cast<const char*>(&n_positions), 4);

    // Number of gammon distributions
    uint32_t n_gammon = static_cast<uint32_t>(gammon_distributions_.size());
    out.write(reinterpret_cast<const char*>(&n_gammon), 4);

    // Distributions: N_POSITIONS × MAX_ROLLS × uint16
    out.write(reinterpret_cast<const char*>(distributions_.data()),
              N_POSITIONS * MAX_ROLLS * sizeof(uint16_t));

    // Mean rolls: N_POSITIONS × float32
    out.write(reinterpret_cast<const char*>(mean_rolls_.data()),
              N_POSITIONS * sizeof(float));

    // Gammon index: N_POSITIONS × int32
    out.write(reinterpret_cast<const char*>(gammon_index_.data()),
              N_POSITIONS * sizeof(int));

    // Gammon distributions: n_gammon × MAX_ROLLS × uint16
    if (n_gammon > 0) {
        out.write(reinterpret_cast<const char*>(gammon_distributions_.data()),
                  n_gammon * MAX_ROLLS * sizeof(uint16_t));
    }

    return out.good();
}

bool BearoffDB::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    uint32_t magic, version, points, checkers_val, max_rolls, n_positions, n_gammon;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&points), 4);
    in.read(reinterpret_cast<char*>(&checkers_val), 4);
    in.read(reinterpret_cast<char*>(&max_rolls), 4);
    in.read(reinterpret_cast<char*>(&n_positions), 4);
    in.read(reinterpret_cast<char*>(&n_gammon), 4);

    if (magic != MAGIC || version != VERSION) return false;
    if (points != POINTS || checkers_val != CHECKERS) return false;
    if (max_rolls != MAX_ROLLS || n_positions != N_POSITIONS) return false;

    in.read(reinterpret_cast<char*>(distributions_.data()),
            N_POSITIONS * MAX_ROLLS * sizeof(uint16_t));

    in.read(reinterpret_cast<char*>(mean_rolls_.data()),
            N_POSITIONS * sizeof(float));

    in.read(reinterpret_cast<char*>(gammon_index_.data()),
            N_POSITIONS * sizeof(int));

    gammon_distributions_.resize(n_gammon);
    if (n_gammon > 0) {
        in.read(reinterpret_cast<char*>(gammon_distributions_.data()),
                n_gammon * MAX_ROLLS * sizeof(uint16_t));
    }

    if (!binom_initialized_) init_binom();
    loaded_ = in.good();
    return loaded_;
}

// --- BearoffStrategy ---

BearoffStrategy::BearoffStrategy(std::shared_ptr<Strategy> base, const BearoffDB* db)
    : base_(std::move(base)), db_(db) {}

double BearoffStrategy::evaluate(const Board& board, bool pre_move_is_race) const {
    if (db_->is_bearoff(board)) {
        // post_move=true: Strategy evaluates post-move boards (opponent rolls next)
        auto probs = db_->lookup_probs(board, /*post_move=*/true);
        return NeuralNetwork::compute_equity(probs);
    }
    return base_->evaluate(board, pre_move_is_race);
}

std::array<float, NUM_OUTPUTS> BearoffStrategy::evaluate_probs(
    const Board& board, bool pre_move_is_race) const
{
    if (db_->is_bearoff(board)) {
        return db_->lookup_probs(board, /*post_move=*/true);
    }
    return base_->evaluate_probs(board, pre_move_is_race);
}

std::array<float, NUM_OUTPUTS> BearoffStrategy::evaluate_probs(
    const Board& board, const Board& pre_move_board) const
{
    if (db_->is_bearoff(board)) {
        return db_->lookup_probs(board, /*post_move=*/true);
    }
    return base_->evaluate_probs(board, pre_move_board);
}

int BearoffStrategy::best_move_index(const std::vector<Board>& candidates,
                                      bool pre_move_is_race) const
{
    // Check if all candidates are bearoff positions — if so, use DB for ranking.
    // Candidates are post-move boards, so use post_move=true.
    bool all_bearoff = true;
    for (const auto& c : candidates) {
        if (!db_->is_bearoff(c)) {
            all_bearoff = false;
            break;
        }
    }

    if (all_bearoff && !candidates.empty()) {
        int best = 0;
        double best_eq = -1e9;
        for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
            auto probs = db_->lookup_probs(candidates[i], /*post_move=*/true);
            double eq = NeuralNetwork::compute_equity(probs);
            if (eq > best_eq) {
                best_eq = eq;
                best = i;
            }
        }
        return best;
    }

    return base_->best_move_index(candidates, pre_move_is_race);
}

int BearoffStrategy::best_move_index(const std::vector<Board>& candidates,
                                      const Board& pre_move_board) const
{
    bool all_bearoff = true;
    for (const auto& c : candidates) {
        if (!db_->is_bearoff(c)) {
            all_bearoff = false;
            break;
        }
    }

    if (all_bearoff && !candidates.empty()) {
        int best = 0;
        double best_eq = -1e9;
        for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
            auto probs = db_->lookup_probs(candidates[i], /*post_move=*/true);
            double eq = NeuralNetwork::compute_equity(probs);
            if (eq > best_eq) {
                best_eq = eq;
                best = i;
            }
        }
        return best;
    }

    return base_->best_move_index(candidates, pre_move_board);
}

} // namespace bgbot
