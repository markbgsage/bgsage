// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/encoding.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include "bgbot/board.h"
#include <algorithm>
#include <cmath>
#include <limits>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cstring>
#include <mutex>
#include <numeric>
#include <utility>

namespace bgbot {

// ======================== Escape Lookup Tables (GNUbg-style) ========================
// Precomputed tables mapping a 12-bit bitmap of blocked points to escape roll counts.
// anPoint: maps checker count to "is blocked" (>=2 means blocked = 1).
static const int anPoint[16] = {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

static int anEscapes[0x1000];
static int anEscapes1[0x1000];
static uint8_t back_escape_counts[6][4096];
static int entry_losses[2][64];
static std::once_flag escape_tables_init_flag;

static void compute_escape_table0() {
    for (int i = 0; i < 0x1000; i++) {
        int c = 0;
        for (int n0 = 0; n0 <= 5; n0++)
            for (int n1 = 0; n1 <= n0; n1++)
                if (!(i & (1 << (n0 + n1 + 1))) &&
                    !((i & (1 << n0)) && (i & (1 << n1))))
                    c += (n0 == n1) ? 1 : 2;
        anEscapes[i] = c;
    }
}

static void compute_escape_table1() {
    anEscapes1[0] = 0;
    for (int i = 1; i < 0x1000; i++) {
        int c = 0;
        int low = 0;
        while (!(i & (1 << low))) ++low;

        for (int n0 = 0; n0 <= 5; n0++)
            for (int n1 = 0; n1 <= n0; n1++)
                if ((n0 + n1 + 1 > low) &&
                    !(i & (1 << (n0 + n1 + 1))) &&
                    !((i & (1 << n0)) && (i & (1 << n1))))
                    c += (n0 == n1) ? 1 : 2;
        anEscapes1[i] = c;
    }
}

void init_escape_tables() {
    std::call_once(escape_tables_init_flag, []() {
        compute_escape_table0();
        compute_escape_table1();
        // The trained back-escape feature tests the 12 points immediately
        // in front of the back checker. A longer double already escapes
        // by its second step, so no farther blocker can change its result.
        for (int back = 19; back <= 24; ++back) {
            Board board{};
            board[back] = 1;
            for (unsigned mask = 0; mask < 4096; ++mask) {
                for (int bit = 0; bit < 12; ++bit)
                    board[back - 12 + bit] = (mask & (1u << bit)) ? -2 : 0;
                back_escape_counts[back - 19][mask] = static_cast<uint8_t>(back_escapes(board));
            }
        }
        for (int bar = 1; bar <= 2; ++bar) {
            Board board{};
            board[25] = bar;
            for (unsigned mask = 0; mask < 64; ++mask) {
                for (int bit = 0; bit < 6; ++bit)
                    board[24 - bit] = (mask & (1u << bit)) ? -2 : 0;
                entry_losses[bar - 1][mask] = enter_loss(board);
            }
        }
    });
}

// Escapes: count rolls that let a checker at position n (0-indexed, GNUbg style)
// escape, given half_board[0..24] (one player's checkers, unsigned counts).
// half_board[k] = number of checkers at position k (0=ace, 23=opponent's ace, 24=bar).
// A position with >=2 checkers blocks escape through that point.
int escapes(const int half_board[25], int n) {
    init_escape_tables();
    int af = 0;
    int m = (n < 12) ? n : 12;
    for (int i = 0; i < m; i++) {
        int idx = 24 + i - n;
        if (idx >= 0 && idx < 25)
            af |= (anPoint[half_board[idx] < 0 ? 0 : (half_board[idx] > 15 ? 15 : half_board[idx])] << i);
    }
    return anEscapes[af];
}

int escapes1(const int half_board[25], int n) {
    init_escape_tables();
    int af = 0;
    int m = (n < 12) ? n : 12;
    for (int i = 0; i < m; i++) {
        int idx = 24 + i - n;
        if (idx >= 0 && idx < 25)
            af |= (anPoint[half_board[idx] < 0 ? 0 : (half_board[idx] > 15 ? 15 : half_board[idx])] << i);
    }
    return anEscapes1[af];
}

// Helper: extract player's half-board (GNUbg format) from our Board.
// GNUbg anBoard[k] = player's checkers at position k, where 0 = ace point.
// Our board[k+1] has player checkers (positive values).
// Returns: hb[0..23] = player checkers on points 1-24, hb[24] = bar.
static void extract_player_half(const Board& board, int hb[25]) {
    for (int k = 0; k < 24; k++) {
        hb[k] = (board[k + 1] > 0) ? board[k + 1] : 0;
    }
    hb[24] = board[25];  // bar
}

// Helper: extract opponent's half-board (GNUbg format) from our Board.
// GNUbg anBoardOpp[k] = opponent's checkers at position k from OPPONENT's perspective.
// Our board has opponent checkers as negative values. Opponent's position k from their
// perspective corresponds to our point 25-k-1 = 24-k.
// So: hb[k] = abs(board[24-k]) if board[24-k] < 0, else 0.
static void extract_opponent_half(const Board& board, int hb[25]) {
    for (int k = 0; k < 24; k++) {
        int pt = 24 - k;
        hb[k] = (board[pt] < 0) ? -board[pt] : 0;
    }
    hb[24] = board[0];  // opponent's bar
}

namespace {
// Point encoding is: [n>=1, n>=2, n>=3, (n-3)/2] with hard cap at 15.
// Precomputing avoids hot-path branches in input encoding loops.
static constexpr float k_point_features[16][4] = {
    {0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 1.0f, 0.0f, 0.0f},
    {1.0f, 1.0f, 1.0f, 0.0f},
    {1.0f, 1.0f, 1.0f, 0.5f},
    {1.0f, 1.0f, 1.0f, 1.0f},
    {1.0f, 1.0f, 1.0f, 1.5f},
    {1.0f, 1.0f, 1.0f, 2.0f},
    {1.0f, 1.0f, 1.0f, 2.5f},
    {1.0f, 1.0f, 1.0f, 3.0f},
    {1.0f, 1.0f, 1.0f, 3.5f},
    {1.0f, 1.0f, 1.0f, 4.0f},
    {1.0f, 1.0f, 1.0f, 4.5f},
    {1.0f, 1.0f, 1.0f, 5.0f},
    {1.0f, 1.0f, 1.0f, 5.5f},
    {1.0f, 1.0f, 1.0f, 6.0f},
};

static inline const float* point_feature_row(int n) {
    if (n < 0) return k_point_features[0];
    if (n > 15) return k_point_features[15];
    return k_point_features[n];
}
} // namespace

// ======================== Tesauro 196-input encoding ========================

std::array<float, TESAURO_INPUTS> compute_tesauro_inputs(const Board& board) {
    std::array<float, TESAURO_INPUTS> inputs;

    for (int i = 0; i < 24; ++i) {
        int n = board[i + 1];  // point i+1; positive = player 1, negative = player 2

        const float* p_feat = point_feature_row(n);
        const int p0 = 4 * i;
        inputs[p0 + 0] = p_feat[0];
        inputs[p0 + 1] = p_feat[1];
        inputs[p0 + 2] = p_feat[2];
        inputs[p0 + 3] = p_feat[3];

        // Player 2 checkers (negative n, use absolute value)
        int m = -n;  // m > 0 when player 2 has checkers here
        const float* m_feat = point_feature_row(m);
        const int p1 = p0 + 98;
        inputs[p1 + 0] = m_feat[0];
        inputs[p1 + 1] = m_feat[1];
        inputs[p1 + 2] = m_feat[2];
        inputs[p1 + 3] = m_feat[3];
    }

    // Bar inputs
    inputs[96]  = board[25] / 2.0f;   // player 1 bar
    inputs[194] = board[0]  / 2.0f;   // player 2 bar

    // Borne-off inputs
    inputs[97]  = player_borne_off(board)   / 15.0f;
    inputs[195] = opponent_borne_off(board) / 15.0f;

    return inputs;
}

// ======================== Helper functions for extended encoding ========================

BarExitProbs prob_no_enter_from_bar(const Board& board) {
    // Count opponent's anchors in their home board (points 19-24 for player, points 1-6 for opp)
    // Player's bar entry is blocked by opponent anchors in points 19-24
    int opp_anchors_in_home = 0;
    for (int i = 19; i <= 24; ++i) {
        if (board[i] < -1) opp_anchors_in_home++;
    }
    float player_prob = (float)(opp_anchors_in_home * opp_anchors_in_home) / 36.0f;

    // Opponent's bar entry is blocked by player anchors in points 1-6
    int player_anchors_in_home = 0;
    for (int i = 1; i <= 6; ++i) {
        if (board[i] > 1) player_anchors_in_home++;
    }
    float opp_prob = (float)(player_anchors_in_home * player_anchors_in_home) / 36.0f;

    return {player_prob, opp_prob};
}

ForwardAnchorPoints forward_anchor_points(const Board& board) {
    // Player's most forward anchor on opponent's side (points 13-24 from player's view)
    // Returned as point number from opponent's perspective: 25-i
    int player_fwd = 0;
    for (int i = 13; i <= 24; ++i) {
        if (board[i] > 1) {
            player_fwd = 25 - i;  // from opponent's perspective
            break;
        }
    }

    // Opponent's most forward anchor on player's side (points 1-12 from player's view)
    // Scan from 12 down to find the most forward (closest to player's home)
    int opp_fwd = 0;
    for (int i = 12; i >= 1; --i) {
        if (board[i] < -1) {
            opp_fwd = i;
            break;
        }
    }

    return {player_fwd, opp_fwd};
}

// ======================== Hitting Shots ========================

// Lookup table for long-distance hitting rolls.
// For distance d (7-24), lists the (d1, d2) pairs (d1 >= d2) that could hit.
struct HittingShotsEntry {
    int count;
    std::pair<int,int> rolls[4];
};

static constexpr HittingShotsEntry HITTING_SHOTS_TABLE[] = {
    // distance 0-6: not used (handled differently)
    {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}},
    // distance 7
    {3, {{6,1}, {5,2}, {4,3}}},
    // distance 8
    {4, {{6,2}, {5,3}, {4,4}, {2,2}}},
    // distance 9
    {3, {{6,3}, {5,4}, {3,3}}},
    // distance 10
    {2, {{6,4}, {5,5}}},
    // distance 11
    {1, {{6,5}}},
    // distance 12
    {3, {{6,6}, {4,4}, {3,3}}},
    // distance 13
    {0, {}},
    // distance 14
    {0, {}},
    // distance 15
    {1, {{5,5}}},
    // distance 16
    {1, {{4,4}}},
    // distance 17
    {0, {}},
    // distance 18
    {1, {{6,6}}},
    // distance 19
    {0, {}},
    // distance 20
    {1, {{5,5}}},
    // distance 21
    {0, {}},
    // distance 22
    {0, {}},
    // distance 23
    {0, {}},
    // distance 24
    {1, {{6,6}}},
};

// Check if a roll (d1, d2) can hit target from source for doubles
static bool hits_double(const Board& board, int d, int target, int source) {
    // For doubles: check if intermediate points are blocked by opponent anchors
    for (int step = 0; step < 4; ++step) {
        int pt = target + d * step;
        if (pt <= source && board[pt] < -1) {
            return false;
        }
    }
    return true;
}

// Check if a non-double roll can hit target from source
static bool hits_regular(const Board& board, int d1, int d2, int target) {
    // Can hit if at least one intermediate landing point (target+d1 or target+d2) is not blocked
    return board[target + d1] >= -1 || board[target + d2] >= -1;
}

// Encode a roll pair as a unique index for deduplication.
// We represent (d1, d2) with d1 >= d2 as d1*7+d2. Range: 0..48.
// Doubles take 1 slot, non-doubles take 2 (we track both orderings via the set).
// Use a 64-bit bitmask for dedup — 49 possible values easily fits.
static inline int roll_key(int d1, int d2) {
    return (d1 > d2) ? d1 * 7 + d2 : d2 * 7 + d1;
}

int hitting_shots(const Board& board) {
    // Use a bitmask to track unique rolls that hit.
    // Keys: d_high * 7 + d_low, max = 6*7+6 = 48, so 49 bits needed → uint64_t.
    uint64_t rolls_mask = 0;

    for (int i = 1; i <= 24; ++i) {
        if (board[i] != -1) continue;  // need exactly one opponent checker (a blot)

        // Look for player checkers that could hit this blot
        for (int j = i + 1; j <= 24; ++j) {
            if (board[j] <= 0) continue;  // need a player checker

            int distance = j - i;

            if (distance <= 6) {
                // Any roll containing this distance hits
                for (int k = 1; k <= 6; ++k) {
                    int hi = (distance > k) ? distance : k;
                    int lo = (distance < k) ? distance : k;
                    rolls_mask |= (1ULL << (hi * 7 + lo));
                }

                // Check short doubles that can reach via multiple steps
                if (distance == 2 && board[i + 1] > -2) {
                    rolls_mask |= (1ULL << (1 * 7 + 1));  // 1-1
                }
                if (distance == 3 && board[i + 1] > -2 && board[i + 2] > -2) {
                    rolls_mask |= (1ULL << (1 * 7 + 1));  // 1-1
                }
                if (distance == 4 && board[i + 1] > -2 && board[i + 2] > -2 && board[i + 3] > -2) {
                    rolls_mask |= (1ULL << (1 * 7 + 1));  // 1-1
                }
                if (distance == 4 && board[i + 2] > -2) {
                    rolls_mask |= (1ULL << (2 * 7 + 2));  // 2-2
                }
                if (distance == 6 && board[i + 2] > -2 && board[i + 4] > -2) {
                    rolls_mask |= (1ULL << (2 * 7 + 2));  // 2-2
                }
                if (distance == 6 && board[i + 3] > -2) {
                    rolls_mask |= (1ULL << (3 * 7 + 3));  // 3-3
                }

                // Check shorter combination rolls
                if (distance == 3 && (board[i + 1] > -2 || board[i + 2] > -2)) {
                    rolls_mask |= (1ULL << (2 * 7 + 1));  // 2-1
                }
                if (distance == 4 && (board[i + 1] > -2 || board[i + 3] > -2)) {
                    rolls_mask |= (1ULL << (3 * 7 + 1));  // 3-1
                }
                if (distance == 5 && (board[i + 1] > -2 || board[i + 4] > -2)) {
                    rolls_mask |= (1ULL << (4 * 7 + 1));  // 4-1
                }
                if (distance == 5 && (board[i + 2] > -2 || board[i + 3] > -2)) {
                    rolls_mask |= (1ULL << (3 * 7 + 2));  // 3-2
                }
                if (distance == 6 && (board[i + 1] > -2 || board[i + 5] > -2)) {
                    rolls_mask |= (1ULL << (5 * 7 + 1));  // 5-1
                }
                if (distance == 6 && (board[i + 2] > -2 || board[i + 4] > -2)) {
                    rolls_mask |= (1ULL << (4 * 7 + 2));  // 4-2
                }
            } else if (distance <= 24) {
                // Long-distance hits: look up possible rolls
                const auto& entry = HITTING_SHOTS_TABLE[distance];
                for (int r = 0; r < entry.count; ++r) {
                    int d1 = entry.rolls[r].first;
                    int d2 = entry.rolls[r].second;
                    bool can_hit;
                    if (d1 == d2) {
                        can_hit = hits_double(board, d1, i, j);
                    } else {
                        can_hit = hits_regular(board, d1, d2, i);
                    }
                    if (can_hit) {
                        rolls_mask |= (1ULL << (d1 * 7 + d2));
                    }
                }
            }
        }
    }

    // Count rolls: doubles count as 1, non-doubles count as 2
    int total = 0;
    for (int d1 = 1; d1 <= 6; ++d1) {
        for (int d2 = 1; d2 <= d1; ++d2) {
            if (rolls_mask & (1ULL << (d1 * 7 + d2))) {
                total += (d1 == d2) ? 1 : 2;
            }
        }
    }
    return total;
}

// ======================== Double Hitting Shots ========================

// Distances are bounded by the board. The same pair occurs repeatedly across
// sibling moves, so compute its double and required steps at compile time.
struct DoubleHitPath { unsigned char die, first_steps, second_steps; };
static constexpr auto double_hit_paths = [] {
    std::array<std::array<DoubleHitPath, 25>, 25> paths{};
    for (int d1 = 1; d1 <= 24; ++d1) {
        for (int d2 = 1; d2 <= 24; ++d2) {
            int die = std::gcd(d1, d2);
            int first = d1 / die, second = d2 / die;
            if (die <= 6 && first + second <= 4)
                paths[d1][d2] = {static_cast<unsigned char>(die),
                                 static_cast<unsigned char>(first), static_cast<unsigned char>(second)};
        }
    }
    return paths;
}();

int double_hitting_shots(const Board& board) {
    // Find all opponent blots and the distances from which player checkers can hit them
    struct BlotInfo {
        int point;          // board point with opponent blot
        int dists[24];      // distances of player checkers that can hit this blot
        int n_dists;
    };
    BlotInfo blots[24];     // max 24 opponent blots
    int n_blots = 0;

    for (int i = 1; i <= 24; ++i) {
        if (board[i] != -1) continue;

        blots[n_blots].point = i;
        blots[n_blots].n_dists = 0;

        for (int d = 1; d <= 24; ++d) {
            if (i + d < 26 && board[i + d] > 0) {  // includes bar(25) if player checker there
                blots[n_blots].dists[blots[n_blots].n_dists++] = d;
            }
        }
        n_blots++;
    }

    // Use bitmask for dedup: key = d1*7+d2 for d1,d2 in 1..6 (both orderings for non-doubles)
    // Need 49 bits → uint64_t
    uint64_t pairs_mask = 0;

    // Case 1: Two different player checkers hit two different blots
    for (int bi = 0; bi < n_blots; ++bi) {
        for (int hi = 0; hi < blots[bi].n_dists; ++hi) {
            int d1 = blots[bi].dists[hi];
            for (int bk = bi + 1; bk < n_blots; ++bk) {
                for (int hk = 0; hk < blots[bk].n_dists; ++hk) {
                    int d2 = blots[bk].dists[hk];

                    // Both must be single die rolls (<=6) for non-double case
                    if (d1 < 7 && d2 < 7) {
                        // Check that the two hits use different source checkers,
                        // or the source point has more than one checker
                        if (blots[bi].point + d1 != blots[bk].point + d2 ||
                            board[blots[bi].point + d1] > 1) {
                            pairs_mask |= (1ULL << (d1 * 7 + d2));
                            if (d1 != d2) {
                                pairs_mask |= (1ULL << (d2 * 7 + d1));
                            }
                        }
                    }

                    // Check doubles: if gcd divides both distances evenly and
                    // total steps < 5, it's possible with a double
                    const auto path = double_hit_paths[d1][d2];
                    if (path.die) {
                        int g = path.die, n1 = path.first_steps, n2 = path.second_steps;
                        // Check intermediate points aren't blocked
                        bool blocked = false;
                        for (int m = 1; m < n1; ++m) {
                            int pt = blots[bi].point + g * m;
                            if (pt < 25 && board[pt] < -1) {
                                blocked = true;
                                break;
                            }
                        }
                        if (!blocked) {
                            for (int m = 1; m < n2; ++m) {
                                int pt = blots[bk].point + g * m;
                                if (pt < 25 && board[pt] < -1) {
                                    blocked = true;
                                    break;
                                }
                            }
                        }
                        if (!blocked) {
                            pairs_mask |= (1ULL << (g * 7 + g));
                        }
                    }
                }
            }
        }
    }

    // Case 2: One checker steps through a blot to hit another blot beyond
    for (int bi = 0; bi < n_blots; ++bi) {
        int pnt = blots[bi].point;

        for (int hi = 0; hi < blots[bi].n_dists; ++hi) {
            int d1 = blots[bi].dists[hi];
            if (d1 > 6) continue;

            // Non-double: check if there's a blot d2 below this blot
            for (int d2 = 1; d2 <= 6; ++d2) {
                if (pnt - d2 > 0 && board[pnt - d2] == -1) {
                    pairs_mask |= (1ULL << (d1 * 7 + d2));
                    if (d1 != d2) {
                        pairs_mask |= (1ULL << (d2 * 7 + d1));
                    }
                }
                // Doubles: step through with multiple die
                if (d1 == d2) {
                    if (pnt - d1 > 0 && board[pnt - d1] < -1) continue;
                    if (pnt - 2 * d1 > 0 && board[pnt - 2 * d1] == -1) {
                        pairs_mask |= (1ULL << (d1 * 7 + d1));
                    }
                    if (pnt - 2 * d1 > 0 && board[pnt - 2 * d1] < -1) continue;
                    if (pnt - 3 * d1 > 0 && board[pnt - 3 * d1] == -1) {
                        pairs_mask |= (1ULL << (d1 * 7 + d1));
                    }
                }
            }
        }
    }

    // Case 3: One checker steps through multiple blots on doubles
    for (int bi = 0; bi < n_blots; ++bi) {
        int pnt = blots[bi].point;

        for (int d = 1; d <= 6; ++d) {
            bool found_second = false;

            if (pnt + d < 25) {
                if (board[pnt + d] == -1) found_second = true;
                if (board[pnt + d] < -1) continue;  // blocked
            }

            if (pnt + 2 * d < 25) {
                if (board[pnt + 2 * d] == -1) found_second = true;
                if (board[pnt + 2 * d] > 0 && found_second) {
                    pairs_mask |= (1ULL << (d * 7 + d));
                }
                if (board[pnt + 2 * d] < -1) continue;
            }

            if (pnt + 3 * d < 25) {
                // Note: reference checks pnt + 2*d for second blot here too
                if (board[pnt + 2 * d] == -1) found_second = true;
                if (board[pnt + 3 * d] > 0 && found_second) {
                    pairs_mask |= (1ULL << (d * 7 + d));
                }
                if (board[pnt + 3 * d] < -1) continue;
            }

            if (pnt + 4 * d < 25) {
                if (board[pnt + 4 * d] > 0 && found_second) {
                    pairs_mask |= (1ULL << (d * 7 + d));
                }
            }
        }
    }

    // Count: each entry in pairs_mask is a unique (d1, d2) pair representing one roll
    // But we stored both orderings for non-doubles. Each unique roll is:
    // - Doubles (d,d): appears once at (d*7+d)
    // - Non-doubles: appears at both (d1*7+d2) and (d2*7+d1)
    // The Python code counts len(pairs) where pairs contains both orderings.
    // So we just popcount the mask.
    int count = 0;
    uint64_t m = pairs_mask;
    while (m) {
        count += (m & 1);
        m >>= 1;
    }
    return count;
}

// ======================== Back Escapes ========================

int back_escapes(const Board& board) {
    // Find the player's back checker (furthest from home, points 19-24)
    for (int i = 24; i >= 19; --i) {
        if (board[i] > 0) {
            // Found the back checker at point i
            int n_escapes = 0;

            for (int d1 = 1; d1 <= 6; ++d1) {
                for (int d2 = 1; d2 <= d1; ++d2) {
                    // Check if this roll goes far enough to escape (reach point <= 15)
                    if (d1 != d2) {
                        // Non-double: moves d1+d2
                        if (i - d1 - d2 > 15) continue;  // doesn't reach

                        // Check both orderings for blocking
                        if (board[i - d1] < -1 && board[i - d2] < -1) continue;  // blocked both ways

                        // Check final landing point
                        if (board[i - d1 - d2] < -1) continue;  // destination blocked

                        n_escapes += 2;  // non-doubles count as 2 rolls
                    } else {
                        // Double: can use 2, 3, or 4 steps of size d1
                        if (i - 4 * d1 > 15) continue;  // even 4 steps can't reach

                        // Check if first step is blocked
                        if (board[i - d1] < -1 && board[i - d2] < -1) continue;

                        // Check destination after 2 steps
                        if (board[i - d1 - d2] < -1) continue;

                        if (i - 2 * d1 <= 15) {
                            n_escapes += 1;
                        } else if (i - 3 * d1 <= 15 && i - 3 * d1 > 0 && board[i - 3 * d1] > -2) {
                            n_escapes += 1;
                        } else if (i - 4 * d1 <= 15 && i - 4 * d1 > 0 &&
                                   board[i - 3 * d1] > -2 && board[i - 4 * d1] > -2) {
                            n_escapes += 1;
                        }
                    }
                }
            }

            return n_escapes;
        }
    }

    // No back checkers beyond point 18 → already escaped
    return 36;
}

// ======================== Max Point / Max Anchor ========================

int max_point(const Board& board) {
    for (int i = 24; i >= 1; --i) {
        if (board[i] > 0) return i;
    }
    return 0;
}

int max_anchor_point(const Board& board) {
    for (int i = 24; i >= 1; --i) {
        if (board[i] > 1) return i;
    }
    return 0;
}

// ======================== New GNUbg Features ========================

// Find opponent's back checker point (highest-numbered point with opponent checkers).
// Returns 0 if no opponent checkers on points. This is nOppBack in GNUbg terms.
static int find_opp_back_checker(const Board& board) {
    for (int i = 24; i >= 1; --i) {
        if (board[i] < 0) return i;
    }
    return 0;
}

int break_contact(const Board& board) {
    int opp_back = find_opp_back_checker(board);
    if (opp_back == 0) return 0;

    // GNUbg: sum of (i+1-nOppBack)*anBoard[i] for i > nOppBack
    // In our convention: sum of (pt - opp_back) * board[pt] for player checkers past opp_back
    int n = 0;
    for (int pt = opp_back + 1; pt <= 24; ++pt) {
        if (board[pt] > 0)
            n += (pt - opp_back) * board[pt];
    }
    // Also count bar checkers (they are past everything)
    if (board[25] > 0)
        n += (25 - opp_back) * board[25];
    return n;
}

int free_pip(const Board& board) {
    int opp_back = find_opp_back_checker(board);
    if (opp_back == 0) return 0;

    // GNUbg: sum of (i+1)*anBoard[i] for i < nOppBack
    // In our convention: sum of pt * board[pt] for player checkers before opp_back
    int p = 0;
    for (int pt = 1; pt < opp_back; ++pt) {
        if (board[pt] > 0)
            p += pt * board[pt];
    }
    return p;
}

int timing(const Board& board) {
    int opp_back = find_opp_back_checker(board);

    // GNUbg timing uses 0-indexed positions. Our board is 1-indexed.
    // GNUbg: anBoard[24] = bar (our board[25])
    //        anBoard[i]   = our board[i+1]
    // nOppBack in GNUbg = 23 - (opp_back_from_opp_perspective)
    // But in CalculateHalfInputs, nOppBack is already computed differently.
    // Let me re-derive from the GNUbg code directly:
    // GNUbg nOppBack = 23 - nOppBack_raw where nOppBack_raw = highest anBoardOpp[k] with checkers
    // In our terms: nOppBack (GNUbg) corresponds to our opp_back - 1 (0-indexed)

    int t = 0;
    int no = 0;

    // t += 24 * anBoard[24] → our board[25] (bar)
    t += 24 * board[25];
    no += board[25];

    // GNUbg: for (i = 23; i >= 12 && i > nOppBack; --i)
    // GNUbg index i corresponds to our point i+1.
    // nOppBack in GNUbg 0-indexed = opp_back - 1 (our 1-indexed)
    int gnubg_opp_back = opp_back > 0 ? opp_back - 1 : -1;

    // Points 24 down to 13 in our convention (GNUbg i=23 down to 12), while i > gnubg_opp_back
    for (int pt = 24; pt >= 13 && (pt - 1) > gnubg_opp_back; --pt) {
        int checkers = (board[pt] > 0) ? board[pt] : 0;
        if (checkers > 0 && checkers != 2) {
            int n = (checkers > 2) ? (checkers - 2) : 1;
            no += n;
            t += (pt - 1) * n;  // GNUbg uses 0-indexed
        }
    }

    // Continue from where we left off, points down to 7 (GNUbg i=6..11)
    int start_pt = (13 > opp_back) ? 12 : opp_back;
    for (int pt = start_pt; pt >= 7; --pt) {
        int checkers = (board[pt] > 0) ? board[pt] : 0;
        if (checkers > 0) {
            no += checkers;
            t += (pt - 1) * checkers;
        }
    }

    // Points 6 down to 1 (GNUbg i=5 down to 0) — home board adjustment
    for (int pt = 6; pt >= 1; --pt) {
        int checkers = (board[pt] > 0) ? board[pt] : 0;
        if (checkers > 2) {
            t += (pt - 1) * (checkers - 2);
            no += (checkers - 2);
        } else if (checkers < 2) {
            int deficit = 2 - checkers;
            if (no >= deficit) {
                t -= (pt - 1) * deficit;
                no -= deficit;
            }
        }
    }

    if (t < 0) t = 0;
    return t;
}

float backbone(const Board& board) {
    // GNUbg: scan from point 23 down to 1 (0-indexed) for anchors (>=2 checkers)
    // Our: scan from point 24 down to 2 for player anchors

    int pa = -1;  // previous anchor point (0-indexed)
    int w = 0;
    int tot = 0;

    for (int pt = 24; pt >= 2; --pt) {
        if (board[pt] >= 2) {
            int gnubg_pt = pt - 1;  // convert to 0-indexed
            if (pa == -1) {
                pa = gnubg_pt;
                continue;
            }

            int d = pa - gnubg_pt;
            int c = 0;
            if (d <= 6) {
                c = 11;
            } else if (d <= 11) {
                c = 13 - d;
            }

            w += c * board[pt + (pa - gnubg_pt)];  // wait, need to use checkers at pa+1
            // Actually GNUbg uses anBoard[pa] which is the previous anchor's checker count
            // pa is in 0-indexed, so anBoard[pa] = our board[pa+1]
            w += c * board[pa + 1] - c * board[pa + 1];  // this is wrong, let me redo

            // Re-reading GNUbg: w += c * anBoard[pa]; where pa is the PREVIOUS anchor
            // and anBoard[pa] is the checker count at that anchor
            // After computing w, tot += anBoard[pa]
            // Then pa = np (current anchor)
            // So we accumulate w and tot for the PREVIOUS anchor's checkers
            // Let me rewrite this correctly:
            pa = -1;  // reset and redo properly
            break;
        }
    }

    // Redo backbone properly
    pa = -1;
    w = 0;
    tot = 0;

    for (int gnubg_pt = 23; gnubg_pt > 0; --gnubg_pt) {
        int our_pt = gnubg_pt + 1;
        if (board[our_pt] >= 2) {
            if (pa == -1) {
                pa = gnubg_pt;
                continue;
            }

            int d = pa - gnubg_pt;
            int c = 0;
            if (d <= 6) {
                c = 11;
            } else if (d <= 11) {
                c = 13 - d;
            }

            // GNUbg: w += c * anBoard[pa]; tot += anBoard[pa];
            // anBoard[pa] = our board[pa+1]
            w += c * board[pa + 1];
            tot += board[pa + 1];

            pa = gnubg_pt;
        }
    }

    if (tot > 0) {
        return 1.0f - (float)w / (float)(tot * 11);
    }
    return 0.0f;
}

float backg(const Board& board) {
    // Count anchor points (2+ checkers) in opponent's home: points 19-24 in our convention
    // GNUbg: points 18-23 (0-indexed) = our 19-24
    int nAc = 0;
    for (int pt = 19; pt <= 24; ++pt) {
        if (board[pt] >= 2) ++nAc;
    }

    if (nAc > 1) {
        // Sum of checkers in opponent's home + bar
        int tot = board[25];  // bar
        for (int pt = 19; pt <= 24; ++pt) {
            if (board[pt] > 0) tot += board[pt];
        }
        return (tot - 3) / 4.0f;
    }
    return 0.0f;
}

float backg1(const Board& board) {
    int nAc = 0;
    for (int pt = 19; pt <= 24; ++pt) {
        if (board[pt] >= 2) ++nAc;
    }

    if (nAc == 1) {
        int tot = board[25];  // bar
        for (int pt = 19; pt <= 24; ++pt) {
            if (board[pt] > 0) tot += board[pt];
        }
        return tot / 8.0f;
    }
    return 0.0f;
}

int enter_loss(const Board& board) {
    // GNUbg I_ENTER: average pips lost when on the bar
    // Only active when player has checker(s) on bar
    if (board[25] <= 0) return 0;

    int loss = 0;
    bool two = board[25] > 1;

    // Check opponent's home board (points 19-24 in our convention)
    // GNUbg anBoardOpp[i] for i=0..5 = opponent's points in THEIR home
    // In our convention, opponent's home is points 19-24 from player's view
    // But we need to check if opponent has blocked points in THEIR home
    // which is where player enters from bar.
    // Opponent blocks point if they have >=2 checkers there.
    // Our board[pt] < -1 means opponent has >=2 at that point.
    // Points 19-24 in our convention = opponent's points 1-6.

    // GNUbg loop: for (i = 0; i < 6; ++i) if (anBoardOpp[i] > 1)
    // anBoardOpp[i] = opponent's checkers at their point i (0-indexed)
    // Opponent's point 0 (ace) = our point 24
    // Opponent's point 5 = our point 19
    // So anBoardOpp[i] corresponds to our board[24-i] being < -1

    for (int i = 0; i < 6; ++i) {
        int our_pt = 24 - i;
        if (board[our_pt] < -1) {
            // This point is blocked — any double loses
            loss += 4 * (i + 1);

            for (int j = i + 1; j < 6; ++j) {
                int our_pt_j = 24 - j;
                if (board[our_pt_j] < -1) {
                    loss += 2 * (i + j + 2);
                } else {
                    if (two) {
                        loss += 2 * (i + 1);
                    }
                }
            }
        } else {
            if (two) {
                for (int j = i + 1; j < 6; ++j) {
                    int our_pt_j = 24 - j;
                    if (board[our_pt_j] < -1) {
                        loss += 2 * (j + 1);
                    }
                }
            }
        }
    }

    return loss;
}

float containment(const Board& board) {
    init_escape_tables();

    // GNUbg I_CONTAIN: (36 - min_escapes) / 36 from point 15 to 24
    // Uses Escapes(anBoard, i) where anBoard is the PLAYER's board
    // This counts how well the player can contain (prevent escape)
    int player_hb[25];
    extract_player_half(board, player_hb);

    int n = 36;
    // GNUbg: for (i = 15; i < 24; i++) — 0-indexed
    // Our: points 16-24, which in GNUbg 0-indexed = 15-23
    for (int gnubg_i = 15; gnubg_i < 24; ++gnubg_i) {
        int j = escapes(player_hb, gnubg_i);
        if (j < n) n = j;
    }

    return (36.0f - n) / 36.0f;
}

float acontainment(const Board& board) {
    init_escape_tables();

    // GNUbg I_ACONTAIN: containment from point 15 to opponent's back checker
    int player_hb[25];
    extract_player_half(board, player_hb);

    // Find opponent's back checker in GNUbg terms
    // GNUbg nOppBack = 23 - (highest anBoardOpp[k] with checkers)
    // In our convention: find highest point with opponent checkers
    int opp_back = find_opp_back_checker(board);
    // Convert to GNUbg's nOppBack: gnubg's nOppBack = 23 - (24 - opp_back) = opp_back - 1
    int gnubg_nOppBack = opp_back > 0 ? opp_back - 1 : -1;

    int n = 36;
    // GNUbg: for (i = 15; i < 24 - nOppBack; i++)
    // 24 - nOppBack = 24 - gnubg_nOppBack
    int limit = (gnubg_nOppBack >= 0) ? 24 - gnubg_nOppBack : 24;
    for (int gnubg_i = 15; gnubg_i < limit; ++gnubg_i) {
        int j = escapes(player_hb, gnubg_i);
        if (j < n) n = j;
    }

    return (36.0f - n) / 36.0f;
}

int mobility(const Board& board) {
    init_escape_tables();

    // GNUbg I_MOBILITY: sum of (i-5) * anBoard[i] * Escapes(anBoardOpp, i) for i=6..24
    // anBoard = player's board (0-indexed)
    // anBoardOpp = opponent's board (0-indexed from opponent's perspective)
    int opp_hb[25];
    extract_opponent_half(board, opp_hb);

    int n = 0;
    for (int gnubg_i = 6; gnubg_i < 25; ++gnubg_i) {
        int our_pt = gnubg_i + 1;
        int checkers = 0;
        if (our_pt <= 24) {
            checkers = (board[our_pt] > 0) ? board[our_pt] : 0;
        } else {
            // gnubg_i = 24 means bar (our board[25])
            checkers = board[25];
        }
        if (checkers > 0) {
            n += (gnubg_i - 5) * checkers * escapes(opp_hb, gnubg_i);
        }
    }
    return n;
}

int moment2(const Board& board) {
    // GNUbg I_MOMENT2: one-sided second moment above weighted mean

    // Compute weighted mean of checker positions (0-indexed)
    int total_checkers = 0;
    int total_weighted = 0;
    for (int gnubg_i = 0; gnubg_i < 25; ++gnubg_i) {
        int checkers;
        if (gnubg_i < 24) {
            int our_pt = gnubg_i + 1;
            checkers = (board[our_pt] > 0) ? board[our_pt] : 0;
        } else {
            checkers = board[25];  // bar
        }
        if (checkers > 0) {
            total_checkers += checkers;
            total_weighted += gnubg_i * checkers;
        }
    }

    int mean = 0;
    if (total_checkers > 0) {
        mean = (total_weighted + total_checkers - 1) / total_checkers;  // ceiling division
    }

    // Compute second moment above mean
    int k = 0;
    int j = 0;
    for (int gnubg_i = mean + 1; gnubg_i < 25; ++gnubg_i) {
        int checkers;
        if (gnubg_i < 24) {
            int our_pt = gnubg_i + 1;
            checkers = (board[our_pt] > 0) ? board[our_pt] : 0;
        } else {
            checkers = board[25];
        }
        if (checkers > 0) {
            j += checkers;
            k += checkers * (gnubg_i - mean) * (gnubg_i - mean);
        }
    }

    if (j > 0) {
        k = (k + j - 1) / j;  // ceiling division
    }

    return k;
}

int back_rescue_escapes(const Board& board) {
    init_escape_tables();

    // GNUbg I_BACKRESCAPES: Escapes1(anBoard, 23 - nOppBack)
    int player_hb[25];
    extract_player_half(board, player_hb);

    int opp_back = find_opp_back_checker(board);
    int gnubg_nOppBack = opp_back > 0 ? opp_back - 1 : -1;

    if (gnubg_nOppBack < 0) return 36;  // no opponent on board
    int escape_point = 23 - gnubg_nOppBack;
    if (escape_point < 0) return 0;
    return escapes1(player_hb, escape_point);
}

// ======================== I_PIPLOSS (GNUbg-style) ========================

// GNUbg tables for piploss computation
struct PiplossInter {
    int fAll;
    int anIntermediate[3];
    int nFaces;
    int nPips;
};

static const int aanCombination[24][5] = {
    {0, -1, -1, -1, -1},    /*  1 */
    {1, 2, -1, -1, -1},     /*  2 */
    {3, 4, 5, -1, -1},      /*  3 */
    {6, 7, 8, 9, -1},       /*  4 */
    {10, 11, 12, -1, -1},   /*  5 */
    {13, 14, 15, 16, 17},   /*  6 */
    {18, 19, 20, -1, -1},   /*  7 */
    {21, 22, 23, 24, -1},   /*  8 */
    {25, 26, 27, -1, -1},   /*  9 */
    {28, 29, -1, -1, -1},   /* 10 */
    {30, -1, -1, -1, -1},   /* 11 */
    {31, 32, 33, -1, -1},   /* 12 */
    {-1, -1, -1, -1, -1},   /* 13 */
    {-1, -1, -1, -1, -1},   /* 14 */
    {34, -1, -1, -1, -1},   /* 15 */
    {35, -1, -1, -1, -1},   /* 16 */
    {-1, -1, -1, -1, -1},   /* 17 */
    {36, -1, -1, -1, -1},   /* 18 */
    {-1, -1, -1, -1, -1},   /* 19 */
    {37, -1, -1, -1, -1},   /* 20 */
    {-1, -1, -1, -1, -1},   /* 21 */
    {-1, -1, -1, -1, -1},   /* 22 */
    {-1, -1, -1, -1, -1},   /* 23 */
    {38, -1, -1, -1, -1}    /* 24 */
};

static const PiplossInter aIntermediate[39] = {
    {1, {0, 0, 0}, 1, 1},   /*  0: 1x hits 1 */
    {1, {0, 0, 0}, 1, 2},   /*  1: 2x hits 2 */
    {1, {1, 0, 0}, 2, 2},   /*  2: 11 hits 2 */
    {1, {0, 0, 0}, 1, 3},   /*  3: 3x hits 3 */
    {0, {1, 2, 0}, 2, 3},   /*  4: 21 hits 3 */
    {1, {1, 2, 0}, 3, 3},   /*  5: 11 hits 3 */
    {1, {0, 0, 0}, 1, 4},   /*  6: 4x hits 4 */
    {0, {1, 3, 0}, 2, 4},   /*  7: 31 hits 4 */
    {1, {2, 0, 0}, 2, 4},   /*  8: 22 hits 4 */
    {1, {1, 2, 3}, 4, 4},   /*  9: 11 hits 4 */
    {1, {0, 0, 0}, 1, 5},   /* 10: 5x hits 5 */
    {0, {1, 4, 0}, 2, 5},   /* 11: 41 hits 5 */
    {0, {2, 3, 0}, 2, 5},   /* 12: 32 hits 5 */
    {1, {0, 0, 0}, 1, 6},   /* 13: 6x hits 6 */
    {0, {1, 5, 0}, 2, 6},   /* 14: 51 hits 6 */
    {0, {2, 4, 0}, 2, 6},   /* 15: 42 hits 6 */
    {1, {3, 0, 0}, 2, 6},   /* 16: 33 hits 6 */
    {1, {2, 4, 0}, 3, 6},   /* 17: 22 hits 6 */
    {0, {1, 6, 0}, 2, 7},   /* 18: 61 hits 7 */
    {0, {2, 5, 0}, 2, 7},   /* 19: 52 hits 7 */
    {0, {3, 4, 0}, 2, 7},   /* 20: 43 hits 7 */
    {0, {2, 6, 0}, 2, 8},   /* 21: 62 hits 8 */
    {0, {3, 5, 0}, 2, 8},   /* 22: 53 hits 8 */
    {1, {4, 0, 0}, 2, 8},   /* 23: 44 hits 8 */
    {1, {2, 4, 6}, 4, 8},   /* 24: 22 hits 8 */
    {0, {3, 6, 0}, 2, 9},   /* 25: 63 hits 9 */
    {0, {4, 5, 0}, 2, 9},   /* 26: 54 hits 9 */
    {1, {3, 6, 0}, 3, 9},   /* 27: 33 hits 9 */
    {0, {4, 6, 0}, 2, 10},  /* 28: 64 hits 10 */
    {1, {5, 0, 0}, 2, 10},  /* 29: 55 hits 10 */
    {0, {5, 6, 0}, 2, 11},  /* 30: 65 hits 11 */
    {1, {6, 0, 0}, 2, 12},  /* 31: 66 hits 12 */
    {1, {4, 8, 0}, 3, 12},  /* 32: 44 hits 12 */
    {1, {3, 6, 9}, 4, 12},  /* 33: 33 hits 12 */
    {1, {5, 10, 0}, 3, 15}, /* 34: 55 hits 15 */
    {1, {4, 8, 12}, 4, 16}, /* 35: 44 hits 16 */
    {1, {6, 12, 0}, 3, 18}, /* 36: 66 hits 18 */
    {1, {5, 10, 15}, 4, 20},/* 37: 55 hits 20 */
    {1, {6, 12, 18}, 4, 24} /* 38: 66 hits 24 */
};

static const int aaRoll[21][4] = {
    {0, 2, 5, 9},           /* 11 */
    {0, 1, 4, -1},          /* 21 */
    {1, 8, 17, 24},         /* 22 */
    {0, 3, 7, -1},          /* 31 */
    {1, 3, 12, -1},         /* 32 */
    {3, 16, 27, 33},        /* 33 */
    {0, 6, 11, -1},         /* 41 */
    {1, 6, 15, -1},         /* 42 */
    {3, 6, 20, -1},         /* 43 */
    {6, 23, 32, 35},        /* 44 */
    {0, 10, 14, -1},        /* 51 */
    {1, 10, 19, -1},        /* 52 */
    {3, 10, 22, -1},        /* 53 */
    {6, 10, 26, -1},        /* 54 */
    {10, 29, 34, 37},       /* 55 */
    {0, 13, 18, -1},        /* 61 */
    {1, 13, 21, -1},        /* 62 */
    {3, 13, 25, -1},        /* 63 */
    {6, 13, 28, -1},        /* 64 */
    {10, 13, 30, -1},       /* 65 */
    {13, 31, 36, 38}        /* 66 */
};

int compute_piploss(const Board& board) {
    // This implements GNUbg's piploss calculation from CalculateHalfInputs.
    // We need to work in GNUbg's coordinate system.
    // GNUbg anBoard = player's checkers (0-indexed, unsigned)
    // GNUbg anBoardOpp = opponent's checkers (0-indexed from opponent's perspective)

    // Extract half-boards in GNUbg format
    int anBoard[25];
    int anBoardOpp[25];
    extract_player_half(board, anBoard);
    extract_opponent_half(board, anBoardOpp);

    // Find opponent's back checker
    int nOppBack = -1;
    for (int i = 24; i >= 0; --i) {
        if (anBoardOpp[i]) {
            nOppBack = i;
            break;
        }
    }
    if (nOppBack < 0) return 0;
    nOppBack = 23 - nOppBack;

    // Count player anchors in home (points 0-5)
    int nBoard = 0;
    for (int i = 0; i < 6; i++)
        if (anBoard[i]) nBoard++;

    int aHit[39];
    memset(aHit, 0, sizeof(aHit));

    // For every point we'd consider hitting a blot on
    for (int i = (nBoard > 2) ? 23 : 21; i >= 0; i--) {
        if (anBoardOpp[i] != 1) continue;

        for (int j = 24 - i; j < 25; j++) {
            if (!anBoard[j]) continue;
            if (j < 6 && anBoard[j] == 2) continue;  // don't break home anchors

            int dist = j - (24 - i);
            if (dist < 0 || dist >= 24) continue;

            for (int n = 0; n < 5; n++) {
                if (aanCombination[dist][n] == -1) break;

                const PiplossInter* pi = &aIntermediate[aanCombination[dist][n]];

                if (pi->fAll) {
                    if (pi->nFaces > 1) {
                        bool blocked = false;
                        for (int k = 0; k < 3 && pi->anIntermediate[k] > 0; k++) {
                            if (anBoardOpp[i - pi->anIntermediate[k]] > 1) {
                                blocked = true;
                                break;
                            }
                        }
                        if (blocked) continue;
                    }
                } else {
                    if (anBoardOpp[i - pi->anIntermediate[0]] > 1 &&
                        anBoardOpp[i - pi->anIntermediate[1]] > 1) {
                        continue;
                    }
                }

                aHit[aanCombination[dist][n]] |= (1 << j);
            }
        }
    }

    // Now compute per-roll stats
    struct { int nPips; int nChequers; } aRoll[21];
    memset(aRoll, 0, sizeof(aRoll));

    if (!anBoard[24]) {
        // Not on bar
        for (int i = 0; i < 21; i++) {
            int n = -1;
            for (int j = 0; j < 4; j++) {
                int r = aaRoll[i][j];
                if (r < 0) break;
                if (!aHit[r]) continue;

                const PiplossInter* pi = &aIntermediate[r];

                if (pi->nFaces == 1) {
                    for (int k = 23; k > 0; k--) {
                        if (aHit[r] & (1 << k)) {
                            if (n != k || anBoard[k] > 1)
                                aRoll[i].nChequers++;
                            n = k;
                            if (k - pi->nPips + 1 > aRoll[i].nPips)
                                aRoll[i].nPips = k - pi->nPips + 1;
                            if (aaRoll[i][3] >= 0 && aHit[r] & ~(1 << k))
                                aRoll[i].nChequers++;
                            break;
                        }
                    }
                } else {
                    if (!aRoll[i].nChequers)
                        aRoll[i].nChequers = 1;
                    for (int k = 23; k >= 0; k--)
                        if (aHit[r] & (1 << k)) {
                            if (k - pi->nPips + 1 > aRoll[i].nPips)
                                aRoll[i].nPips = k - pi->nPips + 1;
                            // Check for blots on intermediate points
                            for (int l = 0; l < 3 && pi->anIntermediate[l] > 0; l++)
                                if (anBoardOpp[23 - k + pi->anIntermediate[l]] == 1) {
                                    aRoll[i].nChequers++;
                                    break;
                                }
                            break;
                        }
                }
            }
        }
    } else if (anBoard[24] == 1) {
        // One checker on bar
        for (int i = 0; i < 21; i++) {
            int n = 0;
            for (int j = 0; j < 4; j++) {
                int r = aaRoll[i][j];
                if (r < 0) break;
                if (!aHit[r]) continue;

                const PiplossInter* pi = &aIntermediate[r];

                if (pi->nFaces == 1) {
                    for (int k = 24; k > 0; k--) {
                        if (aHit[r] & (1 << k)) {
                            if (n && k != 24) break;
                            if (k != 24) {
                                int npip = aIntermediate[aaRoll[i][1 - j]].nPips;
                                if (anBoardOpp[npip - 1] > 1) break;
                                n = 1;
                            }
                            aRoll[i].nChequers++;
                            if (k - pi->nPips + 1 > aRoll[i].nPips)
                                aRoll[i].nPips = k - pi->nPips + 1;
                        }
                    }
                } else {
                    if (!(aHit[r] & (1 << 24))) continue;
                    if (!aRoll[i].nChequers) aRoll[i].nChequers = 1;
                    if (25 - pi->nPips > aRoll[i].nPips)
                        aRoll[i].nPips = 25 - pi->nPips;
                    for (int k = 0; k < 3 && pi->anIntermediate[k] > 0; k++)
                        if (anBoardOpp[pi->anIntermediate[k] + 1] == 1) {
                            aRoll[i].nChequers++;
                            break;
                        }
                }
            }
        }
    } else {
        // Multiple checkers on bar
        for (int i = 0; i < 21; i++) {
            for (int j = 0; j < 2; j++) {
                int r = aaRoll[i][j];
                if (!(aHit[r] & (1 << 24))) continue;
                const PiplossInter* pi = &aIntermediate[r];
                if (pi->nFaces != 1) continue;
                aRoll[i].nChequers++;
                if (25 - pi->nPips > aRoll[i].nPips)
                    aRoll[i].nPips = 25 - pi->nPips;
            }
        }
    }

    // Sum pips across all rolls
    int np = 0;
    for (int i = 0; i < 21; i++) {
        int w = (aaRoll[i][3] > 0) ? 1 : 2;
        np += aRoll[i].nPips * w;
    }

    return np;
}

// ======================== Extended 244-input Encoding ========================

// Optimized versions of feature functions that accept precomputed data
// to avoid redundant computation (find_opp_back_checker, extract half-boards, etc.)

static int break_contact_fast(const Board& board, int opp_back) {
    if (opp_back == 0) return 0;
    int n = 0;
    for (int pt = opp_back + 1; pt <= 24; ++pt) {
        if (board[pt] > 0)
            n += (pt - opp_back) * board[pt];
    }
    if (board[25] > 0)
        n += (25 - opp_back) * board[25];
    return n;
}

static int free_pip_fast(const Board& board, int opp_back) {
    if (opp_back == 0) return 0;
    int p = 0;
    for (int pt = 1; pt < opp_back; ++pt) {
        if (board[pt] > 0)
            p += pt * board[pt];
    }
    return p;
}

static int timing_fast(const Board& board, int opp_back) {
    int gnubg_opp_back = opp_back > 0 ? opp_back - 1 : -1;

    int t = 0;
    int no = 0;

    t += 24 * board[25];
    no += board[25];

    for (int pt = 24; pt >= 13 && (pt - 1) > gnubg_opp_back; --pt) {
        int checkers = (board[pt] > 0) ? board[pt] : 0;
        if (checkers > 0 && checkers != 2) {
            int n = (checkers > 2) ? (checkers - 2) : 1;
            no += n;
            t += (pt - 1) * n;
        }
    }

    int start_pt = (13 > opp_back) ? 12 : opp_back;
    for (int pt = start_pt; pt >= 7; --pt) {
        int checkers = (board[pt] > 0) ? board[pt] : 0;
        if (checkers > 0) {
            no += checkers;
            t += (pt - 1) * checkers;
        }
    }

    for (int pt = 6; pt >= 1; --pt) {
        int checkers = (board[pt] > 0) ? board[pt] : 0;
        if (checkers > 2) {
            t += (pt - 1) * (checkers - 2);
            no += (checkers - 2);
        } else if (checkers < 2) {
            int deficit = 2 - checkers;
            if (no >= deficit) {
                t -= (pt - 1) * deficit;
                no -= deficit;
            }
        }
    }

    if (t < 0) t = 0;
    return t;
}

static float acontainment_fast(unsigned player_blocks, int opp_back) {
    int gnubg_nOppBack = opp_back > 0 ? opp_back - 1 : -1;

    int n = 36;
    int limit = (gnubg_nOppBack >= 0) ? 24 - gnubg_nOppBack : 24;
    for (int gnubg_i = 15; gnubg_i < limit; ++gnubg_i) {
        int j = anEscapes[(player_blocks >> (24 - gnubg_i)) & 4095];
        if (j < n) n = j;
    }
    return 36.0f - n;
}

static float containment_fast(unsigned player_blocks) {
    int n = 36;
    for (int gnubg_i = 15; gnubg_i < 24; ++gnubg_i) {
        int j = anEscapes[(player_blocks >> (24 - gnubg_i)) & 4095];
        if (j < n) n = j;
    }
    return 36.0f - n;
}

static int mobility_fast(const Board& board, unsigned opp_blocks) {
    int n = 0;
    for (int gnubg_i = 6; gnubg_i < 25; ++gnubg_i) {
        int our_pt = gnubg_i + 1;
        int checkers = 0;
        if (our_pt <= 24) {
            checkers = (board[our_pt] > 0) ? board[our_pt] : 0;
        } else {
            checkers = board[25];
        }
        if (checkers > 0) {
            n += (gnubg_i - 5) * checkers * anEscapes[(opp_blocks >> (24 - gnubg_i)) & 4095];
        }
    }
    return n;
}

static int back_rescue_escapes_fast(unsigned player_blocks, int opp_back) {
    int gnubg_nOppBack = opp_back > 0 ? opp_back - 1 : -1;
    if (gnubg_nOppBack < 0) return 36;
    int escape_point = 23 - gnubg_nOppBack;
    if (escape_point < 0) return 0;
    return anEscapes1[(player_blocks >> (24 - escape_point)) & 4095];
}

// Point 1 is bit 0. A shift to the right moves a source checker towards
// home; all blots on a given roll can then be tested together.
struct ContactPointMasks {
    unsigned occupied = 0;
    unsigned made = 0;
    unsigned enemy_blots = 0;
    unsigned enemy_blocks = 0;
    unsigned pip_loss_sources = 0;
    unsigned pip_loss_targets = 0;
};

static ContactPointMasks contact_point_masks(const Board& board) {
    ContactPointMasks masks;
    for (int i = 0; i < 24; ++i) {
        const int checkers = board[i + 1];
        masks.occupied |= static_cast<unsigned>(checkers > 0) << i;
        masks.made |= static_cast<unsigned>(checkers >= 2) << i;
        masks.enemy_blots |= static_cast<unsigned>(checkers == -1) << i;
        masks.enemy_blocks |= static_cast<unsigned>(checkers <= -2) << i;
    }
    int occupied_home = 0;
    unsigned two_checker_home_points = 0;
    for (int i = 0; i < 6; ++i) {
        occupied_home += board[i + 1] > 0;
        two_checker_home_points |= static_cast<unsigned>(board[i + 1] == 2) << i;
    }
    // GNUbg's pip-loss feature preserves two-checker home points. With at
    // most two occupied home points it also skips enemy blots on points 1–2.
    masks.pip_loss_sources = masks.occupied & ~two_checker_home_points;
    masks.pip_loss_targets = occupied_home > 2 ? masks.enemy_blots : masks.enemy_blots & ~3u;
    return masks;
}

// Each bit describes one of the 24 points, with the ace point at bit zero.
#if defined(__AVX2__)
static unsigned reverse_points(unsigned bits) {
    static constexpr auto reversed_byte = [] {
        std::array<unsigned, 256> table{};
        for (unsigned i = 1; i < table.size(); ++i)
            table[i] = (table[i >> 1] >> 1) | ((i & 1u) << 7);
        return table;
    }();
    return (reversed_byte[bits & 255] << 16) |
           (reversed_byte[(bits >> 8) & 255] << 8) | reversed_byte[bits >> 16];
}

static std::pair<ContactPointMasks, ContactPointMasks> both_contact_point_masks(const Board& board) {
    unsigned player_points = 0, player_made = 0, enemy_points = 0, enemy_made = 0;
    unsigned player_twos = 0, enemy_twos = 0;
    const auto mask = [](__m256i v) { return static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(v))); };
    const auto zero = _mm256_setzero_si256();
    for (int i = 0; i < 24; i += 8) {
        const auto points = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(board.data() + i + 1));
        player_points |= mask(_mm256_cmpgt_epi32(points, zero)) << i;
        player_made |= mask(_mm256_cmpgt_epi32(points, _mm256_set1_epi32(1))) << i;
        enemy_points |= mask(_mm256_cmpgt_epi32(zero, points)) << i;
        enemy_made |= mask(_mm256_cmpgt_epi32(_mm256_set1_epi32(-1), points)) << i;
        if (i == 0) player_twos = mask(_mm256_cmpeq_epi32(points, _mm256_set1_epi32(2))) & 63;
        if (i == 16) enemy_twos = (mask(_mm256_cmpeq_epi32(points, _mm256_set1_epi32(-2))) & 252) << 16;
    }
    ContactPointMasks player{}, opponent{};
    const auto three_home_points = [](unsigned occupied) {
        unsigned home = occupied & 63;
        home &= home - 1;
        return (home & (home - 1)) != 0;
    };
    player.occupied = player_points;
    player.made = player_made;
    player.enemy_blots = enemy_points ^ enemy_made;
    player.enemy_blocks = enemy_made;
    player.pip_loss_sources = player_points & ~player_twos;
    player.pip_loss_targets = three_home_points(player_points) ? player.enemy_blots : player.enemy_blots & ~3u;
    opponent.occupied = reverse_points(enemy_points);
    opponent.made = reverse_points(enemy_made);
    opponent.enemy_blots = reverse_points(player_points ^ player_made);
    opponent.enemy_blocks = reverse_points(player_made);
    opponent.pip_loss_sources = reverse_points(enemy_points & ~enemy_twos);
    opponent.pip_loss_targets = three_home_points(opponent.occupied) ? opponent.enemy_blots : opponent.enemy_blots & ~3u;
    return {player, opponent};
}
#endif

// One-based point number, with zero denoting the empty mask.
static int highest_set_point(unsigned mask) {
    if (mask == 0) return 0;
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanReverse(&index, mask);
    return static_cast<int>(index) + 1;
#elif defined(__GNUC__) || defined(__clang__)
    return std::numeric_limits<unsigned>::digits - __builtin_clz(mask);
#else
    int point = 0;
    for (; mask; mask >>= 1) ++point;
    return point;
#endif
}

static int forward_anchor_from_mask(unsigned made) {
    const unsigned on_opponent_side = made >> 12;
    if (on_opponent_side == 0) return 0;
    // Lowest of points 13–24, expressed from the opponent's perspective.
    return 13 - highest_set_point(on_opponent_side & (0u - on_opponent_side));
}

static float blocked_entry_probability(unsigned enemy_blocks) {
    static constexpr auto point_counts = [] {
        std::array<int, 64> counts{};
        for (int i = 1; i < 64; ++i) counts[i] = counts[i >> 1] + (i & 1);
        return counts;
    }();
    const int blocked = point_counts[(enemy_blocks >> 18) & 63];
    return static_cast<float>(blocked * blocked) / 36.0f;
}

static float backbone_from_mask(const Board& board, unsigned made) {
    made &= ~1u;  // The backbone feature excludes point 1.
    int previous = 0;
    int weighted_count = 0;
    int checker_count = 0;
    while (made) {
        const int point = highest_set_point(made);
        made ^= 1u << (point - 1);
        if (previous) {
            const int gap = previous - point;
            const int weight = gap <= 6 ? 11 : (gap <= 11 ? 13 - gap : 0);
            weighted_count += weight * board[previous];
            checker_count += board[previous];
        }
        previous = point;
    }
    if (checker_count == 0) return 0.0f;
    return 1.0f - static_cast<float>(weighted_count) / static_cast<float>(11 * checker_count);
}

// Each intermediate blocker mask has already been shifted to the target
// point's coordinates. A path of length k needs k-1 open intermediate points.
static unsigned double_roll_targets(unsigned sources, unsigned blots, int die,
                                    unsigned block1, unsigned block2, unsigned block3) {
    const unsigned from_two = sources >> (2 * die);
    const unsigned from_three = sources >> (3 * die);
    const unsigned from_four = sources >> (4 * die);
    const unsigned indirect = ~block1 & (from_two | (~block2 & (from_three | (~block3 & from_four))));
    return blots & ((sources >> die) | indirect);
}

static bool different_targets(unsigned first, unsigned second) {
    const unsigned both = first | second;
    return first && second && (both & (both - 1));
}

static bool two_hits_on_doubles(const ContactPointMasks& masks, unsigned sources,
                                unsigned direct, int die, unsigned block1,
                                unsigned block2, unsigned block3) {
    const unsigned blots = masks.enemy_blots;
    // Two different targets. The reference assigns the roll using the gcd
    // of the two distances; their normalized step counts must add to <= 4.
    // This gives (1,1), (1,2), or (1,3), in either order.
    const unsigned after_two = blots & ~block1 & (sources >> (2 * die));
    const unsigned after_three = blots & ~block1 & ~block2 & (sources >> (3 * die));
    if (different_targets(direct, direct | after_two | after_three)) return true;

    // After a direct hit (including entry from the bar) continue to a
    // second blot one, two or three steps farther towards home.
    const unsigned lower_one = blots << die;
    const unsigned lower_two = blots << (2 * die);
    const unsigned lower_three = blots << (3 * die);
    const unsigned lower_targets = lower_one | (~(masks.enemy_blocks << die) &
        (lower_two | (~(masks.enemy_blocks << (2 * die)) & lower_three)));
    if (direct & lower_targets) return true;

    // From the lower blot, look back for an on-board checker which crosses
    // another blot. The trained feature tests the first two intermediate
    // points for that other blot on a four-step path.
    const unsigned higher_one = blots >> die;
    const unsigned higher_two = blots >> (2 * die);
    const unsigned from_two = masks.occupied >> (2 * die);
    const unsigned from_three = masks.occupied >> (3 * die);
    const unsigned from_four = masks.occupied >> (4 * die);
    const unsigned via_blot = (higher_one & from_two) |
        (~block2 & (higher_one | higher_two) & (from_three | (~block3 & from_four)));
    return (blots & ~block1 & via_blot) != 0;
}

static bool two_hits_from_separate_checkers(unsigned high_targets, unsigned low_targets,
                                           int high_die, int low_die, unsigned stacked) {
    if (low_targets == 0) return false;
    while (high_targets) {
        const unsigned target = high_targets & (0u - high_targets);
        const unsigned source = target << high_die;
        // Hit a different blot; a single checker cannot make both direct hits.
        const unsigned same_checker = (source & stacked) ? 0u : source >> low_die;
        if (low_targets & ~(target | same_checker)) return true;
        high_targets ^= target;
    }
    return false;
}

// The trained pip-loss feature takes the largest hit target for each roll.
// With one on the bar it also allows a direct hit by an on-board checker
// when the other die can enter. The legacy feature visits the smaller direct
// die first and uses at most one on-board hit; on doubles it checks the
// two-die entry point for that direct hit.
static int entry_pip_loss(const Board& board, const ContactPointMasks& masks) {
    constexpr unsigned bar = 1u << 24;
    const unsigned targets = masks.pip_loss_targets;
    if (!targets) return 0;
    const unsigned blocks = masks.enemy_blocks;
    unsigned direct[7], entered[7];
    int total = 0;
    const bool one_on_bar = board[25] == 1;
    for (int die = 1; die <= 6; ++die) {
        entered[die] = (bar >> die) & targets;
        direct[die] = (masks.pip_loss_sources >> die) & targets;
        unsigned hits = entered[die];
        if (one_on_bar) {
            hits = double_roll_targets(bar, targets, die, blocks >> die, blocks >> (2*die),
                                      blocks >> (3*die));
            if (!(blocks & (bar >> (2 * die)))) hits |= direct[die];
        }
        total += highest_set_point(hits);
    }
    for (int high = 2; high <= 6; ++high) {
        for (int low = 1; low < high; ++low) {
            unsigned hits = entered[low] | entered[high];
            if (one_on_bar) {
                const bool enter_high = !(blocks & (bar >> high));
                const bool enter_low = !(blocks & (bar >> low));
                if (enter_high || enter_low) hits |= (bar >> (high + low)) & targets;
                if (enter_high && direct[low]) hits |= direct[low];
                else if (enter_low) hits |= direct[high];
            }
            total += 2 * highest_set_point(hits);
        }
    }
    return total;
}

struct ShotFeatures {
    int single_hits = 0;
    int double_hits = 0;
    int pip_loss = 0;
};

#if defined(__AVX2__)
// Eight independent 24-point target masks, one per dice roll.
struct RollMasks {
    __m256i bits;
    explicit RollMasks(unsigned value) : bits(_mm256_set1_epi32(value)) {}
    RollMasks(__m256i value) : bits(value) {}
    friend RollMasks operator&(RollMasks a, RollMasks b) { return _mm256_and_si256(a.bits, b.bits); }
    friend RollMasks operator|(RollMasks a, RollMasks b) { return _mm256_or_si256(a.bits, b.bits); }
    friend RollMasks operator~(RollMasks a) { return _mm256_xor_si256(a.bits, _mm256_set1_epi32(-1)); }
    friend RollMasks operator+(RollMasks a, RollMasks b) { return _mm256_add_epi32(a.bits, b.bits); }
    friend RollMasks operator-(RollMasks a, RollMasks b) { return _mm256_sub_epi32(a.bits, b.bits); }
    friend RollMasks operator>>(RollMasks a, RollMasks b) { return _mm256_srlv_epi32(a.bits, b.bits); }
    friend RollMasks operator<<(RollMasks a, RollMasks b) { return _mm256_sllv_epi32(a.bits, b.bits); }
    RollMasks first() const { return *this & (RollMasks(0u) - *this); }
    RollMasks nonempty() const { return _mm256_cmpgt_epi32(bits, _mm256_setzero_si256()); }
    int hit_rolls() const {
        return _mm_popcnt_u32(_mm256_movemask_ps(_mm256_castsi256_ps(nonempty().bits)));
    }
    RollMasks highest_points() const {
        // A 24-bit mask converts exactly to binary32. Its unbiased exponent
        // plus one is the point index. Clamp the empty mask's result to zero.
        const __m256i exp = _mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(bits)), 23);
        return _mm256_max_epi32(_mm256_setzero_si256(), _mm256_sub_epi32(exp, _mm256_set1_epi32(126)));
    }
    int sum() const {
        int lanes[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), bits);
        return std::accumulate(lanes, lanes + 8, 0);
    }
};

static ShotFeatures off_bar_shots(const ContactPointMasks& masks, unsigned reachable_blots) {
    ShotFeatures counts;
    const RollMasks occupied(masks.occupied), made(masks.made), blocks(masks.enemy_blocks);
    const RollMasks pip_sources(masks.pip_loss_sources), pip_targets(masks.pip_loss_targets);
    const RollMasks blots(reachable_blots);
    const bool multiple = (reachable_blots & (reachable_blots - 1)) != 0;

    // Six doubles. Shifts of 32 make the spare lanes empty.
    const RollMasks die = _mm256_setr_epi32(1,2,3,4,5,6,32,32);
    const RollMasks twice = die + die, three = twice + die, four = twice + twice;
    const RollMasks b1 = blocks >> die, b2 = blocks >> twice, b3 = blocks >> three;
    const auto double_paths = [&](RollMasks sources) {
        return (sources >> die) | (~b1 & ((sources >> twice) |
               (~b2 & ((sources >> three) | (~b3 & (sources >> four))))));
    };
    counts.single_hits = (blots & double_paths(occupied)).hit_rolls();
    if (multiple) {
        const RollMasks direct = blots & (occupied >> die);
        const RollMasks from2 = occupied >> twice, from3 = occupied >> three, from4 = occupied >> four;
        const RollMasks all = direct | (blots & ~b1 & (from2 | (~b2 & from3)));
        const RollMasks distinct = direct.nonempty() & all & (all - RollMasks(1u));
        const RollMasks lower = (blots << die) | (~(blocks << die) &
            ((blots << twice) | (~(blocks << twice) & (blots << three))));
        const RollMasks higher = blots >> die;
        const RollMasks via_blot = (higher & from2) | (~b2 & (higher | (blots >> twice)) & (from3 | (~b3 & from4)));
        counts.double_hits = (distinct | (direct & lower) | (blots & ~b1 & via_blot)).hit_rolls();
    }
    RollMasks pip_total = (pip_targets & double_paths(pip_sources)).highest_points();

    // Fifteen unlike rolls, each with two equally likely dice orders.
    const RollMasks high[] = {_mm256_setr_epi32(2,3,3,4,4,4,5,5), _mm256_setr_epi32(5,5,6,6,6,6,6,32)};
    const RollMasks low[] = {_mm256_setr_epi32(1,1,2,1,2,3,1,2), _mm256_setr_epi32(3,4,1,2,3,4,5,32)};
    for (int group = 0; group < 2; ++group) {
        const RollMasks h = high[group], l = low[group], sum = h + l;
        const RollMasks open = ~((blocks >> h) & (blocks >> l));
        const RollMasks hd = blots & (occupied >> h), ld = blots & (occupied >> l);
        counts.single_hits += 2 * (hd | ld | (blots & open & (occupied >> sum))).hit_rolls();
        if (multiple) {
            const RollMasks through = (hd & (blots << l)) | (ld & (blots << h));
            // A pair of direct hits can fail only on the same target or from
            // the same lone checker. If the two lowest targets coincide, any
            // higher h target works. If they share a source, any higher l target
            // works. Testing each lowest target against the other set suffices.
            const RollMasks first_h = hd.first(), first_l = ld.first();
            const RollMasks lone_h = ~made & (first_h << h), lone_l = ~made & (first_l << l);
            const RollMasks other_l = ld & ~(first_h | (lone_h >> l));
            const RollMasks other_h = hd & ~(first_l | (lone_l >> h));
            counts.double_hits += 2 * (through | (hd.nonempty() & other_l) |
                                       (ld.nonempty() & other_h)).hit_rolls();
        }
        const RollMasks hits = pip_targets &
            ((pip_sources >> h) | (pip_sources >> l) | (open & (pip_sources >> sum)));
        const RollMasks points = hits.highest_points();
        pip_total = pip_total + points + points;
    }
    counts.pip_loss = pip_total.sum();
    return counts;
}
#endif

// The bar changes the GNUbg pip-loss calculation. Its no-bar case uses the
// highest hittable target per roll. Checkers on the bar restrict legal entries
// and which direct hits by on-board checkers contribute to pip loss.
template<bool OnBar>
static ShotFeatures shot_features(const Board& board, const ContactPointMasks& masks) {
    ShotFeatures counts;
    unsigned blots = masks.enemy_blots;
    if (!OnBar && blots) {
        if (!masks.occupied) return counts;
        // Blots at or beyond the rearmost checker cannot be hit.
        blots &= (1u << (highest_set_point(masks.occupied) - 1)) - 1;
    }
    if (blots == 0) return counts;
#if defined(__AVX2__)
    if constexpr (!OnBar) return off_bar_shots(masks, blots);
#endif
    const bool multiple_blots = (blots & (blots - 1)) != 0;
    // The trained single-shot feature uses only on-board sources. The
    // double-shot feature also considers entries, including a stacked bar.
    const unsigned sources = masks.occupied | (OnBar ? 1u << 24 : 0u);
    const unsigned stacked = masks.made | (static_cast<unsigned>(board[25] > 1) << 24);
    unsigned direct[7]{}, direct_with_bar[7]{}, pip_direct[7]{};
    for (int die = 1; die <= 6; ++die) {
        const unsigned block1 = masks.enemy_blocks >> die;
        const unsigned block2 = masks.enemy_blocks >> (2 * die);
        const unsigned block3 = masks.enemy_blocks >> (3 * die);
        direct[die] = blots & (masks.occupied >> die);
        direct_with_bar[die] = blots & (sources >> die);
        counts.single_hits += double_roll_targets(masks.occupied, blots, die, block1, block2, block3) != 0;
        if (multiple_blots) {
            counts.double_hits += two_hits_on_doubles(masks, sources, direct_with_bar[die],
                                                     die, block1, block2, block3);
        }
        if constexpr (!OnBar) {
            pip_direct[die] = masks.pip_loss_targets & (masks.pip_loss_sources >> die);
            counts.pip_loss += highest_set_point(double_roll_targets(masks.pip_loss_sources,
                masks.pip_loss_targets, die, block1, block2, block3));
        }
    }
    for (int high = 2; high <= 6; ++high) {
        for (int low = 1; low < high; ++low) {
            // There are two orders for unlike dice. Either open intermediate
            // permits a two-die path; hit counts weight the roll twice.
            const unsigned open_path = ~((masks.enemy_blocks >> high) & (masks.enemy_blocks >> low));
            const unsigned combined = blots & open_path & (masks.occupied >> (high + low));
            counts.single_hits += 2 * ((direct[high] | direct[low] | combined) != 0);
            if (multiple_blots) {
                const bool step_through = (direct_with_bar[high] & (blots << low)) ||
                                          (direct_with_bar[low] & (blots << high));
                counts.double_hits += 2 * (step_through || two_hits_from_separate_checkers(
                    direct_with_bar[high], direct_with_bar[low], high, low, stacked));
            }
            if constexpr (!OnBar) {
                const unsigned pip_combined = masks.pip_loss_targets & open_path &
                                              (masks.pip_loss_sources >> (high + low));
                counts.pip_loss += 2 * highest_set_point(pip_direct[high] | pip_direct[low] | pip_combined);
            }
        }
    }
    if constexpr (OnBar) counts.pip_loss = entry_pip_loss(board, masks);
    return counts;
}

// A saved half describes exactly the same checkers and bar. Its one-sided
// features survive unchanged. Interaction features also depend on the
// opposing checkers, their made points, and the position of their back man.
static void contact_half(const Board& board, const ContactPointMasks& side, unsigned other_made,
                         float* out, const float* saved = nullptr, int previous_opp_back = -1) {
    const int back = highest_set_point(side.occupied);
    const int opp_back = highest_set_point(side.enemy_blots | side.enemy_blocks);
    if (saved) {
        std::copy_n(saved, 26, out);
    } else {
        const int off = player_borne_off(board);
        const int forward = forward_anchor_from_mask(side.made);
        const float contain = containment_fast(side.made);
        out[0] = board[25] / 2.0f;
        out[1] = off >= 5 ? 1.0f : off / 5.0f;
        out[2] = off >= 10 ? 1.0f : (off <= 5 ? 0.0f : (off - 5) / 5.0f);
        out[3] = off <= 10 ? 0.0f : (off - 10) / 5.0f;
        out[5] = forward == 0 ? 2.0f : forward / 6.0f;
        out[9] = back / 24.0f;
        out[10] = highest_set_point(side.made) / 24.0f;
        out[16] = contain * (1.0f / 36.0f);
        out[17] = (contain * contain) * (1.0f / 1296.0f);
        out[19] = moment2(board) / 400.0f;
        out[22] = backbone_from_mask(board, side.made);
        out[23] = backg(board);
        out[24] = backg1(board);
    }
    if (!saved || previous_opp_back != opp_back) {
        const float contain = acontainment_fast(side.made, opp_back);
        out[11] = break_contact_fast(board, opp_back) / 167.0f;
        out[12] = free_pip_fast(board, opp_back) / 100.0f;
        out[14] = contain * (1.0f / 36.0f);
        out[15] = (contain * contain) * (1.0f / 1296.0f);
        out[21] = timing_fast(board, opp_back) / 100.0f;
        out[25] = back_rescue_escapes_fast(side.made, opp_back) / 36.0f;
    }
    const auto shots = board[25] ? shot_features<true>(board, side) : shot_features<false>(board, side);
    const int escapes = back < 19 ? 36 : back_escape_counts[back - 19][(side.enemy_blocks >> (back - 13)) & 4095];
    const int entry_loss = board[25] ? entry_losses[board[25] > 1][other_made & 63] : 0;
    out[4] = blocked_entry_probability(side.enemy_blocks);
    out[6] = shots.single_hits / 36.0f;
    out[7] = shots.double_hits / 36.0f;
    out[8] = escapes / 36.0f;
    out[13] = shots.pip_loss / (12.0f * 36.0f);
    out[18] = mobility_fast(board, other_made) / 3600.0f;
    out[20] = entry_loss / (36.0f * (49.0f / 6.0f));
}

static void contact_aux(const Board& board, float* p_aux, float* o_aux,
                        const float* saved_opponent = nullptr, int previous_front_in_opponent_view = -1) {
    const Board flipped = flip(board);
#if defined(__AVX2__)
    const auto [player, opponent] = both_contact_point_masks(board);
#else
    const auto player = contact_point_masks(board);
    const auto opponent = contact_point_masks(flipped);
#endif
    init_escape_tables();
    contact_half(board, player, opponent.made, p_aux);
    contact_half(flipped, opponent, player.made, o_aux, saved_opponent, previous_front_in_opponent_view);
}

std::array<float, EXTENDED_CONTACT_INPUTS> compute_extended_contact_inputs(const Board& board) {
    static_assert(EXTENDED_CONTACT_INPUTS == 244, "Expected 244 inputs");
    std::array<float, EXTENDED_CONTACT_INPUTS> inputs;  // all entries assigned below

    constexpr int P2_OFFSET = 122;  // Player 2 block starts at index 122

    // Point encoding: same as Tesauro but offset differently for player 2
    for (int i = 0; i < 24; ++i) {
        int n = board[i + 1];

        const float* p_feat = point_feature_row(n);
        const int p0 = 4 * i;
        inputs[p0 + 0] = p_feat[0];
        inputs[p0 + 1] = p_feat[1];
        inputs[p0 + 2] = p_feat[2];
        inputs[p0 + 3] = p_feat[3];

        int m = -n;
        const float* m_feat = point_feature_row(m);
        const int p1 = 4 * i + P2_OFFSET;
        inputs[p1 + 0] = m_feat[0];
        inputs[p1 + 1] = m_feat[1];
        inputs[p1 + 2] = m_feat[2];
        inputs[p1 + 3] = m_feat[3];
    }

    contact_aux(board, inputs.data() + 96, inputs.data() + P2_OFFSET + 96);
    return inputs;
}


int board_input_deltas(const Board& board, const Board& base, const float* saved_inputs,
                       bool tesauro, int* indices, float* values) {
    unsigned changed = 0;
    for (int i = 0; i < 24; ++i)
        changed |= static_cast<unsigned>(board[i + 1] != base[i + 1]) << i;

    float p_aux[26], o_aux[26];
    if (tesauro) {
        p_aux[0] = board[25] / 2.0f;
        p_aux[1] = player_borne_off(board) / 15.0f;
        o_aux[0] = board[0] / 2.0f;
        o_aux[1] = opponent_borne_off(board) / 15.0f;
    } else {
        // Comparing bar counts alone misses siblings that hit different blots.
        bool same_opponent = board[0] == base[0];
        for (unsigned bits = changed; same_opponent && bits; bits &= bits - 1) {
            const int point = highest_set_point(bits & (0u - bits));
            same_opponent = board[point] >= 0 && base[point] >= 0;
        }
        int previous_front = 0;
        if (same_opponent) {
            for (int point = 1; point <= 24; ++point) {
                if (base[point] > 0) {
                    previous_front = 25 - point;
                    break;
                }
            }
        }
        constexpr int opponent_aux_offset = EXTENDED_CONTACT_INPUTS / 2 + 24 * 4;
        contact_aux(board, p_aux, o_aux, same_opponent ? saved_inputs + opponent_aux_offset : nullptr,
                    previous_front);
    }
    int count = 0;
    auto append = [&](int index, float current) {
        const float delta = current - saved_inputs[index];
        if (delta != 0.0f && count < INPUT_DELTA_LIMIT) {
            indices[count] = index;
            values[count++] = delta;
        }
    };
    const int aux_count = tesauro ? 2 : 26;
    for (int side = 0; side < 2; ++side) {
        const int offset = side * (96 + aux_count);
        for (unsigned bits = changed; bits; bits &= bits - 1) {
            const unsigned first = bits & (0u - bits);
            const int point = highest_set_point(first);
            const int n = side ? -board[point] : board[point];
            const int previous = side ? -base[point] : base[point];
            if (n <= 0 && previous <= 0) continue;
            const float* row = point_feature_row(n);
            for (int k = 0; k < 4; ++k) append(offset + 4 * (point - 1) + k, row[k]);
        }
        const float* aux = side ? o_aux : p_aux;
        for (int k = 0; k < aux_count; ++k) append(offset + 96 + k, aux[k]);
    }
    return count;
}

// ======================== Game Plan Classification ========================

const char* game_plan_name(GamePlan gp) {
    switch (gp) {
        case GamePlan::PURERACE:  return "purerace";
        case GamePlan::RACING:    return "racing";
        case GamePlan::ATTACKING: return "attacking";
        case GamePlan::PRIMING:   return "priming";
        case GamePlan::ANCHORING: return "anchoring";
        default:                  return "unknown";
    }
}

GamePlan classify_game_plan(const Board& board) {
    if (is_race(board)) return GamePlan::PURERACE;
    auto [player_pips, opponent_pips] = pip_counts(board);
    return classify_contact_game_plan(board, player_pips, opponent_pips);
}

GamePlan classify_contact_game_plan(const Board& board, int player_pips, int opponent_pips) {
    // Count opponent back checkers (opponent checkers in player's home board + bar)
    int n_opp_back = board[0];  // opponent on bar
    for (int i = 1; i <= 6; ++i) {
        if (board[i] < 0) n_opp_back -= board[i];
    }

    if (n_opp_back == 0) {
        // No opponent back checkers to prime or blitz
        if (player_pips < opponent_pips) {
            return GamePlan::RACING;
        }

        // Does player have at least two back checkers?
        int n_player_back = 0;
        for (int i = 19; i <= 24; ++i) {
            if (board[i] > 0) n_player_back += board[i];
        }
        // bar counts as back too
        n_player_back += board[25];

        if (n_player_back > 1) {
            return GamePlan::ANCHORING;
        }

        // Check if opponent has more borne off and no offside checkers
        int n_player_borne = player_borne_off(board);
        int n_opp_borne = opponent_borne_off(board);
        int n_opp_offside = board[0];  // bar counts
        for (int i = 1; i <= 13; ++i) {
            if (board[i] < 0) n_opp_offside -= board[i];
        }

        if (n_opp_borne > n_player_borne && n_opp_offside == 0) {
            return GamePlan::ATTACKING;
        }
        return GamePlan::RACING;
    }

    // Opponent has back checkers
    if (player_pips < opponent_pips) {
        // Player is ahead in the race
        // Check if this is an endgame bearing-in situation
        int n_out = 0;
        for (int i = 7; i <= 12; ++i) {
            if (board[i] > 0) n_out += board[i];
        }
        int n_rest = 0;
        for (int i = 13; i <= 24; ++i) {
            if (board[i] > 0) n_rest += board[i];
        }
        n_rest += board[25];  // bar

        if (n_rest == 0 && n_out < 4) {
            return GamePlan::RACING;
        }

        // Count player checkers in their zone (points 1-12)
        int n_zone = 0;
        for (int i = 1; i <= 12; ++i) {
            if (board[i] > 0) n_zone += board[i];
        }

        if (n_zone >= 8) {
            // Player has ammunition. Attack, race, prime, or anchor?

            // Opponent holding at least two anchors (back game)? -> Race
            int n_opp_anchors = 0;
            for (int i = 1; i <= 6; ++i) {
                if (board[i] < -1) n_opp_anchors++;
            }
            if (n_opp_anchors >= 2) {
                return GamePlan::RACING;
            }

            // Opponent has blot on the bar? -> Attack
            if (board[0] > 0) {
                return GamePlan::ATTACKING;
            }

            // Opponent has blot in player's home board?
            int opp_blot_pnt = -1;
            for (int i = 1; i <= 6; ++i) {
                if (board[i] == -1) {
                    opp_blot_pnt = i;
                    break;
                }
            }

            if (opp_blot_pnt >= 0) {
                // Does player have a checker above the blot?
                bool player_checker_above = false;
                for (int i = opp_blot_pnt + 1; i <= 6; ++i) {
                    if (board[i] > 0) {
                        player_checker_above = true;
                        break;
                    }
                }
                return player_checker_above ? GamePlan::ATTACKING : GamePlan::RACING;
            }

            // Check for priming possibility
            int n_player_back = 0;
            for (int i = 19; i <= 24; ++i) {
                if (board[i] > 0) n_player_back += board[i];
            }
            n_player_back += board[25];

            if (n_player_back < 2) {
                // Prime if at least a 3-prime in front of opponent's highest checker
                int opp_max = board[0] > 0 ? 0 : -1;
                for (int i = 1; i <= 5; ++i) {
                    if (board[i] < 0) opp_max = i;
                }

                if (opp_max != -1) {
                    int max_prime = 0, n_prime = 0;
                    for (int i = opp_max + 1; i <= 11; ++i) {
                        if (board[i] > 1) {
                            n_prime++;
                            if (n_prime > max_prime) max_prime = n_prime;
                        } else {
                            n_prime = 0;
                        }
                    }
                    if (max_prime >= 3) {
                        return GamePlan::PRIMING;
                    }
                }
                return GamePlan::RACING;
            }

            // Otherwise attack
            return GamePlan::ATTACKING;
        } else {
            // Not enough ammo
            return GamePlan::RACING;
        }
    } else {
        // Player is behind in the race (player_pips >= opponent_pips)
        int n_player_borne = player_borne_off(board);
        int n_opp_borne = opponent_borne_off(board);
        int n_player_offside = board[25];  // bar
        for (int i = 13; i <= 24; ++i) {
            if (board[i] > 0) n_player_offside += board[i];
        }
        int n_opp_offside = board[0];
        for (int i = 1; i <= 13; ++i) {
            if (board[i] < 0) n_opp_offside -= board[i];
        }

        // End game: player has more borne off and no offside checkers
        if (n_player_borne > n_opp_borne && n_player_offside == 0) {
            return GamePlan::RACING;
        }

        // Check for deep home board point with gap
        int n_home_anchors = 0;
        for (int i = 1; i <= 6; ++i) {
            if (board[i] > 1) n_home_anchors++;
        }

        bool has_deep_point = false;
        for (int i = 1; i <= 3; ++i) {
            if (board[i] >= 2) { has_deep_point = true; break; }
        }

        bool has_gap = false;
        for (int i = 4; i <= 6; ++i) {
            if (board[i] <= 0) { has_gap = true; break; }
        }

        if (n_home_anchors < 4 && has_deep_point && has_gap) {
            double pip_diff_ratio = (player_pips + opponent_pips > 0) ?
                2.0 * (player_pips - opponent_pips) / (player_pips + opponent_pips) : 0.0;

            if (pip_diff_ratio <= 0.1) {
                return GamePlan::ATTACKING;
            }

            int n_player_back = 0;
            for (int i = 19; i <= 24; ++i) {
                if (board[i] > 0) n_player_back += board[i];
            }
            n_player_back += board[25];

            return (n_player_back < 2) ? GamePlan::RACING : GamePlan::ANCHORING;
        }

        // Check for opponent high anchor blocking priming
        bool opp_has_high_anchor = false;
        for (int i = 4; i <= 6; ++i) {
            if (board[i] <= -2) { opp_has_high_anchor = true; break; }
        }

        if (opp_has_high_anchor) {
            // No priming possible
            int n_player_back = 0;
            for (int i = 19; i <= 24; ++i) {
                if (board[i] > 0) n_player_back += board[i];
            }
            n_player_back += board[25];

            if (n_player_back >= 2) {
                return GamePlan::ANCHORING;
            }

            // Attack if opponent has blots and pip diff is small
            int n_opp_blots = board[0] > 0 ? 1 : 0;
            for (int i = 1; i <= 12; ++i) {
                if (board[i] == -1) n_opp_blots++;
            }

            double pip_diff_ratio = (player_pips + opponent_pips > 0) ?
                2.0 * (player_pips - opponent_pips) / (player_pips + opponent_pips) : 0.0;

            if (n_opp_blots > 0 && pip_diff_ratio <= 0.1) {
                return GamePlan::ATTACKING;
            }
            return GamePlan::RACING;
        }

        // No opponent high anchor - check for priming position
        int n_primers = 0;
        for (int i = 4; i <= 12; ++i) {
            if (board[i] >= 1) n_primers++;
        }
        int n_primers_bar = 0;
        for (int i = 5; i <= 8; ++i) {
            if (board[i] >= 1) n_primers_bar++;
        }

        if (n_primers >= 4 || n_primers_bar >= 3) {
            return GamePlan::PRIMING;
        }

        // Attack, anchor, or prime fallback
        int n_opp_blots = board[0] > 0 ? 1 : 0;
        for (int i = 1; i <= 12; ++i) {
            if (board[i] == -1) n_opp_blots++;
        }

        double pip_diff_ratio = (player_pips + opponent_pips > 0) ?
            2.0 * (player_pips - opponent_pips) / (player_pips + opponent_pips) : 0.0;

        if (n_opp_blots > 0 && pip_diff_ratio <= 0.1) {
            return GamePlan::ATTACKING;
        }

        int n_player_back = 0;
        for (int i = 19; i <= 24; ++i) {
            if (board[i] > 0) n_player_back += board[i];
        }
        n_player_back += board[25];

        return (n_player_back >= 2) ? GamePlan::ANCHORING : GamePlan::PRIMING;
    }
}

// --- Game plan pair strategy helpers ---

int game_plan_pair_index(GamePlan player, GamePlan opponent) {
    int p = static_cast<int>(player);   // RACING=1..ANCHORING=4
    int o = static_cast<int>(opponent);
    return (p - 1) * 4 + (o - 1);
}

static const std::array<const char*, NUM_PAIR_NNS> PAIR_NAMES = {
    "purerace",
    "race_race",  "race_att",  "race_prim",  "race_anch",
    "att_race",   "att_att",   "att_prim",   "att_anch",
    "prim_race",  "prim_att",  "prim_prim",  "prim_anch",
    "anch_race",  "anch_att",  "anch_prim",  "anch_anch",
};

const char* game_plan_pair_name(int pair_idx) {
    if (pair_idx < 0 || pair_idx >= NUM_PAIR_NNS) return "unknown";
    return PAIR_NAMES[pair_idx];
}

const std::array<const char*, NUM_PAIR_NNS>& game_plan_pair_names() {
    return PAIR_NAMES;
}

} // namespace bgbot
