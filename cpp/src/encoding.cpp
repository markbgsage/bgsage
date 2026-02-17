#include "bgbot/encoding.h"
#include "bgbot/board.h"
#include <algorithm>
#include <cmath>
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
                    int g = std::gcd(d1, d2);
                    int n1 = d1 / g;
                    int n2 = d2 / g;
                    if (g < 7 && n1 + n2 < 5) {
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

static float acontainment_fast(const int player_hb[25], int opp_back) {
    int gnubg_nOppBack = opp_back > 0 ? opp_back - 1 : -1;

    int n = 36;
    int limit = (gnubg_nOppBack >= 0) ? 24 - gnubg_nOppBack : 24;
    for (int gnubg_i = 15; gnubg_i < limit; ++gnubg_i) {
        int j = escapes(player_hb, gnubg_i);
        if (j < n) n = j;
    }
    return (36.0f - n) / 36.0f;
}

static float containment_fast(const int player_hb[25]) {
    int n = 36;
    for (int gnubg_i = 15; gnubg_i < 24; ++gnubg_i) {
        int j = escapes(player_hb, gnubg_i);
        if (j < n) n = j;
    }
    return (36.0f - n) / 36.0f;
}

static int mobility_fast(const Board& board, const int opp_hb[25]) {
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
            n += (gnubg_i - 5) * checkers * escapes(opp_hb, gnubg_i);
        }
    }
    return n;
}

static int back_rescue_escapes_fast(const int player_hb[25], int opp_back) {
    int gnubg_nOppBack = opp_back > 0 ? opp_back - 1 : -1;
    if (gnubg_nOppBack < 0) return 36;
    int escape_point = 23 - gnubg_nOppBack;
    if (escape_point < 0) return 0;
    return escapes1(player_hb, escape_point);
}

// Compute all extended features for one side (player) of the board.
// `board` is from the player's perspective. `opp_back` is the highest point
// with opponent checkers (from player's view). `player_hb` and `opp_hb` are
// precomputed half-board arrays.
static void compute_side_features(
    const Board& board,
    int opp_back,
    const int player_hb[25],
    const int opp_hb[25],
    // Outputs:
    int& out_break_contact,
    int& out_free_pip,
    int& out_piploss,
    float& out_acontain,
    float& out_contain,
    int& out_mobility,
    int& out_moment2_val,
    int& out_enter,
    int& out_timing_val,
    float& out_backbone_val,
    float& out_backg_val,
    float& out_backg1_val,
    int& out_backrescapes)
{
    out_break_contact = break_contact_fast(board, opp_back);
    out_free_pip = free_pip_fast(board, opp_back);
    out_piploss = compute_piploss(board);
    out_acontain = acontainment_fast(player_hb, opp_back);
    out_contain = containment_fast(player_hb);
    out_mobility = mobility_fast(board, opp_hb);
    out_moment2_val = moment2(board);
    out_enter = enter_loss(board);
    out_timing_val = timing_fast(board, opp_back);
    out_backbone_val = backbone(board);
    out_backg_val = backg(board);
    out_backg1_val = backg1(board);
    out_backrescapes = back_rescue_escapes_fast(player_hb, opp_back);
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

    // Precompute shared data once
    // Extract half-boards once (used by containment, acontainment, mobility, backrescapes).
    int player_hb[25], opp_hb[25];
    extract_player_half(board, player_hb);
    extract_opponent_half(board, opp_hb);

    // Flipped half-boards are a permutation of the same arrays:
    // player of flipped == opponent half-board of original, and vice versa.
    int flipped_player_hb[25], flipped_opp_hb[25];
    std::copy(std::begin(opp_hb), std::end(opp_hb), std::begin(flipped_player_hb));
    std::copy(std::begin(player_hb), std::end(player_hb), std::begin(flipped_opp_hb));

    // Compute both back-checker values in one pass:
    // p_opp_back: highest point with opponent checkers (board perspective).
    // o_opp_back: highest point with opponent checkers in flipped board
    //             => lowest positive point for the original player.
    int p_opp_back = 0;
    int o_opp_back = 0;
    for (int i = 1; i <= 24; ++i) {
        const int fwd_pt = i;
        const int rev_pt = 25 - i;
        if (p_opp_back == 0 && board[rev_pt] < 0) p_opp_back = rev_pt;
        if (o_opp_back == 0 && board[fwd_pt] > 0) o_opp_back = 25 - fwd_pt;
    }

    const Board flipped = flip(board);

    // Initialize escape tables once
    init_escape_tables();

    // Count borne-off checkers
    int player_off = player_borne_off(board);
    int opp_off = opponent_borne_off(board);

    // Simple features (cheap)
    auto exit_probs = prob_no_enter_from_bar(board);
    auto fwd_anchors = forward_anchor_points(board);
    int p_back_esc = back_escapes(board);
    int o_back_esc = back_escapes(flipped);
    int p_max_pt = max_point(board);
    int o_max_pt = max_point(flipped);
    int p_max_anch = max_anchor_point(board);
    int o_max_anch = max_anchor_point(flipped);

    // Hitting shots (expensive but no redundant computation to eliminate)
    int p_shots = hitting_shots(board);
    int o_shots = hitting_shots(flipped);
    int p_dbl_shots = double_hitting_shots(board);
    int o_dbl_shots = double_hitting_shots(flipped);

    // Compute all GNUbg features for player using precomputed data
    int p_break_contact, p_free_pip, p_piploss, p_mobility_val, p_moment2_val, p_enter, p_timing_val, p_backrescapes;
    float p_acontain, p_contain, p_backbone_val, p_backg_val, p_backg1_val;
    compute_side_features(board, p_opp_back, player_hb, opp_hb,
        p_break_contact, p_free_pip, p_piploss, p_acontain, p_contain,
        p_mobility_val, p_moment2_val, p_enter, p_timing_val,
        p_backbone_val, p_backg_val, p_backg1_val, p_backrescapes);

    // Compute all GNUbg features for opponent using precomputed data
    int o_break_contact, o_free_pip, o_piploss, o_mobility_val, o_moment2_val, o_enter, o_timing_val, o_backrescapes;
    float o_acontain, o_contain, o_backbone_val, o_backg_val, o_backg1_val;
    compute_side_features(flipped, o_opp_back, flipped_player_hb, flipped_opp_hb,
        o_break_contact, o_free_pip, o_piploss, o_acontain, o_contain,
        o_mobility_val, o_moment2_val, o_enter, o_timing_val,
        o_backbone_val, o_backg_val, o_backg1_val, o_backrescapes);

    // === Player 1 features [96-121] ===
    inputs[96]  = board[25] / 2.0f;  // bar
    inputs[97]  = player_off >= 5 ? 1.0f : player_off / 5.0f;
    inputs[98]  = player_off >= 10 ? 1.0f : (player_off <= 5 ? 0.0f : (player_off - 5) / 5.0f);
    inputs[99]  = player_off <= 10 ? 0.0f : (player_off - 10) / 5.0f;
    inputs[100] = exit_probs.player;                                          // I_ENTER2
    inputs[101] = fwd_anchors.player == 0 ? 2.0f : fwd_anchors.player / 6.0f; // I_FORWARD_ANCHOR
    inputs[102] = p_shots / 36.0f;                                            // I_P1
    inputs[103] = p_dbl_shots / 36.0f;                                        // I_P2
    inputs[104] = p_back_esc / 36.0f;                                         // I_BACKESCAPES
    inputs[105] = p_max_pt / 24.0f;                                           // I_BACK_CHEQUER
    inputs[106] = p_max_anch / 24.0f;                                         // I_BACK_ANCHOR
    inputs[107] = p_break_contact / 167.0f;                                   // I_BREAK_CONTACT
    inputs[108] = p_free_pip / 100.0f;                                        // I_FREEPIP
    inputs[109] = p_piploss / (12.0f * 36.0f);                                // I_PIPLOSS
    inputs[110] = p_acontain;                                                  // I_ACONTAIN
    inputs[111] = p_acontain * p_acontain;                                     // I_ACONTAIN2
    inputs[112] = p_contain;                                                   // I_CONTAIN
    inputs[113] = p_contain * p_contain;                                       // I_CONTAIN2
    inputs[114] = p_mobility_val / 3600.0f;                                    // I_MOBILITY
    inputs[115] = p_moment2_val / 400.0f;                                      // I_MOMENT2
    inputs[116] = p_enter / (36.0f * (49.0f / 6.0f));                          // I_ENTER
    inputs[117] = p_timing_val / 100.0f;                                       // I_TIMING
    inputs[118] = p_backbone_val;                                              // I_BACKBONE
    inputs[119] = p_backg_val;                                                 // I_BACKG
    inputs[120] = p_backg1_val;                                                // I_BACKG1
    inputs[121] = p_backrescapes / 36.0f;                                      // I_BACKRESCAPES

    // === Player 2 features [218-243] ===
    inputs[P2_OFFSET + 96]  = board[0] / 2.0f;  // bar (index 218)
    inputs[P2_OFFSET + 97]  = opp_off >= 5 ? 1.0f : opp_off / 5.0f;
    inputs[P2_OFFSET + 98]  = opp_off >= 10 ? 1.0f : (opp_off <= 5 ? 0.0f : (opp_off - 5) / 5.0f);
    inputs[P2_OFFSET + 99]  = opp_off <= 10 ? 0.0f : (opp_off - 10) / 5.0f;
    inputs[P2_OFFSET + 100] = exit_probs.opponent;
    inputs[P2_OFFSET + 101] = fwd_anchors.opponent == 0 ? 2.0f : fwd_anchors.opponent / 6.0f;
    inputs[P2_OFFSET + 102] = o_shots / 36.0f;
    inputs[P2_OFFSET + 103] = o_dbl_shots / 36.0f;
    inputs[P2_OFFSET + 104] = o_back_esc / 36.0f;
    inputs[P2_OFFSET + 105] = o_max_pt / 24.0f;
    inputs[P2_OFFSET + 106] = o_max_anch / 24.0f;
    inputs[P2_OFFSET + 107] = o_break_contact / 167.0f;
    inputs[P2_OFFSET + 108] = o_free_pip / 100.0f;
    inputs[P2_OFFSET + 109] = o_piploss / (12.0f * 36.0f);
    inputs[P2_OFFSET + 110] = o_acontain;
    inputs[P2_OFFSET + 111] = o_acontain * o_acontain;
    inputs[P2_OFFSET + 112] = o_contain;
    inputs[P2_OFFSET + 113] = o_contain * o_contain;
    inputs[P2_OFFSET + 114] = o_mobility_val / 3600.0f;
    inputs[P2_OFFSET + 115] = o_moment2_val / 400.0f;
    inputs[P2_OFFSET + 116] = o_enter / (36.0f * (49.0f / 6.0f));
    inputs[P2_OFFSET + 117] = o_timing_val / 100.0f;
    inputs[P2_OFFSET + 118] = o_backbone_val;
    inputs[P2_OFFSET + 119] = o_backg_val;
    inputs[P2_OFFSET + 120] = o_backg1_val;
    inputs[P2_OFFSET + 121] = o_backrescapes / 36.0f;

    return inputs;
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

} // namespace bgbot
