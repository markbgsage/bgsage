#pragma once

#include <array>

namespace bgbot {

// Forward declaration
enum class CubeOwner;

constexpr int MAX_MATCH_LENGTH = 25;

// Match state from the perspective of the player on roll.
// away1/away2 = 0 means money game (no match).
struct MatchInfo {
    int away1 = 0;          // Points player needs to win (0 = money game)
    int away2 = 0;          // Points opponent needs to win (0 = money game)
    bool is_crawford = false;

    bool is_money() const { return away1 == 0 && away2 == 0; }
    bool is_post_crawford() const {
        return !is_crawford && (away1 == 1 || away2 == 1);
    }
    MatchInfo flip() const { return {away2, away1, is_crawford}; }
};

// ---------------------------------------------------------------------------
// Kazaross-XG2 Match Equity Table (hardcoded)
// ---------------------------------------------------------------------------

// Pre-Crawford MET: MET_PRE[i][j] = MWC for player needing (i+1) points
// when opponent needs (j+1) points. 0-indexed.
// Source: gnubg_src/met/Kazaross-XG2.xml
extern const std::array<std::array<float, MAX_MATCH_LENGTH>, MAX_MATCH_LENGTH> MET_PRE;

// Post-Crawford MET: MET_POST_CRAWFORD[i] = trailer's MWC when trailer needs
// (i+1) points and leader needs 1 point.
// Index 0 = DMP (1-away, 1-away) = 0.5
extern const std::array<float, MAX_MATCH_LENGTH> MET_POST_CRAWFORD;

// ---------------------------------------------------------------------------
// MET lookup functions
// ---------------------------------------------------------------------------

// Core MET lookup: returns MWC for the player needing away1 points.
// away1 <= 0 → 1.0 (player won); away2 <= 0 → 0.0 (opponent won).
// is_post_crawford: true only when the Crawford game has already been played
// and someone is still 1-away. The pre-Crawford table (MET_PRE) has correct
// values for 1-away positions before Crawford occurs.
float get_met(int away1, int away2, bool is_post_crawford = false);

// MWC after nPoints are scored.
// player_wins=true: player wins nPoints (player's away decreases).
// player_wins=false: opponent wins nPoints (opponent's away decreases).
float get_met_after(int away1, int away2, int nPoints,
                    bool player_wins, bool is_crawford);

// Cubeless MWC from 5 NN probability outputs and MET.
// Decomposes cumulative probs into exclusive probs, weights by 6 outcome MWCs.
float cubeless_mwc(const std::array<float, 5>& probs,
                   int away1, int away2, int cube_value, bool is_crawford);

// Linear equity ↔ MWC conversions, anchored at win/lose cube_value points.
float eq2mwc(float equity, int away1, int away2, int cube_value,
             bool is_crawford);
float mwc2eq(float mwc, int away1, int away2, int cube_value,
             bool is_crawford);

// Can the player on roll legally double in match play?
bool can_double_match(int away1, int away2, int cube_value,
                      CubeOwner owner, bool is_crawford);

// Double/Pass MWC: MWC for the doubler when opponent passes.
// (Player wins cube_value points.)
float dp_mwc(int away1, int away2, int cube_value, bool is_crawford);

} // namespace bgbot
