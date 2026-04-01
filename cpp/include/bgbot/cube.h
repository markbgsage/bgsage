#pragma once

#include "strategy.h"
#include "multipy.h"
#include "match_equity.h"
#include "types.h"
#include <array>
#include <memory>

namespace bgbot {

// Who owns the doubling cube.
// CENTERED: either player can double (initial state).
// PLAYER: the player on roll owns the cube.
// OPPONENT: the opponent owns the cube.
enum class CubeOwner { CENTERED, PLAYER, OPPONENT };

// Current state of the doubling cube (money game or match play).
struct CubeInfo {
    int cube_value = 1;                      // 1, 2, 4, 8, ...
    CubeOwner owner = CubeOwner::CENTERED;
    MatchInfo match;                         // Default: {0,0,false} = money game
    float cube_x_override = -1.0f;          // If >= 0, override auto-detected cube efficiency
    bool jacoby = false;                     // Jacoby rule (money games only)
    bool beaver = false;                     // Beaver rule (money games only)
    int max_cube_value = 0;                  // 0 = unlimited, >0 = cap (1 = cubeless)

    bool is_money() const { return match.is_money(); }

    // Is Jacoby rule currently active? Only when: money game + Jacoby enabled +
    // cube has never been turned (centered). Once the cube is turned, gammons count.
    bool jacoby_active() const {
        return jacoby && is_money() && owner == CubeOwner::CENTERED;
    }
};

// Flip cube ownership (PLAYER <-> OPPONENT, CENTERED stays).
// Used when flipping the board perspective after a half-move.
inline CubeOwner flip_owner(CubeOwner o) {
    if (o == CubeOwner::PLAYER) return CubeOwner::OPPONENT;
    if (o == CubeOwner::OPPONENT) return CubeOwner::PLAYER;
    return CubeOwner::CENTERED;
}

// Flip a CubeInfo to the opponent's perspective.
// PLAYER ↔ OPPONENT, CENTERED stays. Match away values swapped.
inline CubeInfo flip_cube_perspective(const CubeInfo& cube) {
    CubeInfo opp;
    opp.cube_value = cube.cube_value;
    opp.owner = flip_owner(cube.owner);
    opp.match = cube.match.flip();
    opp.cube_x_override = cube.cube_x_override;
    opp.jacoby = cube.jacoby;
    opp.beaver = cube.beaver;
    opp.max_cube_value = cube.max_cube_value;
    return opp;
}

// Can the player on roll legally double?
inline bool can_double(const CubeInfo& ci) {
    // Cube at or above max → cannot double (max_cube_value=1 means cubeless)
    if (ci.max_cube_value > 0 && ci.cube_value >= ci.max_cube_value)
        return false;
    if (!ci.is_money()) {
        return can_double_match(ci.match.away1, ci.match.away2,
                                ci.cube_value, ci.owner, ci.match.is_crawford);
    }
    return ci.owner == CubeOwner::CENTERED || ci.owner == CubeOwner::PLAYER;
}

// True when the cube can never be turned again.
// Used to skip all cubeful overhead (Janowski, cubeful VR, cube checks).
// Typical use: max_cube_value=1 makes the entire game cubeless.
inline bool cube_is_dead(const CubeInfo& ci) {
    return ci.max_cube_value > 0 && ci.cube_value >= ci.max_cube_value;
}

// Compute W (average win value) and L (average loss value) from cubeless probs.
// W = 1 + (P(gw) + P(bw)) / P(win)
// L = 1 + (P(gl) + P(bl)) / (1 - P(win))
// Edge cases: if P(win)=0, W=1; if P(win)=1, L=1.
void compute_WL(const std::array<float, NUM_OUTPUTS>& probs, float& W, float& L);

// Live cube equity (piecewise linear, depends on cube ownership).
// Returns equity normalized to cube value 1.
float money_live(float W, float L, float p_win, CubeOwner owner,
                 bool jacoby = false);

// Cubeless equity from probabilities.
// E = 2*P(win) - 1 + P(gw) - P(gl) + P(bw) - P(bl)
inline float cubeless_equity(const std::array<float, NUM_OUTPUTS>& probs) {
    return 2.0f * probs[0] - 1.0f + probs[1] - probs[3] + probs[2] - probs[4];
}

// Cubeless-to-cubeful conversion (Janowski interpolation).
// E_cubeful = E_dead * (1 - x) + E_live * x
// Returns cubeful equity normalized to cube value 1.
float cl2cf_money(const std::array<float, NUM_OUTPUTS>& probs,
                  CubeOwner owner, float cube_x);

// Jacoby-aware overload: when jacoby_active, gammons/backgammons are zeroed
// (W=1, L=1, dead-cube equity = 2*P(win)-1).
float cl2cf_money(const std::array<float, NUM_OUTPUTS>& probs,
                  CubeOwner owner, float cube_x, bool jacoby_active);

// Cube efficiency for a position.
// Contact/crashed: 0.68
// Race: 0.55 + 0.00125 * roller_pip_count, clamped to [0.6, 0.7]
float cube_efficiency(const Board& board, bool is_race_pos);

// ---------------------------------------------------------------------------
// Match play cubeful evaluation (Janowski in MWC space)
// ---------------------------------------------------------------------------

// Cubeless-to-cubeful for match play (returns MWC).
// Uses Janowski interpolation in MWC space with MET-based anchor points.
// Three internal variants by ownership (centered/owned/unavailable).
float cl2cf_match(const std::array<float, NUM_OUTPUTS>& probs,
                  const CubeInfo& cube, float cube_x);

// Unified cubeless-to-cubeful: dispatches to cl2cf_money() or cl2cf_match().
// For money game: returns cubeful equity.
// For match play: returns cubeful equity (MWC converted to equity via mwc2eq).
float cl2cf(const std::array<float, NUM_OUTPUTS>& probs,
            const CubeInfo& cube, float cube_x);

// Result of a cube decision analysis.
struct CubeDecision {
    float equity_nd;      // No Double equity (normalized to cube=1)
    float equity_dt;      // Double/Take equity (or Double/Beaver if is_beaver)
    float equity_dp;      // Double/Pass equity (+1.0 for money, MET-based for match)
    bool should_double;   // True if doubling is correct
    bool should_take;     // True if opponent should take (vs pass/beaver counts as take)
    float optimal_equity; // Cubeful equity after optimal play by both sides
    bool is_beaver = false; // True if opponent would beaver (DT field = DB equity)
};

// ---------------------------------------------------------------------------
// 2-ply detail structures for cube decision analysis
// ---------------------------------------------------------------------------

// Detail for a single opponent roll within the 2-ply detail view.
struct OpponentRollDetail {
    int die1, die2;
    Board post_move_board;    // Player's perspective (original player, not opponent)
    float cubeful_equity;     // Player's perspective, per-initial-cube
};

// Detail for a single player roll within the 2-ply detail view.
struct PlayerRollDetail {
    int die1, die2;
    Board post_move_board;    // Player's perspective
    float cubeful_equity;     // Player's perspective, per-initial-cube,
                              // incorporates opponent's optimal cube decision
    bool is_terminal = false; // Player's move ended the game (no opponent rolls)
    bool opponent_dp = false; // Opponent has Double/Pass (game over, no opponent rolls)
    std::vector<OpponentRollDetail> opponent_rolls; // 21 entries (empty if terminal/DP)
};

// Per-roll details for the first two turns of a cube decision analysis (ND scenario).
struct TwoPlyDetails {
    std::vector<PlayerRollDetail> player_rolls; // 21 entries
};

// Compute cube decision for a pre-roll position (1-ply / raw NN).
// `probs` are cubeless pre-roll probabilities from the player's perspective.
// Uses Janowski interpolation with the given cube efficiency.
CubeDecision cube_decision_1ply(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x);

// Convenience: compute cube decision using auto-detected cube efficiency.
CubeDecision cube_decision_1ply(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    const Board& board,
    bool is_race_pos);

// ---------------------------------------------------------------------------
// N-ply cubeful evaluation
// ---------------------------------------------------------------------------

// Compute cubeful equity for a pre-roll position at N-ply depth (money game).
// The position is from the player-on-roll's perspective (before rolling).
// Cube efficiency x is only used at 1-ply leaves (Janowski).
// At internal nodes, cube decisions emerge naturally from recursion.
//
// `owner` is from the player-on-roll's perspective:
//   PLAYER   = the player on roll owns the cube
//   OPPONENT = the opponent owns the cube
//   CENTERED = cube is in the center
//
// Returns cubeful equity normalized to cube value 1.
//
// Checker play uses cubeless evaluation (negligible impact on move selection).
float cubeful_equity_nply(
    const Board& board,           // pre-roll, player's perspective
    CubeOwner owner,              // cube ownership from player's perspective
    const Strategy& strategy,     // cubeless base strategy
    int n_plies,
    const MoveFilter& filter = MoveFilters::TINY,
    int n_threads = 1,
    const Strategy* move_filter = nullptr);

// Compute cubeful equity for a pre-roll position at N-ply depth (money or match).
// Uses full CubeInfo including match state. For money game, dispatches to
// the CubeOwner overload. For match play, works in MWC space internally
// and returns cubeful equity (MWC converted via mwc2eq).
float cubeful_equity_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter = MoveFilters::TINY,
    int n_threads = 1,
    const Strategy* move_filter = nullptr);

// Compute full cube decision at N-ply depth.
// `board` is pre-roll from the player's perspective.
// Returns CubeDecision with ND/DT/DP equities and optimal play decisions.
// move_filter: optional cheap strategy (e.g. PubEval) for pre-filtering candidates
// in move selection before evaluating survivors with the full strategy.
CubeDecision cube_decision_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter = MoveFilters::TINY,
    int n_threads = 1,
    const Strategy* move_filter = nullptr);

// Compute full cube decision at N-ply depth, with per-roll details for the
// first two turns (player + opponent) under the No Double scenario.
// Requires n_plies >= 3. Populates `details` with 21 player-roll entries,
// each containing 21 opponent-roll entries (unless terminal or opponent D/P).
// All equities are per-initial-cube, from the player's perspective.
CubeDecision cube_decision_nply_with_details(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    TwoPlyDetails& details,
    const Strategy* move_filter = nullptr);

// Compute cube decision from cubeless pre-roll probabilities (Janowski conversion).
// This is the simplest form of cubeful evaluation from any source of cubeless probs
// (rollout, N-ply, etc.) — just applies Janowski to the given probs.
// Same as cube_decision_1ply but with auto-detected cube efficiency.
inline CubeDecision cube_decision_from_probs(
    const std::array<float, NUM_OUTPUTS>& cubeless_probs,
    const CubeInfo& cube,
    const Board& board,
    bool is_race_pos)
{
    return cube_decision_1ply(cubeless_probs, cube, board, is_race_pos);
}

// Profiling counters for cubeful recursion (debug only)
void reset_cubeful_counters();
void print_cubeful_counters();

} // namespace bgbot
