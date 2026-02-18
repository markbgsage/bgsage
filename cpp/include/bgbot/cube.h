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

    bool is_money() const { return match.is_money(); }
};

// Can the player on roll legally double?
inline bool can_double(const CubeInfo& ci) {
    if (!ci.is_money()) {
        return can_double_match(ci.match.away1, ci.match.away2,
                                ci.cube_value, ci.owner, ci.match.is_crawford);
    }
    return ci.owner == CubeOwner::CENTERED || ci.owner == CubeOwner::PLAYER;
}

// Compute W (average win value) and L (average loss value) from cubeless probs.
// W = 1 + (P(gw) + P(bw)) / P(win)
// L = 1 + (P(gl) + P(bl)) / (1 - P(win))
// Edge cases: if P(win)=0, W=1; if P(win)=1, L=1.
void compute_WL(const std::array<float, NUM_OUTPUTS>& probs, float& W, float& L);

// Live cube equity (piecewise linear, depends on cube ownership).
// No Jacoby rule. Returns equity normalized to cube value 1.
float money_live(float W, float L, float p_win, CubeOwner owner);

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
    float equity_dt;      // Double/Take equity (normalized to cube=1)
    float equity_dp;      // Double/Pass equity (+1.0 for money, MET-based for match)
    bool should_double;   // True if doubling is correct
    bool should_take;     // True if opponent should take (vs pass)
    float optimal_equity; // Cubeful equity after optimal play by both sides
};

// Compute cube decision for a pre-roll position (0-ply).
// `probs` are cubeless pre-roll probabilities from the player's perspective.
// Uses Janowski interpolation with the given cube efficiency.
CubeDecision cube_decision_0ply(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x);

// Convenience: compute cube decision using auto-detected cube efficiency.
CubeDecision cube_decision_0ply(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    const Board& board,
    bool is_race_pos);

// ---------------------------------------------------------------------------
// N-ply cubeful evaluation
// ---------------------------------------------------------------------------

// Compute cubeful equity for a pre-roll position at N-ply depth (money game).
// The position is from the player-on-roll's perspective (before rolling).
// Cube efficiency x is only used at 0-ply leaves (Janowski).
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
    int n_threads = 1);

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
    int n_threads = 1);

// Compute full cube decision at N-ply depth.
// `board` is pre-roll from the player's perspective.
// Returns CubeDecision with ND/DT/DP equities and optimal play decisions.
CubeDecision cube_decision_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter = MoveFilters::TINY,
    int n_threads = 1);

// Compute cube decision from cubeless pre-roll probabilities (Janowski conversion).
// This is the simplest form of cubeful evaluation from any source of cubeless probs
// (rollout, N-ply, etc.) — just applies Janowski to the given probs.
// Same as cube_decision_0ply but with auto-detected cube efficiency.
inline CubeDecision cube_decision_from_probs(
    const std::array<float, NUM_OUTPUTS>& cubeless_probs,
    const CubeInfo& cube,
    const Board& board,
    bool is_race_pos)
{
    return cube_decision_0ply(cubeless_probs, cube, board, is_race_pos);
}

} // namespace bgbot
