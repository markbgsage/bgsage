#pragma once

#include "types.h"
#include <array>

namespace bgbot {

constexpr int TESAURO_INPUTS = 196;
constexpr int EXTENDED_CONTACT_INPUTS = 244;

// Compute the 196 Tesauro inputs from a board position.
// The board is always from player 1's perspective (player on roll).
//
// Layout:
//   Indices 0-95:   player 1 checkers on points 1-24 (4 inputs per point)
//   Index 96:       player 1 checkers on bar / 2.0
//   Index 97:       player 1 checkers borne off / 15.0
//   Indices 98-193: player 2 checkers on points 1-24 (4 inputs per point)
//   Index 194:      player 2 checkers on bar / 2.0
//   Index 195:      player 2 checkers borne off / 15.0
//
// Per-point encoding (4 inputs):
//   Input 0: 1.0 if >= 1 checker, else 0
//   Input 1: 1.0 if >= 2 checkers, else 0
//   Input 2: 1.0 if >= 3 checkers, else 0
//   Input 3: (n - 3) / 2.0 if >= 4 checkers, else 0
std::array<float, TESAURO_INPUTS> compute_tesauro_inputs(const Board& board);

// Compute the 244 extended contact inputs for contact/crashed positions.
// Extends Tesauro encoding with GNUbg-style features (22 per player).
//
// Layout (122 features per player, player 2 offset = 122):
//   [0-95]:    Tesauro point encoding (4 per point × 24)
//   [96]:      Bar / 2.0
//   [97-99]:   Borne-off (3 buckets: 0-5, 5-10, 10-15)
//   [100]:     I_ENTER2: bar-escape probability
//   [101]:     I_FORWARD_ANCHOR: forward anchor / 6
//   [102]:     I_P1: hitting shots / 36
//   [103]:     I_P2: double hitting shots / 36
//   [104]:     I_BACKESCAPES: back escapes / 36
//   [105]:     I_BACK_CHEQUER: max point / 24
//   [106]:     I_BACK_ANCHOR: max anchor point / 24
//   [107]:     I_BREAK_CONTACT: break contact / 167
//   [108]:     I_FREEPIP: free pips / 100
//   [109]:     I_PIPLOSS: piploss / (12*36)
//   [110]:     I_ACONTAIN: containment to opp back
//   [111]:     I_ACONTAIN2: acontain squared
//   [112]:     I_CONTAIN: containment to home
//   [113]:     I_CONTAIN2: contain squared
//   [114]:     I_MOBILITY: mobility / 3600
//   [115]:     I_MOMENT2: second moment / 400
//   [116]:     I_ENTER: bar entry loss / (36*49/6)
//   [117]:     I_TIMING: timing / 100
//   [118]:     I_BACKBONE: backbone [0,1]
//   [119]:     I_BACKG: backgame 2+ anchors
//   [120]:     I_BACKG1: backgame 1 anchor
//   [121]:     I_BACKRESCAPES: back rescue escapes / 36
//   [122-243]: Player 2 (same layout, offset by 122)
std::array<float, EXTENDED_CONTACT_INPUTS> compute_extended_contact_inputs(const Board& board);

// --- Helper functions for extended encoding ---
// These are exposed for testing/validation.

// Probability that a checker on the bar fails to enter.
// Returns (player_prob, opponent_prob).
// Computed as n_anchors² / 36 where n_anchors is opponent's anchors in their home board.
struct BarExitProbs {
    float player;    // prob that player fails to enter from bar
    float opponent;  // prob that opponent fails to enter from bar
};
BarExitProbs prob_no_enter_from_bar(const Board& board);

// Forward anchor points — the most forward anchor (2+ checkers) on the opponent's
// side of the board. Returns point number from opponent's perspective, or 0 if none.
struct ForwardAnchorPoints {
    int player;
    int opponent;
};
ForwardAnchorPoints forward_anchor_points(const Board& board);

// Number of rolls (out of 36) that hit at least one opponent blot.
int hitting_shots(const Board& board);

// Number of distinct rolls (out of 36) that hit at least two opponent blots.
int double_hitting_shots(const Board& board);

// Number of rolls (out of 36) that allow the player's back checker to escape to
// at least point 15 (past the mid-point). If no back checkers exist, returns 36.
int back_escapes(const Board& board);

// Point furthest from home that has at least one player checker (not including bar).
int max_point(const Board& board);

// Point furthest from home that has at least two player checkers (anchor; not including bar).
int max_anchor_point(const Board& board);

// --- GNUbg-style escape counting helpers (lookup table based) ---
// These use precomputed 4096-entry lookup tables for fast roll counting.

// Initialize the escape lookup tables. Must be called once before using
// escapes() or escapes1(). Thread-safe (uses std::call_once).
void init_escape_tables();

// Count rolls (out of 36) that let a checker escape from point `n` (1-24),
// considering blocked points (>=2 checkers) in the given half-board array.
// half_board[0..24] has checker counts for one player (unsigned).
// This matches GNUbg's Escapes() function.
int escapes(const int half_board[25], int n);

// Variant of escapes() that requires clearing past the first unblocked point.
// Matches GNUbg's Escapes1() function.
int escapes1(const int half_board[25], int n);

// Compute GNUbg-style piploss (average pips opponent loses from hits).
// Returns the raw pip sum (not normalized). Normalize by dividing by (12*36).
// Takes the board in our standard convention.
int compute_piploss(const Board& board);

// --- New GNUbg encoding features (per-player) ---

// Break contact: sum of (distance past opponent's back checker * checkers).
// Returns raw value; normalize by / 167.
int break_contact(const Board& board);

// Free pips: pips of checkers already past opponent's back checker.
// Returns raw value; normalize by / 100.
int free_pip(const Board& board);

// Timing: wastage heuristic for checker placement.
// Returns raw value; normalize by / 100.
int timing(const Board& board);

// Backbone: connectivity of anchor chain. Returns raw float in [0,1].
float backbone(const Board& board);

// Backgame strength with 2+ anchors in opponent's home.
// Returns raw value; normalize by / 4.
float backg(const Board& board);

// Backgame strength with exactly 1 anchor in opponent's home.
// Returns raw value; normalize by / 8.
float backg1(const Board& board);

// Average pips lost when on the bar (I_ENTER).
// Returns raw loss value; normalize by / (36 * 49/6).
int enter_loss(const Board& board);

// Containment: (36 - min_escapes) / 36 from point 15 to point 24.
// Returns float in [0, 1].
float containment(const Board& board);

// Containment from point 15 to opponent's back checker.
// Returns float in [0, 1].
float acontainment(const Board& board);

// Mobility: sum of (distance_from_home * checkers * escapes).
// Returns raw value; normalize by / 3600.
int mobility(const Board& board);

// Second moment of checker distribution (spread above mean).
// Returns raw value; normalize by / 400.
int moment2(const Board& board);

// Back rescue escapes using Escapes1 variant.
// Returns raw escape count; normalize by / 36.
int back_rescue_escapes(const Board& board);

// Game plan classification for 5-NN strategy.
// Classifies the player's game plan from the pre-roll checker layout.
// PURERACE = contact broken (is_race() true), RACING = game plan racing with contact.
enum class GamePlan { PURERACE, RACING, ATTACKING, PRIMING, ANCHORING };

// Classify the game plan for the player on roll.
// Board must be from the player's perspective (player = positive checkers).
GamePlan classify_game_plan(const Board& board);

// Human-readable name for a game plan.
const char* game_plan_name(GamePlan gp);

} // namespace bgbot
