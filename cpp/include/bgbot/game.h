#pragma once

#include "types.h"
#include "strategy.h"
#include <random>

namespace bgbot {

// Statistics from a set of games, from player 1's perspective.
struct GameStats {
    int n_games = 0;
    int p1_wins = 0;        // single wins
    int p1_gammons = 0;     // gammon wins
    int p1_backgammons = 0; // backgammon wins
    int p2_wins = 0;        // single losses (p2 wins)
    int p2_gammons = 0;     // gammon losses
    int p2_backgammons = 0; // backgammon losses

    double p1_win_frac() const { return n_games ? (double)p1_wins / n_games : 0; }
    double p1_gammon_frac() const { return n_games ? (double)p1_gammons / n_games : 0; }
    double p1_backgammon_frac() const { return n_games ? (double)p1_backgammons / n_games : 0; }
    double p2_win_frac() const { return n_games ? (double)p2_wins / n_games : 0; }
    double p2_gammon_frac() const { return n_games ? (double)p2_gammons / n_games : 0; }
    double p2_backgammon_frac() const { return n_games ? (double)p2_backgammons / n_games : 0; }

    // Average points per game for player 1
    double avg_ppg() const;

    void record(GameResult result);
    void merge(const GameStats& other);
};

// Play a single game of backgammon. No doubling cube (cubeless).
// Both strategies see the board from "their" perspective (player on roll = player 1).
// Returns the game result from player 1's perspective.
GameResult play_game(const Strategy& strat1, const Strategy& strat2, std::mt19937& rng);

// Play n_games and accumulate statistics (serial).
GameStats play_games(const Strategy& strat1, const Strategy& strat2,
                     int n_games, uint32_t seed);

// Play n_games in parallel across n_threads. Each thread gets an independent RNG.
// n_threads=0 means auto-detect.
GameStats play_games_parallel(const Strategy& strat1, const Strategy& strat2,
                              int n_games, uint32_t seed, int n_threads = 0);

} // namespace bgbot
