#include "bgbot/game.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include <cassert>
#include <thread>
#include <algorithm>

namespace bgbot {

double GameStats::avg_ppg() const {
    if (n_games == 0) return 0;
    int total = p1_wins + 2 * p1_gammons + 3 * p1_backgammons
              - p2_wins - 2 * p2_gammons - 3 * p2_backgammons;
    return (double)total / n_games;
}

void GameStats::record(GameResult result) {
    n_games++;
    switch (result) {
        case GameResult::WIN_SINGLE:      p1_wins++; break;
        case GameResult::WIN_GAMMON:       p1_gammons++; break;
        case GameResult::WIN_BACKGAMMON:   p1_backgammons++; break;
        case GameResult::LOSS_SINGLE:      p2_wins++; break;
        case GameResult::LOSS_GAMMON:      p2_gammons++; break;
        case GameResult::LOSS_BACKGAMMON:  p2_backgammons++; break;
        default: break;
    }
}

// Make a move for the player on roll (always player 1 from the board's perspective).
// Returns the board AFTER the move, still from player 1's perspective.
static Board make_move(const Board& board, const Strategy& strat,
                       int die1, int die2) {
    auto candidates = possible_boards(board, die1, die2);

    if (candidates.size() == 1) {
        return candidates[0];
    }

    int idx = strat.best_move_index(candidates, board);
    return candidates[idx];
}

GameResult play_game(const Strategy& strat1, const Strategy& strat2,
                     std::mt19937& rng) {
    std::uniform_int_distribution<int> die(1, 6);

    Board board = STARTING_BOARD;

    // Determine who goes first: roll until dice differ
    int d1, d2;
    bool player1_on_roll;
    do {
        d1 = die(rng);
        d2 = die(rng);
    } while (d1 == d2);

    player1_on_roll = (d1 > d2);

    // If player 2 goes first, flip the board so that the on-roll player
    // is always "player 1" from the board's perspective.
    if (!player1_on_roll) {
        board = flip(board);
    }

    // Make the opening move with the dice we already rolled
    const Strategy& current_strat = player1_on_roll ? strat1 : strat2;
    board = make_move(board, current_strat, d1, d2);

    // Check if game over
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        // If player 2 was on roll, flip the result sign
        if (!player1_on_roll) {
            result = static_cast<GameResult>(-static_cast<int>(result));
        }
        return result;
    }

    // Flip board for the other player's turn
    board = flip(board);
    player1_on_roll = !player1_on_roll;

    // Continue playing
    for (;;) {
        d1 = die(rng);
        d2 = die(rng);

        const Strategy& strat = player1_on_roll ? strat1 : strat2;
        board = make_move(board, strat, d1, d2);

        result = check_game_over(board);
        if (result != GameResult::NOT_OVER) {
            // Result is from the perspective of "player 1 on the board",
            // which is whoever was on roll. If player 2 was on roll,
            // flip the sign.
            if (!player1_on_roll) {
                result = static_cast<GameResult>(-static_cast<int>(result));
            }
            return result;
        }

        // Flip for next player
        board = flip(board);
        player1_on_roll = !player1_on_roll;
    }
}

void GameStats::merge(const GameStats& other) {
    n_games += other.n_games;
    p1_wins += other.p1_wins;
    p1_gammons += other.p1_gammons;
    p1_backgammons += other.p1_backgammons;
    p2_wins += other.p2_wins;
    p2_gammons += other.p2_gammons;
    p2_backgammons += other.p2_backgammons;
}

GameStats play_games(const Strategy& strat1, const Strategy& strat2,
                     int n_games, uint32_t seed) {
    std::mt19937 rng(seed);
    GameStats stats;

    for (int i = 0; i < n_games; ++i) {
        GameResult result = play_game(strat1, strat2, rng);
        stats.record(result);
    }

    return stats;
}

GameStats play_games_parallel(const Strategy& strat1, const Strategy& strat2,
                              int n_games, uint32_t seed, int n_threads) {
    if (n_threads <= 0) {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;
        if (n_threads > 16) n_threads = n_threads / 2;
    }
    n_threads = std::min(n_threads, n_games);
    if (n_threads <= 1) {
        return play_games(strat1, strat2, n_games, seed);
    }

    const int base_chunk = n_games / n_threads;
    const int remainder = n_games % n_threads;

    std::vector<GameStats> thread_stats(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    int offset = 0;
    for (int t = 0; t < n_threads; ++t) {
        const int chunk = base_chunk + (t < remainder ? 1 : 0);
        // Each thread gets a unique seed derived from the base seed
        uint32_t thread_seed = seed + static_cast<uint32_t>(t) * 1000003u;

        threads.emplace_back([&strat1, &strat2, chunk, thread_seed,
                              &result = thread_stats[t]]() {
            result = play_games(strat1, strat2, chunk, thread_seed);
        });

        offset += chunk;
    }

    for (auto& t : threads) t.join();

    GameStats stats;
    for (const auto& ts : thread_stats) stats.merge(ts);
    return stats;
}

} // namespace bgbot
