// Standalone profiler for TD training hot path.
// Runs a few hundred games and times each component.

#include "bgbot/types.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/encoding.h"
#include "bgbot/neural_net.h"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

using namespace bgbot;
using Clock = std::chrono::steady_clock;

int main() {
    auto nn = std::make_shared<NeuralNetwork>(120, 0.1f, 42);
    NNStrategy strat(nn);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> die(1, 6);

    const int N_GAMES = 500;

    double t_flip = 0, t_encode = 0, t_fwd_grad = 0, t_fwd = 0, t_td_update = 0;
    double t_movegen = 0, t_bestmove = 0, t_gameover = 0, t_is_race = 0;
    int n_moves = 0;
    int n_candidates_total = 0;

    auto tick = []() { return Clock::now(); };
    auto elapsed = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };

    auto t_total_start = tick();

    for (int g = 0; g < N_GAMES; ++g) {
        Board board = STARTING_BOARD;

        int d1, d2;
        do { d1 = die(rng); d2 = die(rng); } while (d1 == d2);
        if (d2 > d1) board = flip(board);

        bool game_over = false;
        bool first_move = true;

        while (!game_over) {
            if (!first_move) {
                d1 = die(rng);
                d2 = die(rng);
            }
            first_move = false;
            ++n_moves;

            // Step A: flip + encode + forward_with_gradients
            auto t0 = tick();
            Board flipped = flip(board);
            auto t1 = tick();
            auto flipped_inputs = compute_tesauro_inputs(flipped);
            auto t2 = tick();
            nn->forward_with_gradients(flipped_inputs);
            auto t3 = tick();

            t_flip += elapsed(t0, t1);
            t_encode += elapsed(t1, t2);
            t_fwd_grad += elapsed(t2, t3);

            // Step B: move generation + best move selection
            t0 = tick();
            auto candidates = possible_boards(board, d1, d2);
            t1 = tick();
            t_movegen += elapsed(t0, t1);
            n_candidates_total += candidates.size();

            if (candidates.size() > 1) {
                t0 = tick();
                t1 = tick();
                t_is_race += elapsed(t0, t1);

                t0 = tick();
                int idx = strat.best_move_index(candidates, board);
                t1 = tick();
                t_bestmove += elapsed(t0, t1);
                board = candidates[idx];
            } else {
                board = candidates[0];
            }

            // Step C: check game over
            t0 = tick();
            GameResult result = check_game_over(board);
            t1 = tick();
            t_gameover += elapsed(t0, t1);

            if (result != GameResult::NOT_OVER) {
                t0 = tick();
                std::array<float, NN_OUTPUTS> targets;
                switch (result) {
                    case GameResult::WIN_SINGLE:      targets = {0,0,0,0,0}; break;
                    case GameResult::WIN_GAMMON:       targets = {0,0,0,1,0}; break;
                    case GameResult::WIN_BACKGAMMON:   targets = {0,0,0,1,1}; break;
                    case GameResult::LOSS_SINGLE:      targets = {1,0,0,0,0}; break;
                    case GameResult::LOSS_GAMMON:      targets = {1,1,0,0,0}; break;
                    case GameResult::LOSS_BACKGAMMON:  targets = {1,1,1,0,0}; break;
                    default:                           targets = {0.5f,0,0,0,0}; break;
                }
                nn->td_update(targets, 0.1f);
                t1 = tick();
                t_td_update += elapsed(t0, t1);
                game_over = true;
            } else {
                // Non-terminal target
                t0 = tick();
                auto post_inputs = compute_tesauro_inputs(board);
                t1 = tick();
                t_encode += elapsed(t0, t1);

                t0 = tick();
                auto post_outputs = nn->forward(post_inputs);
                t1 = tick();
                t_fwd += elapsed(t0, t1);

                t0 = tick();
                std::array<float, NN_OUTPUTS> targets = {
                    1.0f - post_outputs[0], post_outputs[3], post_outputs[4],
                    post_outputs[1], post_outputs[2]
                };
                nn->td_update(targets, 0.1f);
                t1 = tick();
                t_td_update += elapsed(t0, t1);

                board = flip(board);
            }
        }
    }

    auto t_total_end = tick();
    double t_total = elapsed(t_total_start, t_total_end);

    double avg_cand = (double)n_candidates_total / n_moves;
    double moves_per_game = (double)n_moves / N_GAMES;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "=== TD Training Profile (" << N_GAMES << " games, "
              << n_moves << " moves) ===\n\n";
    std::cout << "Avg moves/game:      " << moves_per_game << "\n";
    std::cout << "Avg candidates/move: " << std::setprecision(2) << avg_cand << "\n\n";

    std::cout << std::setprecision(1);
    std::cout << "Total time:          " << t_total / 1000.0 << " ms\n\n";

    auto pct = [&](double t) { return 100.0 * t / t_total; };
    auto per_move = [&](double t) { return t / n_moves; };

    std::cout << "Component breakdown (per-move us, total ms, %):\n";
    std::cout << "  flip:              " << std::setw(6) << per_move(t_flip)
              << " us   " << std::setw(8) << t_flip/1000.0
              << " ms   " << std::setw(5) << pct(t_flip) << "%\n";
    std::cout << "  encoding:          " << std::setw(6) << per_move(t_encode)
              << " us   " << std::setw(8) << t_encode/1000.0
              << " ms   " << std::setw(5) << pct(t_encode) << "%\n";
    std::cout << "  fwd_with_grads:    " << std::setw(6) << per_move(t_fwd_grad)
              << " us   " << std::setw(8) << t_fwd_grad/1000.0
              << " ms   " << std::setw(5) << pct(t_fwd_grad) << "%\n";
    std::cout << "  forward (eval):    " << std::setw(6) << per_move(t_fwd)
              << " us   " << std::setw(8) << t_fwd/1000.0
              << " ms   " << std::setw(5) << pct(t_fwd) << "%\n";
    std::cout << "  td_update:         " << std::setw(6) << per_move(t_td_update)
              << " us   " << std::setw(8) << t_td_update/1000.0
              << " ms   " << std::setw(5) << pct(t_td_update) << "%\n";
    std::cout << "  move_gen:          " << std::setw(6) << per_move(t_movegen)
              << " us   " << std::setw(8) << t_movegen/1000.0
              << " ms   " << std::setw(5) << pct(t_movegen) << "%\n";
    std::cout << "  best_move (nn*N):  " << std::setw(6) << per_move(t_bestmove)
              << " us   " << std::setw(8) << t_bestmove/1000.0
              << " ms   " << std::setw(5) << pct(t_bestmove) << "%\n";
    std::cout << "  is_race:           " << std::setw(6) << per_move(t_is_race)
              << " us   " << std::setw(8) << t_is_race/1000.0
              << " ms   " << std::setw(5) << pct(t_is_race) << "%\n";
    std::cout << "  check_game_over:   " << std::setw(6) << per_move(t_gameover)
              << " us   " << std::setw(8) << t_gameover/1000.0
              << " ms   " << std::setw(5) << pct(t_gameover) << "%\n";

    double accounted = t_flip + t_encode + t_fwd_grad + t_fwd + t_td_update
                     + t_movegen + t_bestmove + t_is_race + t_gameover;
    std::cout << "\n  accounted:         " << std::setw(14) << accounted/1000.0
              << " ms   " << std::setw(5) << pct(accounted) << "%\n";
    std::cout << "  unaccounted:       " << std::setw(14) << (t_total - accounted)/1000.0
              << " ms   " << std::setw(5) << pct(t_total - accounted) << "%\n";

    double ms_per_game = t_total / 1000.0 / N_GAMES;
    double games_per_sec = 1000.0 / ms_per_game;
    std::cout << "\n" << std::setprecision(2)
              << ms_per_game << " ms/game  (" << games_per_sec << " games/sec)\n";

    return 0;
}
