#include "bgbot/pubeval.h"
#include "bgbot/game.h"
#include <iostream>
#include <iomanip>
#include <chrono>

int main() {
    using namespace bgbot;

    const int N_GAMES = 100;
    const uint32_t SEED = 42;

    PubEval strat1(PubEval::WeightSource::TESAURO);
    PubEval strat2(PubEval::WeightSource::TESAURO);

    auto t0 = std::chrono::high_resolution_clock::now();
    GameStats stats = play_games(strat1, strat2, N_GAMES, SEED);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== PubEval vs PubEval: " << N_GAMES << " games ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Player 1:" << std::endl;
    std::cout << "  Wins (single):     " << stats.p1_wins
              << "  (" << stats.p1_win_frac() << ")" << std::endl;
    std::cout << "  Wins (gammon):     " << stats.p1_gammons
              << "  (" << stats.p1_gammon_frac() << ")" << std::endl;
    std::cout << "  Wins (backgammon): " << stats.p1_backgammons
              << "  (" << stats.p1_backgammon_frac() << ")" << std::endl;
    std::cout << std::endl;
    std::cout << "Player 2:" << std::endl;
    std::cout << "  Wins (single):     " << stats.p2_wins
              << "  (" << stats.p2_win_frac() << ")" << std::endl;
    std::cout << "  Wins (gammon):     " << stats.p2_gammons
              << "  (" << stats.p2_gammon_frac() << ")" << std::endl;
    std::cout << "  Wins (backgammon): " << stats.p2_backgammons
              << "  (" << stats.p2_backgammon_frac() << ")" << std::endl;
    std::cout << std::endl;
    std::cout << "Avg ppg (player 1):  " << stats.avg_ppg() << std::endl;
    std::cout << "Time:                " << std::setprecision(1) << elapsed_ms << " ms"
              << " (" << std::setprecision(1) << elapsed_ms / N_GAMES << " ms/game)" << std::endl;

    return 0;
}
