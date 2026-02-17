// Test: score benchmark from .bm file with manual threading
#include "bgbot/types.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/pubeval.h"
#include "bgbot/benchmark.h"
#include <cstdio>
#include <thread>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace bgbot;

static Board decode_gnubg_pos(const char* pos_str) {
    unsigned char key[10];
    for (int i = 0; i < 10; ++i)
        key[i] = ((pos_str[2*i] - 'A') << 4) + (pos_str[2*i+1] - 'A');
    Board checkers = {};
    int player = 0, point = 0;
    for (int ind = 0; ind < 10; ++ind) {
        unsigned char cur = key[ind];
        for (int bit = 0; bit < 8; ++bit) {
            if (cur & 0x1) {
                if (point < 24) {
                    if (player == 0) checkers[24 - point] -= 1;
                    else             checkers[point + 1] += 1;
                } else {
                    if (player == 0) checkers[0] += 1;
                    else             checkers[25] += 1;
                }
            } else {
                point++;
                if (point == 25) { player++; point = 0; }
            }
            cur >>= 1;
        }
    }
    return checkers;
}

static void score_worker(const PubEval& strat,
                         const BenchmarkScenario* scenarios,
                         int count, int id) {
    std::vector<Board> candidates;
    candidates.reserve(32);
    int scored = 0;
    for (int s = 0; s < count; ++s) {
        const auto& sc = scenarios[s];
        possible_boards(sc.start_board, sc.die1, sc.die2, candidates);
        if (candidates.size() > 1) {
            int idx = strat.best_move_index(candidates, sc.start_board);
            (void)idx;
        }
        scored++;
    }
    printf("  Worker %d: scored %d\n", id, scored);
    fflush(stdout);
}

int main() {
    // Parse .bm
    std::ifstream f("data/contact.bm");
    if (!f) { printf("Cannot open file\n"); return 1; }

    std::vector<BenchmarkScenario> scenarios;
    std::string line;
    while (std::getline(f, line)) {
        if (line.size() < 2 || line[0] != 'm' || line[1] != ' ') continue;
        std::istringstream iss(line);
        std::string token, pos_str;
        int die1, die2;
        iss >> token >> pos_str >> die1 >> die2;
        BenchmarkScenario s;
        s.start_board = decode_gnubg_pos(pos_str.c_str());
        s.die1 = die1;
        s.die2 = die2;
        while (iss >> pos_str) {
            double err = 0.0;
            iss >> err;
            s.ranked_boards.push_back(decode_gnubg_pos(pos_str.c_str()));
            s.ranked_errors.push_back(err);
        }
        scenarios.push_back(std::move(s));
    }

    printf("Loaded %zu scenarios\n", scenarios.size());
    fflush(stdout);

    PubEval strat;
    int n = (int)scenarios.size();

    // Test: manual 2-thread split
    printf("Manual 2-thread test...\n");
    fflush(stdout);
    {
        int half = n / 2;
        std::thread t1(score_worker, std::cref(strat), scenarios.data(), half, 0);
        std::thread t2(score_worker, std::cref(strat), scenarios.data() + half, n - half, 1);
        t1.join();
        t2.join();
    }
    printf("Manual 2-thread OK\n");
    fflush(stdout);

    // Test: score_benchmarks with 2 threads
    printf("score_benchmarks 2 threads...\n");
    fflush(stdout);
    auto result = score_benchmarks(strat, scenarios, 2);
    printf("Result: %.2f (%d)\n", result.score(), result.count);
    fflush(stdout);

    printf("Done!\n");
    return 0;
}
