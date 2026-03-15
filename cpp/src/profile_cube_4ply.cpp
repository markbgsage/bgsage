// Profile a single reference cube decision and report wall-clock runtime.
//
// Usage:
//   bgbot_profile_4ply [iterations] [threads] [ply] [model_dir]
//   [threads]=0 means all hardware threads
//   [ply] = 2 | 3 | 4 (default 4)
//
// Defaults:
//   iterations = 1
//   threads    = all available hardware threads
//   ply        = 4
//   model_dir  = <repo>/models

#include "bgbot/cube.h"
#include "bgbot/encoding.h"
#include "bgbot/neural_net.h"
#include "bgbot/board.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

using namespace bgbot;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
    int iterations = 1;
    int n_ply = 4;
    const int default_threads = static_cast<int>(
        std::thread::hardware_concurrency() > 0
            ? std::thread::hardware_concurrency()
            : 1
    );
    int n_threads = default_threads;

    if (argc > 1) {
        iterations = std::atoi(argv[1]);
    }
    if (argc > 2) {
        int requested_threads = std::atoi(argv[2]);
        n_threads = (requested_threads > 0) ? requested_threads : default_threads;
    }
    if (argc > 3) {
        int requested_ply = std::atoi(argv[3]);
        if (requested_ply >= 2 && requested_ply <= 4) {
            n_ply = requested_ply;
        }
    }

    if (iterations <= 0) iterations = 1;

    // Resolve weight directory. Default to "<repo>/models" relative to build directory.
    std::filesystem::path model_dir;
    if (argc > 4) {
        model_dir = std::filesystem::path(argv[4]);
    } else {
        model_dir = std::filesystem::absolute(
            std::filesystem::path(argv[0])).parent_path().parent_path() / "models";
    }

    // Reference position requested for this benchmark.
    Board board = {
        0,-2,0,0,0,2,4,2,2,0,0,0,-4,3,0,0,0,-1,-2,-4,-2,2,0,0,0,0
    };

    const std::filesystem::path purerace_path = model_dir / "sl_s5_purerace.weights.best";
    const std::filesystem::path racing_path   = model_dir / "sl_s5_racing.weights.best";
    const std::filesystem::path attacking_path = model_dir / "sl_s5_attacking.weights.best";
    const std::filesystem::path priming_path   = model_dir / "sl_s5_priming.weights.best";
    const std::filesystem::path anchoring_path = model_dir / "sl_s5_anchoring.weights.best";

    // Ensure extended-encoding helpers are initialized before timing the benchmark loop.
    init_escape_tables();

    GamePlanStrategy strategy(
        purerace_path.string(),
        racing_path.string(),
        attacking_path.string(),
        priming_path.string(),
        anchoring_path.string(),
        200, 400, 400, 400, 400
    );

    // Multi-ply, money game, centered cube, Jacoby + beaver enabled.
    CubeInfo cube;
    cube.cube_value = 1;
    cube.owner = CubeOwner::CENTERED;
    cube.jacoby = true;
    cube.beaver = true;

    CubeDecision decision = cube_decision_nply(board, cube, strategy, n_ply, MoveFilters::TINY, n_threads);

    printf("Reference position: ");
    for (int i = 0; i < 26; i++) {
        printf("%d%s", board[i], (i + 1 == 26 ? "\n" : ","));
    }
    printf("Model dir: %s\n", model_dir.string().c_str());
    printf("Iterations: %d\n", iterations);
    printf("Threads: %d\n", n_threads);
    printf("%d-ply cube decision baseline: ND=%.8f DT=%.8f DP=%.8f Optimal=%.8f\n",
           n_ply,
           decision.equity_nd, decision.equity_dt, decision.equity_dp, decision.optimal_equity);
    printf("Action: should_double=%s should_take=%s is_beaver=%s\n",
           decision.should_double ? "true" : "false",
           decision.should_take ? "true" : "false",
           decision.is_beaver ? "true" : "false");

    // Warm-up (one warm execution to stabilize timing noise)
    (void)cube_decision_nply(board, cube, strategy, n_ply, MoveFilters::TINY, n_threads);

    auto t0 = Clock::now();
    for (int i = 0; i < iterations; i++) {
        decision = cube_decision_nply(board, cube, strategy, n_ply, MoveFilters::TINY, n_threads);
    }
    auto t1 = Clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double ms_per = (elapsed * 1000.0) / iterations;
    printf("Wall time: %.6f s for %d run(s) (%.6f ms/run)\n", elapsed, iterations, ms_per);
    printf("Latest run: ND=%.8f DT=%.8f DP=%.8f Optimal=%.8f\n",
           decision.equity_nd, decision.equity_dt, decision.equity_dp, decision.optimal_equity);

    return 0;
}
