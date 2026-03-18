// Benchmark: N-ply cube action analytics (parallel).
//
// Reference position: checkers=[0,0,0,2,2,-2,3,2,2,0,0,0,-3,4,0,0,0,-3,0,-3,-2,-2,0,0,0,0]
// Cube centered at 1, money game, jacoby on, beavers on.
//
// Usage:
//   benchmark_4ply [n_threads] [n_ply] [model_dir]
//   Defaults: n_threads=12, n_ply=4, model_dir=<exe_dir>/../models

#include "bgbot/cube.h"
#include "bgbot/encoding.h"
#include "bgbot/neural_net.h"
#include "bgbot/board.h"
#include "bgbot/multipy.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace bgbot;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
    int n_threads = 12;
    int n_ply = 4;
    if (argc > 1) {
        int t = std::atoi(argv[1]);
        if (t > 0) n_threads = t;
    }
    if (argc > 2) {
        int p = std::atoi(argv[2]);
        if (p >= 2) n_ply = p;
    }

    std::filesystem::path model_dir;
    if (argc > 3) {
        model_dir = std::filesystem::path(argv[3]);
    } else {
        model_dir = std::filesystem::absolute(
            std::filesystem::path(argv[0])).parent_path().parent_path() / "models";
    }

    Board board = {
        0, 0, 0, 2, 2, -2, 3, 2, 2, 0, 0, 0, -3, 4, 0, 0, 0, -3, 0, -3, -2, -2, 0, 0, 0, 0
    };

    const auto pr = (model_dir / "sl_s5_purerace.weights.best").string();
    const auto rc = (model_dir / "sl_s5_racing.weights.best").string();
    const auto at = (model_dir / "sl_s5_attacking.weights.best").string();
    const auto pm = (model_dir / "sl_s5_priming.weights.best").string();
    const auto an = (model_dir / "sl_s5_anchoring.weights.best").string();

    init_escape_tables();

    GamePlanStrategy strategy(pr, rc, at, pm, an, 200, 400, 400, 400, 400);

    CubeInfo cube;
    cube.cube_value = 1;
    cube.owner = CubeOwner::CENTERED;
    cube.jacoby = true;
    cube.beaver = true;

    printf("=== %d-ply Cube Benchmark ===\n", n_ply);
    printf("Position: ");
    for (int i = 0; i < 26; i++) printf("%d%s", board[i], i < 25 ? "," : "\n");
    printf("Threads: %d\n", n_threads);
    printf("Model dir: %s\n", model_dir.string().c_str());

    // Warm-up run
    (void)cube_decision_nply(board, cube, strategy, n_ply, MoveFilters::TINY, n_threads);

    auto start = Clock::now();
    auto decision = cube_decision_nply(board, cube, strategy, n_ply, MoveFilters::TINY, n_threads);
    auto end = Clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();

    printf("\n--- Results ---\n");
    printf("ND equity:       %+.8f\n", decision.equity_nd);
    printf("DT equity:       %+.8f\n", decision.equity_dt);
    printf("DP equity:       %+.8f\n", decision.equity_dp);
    printf("Optimal equity:  %+.8f\n", decision.optimal_equity);
    printf("Decision:        %s / %s%s\n",
           decision.should_double ? "Double" : "No Double",
           decision.should_take ? "Take" : "Pass",
           decision.is_beaver ? " (Beaver)" : "");

    printf("\nWall time:       %.3f s\n", elapsed);

    return 0;
}
