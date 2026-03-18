// Benchmark: 3T cube action analytics (XG Roller++ Cube equivalent).
//
// 360 trials, truncation_depth=7, decision_ply=3, late_ply=2, late_threshold=2
// Reference position: checkers=[0,0,0,2,2,-2,3,2,2,0,0,0,-3,4,0,0,0,-3,0,-3,-2,-2,0,0,0,0]
// Cube centered at 1, money game, jacoby on, beavers on.
//
// Usage:
//   benchmark_3t [n_threads] [model_dir]
//   Defaults: n_threads=16, model_dir=<exe_dir>/../models

#include "bgbot/rollout.h"
#include "bgbot/cube.h"
#include "bgbot/encoding.h"
#include "bgbot/board.h"
#include "bgbot/neural_net.h"
#include "bgbot/multipy.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>


using namespace bgbot;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
    int n_threads = 16;
    if (argc > 1) {
        int t = std::atoi(argv[1]);
        if (t > 0) n_threads = t;
    }

    std::filesystem::path model_dir;
    if (argc > 2) {
        model_dir = std::filesystem::path(argv[2]);
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

    auto base = std::make_shared<GamePlanStrategy>(pr, rc, at, pm, an, 200, 400, 400, 400, 400);

    CubeInfo cube;
    cube.cube_value = 1;
    cube.owner = CubeOwner::CENTERED;
    cube.jacoby = true;
    cube.beaver = true;

    // Clear caches before benchmark
    {
        auto tmp = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilters::TINY);
        tmp->clear_cache();
    }

    RolloutConfig config;
    config.n_trials = 360;
    config.truncation_depth = 7;
    config.decision_ply = 3;
    config.truncation_ply = 2;
    config.enable_vr = true;
    config.filter = MoveFilters::TINY;
    config.n_threads = n_threads;
    config.seed = 42;
    config.late_ply = 2;
    config.late_threshold = 2;

    auto rollout = std::make_shared<RolloutStrategy>(base, config);

    printf("=== 3T Benchmark (XG Roller++ Cube) ===\n");
    printf("Position: ");
    for (int i = 0; i < 26; i++) printf("%d%s", board[i], i < 25 ? "," : "\n");
    printf("Config: 360 trials, trunc=7, decision=3ply, late=2ply@2, VR on\n");
    printf("Threads: %d\n", n_threads);
    printf("Model dir: %s\n", model_dir.string().c_str());

    auto start = Clock::now();
    auto cfr = rollout->cubeful_cube_decision(board, cube);
    auto end = Clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();

    printf("\n--- Results ---\n");
    printf("ND equity:       %+.6f  SE: %.6f\n", cfr.nd_equity, cfr.nd_se);
    printf("DT equity:       %+.6f  SE: %.6f\n", cfr.dt_equity, cfr.dt_se);
    printf("DP equity:       %+.6f\n", 1.0);
    printf("CL equity:       %+.6f  SE: %.6f\n", cfr.cubeless.equity, cfr.cubeless.std_error);
    const char* prob_names[] = {"P(win)", "P(gw)", "P(bw)", "P(gl)", "P(bl)"};
    for (int i = 0; i < 5; ++i) {
        printf("%-16s %.6f  SE: %.6f\n",
               prob_names[i],
               cfr.cubeless.mean_probs[i],
               cfr.cubeless.prob_std_errors[i]);
    }

    bool should_double = (std::min(cfr.dt_equity, 1.0) > cfr.nd_equity);
    bool should_take = (cfr.dt_equity <= 1.0);
    printf("Decision:        %s / %s\n",
           should_double ? "Double" : "No Double",
           should_take ? "Take" : "Pass");

    printf("\nWall time:       %.3f s\n", elapsed);

    return 0;
}
