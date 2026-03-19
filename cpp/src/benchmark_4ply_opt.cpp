// Benchmark: 4-ply cube action analytics with full output.
//
// Runs 4-ply cube decision 3 times and reports average time,
// cubeful equities, and cubeless probabilities.
//
// Usage:
//   benchmark_4ply_opt [threads] [iterations] [model_dir]
//   Defaults: threads=16, iterations=3

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
#include <vector>

using namespace bgbot;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
    int n_threads = 16;
    int iterations = 3;

    if (argc > 1) {
        int t = std::atoi(argv[1]);
        if (t > 0) n_threads = t;
    }
    if (argc > 2) {
        int it = std::atoi(argv[2]);
        if (it > 0) iterations = it;
    }

    std::filesystem::path model_dir;
    if (argc > 3) {
        model_dir = std::filesystem::path(argv[3]);
    } else {
        model_dir = std::filesystem::absolute(
            std::filesystem::path(argv[0])).parent_path().parent_path() / "models";
    }

    // Reference position from the task specification
    Board board = {
        0,-2,0,0,0,0,5,2,3,0,0,0,-3,3,0,0,0,-3,2,-5,-2,0,0,0,0,0
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

    printf("=== 4-ply Cube Decision Benchmark ===\n");
    printf("Position: ");
    for (int i = 0; i < 26; i++) printf("%d%s", board[i], i < 25 ? "," : "\n");
    printf("Threads: %d\n", n_threads);
    printf("Iterations: %d\n", iterations);
    printf("Model dir: %s\n", model_dir.string().c_str());
    printf("Filter: TINY (max_moves=%d, threshold=%.2f)\n",
           MoveFilters::TINY.max_moves, MoveFilters::TINY.threshold);
    printf("\n");

    // Warm-up run
    printf("Warming up...\n");
    auto warmup_decision = cube_decision_nply(board, cube, strategy, 4, MoveFilters::TINY, n_threads);
    printf("Warm-up done.\n\n");

    // Get cubeless probs at 4-ply
    printf("Computing 4-ply cubeless probabilities...\n");
    Board flipped = flip(board);
    auto base_strat = std::make_shared<GamePlanStrategy>(pr, rc, at, pm, an, 200, 400, 400, 400, 400);
    MultiPlyStrategy multipy(base_strat, 4, MoveFilters::TINY, false, true, n_threads);
    auto post_probs = multipy.evaluate_probs(flipped, flipped);
    multipy.clear_cache();
    auto cubeless_probs = invert_probs(post_probs);
    float cl_eq = cubeless_equity(cubeless_probs);

    printf("Cubeless probs: W=%.6f Gw=%.6f Bw=%.6f Gl=%.6f Bl=%.6f\n",
           cubeless_probs[0], cubeless_probs[1], cubeless_probs[2],
           cubeless_probs[3], cubeless_probs[4]);
    printf("Cubeless equity: %+.6f\n\n", cl_eq);

    // Timed runs
    CubeDecision decision;
    std::vector<double> times;
    times.reserve(iterations);
    double total_time = 0.0;

    for (int i = 0; i < iterations; i++) {
        reset_cubeful_counters();
        auto start = Clock::now();
        decision = cube_decision_nply(board, cube, strategy, 4, MoveFilters::TINY, n_threads);
        auto end = Clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        times.push_back(elapsed);
        total_time += elapsed;

        printf("Run %d: %.3f s  ND=%+.8f  DT=%+.8f  DP=%+.8f  Opt=%+.8f\n",
               i + 1, elapsed,
               decision.equity_nd, decision.equity_dt,
               decision.equity_dp, decision.optimal_equity);
        if (i == 0) {
            printf("  Profiling counters:\n");
            print_cubeful_counters();
        }
    }

    double avg_time = total_time / iterations;

    printf("\n=== Summary ===\n");
    printf("Average time: %.3f s\n", avg_time);
    printf("ND equity:    %+.8f\n", decision.equity_nd);
    printf("DT equity:    %+.8f\n", decision.equity_dt);
    printf("DP equity:    %+.8f\n", decision.equity_dp);
    printf("Optimal eq:   %+.8f\n", decision.optimal_equity);
    printf("Decision:     %s / %s%s\n",
           decision.should_double ? "Double" : "No Double",
           decision.should_take ? "Take" : "Pass",
           decision.is_beaver ? " (Beaver)" : "");
    printf("Cubeless probs: W=%.6f Gw=%.6f Bw=%.6f Gl=%.6f Bl=%.6f\n",
           cubeless_probs[0], cubeless_probs[1], cubeless_probs[2],
           cubeless_probs[3], cubeless_probs[4]);
    printf("Cubeless equity: %+.6f\n", cl_eq);

    return 0;
}
