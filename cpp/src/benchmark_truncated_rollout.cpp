// Benchmark: XG Roller++ Cube truncated rollout.
// Tests determinism and measures wall-clock time for various configurations.

#include "bgbot/rollout.h"
#include "bgbot/cube.h"
#include "bgbot/encoding.h"
#include "bgbot/board.h"
#include "bgbot/neural_net.h"
#include "bgbot/multipy.h"
#include "bgbot/moves.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

using namespace bgbot;

static void clear_pos_cache(std::shared_ptr<Strategy> base) {
    auto tmp = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilters::TINY);
    tmp->clear_cache();
}

struct BenchResult {
    double nd_equity, dt_equity, cl_equity;
    double nd_se, dt_se, cl_se;
    double elapsed;
};

static BenchResult run_benchmark(
    std::shared_ptr<GamePlanStrategy> base,
    const Board& board, const CubeInfo& cube,
    int n_trials, int decision_ply, int late_ply,
    int n_threads, bool parallelize_trials)
{
    clear_pos_cache(base);

    RolloutConfig config;
    config.n_trials = n_trials;
    config.truncation_depth = 7;
    config.decision_ply = decision_ply;
    config.enable_vr = true;
    config.filter = MoveFilters::TINY;
    config.n_threads = n_threads;
    config.seed = 42;
    config.late_ply = late_ply;
    config.late_threshold = 2;
    config.parallelize_trials = parallelize_trials;

    auto rollout = std::make_shared<RolloutStrategy>(base, config);

    auto start = std::chrono::high_resolution_clock::now();
    auto cfr = rollout->cubeful_cube_decision(board, cube);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    return {cfr.nd_equity, cfr.dt_equity, cfr.cubeless.equity,
            cfr.nd_se, cfr.dt_se, cfr.cubeless.std_error, elapsed};
}

int main(int argc, char* argv[]) {
    std::filesystem::path model_dir;
    if (argc > 1) {
        model_dir = std::filesystem::path(argv[1]);
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

    // ============================================================
    // Determinism test: 3 runs of 36-trial
    // ============================================================
    printf("=== Determinism: 36-trial 3-ply/2-ply rollout ===\n");
    for (int run = 0; run < 3; ++run) {
        auto r = run_benchmark(base, board, cube, 36, 3, 2, 1, false);
        printf("  Run %d: ND=%.9f  DT=%.9f  CL=%.9f  (%.1fs)\n",
               run, r.nd_equity, r.dt_equity, r.cl_equity, r.elapsed);
    }

    // ============================================================
    // Serial baseline
    // ============================================================
    printf("\n=== Serial baseline: 360 trials, 3/2-ply ===\n");
    auto serial = run_benchmark(base, board, cube, 360, 3, 2, 1, false);
    printf("  %.3f s  ND=%+.5f  DT=%+.5f  CL=%+.5f\n",
           serial.elapsed, serial.nd_equity, serial.dt_equity, serial.cl_equity);
    printf("  SE: ND=%.5f  DT=%.5f  CL=%.5f\n",
           serial.nd_se, serial.dt_se, serial.cl_se);

    // ============================================================
    // Parallel scaling: 360 trials, 3/2-ply
    // ============================================================
    printf("\n=== Parallel scaling: 360 trials, 3/2-ply ===\n");
    int thread_counts[] = {4, 8, 16, 32};
    for (int nt : thread_counts) {
        auto r = run_benchmark(base, board, cube, 360, 3, 2, nt, true);
        printf("  %2d threads: %.3f s (%.1fx)  ND=%+.5f  DT=%+.5f\n",
               nt, r.elapsed, serial.elapsed / r.elapsed, r.nd_equity, r.dt_equity);
    }

    // ============================================================
    // Ply comparison: 360 trials (serial)
    // ============================================================
    printf("\n=== Ply comparison: 360 trials (serial) ===\n");
    {
        auto r1 = run_benchmark(base, board, cube, 360, 1, -1, 1, false);
        printf("  1-ply:     %.3f s  ND=%+.5f  DT=%+.5f\n", r1.elapsed, r1.nd_equity, r1.dt_equity);

        auto r2 = run_benchmark(base, board, cube, 360, 2, -1, 1, false);
        printf("  2-ply:     %.3f s  ND=%+.5f  DT=%+.5f\n", r2.elapsed, r2.nd_equity, r2.dt_equity);

        auto r3 = run_benchmark(base, board, cube, 360, 2, 1, 1, false);
        printf("  2/1-ply:   %.3f s  ND=%+.5f  DT=%+.5f\n", r3.elapsed, r3.nd_equity, r3.dt_equity);

        auto r4 = run_benchmark(base, board, cube, 360, 3, 2, 1, false);
        printf("  3/2-ply:   %.3f s  ND=%+.5f  DT=%+.5f\n", r4.elapsed, r4.nd_equity, r4.dt_equity);
    }

    return 0;
}
