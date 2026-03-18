// Experiment: Compare 3T with different truncation evaluation plies.
// Tests: decision_ply=3 (baseline), decision_ply=2 (faster truncation).

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

struct Config3T {
    const char* name;
    int n_trials;
    int truncation_depth;
    int decision_ply;
    int late_ply;
    int late_threshold;
};

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

    Config3T configs[] = {
        // Baseline: XG Roller++ Cube
        {"3T baseline (d=3,l=2@2,t=7)", 360, 7, 3, 2, 2},
        // Experiment: lower truncation via lower decision_ply
        {"2T+ (d=2,l=1@2,t=7)", 360, 7, 2, 1, 2},
        // Experiment: keep 3-ply for move0 but truncation_depth=5
        {"3T trunc5 (d=3,l=2@2,t=5)", 360, 5, 3, 2, 2},
        // Experiment: more trials at lower ply
        {"2T+ more (d=2,l=1@2,t=7,n=720)", 720, 7, 2, 1, 2},
        // Experiment: higher trial count, reduced depth
        {"2T+ t5 (d=2,l=1@2,t=5)", 360, 5, 2, 1, 2},
    };

    printf("=== 3T Experiment: Truncation Ply Comparison ===\n");
    printf("Position: ");
    for (int i = 0; i < 26; i++) printf("%d%s", board[i], i < 25 ? "," : "\n");
    printf("Threads: %d\n\n", n_threads);

    // Reference values from baseline
    double ref_nd = 0, ref_dt = 0;

    for (const auto& cfg : configs) {
        // Clear caches
        {
            auto tmp = std::make_shared<MultiPlyStrategy>(base, 2, MoveFilters::TINY);
            tmp->clear_cache();
        }

        RolloutConfig rcfg;
        rcfg.n_trials = cfg.n_trials;
        rcfg.truncation_depth = cfg.truncation_depth;
        rcfg.decision_ply = cfg.decision_ply;
        rcfg.enable_vr = true;
        rcfg.filter = MoveFilters::TINY;
        rcfg.n_threads = n_threads;
        rcfg.seed = 42;
        rcfg.late_ply = cfg.late_ply;
        rcfg.late_threshold = cfg.late_threshold;

        auto rollout = std::make_shared<RolloutStrategy>(base, rcfg);

        auto start = Clock::now();
        auto cfr = rollout->cubeful_cube_decision(board, cube);
        auto end = Clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        printf("--- %s ---\n", cfg.name);
        printf("ND: %+.6f (SE %.6f)  DT: %+.6f (SE %.6f)  CL: %+.6f\n",
               cfr.nd_equity, cfr.nd_se,
               cfr.dt_equity, cfr.dt_se,
               cfr.cubeless.equity);
        printf("P(win)=%.6f  P(gw)=%.6f  P(gl)=%.6f\n",
               cfr.cubeless.mean_probs[0],
               cfr.cubeless.mean_probs[1],
               cfr.cubeless.mean_probs[3]);

        if (ref_nd == 0) {
            ref_nd = cfr.nd_equity;
            ref_dt = cfr.dt_equity;
            printf("Time: %.3f s  [BASELINE]\n\n", elapsed);
        } else {
            printf("Time: %.3f s  (%.1fx)  dND=%+.6f  dDT=%+.6f\n\n",
                   elapsed, elapsed > 0 ? (configs[0].n_trials == 360 ? 1.0 : 1.0) : 0,
                   cfr.nd_equity - ref_nd, cfr.dt_equity - ref_dt);
        }
    }

    return 0;
}
