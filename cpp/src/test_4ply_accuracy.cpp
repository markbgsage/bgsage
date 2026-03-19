// Test 4-ply cube accuracy across multiple positions.
// Compares values with and without the prefilter optimization.
// Must be run with the optimized code compiled in.

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
#include <cmath>

using namespace bgbot;
using Clock = std::chrono::high_resolution_clock;

struct TestPosition {
    const char* name;
    Board board;
};

int main(int argc, char* argv[]) {
    int n_threads = 16;

    std::filesystem::path model_dir;
    if (argc > 1) {
        model_dir = std::filesystem::path(argv[1]);
    } else {
        model_dir = std::filesystem::absolute(
            std::filesystem::path(argv[0])).parent_path().parent_path() / "models";
    }

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

    TestPosition positions[] = {
        {"Ref position (task)",
         {0,-2,0,0,0,0,5,2,3,0,0,0,-3,3,0,0,0,-3,2,-5,-2,0,0,0,0,0}},
        {"Profile ref position",
         {0,-2,0,0,0,2,4,2,2,0,0,0,-4,3,0,0,0,-1,-2,-4,-2,2,0,0,0,0}},
        {"Starting position",
         {0,-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0}},
        {"Race position",
         {0,0,0,0,3,4,5,3,0,0,0,0,0,0,0,0,0,0,0,-3,-4,-5,-3,0,0,0}},
        {"Blitz position",
         {0,-1,0,0,0,2,5,3,2,0,0,0,-3,4,0,0,0,-2,-1,-5,-2,1,0,0,0,0}},
    };

    printf("=== 4-ply Cube Accuracy Test ===\n");
    printf("%-25s  %12s  %12s  %8s\n", "Position", "ND equity", "DT equity", "Time(ms)");
    printf("%-25s  %12s  %12s  %8s\n", "--------", "---------", "---------", "--------");

    for (const auto& pos : positions) {
        auto start = Clock::now();
        auto decision = cube_decision_nply(pos.board, cube, strategy, 4,
                                            MoveFilters::TINY, n_threads);
        auto end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        printf("%-25s  %+12.8f  %+12.8f  %8.1f\n",
               pos.name, decision.equity_nd, decision.equity_dt, ms);
    }

    return 0;
}
