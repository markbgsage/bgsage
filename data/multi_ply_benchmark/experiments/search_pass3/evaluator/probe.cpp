// SPDX-License-Identifier: MPL-2.0
// Include the translation unit to time the private, fused shot encoder.
#include "encoding.cpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace bgbot;

static auto probe_masks(const Board& b) {
#if defined(__AVX2__)
    return both_contact_point_masks(b);
#else
    return std::make_pair(contact_point_masks(b), contact_point_masks(flip(b)));
#endif
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    if (std::string(argv[1]) == "audit") {
        // Independently generate disjoint ownership, plausible stacks, both bars
        // and borne-off checkers. Compare the integer features to their original
        // board-based functions before any input normalization or NN rounding.
        std::mt19937 generator(883410);
        for (int sample = 0; sample < 262144; ++sample) {
            Board b{};
            for (int side : {1, -1}) {
                const int count = 1 + generator() % 15;
                const int bar = std::min<int>(count, generator() % 4);
                b[side == 1 ? 25 : 0] = bar;
                for (int i = bar; i < count; ++i) {
                    int point;
                    do point = 1 + generator() % 24; while (b[point] * side < 0);
                    b[point] += side;
                }
            }
            const auto p = contact_point_masks(b);
            const auto shots = b[25] ? shot_features<true>(b,p) : shot_features<false>(b,p);
            if (shots.single_hits != hitting_shots(b) || shots.double_hits != double_hitting_shots(b) ||
                shots.pip_loss != compute_piploss(b)) {
                for (auto n : b) std::cerr << n << ' ';
                std::cerr << "mismatch sample " << sample << ": " << shots.single_hits << "/" << hitting_shots(b)
                          << " " << shots.double_hits << "/" << double_hitting_shots(b) << " " << shots.pip_loss
                          << "/" << compute_piploss(b) << std::endl;
                return 1;
            }
        }
        std::cout << "262144 integer shot triples equal the scalar reference" << std::endl;
        return 0;
    }
    std::vector<Board> boards;
    std::ifstream input(argv[1], std::ios::binary);
    Board board;
    while (input.read(reinterpret_cast<char*>(board.data()), sizeof(board))) boards.push_back(board);
    if (boards.empty()) return 2;
    init_escape_tables();
    if (argc == 3) {
        std::ofstream out(argv[2], std::ios::binary);
        for (const auto& b : boards) {
            auto f = compute_extended_contact_inputs(b);
            out.write(reinterpret_cast<char*>(f.data()), sizeof(f));
        }
        return out ? 0 : 1;
    }
    auto measure = [&](const char* name, auto run) {
        double total = 0;
        auto start = std::chrono::steady_clock::now();
        for (int rep = 0; rep != 4; ++rep)
            for (const auto& b : boards) total += run(b);
        double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
        std::cout << name << " " << ns / (4 * boards.size()) << " " << total << std::endl;
    };
    measure("extended_ns", [](const Board& b){return compute_extended_contact_inputs(b)[109];});
    measure("both_shots_ns", [](const Board& b){
        const auto [p,o] = probe_masks(b);
        const Board f = flip(b);
        const auto a = b[25] ? shot_features<true>(b,p) : shot_features<false>(b,p);
        const auto c = f[25] ? shot_features<true>(f,o) : shot_features<false>(f,o);
        return a.single_hits + a.double_hits + a.pip_loss + c.single_hits + c.double_hits + c.pip_loss;
    });
    measure("both_side_features_ns", [](const Board& b){
        const auto [p,o] = probe_masks(b);
        auto side = [](const Board& b, const ContactPointMasks& p, const ContactPointMasks& o) {
            int i,j,k,l,m,n,q; float a,c,d,e,f;
            compute_side_features(b, highest_set_point(p.enemy_blots | p.enemy_blocks), p.made, o.made,
                i,j,a,c,k,l,m,n,d,e,f,q);
            return i+j+a+c+k+l+m+n+d+e+f+q;
        };
        return side(b,p,o) + side(flip(b),o,p);
    });
    return 0;
}
