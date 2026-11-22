// SPDX-License-Identifier: MPL-2.0
#include "bgbot/board.h"
#include "bgbot/encoding.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
using namespace bgbot;

int main(int argc,char** argv) {
    if (argc<3) return 2;
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(),DWORD_PTR(1)<<14);
#endif
    std::ifstream input(argv[1],std::ios::binary);
    std::vector<Board> boards;
    Board board;
    while (input.read(reinterpret_cast<char*>(board.data()),sizeof(board))) boards.push_back(board);
    if (boards.empty()) return 2;
    if (std::string(argv[2])=="dump") {
        if (argc!=4) return 2;
        std::ofstream out(argv[3],std::ios::binary);
        for (const Board& b : boards) {
            auto c=compute_extended_contact_inputs(b);
            auto r=compute_tesauro_inputs(b);
            out.write(reinterpret_cast<const char*>(c.data()),sizeof(c));
            out.write(reinterpret_cast<const char*>(r.data()),sizeof(r));
        }
        return out ? 0 : 1;
    }
    init_escape_tables();
    volatile double guard=0;
    auto time=[&](const char* name,auto evaluate) {
        auto start=std::chrono::steady_clock::now();
        double total=0;
        for(int rep=0;rep<4;++rep)
            for(const auto& b:boards) total+=evaluate(b,rep);
        auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-start).count();
        guard+=total;
        std::cout<<name<<" "<<static_cast<double>(ns)/(4*boards.size())<<" "<<total<<"\n";
    };
    time("extended",[](const Board& b,int rep){return compute_extended_contact_inputs(b)[102+rep];});
    time("hitting_shots",[](const Board& b,int){return hitting_shots(b);});
    time("double_hitting_shots",[](const Board& b,int){return double_hitting_shots(b);});
    time("piploss",[](const Board& b,int){return compute_piploss(b);});
    time("back_escapes",[](const Board& b,int){return back_escapes(b);});
    time("moment2",[](const Board& b,int){return moment2(b);});
    time("backbone",[](const Board& b,int){return backbone(b);});
    time("classify",[](const Board& b,int){return static_cast<int>(classify_game_plan(b));});
    return guard<0;
}
