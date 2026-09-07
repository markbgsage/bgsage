// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/mobile.h"
#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/cube.h"
#include "bgbot/moves.h"
#include "bgbot/neural_net.h"
#include "bgbot/pubeval.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>

using namespace bgbot;
struct BgsageMobileEngine {
    BearoffDB db;
    std::shared_ptr<Strategy> base;
    std::shared_ptr<Strategy> strategy;
};
namespace {
thread_local std::string last_error;
struct Move { Board board; std::array<double, 5> probs; double equity, cubeless; };
std::array<double, 5> widen(const std::array<float, 5>& p) {
    return {p[0], p[1], p[2], p[3], p[4]};
}
std::array<double, 5> invert(const std::array<float, 5>& p) {
    return {1.0-p[0], p[3], p[4], p[1], p[2]};
}
double cl(const std::array<double,5>& p) { return 2*p[0]-1+p[1]-p[3]+p[2]-p[4]; }
template<typename T, size_t N> void array(std::ostringstream& s, const std::array<T,N>& a) {
    s << '[';
    for(size_t i=0;i<N;++i) { if(i) s << ','; s << a[i]; }
    s << ']';
}
void validate(const BgsageMobileRequest& r, int operation) {
    if(operation<0 || operation>1 || r.ply<1 || r.ply>2 || r.cube_owner<0 || r.cube_owner>2 ||
       r.cube_value<1 || r.cube_value>32768 || (r.cube_value & (r.cube_value-1)) ||
       r.away1<0 || r.away2<0 || r.away1>25 || r.away2>25 ||
       ((r.away1==0)!=(r.away2==0)) || r.budget_ms<1 || r.budget_ms>1000 ||
       (operation==0 && (r.die1<1 || r.die1>6 || r.die2<1 || r.die2>6)))
        throw std::invalid_argument("unsupported request");
    int player=r.board[25], opponent=r.board[0];
    if(player<0 || opponent<0 || player>15 || opponent>15) throw std::invalid_argument("invalid bar");
    for(int i=1;i<25;++i) {
        if(r.board[i]<-15 || r.board[i]>15) throw std::invalid_argument("invalid checker count");
        if(r.board[i]>0) player+=r.board[i]; else opponent-=r.board[i];
    }
    if(player<1 || player>15 || opponent<1 || opponent>15)
        throw std::invalid_argument("invalid or terminal board");
}
}
extern "C" BgsageMobileEngine* bgsage_mobile_create(const char* type,
    const char* const* paths, const int* hidden, int count, const char* bearoff) {
    try {
        if(!type || !paths || !hidden || !bearoff || count<5 || count>32)
            throw std::invalid_argument("invalid model configuration");
        auto e=std::make_unique<BgsageMobileEngine>();
        std::vector<std::string> p; std::vector<int> h;
        for(int i=0;i<count;++i) { if(!paths[i] || hidden[i]<1 || hidden[i]>4096) throw std::invalid_argument("invalid model asset"); p.emplace_back(paths[i]); h.push_back(hidden[i]); }
        const std::string t(type);
        if(t=="backgame_pair_phased") e->base=std::make_shared<BackgameAwarePairStrategy>(p,h,true);
        else if(t=="backgame_pair") e->base=std::make_shared<BackgameAwarePairStrategy>(p,h);
        else if(t=="pair") e->base=std::make_shared<GamePlanPairStrategy>(p,h);
        else if(t=="5nn") e->base=std::make_shared<GamePlanStrategy>(p,h);
        else throw std::invalid_argument("unsupported model strategy");
        if(!e->db.load(bearoff)) throw std::runtime_error("bearoff asset unavailable");
        e->strategy=std::make_shared<BearoffStrategy>(e->base,&e->db);
        last_error.clear(); return e.release();
    } catch(const std::exception& ex) { last_error=ex.what(); return nullptr; }
    catch(...) { last_error="engine initialization failed"; return nullptr; }
}
extern "C" void bgsage_mobile_destroy(BgsageMobileEngine* e) { delete e; }
extern "C" void bgsage_mobile_free(char* p) { std::free(p); }
extern "C" const char* bgsage_mobile_last_error() { return last_error.c_str(); }

extern "C" char* bgsage_mobile_analyze(BgsageMobileEngine* e, int operation,
    const BgsageMobileRequest* request) {
    try {
        if(!e || !request) throw std::invalid_argument("engine unavailable");
        const auto& r=*request; validate(r,operation);
        Board board{}; std::copy(r.board,r.board+26,board.begin());
        CubeInfo ci{r.cube_value,static_cast<CubeOwner>(r.cube_owner),
            {r.away1,r.away2,r.is_crawford!=0},-1.0f,
            r.away1==0 && r.jacoby!=0,r.away1==0 && r.beaver!=0};
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(r.budget_ms);
        const auto check_budget=[&] { if(std::chrono::steady_clock::now()>deadline) throw std::runtime_error("native budget exceeded"); };
        std::ostringstream s; s.imbue(std::locale::classic()); s << std::setprecision(17) << std::boolalpha;
        const MoveFilter filter{5,0.08f};
        if(operation==0) {
            std::vector<Board> candidates; possible_boards(board,r.die1,r.die2,candidates);
            std::vector<Move> moves;
            for(const auto& b:candidates) {
                check_budget(); Move m{}; m.board=b;
                if(r.ply==1) {
                    auto p=e->strategy->evaluate_probs(b,board); m.probs=widen(p);
                    auto [pp,op]=pip_counts(b); const auto x=cube_efficiency(p,is_race(b),pp,op);
                    m.equity=cl2cf(p,ci,x);
                } else {
                    // Mirrors BgBotAnalyzer's cube-aware rescore of EVERY
                    // candidate, including root routing and flipped match state.
                    auto v=cubeful_probs_and_equity_nply(flip(b),flip_cube_perspective(ci),
                        *e->strategy,r.ply,filter,1,nullptr,&board);
                    m.probs=invert(v.probs); m.equity=-v.equity;
                }
                m.cubeless=cl(m.probs); moves.push_back(m);
            }
            std::sort(moves.begin(),moves.end(),[](const Move& a,const Move& b) {
                if(a.equity!=b.equity) return a.equity>b.equity;
                if(a.cubeless!=b.cubeless) return a.cubeless>b.cubeless;
                return a.board<b.board;
            });
            s << "{\"moves\":[";
            for(size_t i=0;i<moves.size();++i) { const auto& m=moves[i]; if(i) s << ',';
                s << "{\"board\":"; array(s,m.board); s << ",\"probs\":"; array(s,m.probs);
                s << ",\"equity\":" << m.equity << ",\"cubeless_equity\":" << m.cubeless
                  << ",\"equity_diff\":" << m.equity-moves[0].equity << ",\"eval_level\":\"" << r.ply << "-ply\"}";
            } s << "]}";
        } else {
            std::array<float,5> p; CubeDecision cd;
            if(r.ply==1) {
                p=invert_probs(e->strategy->evaluate_probs(flip(board),is_race(board)));
                auto [pp,op]=pip_counts(board);
                cd=cube_decision_1ply(p,ci,cube_efficiency(p,is_race(board),pp,op));
            } else {
                PubEval prefilter;
                cd=cube_decision_nply(board,ci,*e->strategy,r.ply,filter,1,&prefilter);
                check_budget();
                MultiPlyStrategy multipy(e->strategy,r.ply,filter,false,false,1);
                multipy.set_bearoff_db(&e->db);
                const Board flipped=flip(board);
                p=invert_probs(multipy.evaluate_probs(flipped,flipped)); multipy.clear_cache();
            }
            const char* action=!cd.should_double ? "No Double" : cd.is_beaver ? "Double/Beaver" : cd.should_take ? "Double/Take" : "Double/Pass";
            s << "{\"probs\":"; array(s,p);
            s << ",\"cubeless_equity\":" << cubeless_equity(p)
              << ",\"equity_nd\":" << cd.equity_nd << ",\"equity_dt\":" << cd.equity_dt
              << ",\"equity_dp\":" << cd.equity_dp << ",\"should_double\":" << cd.should_double
              << ",\"should_take\":" << cd.should_take << ",\"optimal_equity\":" << cd.optimal_equity
              << ",\"optimal_action\":\"" << action << "\",\"is_beaver\":" << cd.is_beaver
              << ",\"eval_level\":\"" << r.ply << "-ply\",\"cubeless_se\":null,\"equity_nd_se\":null,\"equity_dt_se\":null}";
        }
        check_budget(); auto text=s.str(); char* out=static_cast<char*>(std::malloc(text.size()+1));
        if(!out) throw std::bad_alloc(); std::memcpy(out,text.c_str(),text.size()+1);
        last_error.clear(); return out;
    } catch(const std::exception& ex) { last_error=ex.what(); return nullptr; }
    catch(...) { last_error="native evaluation failed"; return nullptr; }
}
