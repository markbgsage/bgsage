// SPDX-License-Identifier: MPL-2.0
// One bit per point, with point 1 at bit 0 and the mover's bar at bit 24.
struct ShotPosition {
    unsigned men = 0;
    unsigned anchors = 0;
    unsigned blots = 0;
    unsigned blocked = 0;
    unsigned pip_sources = 0;
    unsigned pip_blots = 0;
    int bar = 0;
};

static ShotPosition shot_position(const Board& b) {
    ShotPosition s;
    unsigned made_home = 0;
    for (int i=0; i<24; ++i) {
        int v=b[i+1];
        s.men |= static_cast<unsigned>(v>0) << i;
        s.anchors |= static_cast<unsigned>(v>=2) << i;
        s.blots |= static_cast<unsigned>(v==-1) << i;
        s.blocked |= static_cast<unsigned>(v<=-2) << i;
    }
    int occupied_home=0;
    for (int i=0; i<6; ++i) {
        made_home |= static_cast<unsigned>(b[i+1]==2) << i;
        occupied_home += b[i+1]>0;
    }
    s.pip_blots=occupied_home>2 ? s.blots : (s.blots & ~3u);
    s.pip_sources=s.men & ~made_home;
    s.bar=b[25];
    return s;
}

static inline int highest_hit_point(unsigned mask) {
    if (!mask) return 0;
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanReverse(&index, mask);
    return static_cast<int>(index)+1;
#elif defined(__GNUC__) || defined(__clang__)
    return 32-__builtin_clz(mask);
#else
    int bits=0;
    for (;mask;mask>>=1) ++bits;
    return bits;
#endif
}

static inline int first_front_anchor(unsigned anchors) {
    unsigned front=anchors>>12;
    if (!front) return 0;
    return 13-highest_hit_point(front & (0u-front));
}

static float backbone_from_anchors(const Board& board,unsigned anchors) {
    // The CPU backbone excludes the ace point. Each interval has the
    // number of checkers at its previous (higher) anchor as its weight.
    anchors &= ~1u;
    int previous=0, weighted=0, total=0;
    while (anchors) {
        int p=highest_hit_point(anchors);
        anchors ^= 1u<<(p-1);
        if (previous) {
            int gap=previous-p;
            int weight=gap<=6 ? 11 : (gap<=11 ? 13-gap : 0);
            total+=board[previous];
            weighted+=weight*board[previous];
        }
        previous=p;
    }
    return total ? 1.0f-static_cast<float>(weighted)/static_cast<float>(11*total) : 0.0f;
}

static int blocked_home(unsigned blocked) {
    static constexpr auto counts=[] {
        std::array<int,64> n{};
        for (int i=1;i<64;++i) n[i]=n[i>>1]+(i&1);
        return n;
    }();
    return counts[(blocked>>18)&63];
}

static inline bool different_blots(unsigned a,unsigned b) {
    const unsigned u=a|b;
    return a && b && (u & (u-1));
}

// Direct hits with two different dice cannot use the same unstacked source.
static inline bool separate_checkers(unsigned a, unsigned b, int hi, int lo,
                                     unsigned stacked) {
    if (!b) return false;
    while (a) {
        const unsigned target=a & (0u-a);
        const unsigned source=target<<hi;
        const unsigned forbidden=target|((source&stacked) ? 0u : source>>lo);
        if (b & ~forbidden) return true;
        a^=target;
    }
    return false;
}

struct ShotCounts { int single=0, twice=0, pips=0; };

// The two shots features and the no-bar piploss feature use the same
// roll geometries. Each shift checks every potential hit target at once.
template<bool Shots, bool Pips>
static ShotCounts mask_shots(const ShotPosition& s) {
    ShotCounts result;
    if (!s.blots) return result;
    const unsigned men=s.men|(static_cast<unsigned>(s.bar>0)<<24);
    const unsigned stacked=s.anchors|(static_cast<unsigned>(s.bar>1)<<24);
    const bool two_blots=(s.blots & (s.blots-1))!=0;
    unsigned direct[7]{}, bar_direct[7]{}, pip_direct[7]{};
    for (int die=1; die<=6; ++die) {
        const unsigned b1=s.blocked>>die;
        const unsigned b2=s.blocked>>(2*die);
        const unsigned b3=s.blocked>>(3*die);
        if constexpr (Shots) {
            direct[die]=s.blots & (s.men>>die);
            bar_direct[die]=s.blots & (men>>die);
            const unsigned multi=(s.men>>(2*die)) |
                (~b2 & ((s.men>>(3*die)) | (~b3 & (s.men>>(4*die)))));
            result.single+=bool(direct[die] | (s.blots & ~b1 & multi));
            if (two_blots) {
                const unsigned one=bar_direct[die];
                const unsigned two=s.blots & ~b1 & (men>>(2*die));
                const unsigned three=s.blots & ~b1 & ~b2 & (men>>(3*die));
                // Shared doubles admit gcd-normalized distances (1,1),
                // (1,2) and (1,3). The legacy double-shot convention also
                // tests a single checker stepping through multiple blots.
                const unsigned next=s.blots<<die;
                const unsigned below=~(s.blocked<<die) &
                    ((s.blots<<(2*die)) | (~(s.blocked<<(2*die)) & (s.blots<<(3*die))));
                const unsigned higher=s.blots>>die, higher2=s.blots>>(2*die);
                const unsigned stepped= (higher & (s.men>>(2*die))) |
                    (~b2 & (higher|higher2) & ((s.men>>(3*die)) | (~b3 & (s.men>>(4*die)))));
                result.twice+=different_blots(one,one|two|three) ||
                    (one & (next|below)) || (s.blots & ~b1 & stepped);
            }
        }
        if constexpr (Pips) {
            pip_direct[die]=s.pip_blots & (s.pip_sources>>die);
            const unsigned multi=(s.pip_sources>>(2*die)) |
                (~b2 & ((s.pip_sources>>(3*die)) | (~b3 & (s.pip_sources>>(4*die)))));
            result.pips+=highest_hit_point(pip_direct[die] | (s.pip_blots & ~b1 & multi));
        }
    }
    for (int hi=2; hi<=6; ++hi) {
        for (int lo=1; lo<hi; ++lo) {
            const unsigned reachable=~((s.blocked>>hi)&(s.blocked>>lo));
            if constexpr (Shots) {
                const unsigned hits=direct[hi]|direct[lo]|
                    (s.blots & reachable & (s.men>>(hi+lo)));
                result.single+=2*bool(hits);
                if (two_blots) {
                    const unsigned a=bar_direct[hi], b=bar_direct[lo];
                    bool stepped=(a & (s.blots<<lo)) || (b & (s.blots<<hi));
                    result.twice+=2*(stepped||separate_checkers(a,b,hi,lo,stacked));
                }
            }
            if constexpr (Pips) {
                const unsigned hits=pip_direct[hi]|pip_direct[lo]|
                    (s.pip_blots & reachable & (s.pip_sources>>(hi+lo)));
                result.pips+=2*highest_hit_point(hits);
            }
        }
    }
    return result;
}
