// Per-rollout storage. Once a slot is published, no thread modifies it.
// Occupied slots with a different key fall through to ordinary evaluation.
struct VRTable {
    static constexpr int slots = 2048;
    struct Entry {
        Board board;
        std::array<CubeInfo, 4> cubes;
        int ncubes;
        float cube_x;
        std::array<std::array<float, NUM_OUTPUTS>, 21> probs;
        std::array<int, 21> picks;
    };
    std::array<std::atomic<unsigned char>, slots> state{};
    std::array<Entry, slots> data;

    static bool same_cube(const CubeInfo& a, const CubeInfo& b) {
        return a.cube_value == b.cube_value && a.owner == b.owner &&
            a.match.away1 == b.match.away1 && a.match.away2 == b.match.away2 &&
            a.match.is_crawford == b.match.is_crawford &&
            a.cube_x_override == b.cube_x_override && a.jacoby == b.jacoby &&
            a.beaver == b.beaver && a.max_cube_value == b.max_cube_value;
    }
    static int slot_for(const Board& board) {
        uint32_t h = 2166136261u;
        for (int p : board) h = (h ^ static_cast<uint32_t>(p)) * 16777619u;
        return (h ^ (h >> 16)) & (slots - 1);
    }
    const Entry* lookup(int slot, const Board& board, const CubeInfo* cubes,
                        int n, float x) const {
        if (state[slot].load(std::memory_order_acquire) != 2) return nullptr;
        const auto& e = data[slot];
        if (e.board != board || e.ncubes != n || e.cube_x != x) return nullptr;
        for (int i = 0; i < n; ++i) if (!same_cube(e.cubes[i], cubes[i])) return nullptr;
        return &e;
    }
    void put(int slot, const Board& board, const CubeInfo* cubes, int n, float x,
             const std::array<std::array<float, NUM_OUTPUTS>, 21>& probs,
             const std::array<int,21>& picks) {
        unsigned char expected = 0;
        if (!state[slot].compare_exchange_strong(expected, 1, std::memory_order_relaxed)) return;
        auto& e = data[slot];
        e.board = board; e.ncubes = n; e.cube_x = x;
        for (int i = 0; i < n; ++i) e.cubes[i] = cubes[i];
        e.probs = probs; e.picks = picks;
        state[slot].store(2, std::memory_order_release);
    }
};
