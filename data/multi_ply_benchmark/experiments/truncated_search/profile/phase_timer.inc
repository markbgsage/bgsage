// Test-build-only phase accounting. Local counters avoid shared atomics in
// per-half-move switches. Destruction flushes the last phase on early returns.
#include <chrono>
enum class Phase { prefill0, prefill1, cube, movegen, vr_mean, select,
                   vr_luck, truncation, other, count };
constexpr int nphases = static_cast<int>(Phase::count);
static std::atomic<long long> profile_ns[nphases];
static std::atomic<long long> profile_entries[nphases];
class PhaseTimer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_ = Clock::now();
    long long ns[nphases]{}, entries[nphases]{};
    Phase current;
    bool done = false;
    void record(Clock::time_point end) {
        const int i = static_cast<int>(current);
        ns[i] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
        ++entries[i];
    }
public:
    explicit PhaseTimer(Phase first) : current(first) {}
    void next(Phase phase) {
        const auto now = Clock::now();
        record(now);
        current = phase;
        start_ = now;
    }
    void finish() {
        if (done) return;
        done = true;
        record(Clock::now());
        for (int i = 0; i < nphases; ++i) if (entries[i]) {
            profile_ns[i].fetch_add(ns[i], std::memory_order_relaxed);
            profile_entries[i].fetch_add(entries[i], std::memory_order_relaxed);
        }
    }
    ~PhaseTimer() { finish(); }
};
#ifdef _WIN32
#define PROFILE_EXPORT __declspec(dllexport)
#else
#define PROFILE_EXPORT __attribute__((visibility("default")))
#endif
extern "C" PROFILE_EXPORT void take_rollout_profile(long long* ns, long long* entries) {
    for (int i = 0; i < nphases; ++i) {
        ns[i] = profile_ns[i].exchange(0, std::memory_order_relaxed);
        entries[i] = profile_entries[i].exchange(0, std::memory_order_relaxed);
    }
}
