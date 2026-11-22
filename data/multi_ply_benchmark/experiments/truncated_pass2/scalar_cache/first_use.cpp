// SPDX-License-Identifier: MPL-2.0
#include "bgbot/multipy.h"
#include <iostream>

int main() {
    constexpr int workers = 16;
    constexpr int rounds = 8;
    for (int round = 0; round < rounds; ++round) {
        bgbot::SharedPosCache cache;
        cache.clear();
        if (!cache.entries.empty()) return 1;
        std::atomic<int> ready{0};
        std::atomic<bool> start{false};
        std::atomic<bool> ok{true};
        std::vector<std::thread> threads;
        for (int i = 0; i < workers; ++i) {
            threads.emplace_back([&, i] {
                bgbot::Board board{};
                board[1] = i;
                const size_t key = 8 * i + 1;
                const std::array<float, bgbot::NUM_OUTPUTS> probs{static_cast<float>(i)};
                ready.fetch_add(1, std::memory_order_relaxed);
                while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
                if (i % 3 == 0) {
                    auto result = cache.lookup_or_reserve(key, 2, board);
                    if (!result.reservation) {
                        ok.store(false);
                        return;
                    }
                    cache.publish(result.reservation, probs);
                } else {
                    if (i % 3 == 1 && cache.lookup(key, 2, board)) ok.store(false);
                    cache.insert(key, 2, board, probs);
                }
                const auto* read = cache.lookup(key, 2, board);
                if (!read || *read != probs) ok.store(false);
            });
        }
        while (ready.load(std::memory_order_relaxed) != workers) std::this_thread::yield();
        start.store(true, std::memory_order_release);
        for (auto& t : threads) t.join();
        if (!ok.load() || cache.inserts.load() != workers || cache.entries.size() != cache.CAPACITY) return 2;

        const auto* storage = cache.entries.data();
        cache.clear();
        if (storage != cache.entries.data() || cache.inserts.load() != 0) return 3;
        bgbot::Board empty{};
        if (cache.lookup(1, 2, empty)) return 4;
        const std::array<float, bgbot::NUM_OUTPUTS> replaced{0.9f};
        cache.insert(1, 2, empty, replaced);
        const auto* final = cache.lookup(1, 2, empty);
        if (!final || *final != replaced) return 5;
    }
    std::cout << rounds << " fresh tables, " << workers
              << " simultaneous first readers/writers per table; clear and reuse passed\n";
}
