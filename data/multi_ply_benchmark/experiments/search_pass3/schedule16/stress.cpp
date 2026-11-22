// SPDX-License-Identifier: MPL-2.0
#include "bgbot/multipy.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

using bgbot::multipy_cooperative_for;
using bgbot::multipy_parallel_run;

struct Rendezvous {
    std::mutex mutex;
    std::condition_variable changed;
    int count = 0;
    void wait_for(int target) {
        std::unique_lock lock(mutex);
        if (++count == target) changed.notify_all();
        else changed.wait(lock, [&] { return count == target; });
    }
};

int main() {
    const int pool_size = std::max(1u, std::thread::hardware_concurrency());
    const int outer = pool_size + 1;  // all pool workers and the main thread
    int callbacks = 0;
    for (int iteration = 0; iteration < 100; ++iteration) {
        Rendezvous started, finished;
        std::atomic<bool> invalid{false};
        multipy_parallel_run(outer, [&] {
            started.wait_for(outer);
            std::array<std::atomic<int>, 21> seen{};
            for (auto& item : seen) item.store(0);
            auto cookie = std::make_shared<int>(iteration);
            std::weak_ptr<int> captured = cookie;
            bool caught = false;
            try {
                multipy_cooperative_for(21, 4, [&, cookie = std::move(cookie)](int i) {
                    if (*cookie != iteration || ++seen[i] != 1) invalid = true;
                    if (i == iteration % 21) throw std::runtime_error("roll failure");
                });
            } catch (const std::runtime_error&) {
                caught = true;
            } catch (...) {
                invalid = true;
            }
            if (!captured.expired() || !caught) invalid = true;
            for (auto& item : seen) if (item.load() != 1) invalid = true;
            // Queued helpers cannot run until everyone has finished the
            // above checks and relinquished the stack buffers that they used.
            finished.wait_for(outer);
        });
        if (invalid) return 1;
        callbacks += outer * 21;
    }
    std::cout << "100 saturated groups, " << callbacks
              << " roll callbacks, all visited once; exception and capture release checks passed\n";

    for (int budget = 8; budget <= 21; ++budget) {
        for (int count : {2, 3, 4, 5, 12, 21, 40}) {
            const int workers = std::min(count, budget / 2);
            const int lanes = budget / workers;
            std::atomic<int> next{0}, active{0}, peak{0}, total{0};
            multipy_parallel_run(workers, [&] {
                for (int i = next++; i < count; i = next++) {
                    multipy_cooperative_for(21, lanes, [&](int) {
                        int current = ++active;
                        int maximum = peak.load();
                        while (maximum < current && !peak.compare_exchange_weak(maximum, current)) {}
                        std::this_thread::yield();
                        ++total;
                        --active;
                    });
                }
            });
            if (active || total != count * 21 || peak > budget) return 2;
        }
    }
    std::cout << "98 scheduling combinations satisfied callback counts and CPU budgets\n";
}
