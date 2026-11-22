# 4P checker scheduler

Across the 100 checker positions at 16 threads on Linux, sharing the thread budget between checker endpoints and rolls reduced total wall time from 30.893 to 27.838 seconds (9.9%). All 100 complete analytics responses matched. Seventy-four calls finished sooner, and aggregate CPU time increased from 362.9 to 386.7 seconds. [Matched screen](../integrated-schedule/summary.json) · [Full analytics](../integrated-schedule/outputs.zip)

## Scheduling contract

For 4P checker analysis with at least two legal endpoints and a requested budget of 8–21 threads, the engine assigns `min(endpoint_count, floor(budget / 2))` endpoint workers. Each worker gets `floor(budget / endpoint_workers)` roll lanes, including its calling thread. Their product fits the requested budget. Each roll writes to its own slot and the endpoint reduces results in fixed dice order. [Checker dispatch](../../../../../cpp/src/cube_eval.cpp)

The nested roll caller claims work alongside the pool helpers. Even with every pool thread occupied by an endpoint, each endpoint can finish its unclaimed rolls. An endpoint waits for the roll callbacks that have started. Before returning or propagating a worker exception, it releases the callback and its captures. A queued helper that starts later sees a drained job. [Pool implementation](../../../../../cpp/src/multipy.cpp)

## Concurrency test

The native test fills every pool thread and the main thread before starting nested work. In each of 100 rounds, every endpoint checks that its 21 rolls run exactly once, that a deliberately failing roll propagates its exception, and that the callback releases its captured object while every pool thread is still inside the outer call. A second set of tests exercises all thread budgets from 8 through 21 across seven endpoint counts and checks peak simultaneous callbacks against the budget. On the 16-logical-core Linux host, the saturation test verified 35,700 roll callbacks and the budget test passed all 98 combinations. [Test source](stress.cpp) · [Recorded output](stress-output.txt)

Run from the engine root on Linux with CMake and a C++17 compiler. The timeout bounds the run if nested scheduling stalls.

```bash
study=data/multi_ply_benchmark/experiments/search_pass3/schedule16
build=build/schedule16-stress
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DBGBOT_BUILD_PYTHON=OFF -DBGBOT_BUILD_APPS=OFF
cmake --build "$build" --target bgbot_lib bgbot_cuda_stub --parallel 2
c++ -std=c++17 -O2 -Icpp/include "$study/stress.cpp" \
  -Wl,--start-group "$build/libbgbot_lib.a" "$build/libbgbot_cuda_stub.a" -Wl,--end-group \
  -pthread -o "$build/stress"
timeout 120s "$build/stress"
```
