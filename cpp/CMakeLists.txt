cmake_minimum_required(VERSION 3.18)
project(bgbot LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architecture: RTX 4070 SUPER = compute capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

# Allow VS 2026 as host compiler (CUDA 13.1 only officially supports up to VS 2022)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")

# Optimization flags — compiler-specific
if(MSVC)
    set(BGBOT_OPT_FLAGS /O2 /fp:fast /GL /arch:AVX2)
    set(BGBOT_SAFE_OPT_FLAGS /O2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(BGBOT_OPT_FLAGS -march=native -ffast-math -funroll-loops)
    set(BGBOT_SAFE_OPT_FLAGS -march=native -funroll-loops)
endif()

# Core library (CPU only)
add_library(bgbot_lib STATIC
    src/board.cpp
    src/moves.cpp
    src/strategy.cpp
    src/pubeval.cpp
    src/game.cpp
    src/benchmark.cpp
    src/encoding.cpp
    src/neural_net.cpp
    src/training.cpp
    src/multipy.cpp
    src/rollout.cpp
    src/cube.cpp
    src/match_equity.cpp
    src/bearoff.cpp
)
target_include_directories(bgbot_lib PUBLIC include)
target_compile_options(bgbot_lib PRIVATE ${BGBOT_OPT_FLAGS})

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(bgbot_lib PUBLIC Threads::Threads)

# CUDA library for GPU-accelerated supervised learning
find_package(CUDAToolkit REQUIRED)
add_library(bgbot_cuda STATIC
    src/cuda_nn.cu
)
target_include_directories(bgbot_cuda PUBLIC include)
target_link_libraries(bgbot_cuda PUBLIC bgbot_lib CUDA::cublas CUDA::cudart)
# Pass host compiler flags to nvcc for the host code
if(MSVC)
    target_compile_options(bgbot_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O2>)
endif()

# Main executable — PubEval vs PubEval simulation
add_executable(bgbot_sim src/main.cpp)
target_link_libraries(bgbot_sim PRIVATE bgbot_lib)

# Profiler executable
add_executable(bgbot_profile src/profile_training.cpp)
target_link_libraries(bgbot_profile PRIVATE bgbot_lib)

# Test executable for benchmark crash
add_executable(bgbot_test_crash src/test_benchmark_crash.cpp)
target_link_libraries(bgbot_test_crash PRIVATE bgbot_lib)

# 1-ply evaluation profiler
add_executable(bgbot_profile_1ply src/profile_0ply.cpp)
target_link_libraries(bgbot_profile_1ply PRIVATE bgbot_lib)

# 4-ply cube profiler
add_executable(bgbot_profile_4ply src/profile_cube_4ply.cpp)
target_link_libraries(bgbot_profile_4ply PRIVATE bgbot_lib)

# Truncated rollout benchmark
add_executable(benchmark_truncated_rollout src/benchmark_truncated_rollout.cpp)
target_link_libraries(benchmark_truncated_rollout PRIVATE bgbot_lib)

# 3T benchmark (XG Roller++ Cube)
add_executable(benchmark_3t src/benchmark_3t.cpp)
target_link_libraries(benchmark_3t PRIVATE bgbot_lib)

# 4-ply cube benchmark
add_executable(benchmark_4ply src/benchmark_4ply.cpp)
target_link_libraries(benchmark_4ply PRIVATE bgbot_lib)

# 4-ply optimization benchmark
add_executable(benchmark_4ply_opt src/benchmark_4ply_opt.cpp)
target_link_libraries(benchmark_4ply_opt PRIVATE bgbot_lib)

add_executable(test_4ply_accuracy src/test_4ply_accuracy.cpp)
target_link_libraries(test_4ply_accuracy PRIVATE bgbot_lib)

# 3T profiler
add_executable(profile_3t src/profile_3t.cpp)
target_link_libraries(profile_3t PRIVATE bgbot_lib)

# 3T component profiler
add_executable(benchmark_3t_profile src/benchmark_3t_profile.cpp)
target_link_libraries(benchmark_3t_profile PRIVATE bgbot_lib)

# 3T experiment (truncation ply comparison)
add_executable(benchmark_3t_experiment src/benchmark_3t_experiment.cpp)
target_link_libraries(benchmark_3t_experiment PRIVATE bgbot_lib)

# 3T detailed profiler
add_executable(profile_3t_detailed src/profile_3t_detailed.cpp)
target_link_libraries(profile_3t_detailed PRIVATE bgbot_lib)

# pybind11 Python module
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(bgbot_cpp pybind/bindings.cpp)
target_link_libraries(bgbot_cpp PRIVATE bgbot_lib bgbot_cuda)

# On MinGW, statically link the C++ runtime so Python can load the .pyd
if(MINGW)
    target_link_options(bgbot_cpp PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
    target_link_options(bgbot_sim PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
endif()

# MSVC: use LTCG for link-time optimization
if(MSVC)
    target_link_options(bgbot_lib PRIVATE /LTCG)
endif()
