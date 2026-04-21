# bgsage — top-level CMakeLists.txt for the pip (scikit-build-core) build.
#
# This file delegates the compiled-extension build to cpp/CMakeLists_cpu.txt
# (the CPU-only engine build) and then adds the install steps that place the
# pybind11 module, model weights, and bearoff database into the wheel.
#
# The dev / Docker builds continue to use cpp/CMakeLists_cpu.txt directly and
# are unaffected by this file.

cmake_minimum_required(VERSION 3.18)
project(bgsage LANGUAGES CXX)

# Don't build the benchmark / profiler executables in the wheel — just the
# pybind11 module. Dev builds leave this ON (the default in CMakeLists_cpu.txt).
set(BGBOT_BUILD_APPS OFF CACHE BOOL "Build benchmark/profiler executables")

include("${CMAKE_CURRENT_SOURCE_DIR}/cpp/CMakeLists_cpu.txt")

# Install the Python extension at the wheel root so `import bgbot_cpp` works.
# scikit-build-core's default install prefix is the site-packages root.
if(TARGET bgbot_cpp)
    install(TARGETS bgbot_cpp
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .)
endif()

# Bundle the production model weights (stage9) and the bearoff database into
# the wheel, inside the `bgsage` package under `_assets/`.  `weights.py`
# discovers them there when running from an installed wheel.
file(GLOB BGSAGE_STAGE9_WEIGHTS
    "${CMAKE_CURRENT_SOURCE_DIR}/models/sl_s9_*.weights.best")

if(NOT BGSAGE_STAGE9_WEIGHTS)
    message(FATAL_ERROR
        "No stage9 weights found under ${CMAKE_CURRENT_SOURCE_DIR}/models/. "
        "Expected sl_s9_*.weights.best files to be present for the wheel build.")
endif()

install(FILES ${BGSAGE_STAGE9_WEIGHTS}
    DESTINATION bgsage/_assets/models)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/bearoff_1sided.db"
    DESTINATION bgsage/_assets/data)
