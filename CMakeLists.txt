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

# Bundle the production model's weights and the bearoff database into the
# wheel, inside the `bgsage` package under `_assets/`.  `weights.py` discovers
# them there when running from an installed wheel.
#
# The production model is `stage11` (PRODUCTION_MODEL in python/bgsage/weights.py),
# a 24-NN layout: Stage 9's 17 standard pair NNs (sl_s9_*) carried unchanged,
# plus 7 backgame/containment specialists (sl_s11_bg_*).  Both families are
# required — omitting either makes BgBotAnalyzer() fail with FileNotFoundError
# at runtime, which the build itself would not catch.
#
# WHEN PROMOTING A NEW PRODUCTION MODEL: add its weight-file pattern here, and
# keep the BGSAGE_MIN_WEIGHTS floor below in step with the new file count.
file(GLOB BGSAGE_MODEL_WEIGHTS
    "${CMAKE_CURRENT_SOURCE_DIR}/models/sl_s9_*.weights.best"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/sl_s11_*.weights.best")

# Drop training artifacts the registry never loads (e.g. the TD-init variant
# sl_s11_bg_deep_tdinit) — they'd be dead weight in every wheel.
list(FILTER BGSAGE_MODEL_WEIGHTS EXCLUDE REGEX "_tdinit\\.weights\\.best$")

# stage11 resolves to 21 distinct files (14 sl_s9_* after NN sharing + 7
# sl_s11_bg_*).  The glob picks up all 19 sl_s9_* files, so 26 is the expected
# count; require at least 21 so a partial checkout fails the build loudly
# rather than shipping a wheel that cannot construct an analyzer.
set(BGSAGE_MIN_WEIGHTS 21)
list(LENGTH BGSAGE_MODEL_WEIGHTS BGSAGE_N_WEIGHTS)

if(BGSAGE_N_WEIGHTS LESS BGSAGE_MIN_WEIGHTS)
    message(FATAL_ERROR
        "Only ${BGSAGE_N_WEIGHTS} model weight files found under "
        "${CMAKE_CURRENT_SOURCE_DIR}/models/ (need at least ${BGSAGE_MIN_WEIGHTS}). "
        "The production model needs both sl_s9_*.weights.best and "
        "sl_s11_*.weights.best to be present for the wheel build.")
endif()

message(STATUS "bgsage: bundling ${BGSAGE_N_WEIGHTS} model weight files")

install(FILES ${BGSAGE_MODEL_WEIGHTS}
    DESTINATION bgsage/_assets/models)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/bearoff_1sided.db"
    DESTINATION bgsage/_assets/data)
