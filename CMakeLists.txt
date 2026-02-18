cmake_minimum_required(VERSION 3.18)
project(ARMM LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_EXECUTORCH "Enable LLM translation via ExecuTorch + Qwen3" OFF)

# ---------- SIMD flags ----------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -march=armv8-a+simd")
    message(STATUS "Target: ARM64 NEON")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -mavx2 -mfma")
    message(STATUS "Target: x86_64 AVX2")
endif()

# Optimization — no -ffast-math (breaks whisper.cpp)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_C_FLAGS_RELEASE   "-O3 -funroll-loops")

# ---------- Component libraries ----------
# MT before ASR so that if llama.cpp were present its ggml is built first.
add_subdirectory(mt)
add_subdirectory(asr)
add_subdirectory(tts)
add_subdirectory(pipeline)
