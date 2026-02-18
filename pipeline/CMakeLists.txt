cmake_minimum_required(VERSION 3.18)
project(PipelineModule)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    if(NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+simd")
    endif()
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops")

add_library(pipeline STATIC
    pipeline.cpp
    phrase_detector.cpp
    performance_tracker.cpp
)

target_include_directories(pipeline PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../asr
    ${CMAKE_CURRENT_SOURCE_DIR}/../mt
    ${CMAKE_CURRENT_SOURCE_DIR}/../tts
)

target_link_libraries(pipeline asr_wrapper mt_wrapper tts_wrapper)
