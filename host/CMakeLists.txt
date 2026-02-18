cmake_minimum_required(VERSION 3.18)
project(HostTranslation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    if(NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+simd")
    endif()
    message(STATUS "Host: ARM64 NEON")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")
    message(STATUS "Host: x86_64 AVX2")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops")

option(USE_EXECUTORCH "Enable LLM translation via ExecuTorch" OFF)

# PortAudio (optional, real-time mic input)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PORTAUDIO QUIET portaudio-2.0)
endif()
if(PORTAUDIO_FOUND)
    message(STATUS "PortAudio found — real-time mode enabled")
    add_definitions(-DUSE_PORTAUDIO)
else()
    message(STATUS "PortAudio not found — file mode only")
endif()

# Component libraries
add_subdirectory(../mt  mt)
add_subdirectory(../asr asr)
add_subdirectory(../tts tts)
add_subdirectory(../pipeline pipeline)

add_executable(translation_host main.cpp)

target_include_directories(translation_host PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/../asr
    ${CMAKE_CURRENT_SOURCE_DIR}/../mt
    ${CMAKE_CURRENT_SOURCE_DIR}/../tts
)

target_link_libraries(translation_host
    pipeline asr_wrapper mt_wrapper tts_wrapper
)

if(PORTAUDIO_FOUND)
    target_include_directories(translation_host PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(translation_host ${PORTAUDIO_LIBRARIES})
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(translation_host pthread)
endif()
# macOS links pthreads automatically via system libraries

if(APPLE)
    target_link_libraries(translation_host
        "-framework CoreFoundation"
        "-framework Foundation"
    )
    # Add rpaths for dynamic libraries (SentencePiece, ONNX Runtime)
    if(DEFINED ENV{CONDA_PREFIX})
        target_link_options(translation_host PRIVATE "LINKER:-rpath,$ENV{CONDA_PREFIX}/lib")
    endif()
    if(ORT_ROOT_DIR)
        target_link_options(translation_host PRIVATE "LINKER:-rpath,${ORT_ROOT_DIR}/lib")
    endif()
    # ExecuTorch: force-load static libraries that use constructor-based
    # registration (ops, backends). Must come BEFORE their dependencies
    # in the link line, with deps repeated after to resolve new symbols.
    if(USE_EXECUTORCH AND ET_ROOT)
        target_link_libraries(translation_host
            -force_load "${ET_ROOT}/lib/libexecutorch.a"
            -force_load "${ET_ROOT}/lib/libportable_ops_lib.a"
            -force_load "${ET_ROOT}/lib/libxnnpack_backend.a"
            "${ET_ROOT}/lib/libXNNPACK.a"
            "${ET_ROOT}/lib/libxnnpack-microkernels-prod.a"
            "${ET_ROOT}/lib/libpthreadpool.a"
            "${ET_ROOT}/lib/libcpuinfo.a"
            "${ET_ROOT}/lib/libkleidiai.a"
            "${ET_ROOT}/lib/libexecutorch.a"
            "${ET_ROOT}/lib/libexecutorch_core.a"
            "${ET_ROOT}/lib/libkernels_util_all_deps.a"
            "${ET_ROOT}/lib/libportable_kernels.a"
            "${ET_ROOT}/lib/libextension_threadpool.a"
        )
        if(EXISTS "${ET_ROOT}/lib/libquantized_ops_lib.a")
            target_link_libraries(translation_host
                -force_load "${ET_ROOT}/lib/libquantized_ops_lib.a"
                "${ET_ROOT}/lib/libquantized_kernels.a"
            )
        endif()
        # LLM custom ops (llama::custom_sdpa, llama::update_cache)
        if(EXISTS "${ET_ROOT}/lib/libcustom_ops.a")
            target_link_libraries(translation_host
                -force_load "${ET_ROOT}/lib/libcustom_ops.a"
            )
        endif()
        # Force-load regex_lookahead for PCRE2 tokenizer support
        if(EXISTS "${ET_ROOT}/lib/libregex_lookahead.a")
            target_link_libraries(translation_host
                -force_load "${ET_ROOT}/lib/libregex_lookahead.a"
                "${ET_ROOT}/lib/libpcre2-8.a"
            )
        endif()
    endif()
endif()
