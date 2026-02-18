cmake_minimum_required(VERSION 3.18)
project(MTModule)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_EXECUTORCH "Enable LLM translation via ExecuTorch + Qwen3" OFF)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    if(NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+simd")
    endif()
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops")

# Sources
set(MT_SOURCES
    mt_wrapper.cpp
    mt_wrapper.h
    marian_mt_wrapper.cpp
    marian_mt_wrapper.h
)

add_library(mt_wrapper STATIC ${MT_SOURCES})
target_include_directories(mt_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mt_wrapper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../pipeline)

# ============================================================
# Android: use prebuilt imported targets created by parent CMake
# ============================================================
if(ANDROID)
    # ONNX Runtime
    if(TARGET onnxruntime)
        message(STATUS "MT: linking ONNX Runtime (Android prebuilt)")
        target_compile_definitions(mt_wrapper PUBLIC USE_ONNXRUNTIME)
        target_link_libraries(mt_wrapper onnxruntime)
        if(ORT_INCLUDE_DIR)
            target_include_directories(mt_wrapper PUBLIC "${ORT_INCLUDE_DIR}")
        endif()
    else()
        message(STATUS "MT: ONNX Runtime not available — fallback mode")
    endif()

    # SentencePiece
    if(TARGET sentencepiece)
        message(STATUS "MT: linking SentencePiece (Android prebuilt)")
        target_compile_definitions(mt_wrapper PUBLIC USE_SENTENCEPIECE)
        target_link_libraries(mt_wrapper sentencepiece)
        if(SP_INCLUDE_DIR)
            target_include_directories(mt_wrapper PUBLIC "${SP_INCLUDE_DIR}")
        endif()
    else()
        message(STATUS "MT: SentencePiece not available — fallback tokenization")
    endif()

# ============================================================
# Desktop: use pkg-config or conda
# ============================================================
else()
    # ---------- SentencePiece ----------
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(SENTENCEPIECE QUIET sentencepiece)
    endif()

    if(NOT SENTENCEPIECE_FOUND)
        set(SP_ROOT "$ENV{CONDA_PREFIX}")
        if(SP_ROOT AND EXISTS "${SP_ROOT}/include/sentencepiece_processor.h")
            set(SENTENCEPIECE_FOUND TRUE)
            set(SENTENCEPIECE_INCLUDE_DIRS "${SP_ROOT}/include")
            set(SENTENCEPIECE_LIBRARY_DIRS "${SP_ROOT}/lib")
            set(SENTENCEPIECE_LIBRARIES "sentencepiece")
        endif()
    endif()

    if(SENTENCEPIECE_FOUND)
        message(STATUS "SentencePiece found")
        target_compile_definitions(mt_wrapper PUBLIC USE_SENTENCEPIECE)
        target_include_directories(mt_wrapper PUBLIC ${SENTENCEPIECE_INCLUDE_DIRS})
        if(SENTENCEPIECE_LIBRARY_DIRS)
            find_library(SP_LIB sentencepiece PATHS ${SENTENCEPIECE_LIBRARY_DIRS} NO_DEFAULT_PATH)
            if(SP_LIB)
                target_link_libraries(mt_wrapper ${SP_LIB})
            else()
                target_link_directories(mt_wrapper PUBLIC ${SENTENCEPIECE_LIBRARY_DIRS})
                target_link_libraries(mt_wrapper ${SENTENCEPIECE_LIBRARIES})
            endif()
        else()
            target_link_libraries(mt_wrapper ${SENTENCEPIECE_LIBRARIES})
        endif()
    else()
        message(STATUS "SentencePiece not found — MT will use fallback tokenization")
    endif()

    # ---------- ONNX Runtime ----------
    # Try pkg-config first
    if(PkgConfig_FOUND)
        pkg_check_modules(ONNXRUNTIME QUIET onnxruntime)
    endif()

    if(ONNXRUNTIME_FOUND)
        message(STATUS "ONNX Runtime found via pkg-config")
        target_compile_definitions(mt_wrapper PUBLIC USE_ONNXRUNTIME)
        target_include_directories(mt_wrapper PUBLIC ${ONNXRUNTIME_INCLUDE_DIRS})
        target_link_libraries(mt_wrapper ${ONNXRUNTIME_LIBRARIES})
    else()
        # Try ORT_ROOT_DIR (prebuilt download), then CONDA_PREFIX
        if(NOT ORT_ROOT_DIR)
            set(ORT_ROOT_DIR "$ENV{ORT_ROOT_DIR}")
        endif()
        set(_ort_search_paths "${ORT_ROOT_DIR}" "$ENV{CONDA_PREFIX}")
        set(_ort_found FALSE)

        foreach(_ort_path ${_ort_search_paths})
            if(_ort_found OR NOT _ort_path)
                continue()
            endif()

            # Check prebuilt layout (flat: include/onnxruntime_cxx_api.h)
            if(EXISTS "${_ort_path}/include/onnxruntime_cxx_api.h"
               AND (EXISTS "${_ort_path}/lib/libonnxruntime.so" OR EXISTS "${_ort_path}/lib/libonnxruntime.dylib"))
                message(STATUS "ONNX Runtime found (prebuilt): ${_ort_path}")
                target_compile_definitions(mt_wrapper PUBLIC USE_ONNXRUNTIME)
                target_include_directories(mt_wrapper PUBLIC "${_ort_path}/include")
                target_link_directories(mt_wrapper PRIVATE "${_ort_path}/lib")
                find_library(ORT_LIB_MT onnxruntime PATHS "${_ort_path}/lib" NO_DEFAULT_PATH)
                if(ORT_LIB_MT)
                    target_link_libraries(mt_wrapper ${ORT_LIB_MT})
                else()
                    target_link_libraries(mt_wrapper onnxruntime)
                endif()
                set(_ort_found TRUE)
            # Check conda layout (include/onnxruntime/core/session/onnxruntime_cxx_api.h)
            elseif(EXISTS "${_ort_path}/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
                   AND (EXISTS "${_ort_path}/lib/libonnxruntime.so" OR EXISTS "${_ort_path}/lib/libonnxruntime.dylib"))
                message(STATUS "ONNX Runtime found (conda): ${_ort_path}")
                target_compile_definitions(mt_wrapper PUBLIC USE_ONNXRUNTIME)
                target_include_directories(mt_wrapper PUBLIC "${_ort_path}/include")
                target_link_directories(mt_wrapper PRIVATE "${_ort_path}/lib")
                find_library(ORT_LIB_MT onnxruntime PATHS "${_ort_path}/lib" NO_DEFAULT_PATH)
                if(ORT_LIB_MT)
                    target_link_libraries(mt_wrapper ${ORT_LIB_MT})
                else()
                    target_link_libraries(mt_wrapper onnxruntime)
                endif()
                set(_ort_found TRUE)
            endif()
        endforeach()

        if(NOT _ort_found)
            message(STATUS "ONNX Runtime not found — MT will use fallback")
        endif()
    endif()
endif()

# ============================================================
# ExecuTorch (LLM backend) — Qwen3-0.6B
# ============================================================
if(USE_EXECUTORCH)
    # Add LLM translator and translation mode sources
    target_sources(mt_wrapper PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/llm_translator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/llm_translator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/translation_mode.h
    )
    target_compile_definitions(mt_wrapper PUBLIC USE_EXECUTORCH C10_USING_CUSTOM_GENERATED_MACROS)

    if(ANDROID)
        # Android: ExecuTorch comes from Maven AAR or prebuilt .so
        set(ET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libs/executorch")
        if(EXISTS "${ET_DIR}/include")
            target_include_directories(mt_wrapper PUBLIC "${ET_DIR}/include")
        endif()
        if(EXISTS "${ET_DIR}/jni/${ANDROID_ABI}/libexecutorch_jni.so")
            add_library(executorch_lib SHARED IMPORTED)
            set_target_properties(executorch_lib PROPERTIES
                IMPORTED_LOCATION "${ET_DIR}/jni/${ANDROID_ABI}/libexecutorch_jni.so")
            target_link_libraries(mt_wrapper executorch_lib)
        else()
            message(STATUS "MT: ExecuTorch prebuilt not found, will use AAR at runtime")
        endif()
    else()
        # Desktop: find ExecuTorch installation
        set(ET_ROOT "$ENV{EXECUTORCH_ROOT}" CACHE PATH "ExecuTorch installation root")
        if(NOT ET_ROOT)
            set(ET_ROOT "" CACHE PATH "ExecuTorch installation root" FORCE)
        endif()
        if(ET_ROOT AND EXISTS "${ET_ROOT}/lib/libexecutorch.a")
            message(STATUS "ExecuTorch found at: ${ET_ROOT}")
            target_include_directories(mt_wrapper PUBLIC
                "${ET_ROOT}/include"
            )
            # Core ExecuTorch libraries
            # Force-load ops libs so static constructors register operators
            if(APPLE)
                set(_ET_FORCE_LOAD -force_load)
            else()
                set(_ET_FORCE_LOAD -Wl,--whole-archive)
                set(_ET_NO_FORCE_LOAD -Wl,--no-whole-archive)
            endif()
            target_link_libraries(mt_wrapper
                "${ET_ROOT}/lib/libextension_module_static.a"
                "${ET_ROOT}/lib/libextension_tensor.a"
                "${ET_ROOT}/lib/libextension_named_data_map.a"
                "${ET_ROOT}/lib/libextension_data_loader.a"
                "${ET_ROOT}/lib/libextension_flat_tensor.a"
                "${ET_ROOT}/lib/libexecutorch.a"
                "${ET_ROOT}/lib/libexecutorch_core.a"
                "${ET_ROOT}/lib/libportable_ops_lib.a"
                "${ET_ROOT}/lib/libportable_kernels.a"
                "${ET_ROOT}/lib/libkernels_util_all_deps.a"
                "${ET_ROOT}/lib/libxnnpack_backend.a"
                "${ET_ROOT}/lib/libXNNPACK.a"
                "${ET_ROOT}/lib/libpthreadpool.a"
                "${ET_ROOT}/lib/libcpuinfo.a"
                "${ET_ROOT}/lib/libkleidiai.a"
                "${ET_ROOT}/lib/libextension_threadpool.a"
            )
            # Quantized + optimized kernels (required for 4-bit models)
            # Use INTERFACE link options to ensure force-load propagates
            # through the static library chain to the final executable
            if(EXISTS "${ET_ROOT}/lib/libquantized_kernels.a")
                target_link_libraries(mt_wrapper
                    "${ET_ROOT}/lib/libquantized_kernels.a"
                    "${ET_ROOT}/lib/libquantized_ops_lib.a"
                )
                if(APPLE)
                    target_link_options(mt_wrapper INTERFACE
                        -Wl,-force_load,${ET_ROOT}/lib/libquantized_ops_lib.a
                    )
                endif()
                message(STATUS "ExecuTorch: Quantized kernels enabled")
            endif()
            # Optimized kernels linked but NOT force-loaded to avoid
            # duplicate op registration with portable_ops_lib
            if(EXISTS "${ET_ROOT}/lib/liboptimized_kernels.a")
                target_link_libraries(mt_wrapper
                    "${ET_ROOT}/lib/liboptimized_kernels.a"
                    "${ET_ROOT}/lib/liboptimized_portable_kernels.a"
                    "${ET_ROOT}/lib/libcpublas.a"
                    "${ET_ROOT}/lib/libeigen_blas.a"
                )
                message(STATUS "ExecuTorch: Optimized kernels available")
            endif()
            # Export ET_ROOT for the final executable to set up force-loads
            set(ET_ROOT "${ET_ROOT}" CACHE INTERNAL "ExecuTorch install root")
            # HFTokenizer library
            if(EXISTS "${ET_ROOT}/lib/libtokenizers.a")
                # Force-load regex_lookahead and pcre2 to ensure static
                # constructors and all symbols are linked (PCRE2 needed
                # for Qwen3 tokenizer lookahead patterns)
                if(APPLE)
                    target_link_libraries(mt_wrapper
                        "${ET_ROOT}/lib/libtokenizers.a"
                        -force_load "${ET_ROOT}/lib/libregex_lookahead.a"
                        "${ET_ROOT}/lib/libpcre2-8.a"
                        "${ET_ROOT}/lib/libre2.a"
                        "${ET_ROOT}/lib/libsentencepiece.a"
                    )
                else()
                    target_link_libraries(mt_wrapper
                        "${ET_ROOT}/lib/libtokenizers.a"
                        -Wl,--whole-archive
                        "${ET_ROOT}/lib/libregex_lookahead.a"
                        -Wl,--no-whole-archive
                        "${ET_ROOT}/lib/libpcre2-8.a"
                        "${ET_ROOT}/lib/libre2.a"
                        "${ET_ROOT}/lib/libsentencepiece.a"
                    )
                endif()
                # Abseil dependencies (required by re2)
                file(GLOB ABSEIL_LIBS "${ET_ROOT}/lib/libabsl_*.a")
                if(ABSEIL_LIBS)
                    target_link_libraries(mt_wrapper ${ABSEIL_LIBS})
                endif()
                message(STATUS "ExecuTorch: HFTokenizer + regex_lookahead (PCRE2) enabled")
            else()
                message(WARNING "ExecuTorch: tokenizer library not found")
            endif()
        else()
            message(WARNING "ExecuTorch: ET_ROOT not set or libs not found")
            message(WARNING "  Set -DET_ROOT=/path/to/executorch/install")
            message(WARNING "  LLM translator will compile but may not link")
        endif()
    endif()

    message(STATUS "MT: ExecuTorch LLM backend enabled (Qwen3-0.6B)")
endif()
