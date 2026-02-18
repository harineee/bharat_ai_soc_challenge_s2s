cmake_minimum_required(VERSION 3.18)
project(TTSModule)

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

add_library(tts_wrapper STATIC tts_wrapper.cpp)
target_include_directories(tts_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================
# Android: use prebuilt imported target created by parent CMake
# ============================================================
if(ANDROID)
    if(TARGET onnxruntime)
        message(STATUS "TTS: linking ONNX Runtime (Android prebuilt)")
        target_compile_definitions(tts_wrapper PUBLIC USE_ONNXRUNTIME)
        target_link_libraries(tts_wrapper onnxruntime)
        if(ORT_INCLUDE_DIR)
            target_include_directories(tts_wrapper PUBLIC "${ORT_INCLUDE_DIR}")
        endif()
    else()
        message(STATUS "TTS: ONNX Runtime not available — test tone fallback")
    endif()

# ============================================================
# Desktop: use pkg-config or conda
# ============================================================
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ONNXRUNTIME QUIET onnxruntime)
    endif()

    if(ONNXRUNTIME_FOUND)
        message(STATUS "TTS: ONNX Runtime found via pkg-config")
        target_compile_definitions(tts_wrapper PUBLIC USE_ONNXRUNTIME)
        target_include_directories(tts_wrapper PUBLIC ${ONNXRUNTIME_INCLUDE_DIRS})
        target_link_libraries(tts_wrapper ${ONNXRUNTIME_LIBRARIES})
    else()
        # Try ORT_ROOT_DIR (prebuilt download), then CONDA_PREFIX
        if(NOT ORT_ROOT_DIR)
            set(ORT_ROOT_DIR "$ENV{ORT_ROOT_DIR}")
        endif()
        set(_tts_ort_search "${ORT_ROOT_DIR}" "$ENV{CONDA_PREFIX}")
        set(_tts_ort_found FALSE)

        foreach(_ort_path ${_tts_ort_search})
            if(_tts_ort_found OR NOT _ort_path)
                continue()
            endif()

            # Prebuilt layout (flat include)
            if(EXISTS "${_ort_path}/include/onnxruntime_cxx_api.h"
               AND (EXISTS "${_ort_path}/lib/libonnxruntime.so" OR EXISTS "${_ort_path}/lib/libonnxruntime.dylib"))
                message(STATUS "TTS: ONNX Runtime found (prebuilt): ${_ort_path}")
                target_compile_definitions(tts_wrapper PUBLIC USE_ONNXRUNTIME)
                target_include_directories(tts_wrapper PUBLIC "${_ort_path}/include")
                target_link_directories(tts_wrapper PRIVATE "${_ort_path}/lib")
                find_library(ORT_LIB_TTS onnxruntime PATHS "${_ort_path}/lib" NO_DEFAULT_PATH)
                if(ORT_LIB_TTS)
                    target_link_libraries(tts_wrapper ${ORT_LIB_TTS})
                else()
                    target_link_libraries(tts_wrapper onnxruntime)
                endif()
                set(_tts_ort_found TRUE)
            # Conda layout
            elseif(EXISTS "${_ort_path}/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
                   AND (EXISTS "${_ort_path}/lib/libonnxruntime.so" OR EXISTS "${_ort_path}/lib/libonnxruntime.dylib"))
                message(STATUS "TTS: ONNX Runtime found (conda): ${_ort_path}")
                target_compile_definitions(tts_wrapper PUBLIC USE_ONNXRUNTIME)
                target_include_directories(tts_wrapper PUBLIC "${_ort_path}/include")
                target_link_directories(tts_wrapper PRIVATE "${_ort_path}/lib")
                find_library(ORT_LIB_TTS onnxruntime PATHS "${_ort_path}/lib" NO_DEFAULT_PATH)
                if(ORT_LIB_TTS)
                    target_link_libraries(tts_wrapper ${ORT_LIB_TTS})
                else()
                    target_link_libraries(tts_wrapper onnxruntime)
                endif()
                set(_tts_ort_found TRUE)
            endif()
        endforeach()

        if(NOT _tts_ort_found)
            message(STATUS "TTS: ONNX Runtime not found — test tone fallback")
        endif()
    endif()
endif()
