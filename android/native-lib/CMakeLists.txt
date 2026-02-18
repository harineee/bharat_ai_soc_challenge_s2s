cmake_minimum_required(VERSION 3.18)
project(NativeLib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- ARM NEON ----------
if(ANDROID AND "${ANDROID_ABI}" STREQUAL "arm64-v8a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -march=armv8-a+simd")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fPIC -march=armv8-a+simd")
elseif(ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fPIC")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto -funroll-loops -fno-rtti")
set(CMAKE_C_FLAGS_RELEASE   "-O3 -flto -funroll-loops")

# ===================================================================
# Set up ONNX Runtime and SentencePiece BEFORE adding component
# subdirectories, so mt_wrapper and tts_wrapper can link against them.
# ===================================================================

# ---------- ONNX Runtime (prebuilt for Android) ----------
set(ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../libs/onnxruntime")
set(ORT_LIB "${ORT_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so")

if(EXISTS "${ORT_LIB}")
    add_library(onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_LIB}"
    )
    # Determine header path (nested or flat)
    if(EXISTS "${ORT_DIR}/headers/onnxruntime/core/session/onnxruntime_cxx_api.h")
        set(ORT_INCLUDE_DIR "${ORT_DIR}/headers" CACHE PATH "" FORCE)
    elseif(EXISTS "${ORT_DIR}/headers/onnxruntime_cxx_api.h")
        set(ORT_INCLUDE_DIR "${ORT_DIR}/headers" CACHE PATH "" FORCE)
    endif()
    target_include_directories(onnxruntime INTERFACE "${ORT_INCLUDE_DIR}")
    set(ONNXRUNTIME_ANDROID_FOUND TRUE CACHE BOOL "" FORCE)
    message(STATUS "ONNX Runtime: found prebuilt for ${ANDROID_ABI}")
else()
    set(ONNXRUNTIME_ANDROID_FOUND FALSE CACHE BOOL "" FORCE)
    message(WARNING "ONNX Runtime: NOT found at ${ORT_LIB}")
    message(WARNING "  Run ./scripts/setup_android_deps.sh to download")
endif()

# ---------- SentencePiece ----------
set(SP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../libs/sentencepiece")
set(SP_LIB "${SP_DIR}/lib/${ANDROID_ABI}/libsentencepiece.a")
set(SP_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/sentencepiece")

if(EXISTS "${SP_LIB}")
    # Option 1: Use prebuilt static library
    add_library(sentencepiece STATIC IMPORTED GLOBAL)
    set_target_properties(sentencepiece PROPERTIES
        IMPORTED_LOCATION "${SP_LIB}"
    )
    set(SP_INCLUDE_DIR "${SP_DIR}/include" CACHE PATH "" FORCE)
    target_include_directories(sentencepiece INTERFACE "${SP_INCLUDE_DIR}")
    set(SENTENCEPIECE_ANDROID_FOUND TRUE CACHE BOOL "" FORCE)
    message(STATUS "SentencePiece: found prebuilt for ${ANDROID_ABI}")

elseif(EXISTS "${SP_SRC_DIR}/CMakeLists_android.txt")
    # Option 2: Build from source (NDK will cross-compile)
    message(STATUS "SentencePiece: building from source for ${ANDROID_ABI}")
    include("${SP_SRC_DIR}/CMakeLists_android.txt")

    # Create alias target so component CMakes can use 'sentencepiece'
    add_library(sentencepiece ALIAS sentencepiece_source)
    set(SP_INCLUDE_DIR "${SP_SRC_DIR}/src" CACHE PATH "" FORCE)
    set(SENTENCEPIECE_ANDROID_FOUND TRUE CACHE BOOL "" FORCE)

else()
    set(SENTENCEPIECE_ANDROID_FOUND FALSE CACHE BOOL "" FORCE)
    message(WARNING "SentencePiece: NOT found (prebuilt or source)")
    message(WARNING "  Run ./scripts/setup_android_deps.sh to download source")
endif()

# ---------- ExecuTorch (LLM backend) ----------
set(ET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../libs/executorch")
if(EXISTS "${ET_DIR}/jni/${ANDROID_ABI}/libexecutorch_jni.so")
    set(USE_EXECUTORCH ON CACHE BOOL "" FORCE)

    add_library(executorch_jni SHARED IMPORTED GLOBAL)
    set_target_properties(executorch_jni PROPERTIES
        IMPORTED_LOCATION "${ET_DIR}/jni/${ANDROID_ABI}/libexecutorch_jni.so"
    )
    if(EXISTS "${ET_DIR}/include")
        set(ET_INCLUDE_DIR "${ET_DIR}/include" CACHE PATH "" FORCE)
        target_include_directories(executorch_jni INTERFACE "${ET_INCLUDE_DIR}")
    endif()
    set(EXECUTORCH_ANDROID_FOUND TRUE CACHE BOOL "" FORCE)
    message(STATUS "ExecuTorch: found prebuilt for ${ANDROID_ABI}")
else()
    # ExecuTorch will be provided via Maven AAR dependency
    set(EXECUTORCH_ANDROID_FOUND FALSE CACHE BOOL "" FORCE)
    message(STATUS "ExecuTorch: NOT found at ${ET_DIR}")
    message(STATUS "  ExecuTorch will be provided via Maven AAR dependency")
    option(USE_EXECUTORCH "Enable LLM translation via ExecuTorch" OFF)
endif()

# ---------- Component libraries ----------
# These can now find the imported targets above
add_subdirectory(../../mt  mt)
add_subdirectory(../../asr asr)
add_subdirectory(../../tts tts)
add_subdirectory(../../pipeline pipeline)

# ---------- JNI shared library ----------
add_library(native-lib SHARED native-lib.cpp)

target_include_directories(native-lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..            # project root for "pipeline/pipeline.h"
    ${CMAKE_CURRENT_SOURCE_DIR}/../../pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/../../asr
    ${CMAKE_CURRENT_SOURCE_DIR}/../../mt
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tts
)

target_link_libraries(native-lib
    pipeline asr_wrapper mt_wrapper tts_wrapper
    log
)

# Android NDK log
find_library(log-lib log)
if(log-lib)
    target_link_libraries(native-lib ${log-lib})
endif()

# Link ORT .so to native-lib (also ensures it's packaged in APK)
if(ONNXRUNTIME_ANDROID_FOUND)
    target_link_libraries(native-lib onnxruntime)
endif()

# Link ExecuTorch .so to native-lib
if(EXECUTORCH_ANDROID_FOUND)
    target_link_libraries(native-lib executorch_jni)
endif()
