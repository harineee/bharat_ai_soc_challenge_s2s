cmake_minimum_required(VERSION 3.18)
project(ASRModule)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable NEON for Arm (Apple Silicon has NEON by default, no flag needed)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    if(NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+simd")
    endif()
endif()

# Optimization flags (removed -ffast-math for whisper.cpp compatibility)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops")

# whisper.cpp submodule path (adjust if different)
set(WHISPER_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/whisper_cpp)

# Check if whisper.cpp exists
if(EXISTS ${WHISPER_CPP_DIR}/CMakeLists.txt)
    add_subdirectory(${WHISPER_CPP_DIR} whisper_cpp)
    
    # ASR wrapper library
    add_library(asr_wrapper STATIC
        asr_wrapper.cpp
    )
    
    target_include_directories(asr_wrapper PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${WHISPER_CPP_DIR}
    )
    
    target_link_libraries(asr_wrapper
        whisper
    )
else()
    message(WARNING "whisper.cpp not found. Please clone submodule or set WHISPER_CPP_DIR")
    
    # Create stub library
    add_library(asr_wrapper STATIC asr_wrapper.cpp)
    target_include_directories(asr_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Install rules
install(TARGETS asr_wrapper
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES asr_wrapper.h
    DESTINATION include
)
