cmake_minimum_required(VERSION 3.16)  # Minimum required version of CMake
project(DNN-Adapt)                    # Project name

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the source directory to include paths
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/external/concurrentqueue)
include_directories(${PROJECT_SOURCE_DIR}/external/csvparser)

# add onnxruntime library
set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/external/onnxruntime-linux-x64-gpu-1.21.1")

include_directories(${PROJECT_SOURCE_DIR}/external/onnxruntime-linux-x64-gpu-1.21.1/include)
link_directories(${PROJECT_SOURCE_DIR}/external/onnxruntime-linux-x64-gpu-1.21.1/lib)

# add cuda libraries
find_package(CUDAToolkit REQUIRED)

# Add Spdlog library
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found, download")
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.12.0  
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# Add the executable target
add_executable(run src/main.cpp src/Simulator.cpp src/RequestProcessor.cpp src/nexus.cpp src/Logger.cpp)

# add target
target_link_libraries(run 
    PRIVATE 
    onnxruntime
    CUDA::cudart
    spdlog::spdlog
)
