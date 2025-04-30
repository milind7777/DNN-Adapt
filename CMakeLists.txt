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

# Add the executable target
add_executable(run src/main.cpp src/Simulator.cpp src/RequestProcessor.cpp src/nexus.cpp)

# add target
target_link_libraries(run 
    PRIVATE 
    onnxruntime
    CUDA::cudart
)
