cmake_minimum_required(VERSION 3.16)  # Minimum required version of CMake
project(DNN-Adapt)                    # Project name

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the source directory to include paths
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/external/concurrentqueue)
include_directories(${PROJECT_SOURCE_DIR}/external/csvparser)

# Add the executable target
add_executable(run src/main.cpp src/Simulator.cpp src/RequestProcessor.cpp src/nexus.cpp)

# add target
target_link_libraries(run)
