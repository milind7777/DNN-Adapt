cmake_minimum_required(VERSION 3.16)  # Minimum required version of CMake
project(DNN-Adapt)                    # Project name

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the source directory to include paths
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/com)
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

# GRPC source configuration?
# Find Protobuf and gRPC
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG QUIET)

if(NOT gRPC_FOUND)
    message(STATUS "gRPC package not found. Using manual configuration.")
    # Add gRPC libraries manually if the package wasn't found
    set(GRPC_LIBRARIES grpc++ grpc++_reflection)
else()
    message(STATUS "gRPC package found.")
    # Use the imported targets
    set(GRPC_LIBRARIES gRPC::grpc++ gRPC::grpc++_reflection)
endif()

# Define the proto file
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/com/agent_scheduler.proto)
set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/com)


# Create custom command to generate C++ code from proto file
add_custom_command(
    OUTPUT 
        ${PROTO_SRC_DIR}/agent_scheduler.pb.cc
        ${PROTO_SRC_DIR}/agent_scheduler.pb.h
        ${PROTO_SRC_DIR}/agent_scheduler.grpc.pb.cc
        ${PROTO_SRC_DIR}/agent_scheduler.grpc.pb.h
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
    ARGS 
        --grpc_out=${PROTO_SRC_DIR}
        --cpp_out=${PROTO_SRC_DIR}
        -I=${PROTO_SRC_DIR}
        --plugin=protoc-gen-grpc=`which grpc_cpp_plugin`
        ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating gRPC and Protobuf files from ${PROTO_FILE}"
)

# Create a target that depends on generated files
add_custom_target(proto_gen DEPENDS 
    ${PROTO_SRC_DIR}/agent_scheduler.pb.cc
    ${PROTO_SRC_DIR}/agent_scheduler.pb.h
    ${PROTO_SRC_DIR}/agent_scheduler.grpc.pb.cc
    ${PROTO_SRC_DIR}/agent_scheduler.grpc.pb.h
)

# Set generated files
set(PROTO_GENERATED_FILES
    ${PROTO_SRC_DIR}/agent_scheduler.pb.cc
    ${PROTO_SRC_DIR}/agent_scheduler.pb.h
    ${PROTO_SRC_DIR}/agent_scheduler.grpc.pb.cc
    ${PROTO_SRC_DIR}/agent_scheduler.grpc.pb.h
)

# src/agent directory exists
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/agent)

# Add custom command to generate Python gRPC files
add_custom_command(
    OUTPUT 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/agent/agent_scheduler_pb2.py
        ${CMAKE_CURRENT_SOURCE_DIR}/src/agent/agent_scheduler_pb2_grpc.py
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/agent
    COMMAND python
    ARGS 
        -m grpc_tools.protoc
        -I${PROTO_SRC_DIR}
        --python_out=${CMAKE_CURRENT_SOURCE_DIR}/src/agent
        --grpc_python_out=${CMAKE_CURRENT_SOURCE_DIR}/src/agent
        ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating Python gRPC files from ${PROTO_FILE} into src/agent"
)

# Create a custom target for Python gRPC generation
add_custom_target(python_proto_gen 
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/agent/agent_scheduler_pb2.py
        ${CMAKE_CURRENT_SOURCE_DIR}/src/agent/agent_scheduler_pb2_grpc.py
)


# Add the executable target
add_executable(run 
    src/main.cpp 
    src/Simulator.cpp  
    src/RequestProcessor.cpp 
    src/nexus.cpp 
    src/Logger.cpp
    ${PROTO_GENERATED_FILES}

    )


# Add dependency to ensure proto files are generated before building
add_dependencies(run proto_gen)
add_dependencies(run python_proto_gen)

# add target
target_link_libraries(run 
    PRIVATE 
    onnxruntime
    CUDA::cudart
    spdlog::spdlog
    ${Protobuf_LIBRARIES}
    ${GRPC_LIBRARIES}
)

