rm -rf grpc_stubs
mkdir -p grpc_stubs

python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. scheduler.proto
