# ensure build directory exists
mkdir -p build

# clean and enter build directory
rm -rf build/* && cd build

# configure build
cmake ..

# build
cmake --build .

# save run to main directory
cd ../ && cp build/run run
