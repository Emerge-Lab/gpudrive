#!/bin/bash

# Define the build directory
BUILD_DIR="/home/aarav/gpudrive/build"

# Create the build directory if it does not exist
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

# Change to the build directory
cd "$BUILD_DIR"

# Remove everything inside the build directory
rm -rf *

# Run cmake with specific options
cmake ../ -DCMAKE_BUILD_TYPE=release -DENABLE_SANITIZER=OFF

# Run make with 32 jobs
make -j32

./headless CPU 15000 1