#!/bin/bash

# Path to your CMakeLists.txt file
CMAKE_FILE="/home/aarav/gpudrive/CMakeLists.txt"

# Content to add
CONTENT="option(ENABLE_SANITIZER \"Enable Address Sanitizer\" OFF)\n\nif(ENABLE_SANITIZER)\n    set(CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS} -fsanitize=address -g -O1\")\n    set(CMAKE_LINKER_FLAGS \"\${CMAKE_LINKER_FLAGS} -fsanitize=address\")\nendif()\n"

# Insert the content at line 4
sed -i "4i $CONTENT" "$CMAKE_FILE"
