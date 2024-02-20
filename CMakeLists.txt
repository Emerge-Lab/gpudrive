cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)

include("${CMAKE_CURRENT_SOURCE_DIR}/external/madrona/cmake/madrona_init.cmake")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/madrona/external/googletest)

project(Madrona3DExample LANGUAGES C CXX)

enable_testing()  # Enable testing capabilities

include(setup)
include(dependencies)

add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(tests)  # Add your tests directory
