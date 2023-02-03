cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)
include("${CMAKE_CURRENT_SOURCE_DIR}/external/madrona/cmake/cmake_init.cmake")

project(MadronaGPUHideAndSeek LANGUAGES C CXX)

include(setup)
include(dependencies)

add_subdirectory(external)
add_subdirectory(src)
