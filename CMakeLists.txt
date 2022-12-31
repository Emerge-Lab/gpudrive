cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/external/madrona/cmake")

include(build_type)

project(MadronaGPUHideAndSeek LANGUAGES C CXX)

include(setup)
include(dependencies)

option(TRACING "Enable tracing" OFF)
if (TRACING)
  add_compile_definitions(MADRONA_TRACING=1)
endif()

add_subdirectory(external)
add_subdirectory(src)
