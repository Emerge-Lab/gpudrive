if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-w)
endif()

set(MADRONA_ENABLE_TESTS ON)
add_subdirectory("${MADRONA_DIR}" madrona EXCLUDE_FROM_ALL)
add_subdirectory(json)


