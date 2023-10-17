set(SIMULATOR_SRCS
    types.hpp
    sim.hpp sim.cpp
    init.hpp rng.hpp
    level_gen.hpp level_gen.cpp
)

add_library(mad_escape_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(mad_escape_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_viz_system
)

add_library(mad_escape_mgr SHARED
    mgr.hpp mgr.cpp
)

target_link_libraries(mad_escape_mgr 
    PUBLIC
        madrona_python_utils
    PRIVATE
        mad_escape_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            mad_escape_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(mad_escape_mgr PRIVATE
        mad_escape_gpu_srcs
        madrona_mw_gpu
    )
endif ()

target_compile_definitions(mad_escape_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(madrona_escape_room
    bindings.cpp
)

target_link_libraries(madrona_escape_room PRIVATE
    mad_escape_mgr
)

if (TARGET madrona_viewer)
    add_executable(viewer viewer.cpp)
    target_link_libraries(viewer madrona_viewer
        madrona_mw_core mad_escape_mgr)

    target_compile_definitions(viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )
endif()

add_executable(headless headless.cpp)
target_link_libraries(headless madrona_mw_core mad_escape_mgr)
