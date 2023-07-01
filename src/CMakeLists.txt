set(SIMULATOR_SRCS
    sim.hpp sim.cpp
    init.hpp rng.hpp
    geo_gen.hpp geo_gen.inl geo_gen.cpp
    level_gen.hpp level_gen.cpp
)

add_library(madrona_3d_example_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(madrona_3d_example_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_viz_system
)

add_library(madrona_3d_example_mgr SHARED
    mgr.hpp mgr.cpp
)

target_link_libraries(madrona_3d_example_mgr PRIVATE
    madrona_3d_example_cpu_impl
    madrona_mw_cpu
    madrona_python_utils
    madrona_common
    madrona_importer
    madrona_physics_assets
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            madrona_3d_example_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(madrona_3d_example_mgr PRIVATE
        madrona_3d_example_gpu_srcs
        madrona_mw_gpu
    )
endif ()

target_compile_definitions(madrona_3d_example_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(madrona_3d_example_python
    bindings.cpp
)

target_link_libraries(madrona_3d_example_python PRIVATE
    madrona_3d_example_mgr
)

if (TARGET madrona_viewer)
    add_executable(viewer viewer.cpp)
    target_link_libraries(viewer madrona_viewer
        madrona_mw_core madrona_3d_example_mgr)

    target_compile_definitions(viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )
endif()
