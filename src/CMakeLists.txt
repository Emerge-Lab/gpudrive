set(SIMULATOR_SRCS
    types.hpp
    sim.hpp sim.cpp
    init.hpp rng.hpp
    level_gen.hpp level_gen.cpp
    obb.hpp
    utils.hpp
    rasterizer.hpp
    binary_heap.hpp
    knn.hpp
    dynamics.hpp
)

add_library(gpudrive_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(gpudrive_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
)

add_library(gpudrive_mgr STATIC
    mgr.hpp mgr.cpp
    MapReader.hpp MapReader.cpp
)

target_link_libraries(gpudrive_mgr
    PUBLIC
        madrona_python_utils
    PRIVATE
        madrona_json
        gpudrive_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_render
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            gpudrive_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS
        SRCS
            ${SIMULATOR_SRCS}
    )
    target_link_libraries(gpudrive_mgr PRIVATE
        gpudrive_gpu_srcs
        madrona_mw_gpu
    )
endif ()

target_compile_definitions(gpudrive_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../assets/"
)

madrona_python_module(madrona_gpudrive
    bindings.cpp
)

target_link_libraries(madrona_gpudrive PRIVATE
    gpudrive_mgr
)

if (TARGET madrona_viz)
    add_executable(viewer viewer.cpp)
    target_link_libraries(viewer PRIVATE
        madrona_mw_core gpudrive_mgr madrona_viz)

    target_compile_definitions(viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../assets/"
    )

    if (TARGET madrona_cuda)
        target_link_libraries(viewer PRIVATE
            madrona_cuda
        )
    endif()
endif()

add_executable(headless headless.cpp)
target_link_libraries(headless madrona_mw_core gpudrive_mgr)
