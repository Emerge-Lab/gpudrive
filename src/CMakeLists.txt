set(SIMULATOR_SRCS
    types.hpp
    sim.hpp sim.cpp
    init.hpp rng.hpp
    level_gen.hpp level_gen.cpp
    obb.hpp
    utils.hpp
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
        madrona_viz_system
)

add_library(gpudrive_mgr SHARED
    mgr.hpp mgr.cpp
    MapReader.hpp MapReader.cpp
)

target_link_libraries(gpudrive_mgr 
    PUBLIC
        madrona_python_utils
    PRIVATE
        nlohmann_json::nlohmann_json
        gpudrive_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
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
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(gpudrive
    bindings.cpp
)

target_link_libraries(gpudrive PRIVATE
    gpudrive_mgr
)

if (TARGET madrona_viewer)
    add_executable(viewer viewer.cpp)
    target_link_libraries(viewer madrona_viewer
        madrona_mw_core gpudrive_mgr)

    target_compile_definitions(viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )
endif()

add_executable(headless headless.cpp)
target_link_libraries(headless gpudrive_mgr)
