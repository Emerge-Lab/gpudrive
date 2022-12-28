set(SIMULATOR_SRCS
    sim.hpp sim.cpp
)

add_library(gpu_hideseek_mgr SHARED
    mgr.hpp mgr.cpp
)

target_link_libraries(gpu_hideseek_mgr PRIVATE
    madrona_python_utils
    madrona_common
    madrona_importer
    madrona_physics_assets
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            gpu_hideseek_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(gpu_hideseek_mgr PRIVATE
        gpu_hideseek_gpu_srcs
        madrona_mw_gpu
    )
else ()
endif ()


target_compile_definitions(gpu_hideseek_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(gpu_hideseek_python
    bindings.cpp
)

target_link_libraries(gpu_hideseek_python PRIVATE
    madrona_hdrs
    gpu_hideseek_mgr
)
