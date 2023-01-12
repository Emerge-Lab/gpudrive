#include "mgr.hpp"

#include <madrona/macros.hpp>

#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>
#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic pop
#endif

namespace nb = nanobind;

namespace GPUHideSeek {

NB_MODULE(gpu_hideseek_python, m) {
    nb::enum_<Manager::ExecMode>(m, "ExecMode")
        .value("CPU", Manager::ExecMode::CPU)
        .value("CUDA", Manager::ExecMode::CUDA)
        .export_values();

    nb::class_<Manager> (m, "HideAndSeekSimulator")
        .def("__init__", [](Manager *self,
                            Manager::ExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t min_entities_per_world,
                            int64_t max_entities_per_world,
                            int64_t render_width,
                            int64_t render_height, 
                            bool lidar_render,
                            bool debug_compile) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .minEntitiesPerWorld = (uint32_t)min_entities_per_world,
                .maxEntitiesPerWorld = (uint32_t)max_entities_per_world,
                .renderWidth = (uint32_t)render_width,
                .renderHeight = (uint32_t)render_height,
                .lidarRender = lidar_render,
                .debugCompile = debug_compile,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("min_entities_per_world"),
           nb::arg("max_entities_per_world"),
           nb::arg("render_width"),
           nb::arg("render_height"),
           nb::arg("lidar_render") = true,
           nb::arg("debug_compile") = true)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("agent_mask_tensor", &Manager::agentMaskTensor)
        .def("visibility_masks_tensor", &Manager::visibilityMasksTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
    ;
}

}
