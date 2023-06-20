#include "mgr.hpp"

#include <madrona/macros.hpp>

#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic pop
#endif

namespace nb = nanobind;

namespace GPUHideSeek {

NB_MODULE(gpu_hideseek_python, m) {
    nb::module_::import_("madrona_python");

    nb::class_<Manager> (m, "HideAndSeekSimulator")
        .def("__init__", [](Manager *self,
                            madrona::ExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t min_entities_per_world,
                            int64_t max_entities_per_world,
                            int64_t render_width,
                            int64_t render_height, 
                            bool auto_reset,
                            bool enable_batch_render,
                            bool debug_compile) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .minEntitiesPerWorld = (uint32_t)min_entities_per_world,
                .maxEntitiesPerWorld = (uint32_t)max_entities_per_world,
                .renderWidth = (uint32_t)render_width,
                .renderHeight = (uint32_t)render_height,
                .autoReset = auto_reset,
                .enableBatchRender = enable_batch_render,
                .debugCompile = debug_compile,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("min_entities_per_world"),
           nb::arg("max_entities_per_world"),
           nb::arg("render_width"),
           nb::arg("render_height"),
           nb::arg("auto_reset") = false,
           nb::arg("enable_batch_render") = false,
           nb::arg("debug_compile") = false)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("agent_type_tensor", &Manager::agentTypeTensor)
        .def("relative_agent_observations_tensor", &Manager::relativeAgentObservationsTensor)
        .def("relative_button_observations_tensor", &Manager::relativeButtonObservationsTensor)
        .def("relative_destination_observations_tensor", &Manager::relativeDestinationObservationsTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("lidar_tensor", &Manager::lidarTensor)
        .def("seed_tensor", &Manager::seedTensor)
    ;
}

}
