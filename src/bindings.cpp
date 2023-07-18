#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace nb = nanobind;

namespace GPUHideSeek {

NB_MODULE(madrona_3d_example, m) {
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<Manager> (m, "SimManager")
        .def("__init__", [](Manager *self,
                            madrona::py::PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            bool auto_reset,
                            int64_t render_width,
                            int64_t render_height, 
                            bool enable_batch_render) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .renderWidth = (uint32_t)render_width,
                .renderHeight = (uint32_t)render_height,
                .autoReset = auto_reset,
                .enableBatchRender = enable_batch_render,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("auto_reset"),
           nb::arg("render_width") = 0,
           nb::arg("render_height") = 0,
           nb::arg("enable_batch_render") = false)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("position_observation_tensor", &Manager::positionObservationTensor)
        .def("to_other_agents_tensor", &Manager::toOtherAgentsTensor)
        .def("to_buttons_tensor", &Manager::toButtonsTensor)
        .def("to_goal_tensor", &Manager::toGoalTensor)
        .def("lidar_tensor", &Manager::lidarTensor)
        .def("seed_tensor", &Manager::seedTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
    ;
}

}
