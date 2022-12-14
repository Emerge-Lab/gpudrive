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
    nb::class_<Manager> (m, "HideAndSeekSimulator")
        .def("__init__", [](Manager *self, int64_t gpu_id,
                            int64_t num_worlds, int64_t render_width,
                            int64_t render_height) {
            new (self) Manager(Manager::Config {
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .renderWidth = (uint32_t)render_width,
                .renderHeight = (uint32_t)render_height,
            });
        }, nb::arg("gpu_id"), nb::arg("num_worlds"), nb::arg("render_width"),
           nb::arg("render_height"))
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("move_action_tensor", &Manager::moveActionTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
    ;
}

}
