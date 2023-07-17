#pragma once
#ifdef madrona_3d_example_mgr_EXPORTS
#define MGR_EXPORT MADRONA_EXPORT
#else
#define MGR_EXPORT MADRONA_IMPORT
#endif

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/render/mw.hpp>
#include <madrona/viz/system.hpp>

namespace GPUHideSeek {

class Manager {
public:
    struct Config {
        madrona::ExecMode execMode;
        int gpuID;
        uint32_t numWorlds;
        uint32_t renderWidth;
        uint32_t renderHeight;
        bool autoReset;
        bool enableBatchRender;
    };

    MGR_EXPORT Manager(
        const Config &cfg,
        const madrona::viz::VizECSBridge *viz_bridge = nullptr);
    MGR_EXPORT ~Manager();

    MGR_EXPORT void step();

    MGR_EXPORT madrona::py::Tensor resetTensor() const;
    MGR_EXPORT madrona::py::Tensor actionTensor() const;
    MGR_EXPORT madrona::py::Tensor rewardTensor() const;
    MGR_EXPORT madrona::py::Tensor doneTensor() const;
    MGR_EXPORT madrona::py::Tensor toOtherAgentsTensor() const;
    MGR_EXPORT madrona::py::Tensor toButtonsTensor() const;
    MGR_EXPORT madrona::py::Tensor toGoalTensor() const;
    MGR_EXPORT madrona::py::Tensor lidarTensor() const;
    MGR_EXPORT madrona::py::Tensor seedTensor() const;
    MGR_EXPORT madrona::py::Tensor depthTensor() const;
    MGR_EXPORT madrona::py::Tensor rgbTensor() const;

    MGR_EXPORT void triggerReset(int32_t world_idx);
    MGR_EXPORT void setAction(int32_t world_idx, int32_t agent_idx,
                              int32_t x, int32_t y, int32_t r);

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;
};

}
