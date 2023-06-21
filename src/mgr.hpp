#pragma once

#include <memory>

#include <madrona/python.hpp>
#include <madrona/exec_mode.hpp>
#include <madrona/render/mw.hpp>

namespace GPUHideSeek {

class Manager {
public:
    struct Config {
        madrona::ExecMode execMode;
        int gpuID;
        uint32_t numWorlds;
        uint32_t minEntitiesPerWorld;
        uint32_t maxEntitiesPerWorld;
        uint32_t renderWidth;
        uint32_t renderHeight;
        bool autoReset;
        bool enableBatchRender;
        bool debugCompile;
    };

    MADRONA_IMPORT Manager(const Config &cfg,
                           const madrona::render::RendererBridge *viewer_bridge = nullptr);
    MADRONA_IMPORT ~Manager();

    MADRONA_IMPORT void step();

    MADRONA_IMPORT madrona::py::Tensor resetTensor() const;
    MADRONA_IMPORT madrona::py::Tensor doneTensor() const;
    MADRONA_IMPORT madrona::py::Tensor actionTensor() const;
    MADRONA_IMPORT madrona::py::Tensor rewardTensor() const;
    MADRONA_IMPORT madrona::py::Tensor agentTypeTensor() const;
    MADRONA_IMPORT madrona::py::Tensor relativeAgentObservationsTensor() const;
    MADRONA_IMPORT madrona::py::Tensor relativeButtonObservationsTensor() const;
    MADRONA_IMPORT madrona::py::Tensor relativeDestinationObservationsTensor() const;
    MADRONA_IMPORT madrona::py::Tensor depthTensor() const;
    MADRONA_IMPORT madrona::py::Tensor rgbTensor() const;
    MADRONA_IMPORT madrona::py::Tensor lidarTensor() const;
    MADRONA_IMPORT madrona::py::Tensor seedTensor() const;

    MADRONA_IMPORT void triggerReset(int32_t world_idx);
    MADRONA_IMPORT void setAction(int32_t world_idx, int32_t agent_idx,
                                  int32_t x, int32_t y, int32_t r);

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    inline madrona::py::Tensor exportStateTensor(int64_t slot,
        madrona::py::Tensor::ElementType type,
        madrona::Span<const int64_t> dimensions) const;

    Impl *impl_;
};

}
