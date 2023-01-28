#pragma once

#include <memory>

#include <madrona/python.hpp>

namespace GPUHideSeek {

class Manager {
public:
    enum class ExecMode {
        CPU,
        CUDA,
    };

    struct Config {
        ExecMode execMode;
        int gpuID;
        uint32_t numWorlds;
        uint32_t minEntitiesPerWorld;
        uint32_t maxEntitiesPerWorld;
        uint32_t renderWidth;
        uint32_t renderHeight;
        bool enableRender;
        bool debugCompile;
    };

    MADRONA_IMPORT Manager(const Config &cfg);
    MADRONA_IMPORT ~Manager();

    MADRONA_IMPORT void step();

    MADRONA_IMPORT madrona::py::Tensor resetTensor() const;
    MADRONA_IMPORT madrona::py::Tensor doneTensor() const;
    MADRONA_IMPORT madrona::py::Tensor prepCounterTensor() const;
    MADRONA_IMPORT madrona::py::Tensor actionTensor() const;
    MADRONA_IMPORT madrona::py::Tensor rewardTensor() const;
    MADRONA_IMPORT madrona::py::Tensor agentTypeTensor() const;
    MADRONA_IMPORT madrona::py::Tensor agentMaskTensor() const;
    MADRONA_IMPORT madrona::py::Tensor agentDataTensor() const;
    MADRONA_IMPORT madrona::py::Tensor boxDataTensor() const;
    MADRONA_IMPORT madrona::py::Tensor rampDataTensor() const;
    MADRONA_IMPORT madrona::py::Tensor visibleAgentsMaskTensor() const;
    MADRONA_IMPORT madrona::py::Tensor visibleBoxesMaskTensor() const;
    MADRONA_IMPORT madrona::py::Tensor visibleRampsMaskTensor() const;
    MADRONA_IMPORT madrona::py::Tensor globalPositionsTensor() const;
    MADRONA_IMPORT madrona::py::Tensor depthTensor() const;
    MADRONA_IMPORT madrona::py::Tensor rgbTensor() const;
    MADRONA_IMPORT madrona::py::Tensor lidarTensor() const;

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
