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
        uint32_t renderWidth;
        uint32_t renderHeight;
    };

    MADRONA_IMPORT Manager(const Config &cfg);
    MADRONA_IMPORT ~Manager();

    MADRONA_IMPORT void step();

    MADRONA_IMPORT madrona::py::Tensor resetTensor() const;
    MADRONA_IMPORT madrona::py::Tensor moveActionTensor() const;
    MADRONA_IMPORT madrona::py::Tensor depthTensor() const;
    MADRONA_IMPORT madrona::py::Tensor rgbTensor() const;

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    inline madrona::py::Tensor exportStateTensor(int64_t slot,
        madrona::py::Tensor::ElementType type,
        madrona::Span<const int64_t> dimensions) const;

    std::unique_ptr<Impl> impl_;
};

}
