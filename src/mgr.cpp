#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>

#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::py;

namespace GPUHideSeek {

struct Manager::Impl {
    Config cfg;

    static inline Impl * init(const Config &cfg);
};

struct Manager::CPUImpl {
    EpisodeManager *episodeMgr;
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl : Manager::Impl {
    EpisodeManager *episodeMgr;
    MWCudaExecutor mwGPU;
};
#endif

Manager::Impl * Manager::Impl::init(const Config &cfg)
{
    auto sphere_obj = imp::ImportedObject::importObject(
        (std::filesystem::path(DATA_DIR) / "sphere.obj").c_str());

    if (!sphere_obj.has_value()) {
        FATAL("Failed to load sphere");
    }

    switch (cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        EpisodeManager *episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

        HeapArray<WorldInit> world_inits(cfg.numWorlds);

        for (int64_t i = 0; i < (int64_t)cfg.numWorlds; i++) {
            world_inits[i] = WorldInit {
                episode_mgr,
            };
        }

        MWCudaExecutor mwgpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(WorldInit),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = cfg.numWorlds,
            .numExportedBuffers = 2,
            .gpuID = (uint32_t)cfg.gpuID,
            .renderWidth = cfg.renderWidth,
            .renderHeight = cfg.renderHeight,
        }, {
            "",
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::Debug,
            CompileConfig::Executor::TaskGraph,
        });

        DynArray<imp::SourceObject> renderer_objects({
            imp::SourceObject { sphere_obj->meshes },
        });

        mwgpu_exec.loadObjects(renderer_objects);

        return new CUDAImpl {
            { cfg },
            episode_mgr,
            std::move(mwgpu_exec),
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        return nullptr;
    } break;
    default: __builtin_unreachable();
    }
}

MADRONA_EXPORT Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{}

MADRONA_EXPORT Manager::~Manager() {}

MADRONA_EXPORT void Manager::step()
{
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        static_cast<CUDAImpl>(impl_)->mwGPU.run();
#endif
    } break;
    case ExecMode::CPU: {
    } break;
    }
}

MADRONA_EXPORT Tensor Manager::resetTensor() const
{
    return exportStateTensor(0, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds, 1});
}

MADRONA_EXPORT Tensor Manager::moveActionTensor() const
{
    return exportStateTensor(1, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds, 1});
}

MADRONA_EXPORT Tensor Manager::depthTensor() const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr = static_cast<CUDAImpl>(impl_)->mwGPU.depthObservations();
        gpu_id = impl_->cfg.gpuID;
#endif
    } else {
        dev_ptr = nullptr;
    }

    return Tensor(dev_ptr, Tensor::ElementType::Float32,
                     {impl_->cfg.numWorlds, impl_->cfg.renderHeight,
                      impl_->cfg.renderWidth, 1}, gpu_id);
}

MADRONA_EXPORT Tensor Manager::rgbTensor() const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr = static_cast<CUDAImpl>(impl_)->mwGPU.rgbObservations();
        gpu_id = impl_->cfg.gpuID;
#endif
    } else {
        dev_ptr = nullptr;
    }

    return Tensor(dev_ptr, Tensor::ElementType::UInt8,
                  {impl_->cfg.numWorlds, impl_->cfg.renderHeight,
                   impl_->cfg.renderWidth, 4}, gpu_id);
}

Tensor Manager::exportStateTensor(int64_t slot,
                                  Tensor::ElementType type,
                                  Span<const int64_t> dimensions) const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();
    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr = static_cast<CUDAImpl>(impl_)->mwGPU.getExported(slot);
        gpu_id = impl_->cfg.gpuID;
#endif
    } else {
        dev_ptr = nullptr;
    }

    return Tensor(dev_ptr, type, dimensions, gpu_id);
}


}
