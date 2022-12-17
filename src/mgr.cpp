#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_assets.hpp>

#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace GPUHideSeek {

struct Manager::Impl {
    Config cfg;
    PhysicsLoader physicsLoader;
    EpisodeManager *episodeMgr;
    TrainingExecutor mwGPU;

    static inline Impl * init(const Config &cfg);
};

Manager::Impl * Manager::Impl::init(const Config &cfg)
{
    auto sphere_obj = imp::ImportedObject::importObject(
        (std::filesystem::path(DATA_DIR) / "sphere.obj").c_str());

    if (!sphere_obj.has_value()) {
        FATAL("Failed to load sphere");
    }

    PhysicsLoader phys_loader(PhysicsLoader::StorageType::CUDA, 10);

    RigidBodyMetadata sphere_phys_metadata {
        .invInertiaTensor = { 1.f, 1.f, 1.f },
    };

    AABB sphere_aabb {
        .pMin = { -1, -1, -1 },
        .pMax = { 1, 1, 1 },
    };

    phys_loader.loadObjects(&sphere_phys_metadata, &sphere_aabb, 1);

    ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();

    EpisodeManager *episode_mgr = 
        (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
    REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

    HeapArray<WorldInit> world_inits(cfg.numWorlds);

    for (int64_t i = 0; i < (int64_t)cfg.numWorlds; i++) {
        world_inits[i] = WorldInit {
            episode_mgr,
            phys_obj_mgr,
        };
    }

    TrainingExecutor mwgpu_exec({
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

    return new Impl {
        cfg,
        std::move(phys_loader),
        episode_mgr,
        std::move(mwgpu_exec),
    };
}

MADRONA_EXPORT Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{}

MADRONA_EXPORT Manager::~Manager() {}

MADRONA_EXPORT void Manager::step()
{
    impl_->mwGPU.run();
}

MADRONA_EXPORT GPUTensor Manager::resetTensor() const
{
    void *dev_ptr = impl_->mwGPU.getExported(0);

    return GPUTensor(dev_ptr, GPUTensor::ElementType::Int32,
                     {impl_->cfg.numWorlds, 1}, impl_->cfg.gpuID);
}

MADRONA_EXPORT GPUTensor Manager::moveActionTensor() const
{
    void *dev_ptr = impl_->mwGPU.getExported(1);

    return GPUTensor(dev_ptr, GPUTensor::ElementType::Int32,
                     {impl_->cfg.numWorlds, 1}, impl_->cfg.gpuID);
}

MADRONA_EXPORT GPUTensor Manager::depthTensor() const
{
    void *dev_ptr = impl_->mwGPU.depthObservations();

    return GPUTensor(dev_ptr, GPUTensor::ElementType::Float32,
                     {impl_->cfg.numWorlds, impl_->cfg.renderHeight,
                      impl_->cfg.renderWidth, 1}, impl_->cfg.gpuID);
}

MADRONA_EXPORT GPUTensor Manager::rgbTensor() const
{
    void *dev_ptr = impl_->mwGPU.rgbObservations();

    return GPUTensor(dev_ptr, GPUTensor::ElementType::UInt8,
                     {impl_->cfg.numWorlds, impl_->cfg.renderHeight,
                      impl_->cfg.renderWidth, 4}, impl_->cfg.gpuID);
}

}
