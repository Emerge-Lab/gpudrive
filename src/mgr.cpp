#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_assets.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>

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
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace GPUHideSeek {

struct Manager::Impl {
    Config cfg;
    PhysicsLoader physicsLoader;
    EpisodeManager *episodeMgr;

    static inline Impl * init(const Config &cfg);
};

struct Manager::CPUImpl : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, GPUHideSeek::Config, WorldInit>;

    TaskGraphT cpuExec;
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl : Manager::Impl {
    MWCudaExecutor mwGPU;
};
#endif

static void loadPhysicsObjects(PhysicsLoader &loader)
{
    DynArray<RigidBodyMetadata> metadatas(0);
    DynArray<AABB> aabbs(0);
    DynArray<CollisionPrimitive> prims(0);

    { // Sphere: (0)
        metadatas.push_back({
            .invInertiaTensor = { 2.5f, 2.5f, 2.5f },
            .invMass = 1.f,
            .muS = 0.5f,
            .muD = 0.5f,
        });

        aabbs.push_back({
            .pMin = { -1, -1, -1 },
            .pMax = { 1, 1, 1 },
        });

        prims.push_back({
            .type = CollisionPrimitive::Type::Sphere,
            .sphere = {
                .radius = 1.f,
            },
        });
    }

    { // Plane: (1)
        metadatas.push_back({
            .invInertiaTensor = { 0.f, 0.f, 0.f },
            .invMass = 0.f,
            .muS = 0.5f,
            .muD = 0.5f,
        });

        aabbs.push_back({
            .pMin = { -FLT_MAX, -FLT_MAX, -FLT_MAX },
            .pMax = { FLT_MAX, FLT_MAX, 0 },
        });

        prims.push_back({
            .type = CollisionPrimitive::Type::Plane,
            .plane = {},
        });
    }

    { // Cube: (2)
        metadatas.push_back({
            .invInertiaTensor = { 1.5f, 1.5f, 1.5f },
            .invMass = 1.f,
            .muS = 0.5f,
            .muD = 0.5f,
        });

        auto cube_path =
            std::filesystem::path(DATA_DIR) / "cube_collision.obj";

        HeapArray<PhysicsLoader::LoadedHull> cube_hulls =
            loader.importConvexDecompFromDisk(cube_path.c_str());

        aabbs.push_back(cube_hulls[0].aabb);

        prims.push_back({
            .type = CollisionPrimitive::Type::Hull,
            .hull = {
                .halfEdgeMesh = cube_hulls[0].collisionMesh,
            },
        });
    }

    { // Wall: (3)
        metadatas.push_back({
            .invInertiaTensor = { 0.f, 0.f, 0.f },
            .invMass = 0.f,
            .muS = 0.5f,
            .muD = 0.5f,
        });

        auto wall_path =
            std::filesystem::path(DATA_DIR) / "wall_collision.obj";

        HeapArray<PhysicsLoader::LoadedHull> wall_hulls =
            loader.importConvexDecompFromDisk(wall_path.c_str());

        aabbs.push_back(wall_hulls[0].aabb);

        prims.push_back({
            .type = CollisionPrimitive::Type::Hull,
            .hull = {
                .halfEdgeMesh = wall_hulls[0].collisionMesh,
            },
        });
    }

    { // Cylinder: (4)
        metadatas.push_back({
            .invInertiaTensor = { 0.f, 0.f, 1.f }, // FIXME
            .invMass = 1.f,
            .muS = 0.5f,
            .muD = 0.5f,
        });

        auto cylinder_path =
            std::filesystem::path(DATA_DIR) / "cylinder_collision.obj";

        HeapArray<PhysicsLoader::LoadedHull> cylinder_hulls =
            loader.importConvexDecompFromDisk(cylinder_path.c_str());

        aabbs.push_back(cylinder_hulls[0].aabb);

        prims.push_back({
            .type = CollisionPrimitive::Type::Hull,
            .hull = {
                .halfEdgeMesh = cylinder_hulls[0].collisionMesh,
            },
        });
    }

    { // Ramp (5)
        metadatas.push_back({
            .invInertiaTensor = { 1.5f, 1.5f, 1.5f }, // FIXME
            .invMass = 1.f,
            .muS = 0.5f,
            .muD = 0.5f,
        });

        auto ramp_path =
            std::filesystem::path(DATA_DIR) / "ramp_collision.obj";

        HeapArray<PhysicsLoader::LoadedHull> ramp_hulls =
            loader.importConvexDecompFromDisk(ramp_path.c_str());

        aabbs.push_back(ramp_hulls[0].aabb);

        prims.push_back({
            .type = CollisionPrimitive::Type::Hull,
            .hull = {
                .halfEdgeMesh = ramp_hulls[0].collisionMesh,
            },
        });
    }

    { // Elongated box (6)
        float width = 8;
        float height = 2;
        float depth = 1.5;

        Vector3 inv_inertia = 12.f / Vector3 {
            height * height + depth * depth,
            height * height  + width * width,
            width * width + depth * depth,
        };

        metadatas.push_back({
            .invInertiaTensor = inv_inertia,
            .invMass = 1.f,
            .muS = 0.5f,
            .muD = 0.5f,
        });

        auto elongated_path =
            std::filesystem::path(DATA_DIR) / "elongated_collision.obj";

        HeapArray<PhysicsLoader::LoadedHull> elongated_box_hulls =
            loader.importConvexDecompFromDisk(elongated_path.c_str());

        aabbs.push_back(elongated_box_hulls[0].aabb);

        prims.push_back({
            .type = CollisionPrimitive::Type::Hull,
            .hull = {
                .halfEdgeMesh = elongated_box_hulls[0].collisionMesh,
            },
        });
    }

    loader.loadObjects(metadatas.data(), aabbs.data(),
                       prims.data(), metadatas.size());
}

Manager::Impl * Manager::Impl::init(const Config &cfg)
{
    HostEventLogging(HostEvent::initStart);

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk({
        (std::filesystem::path(DATA_DIR) / "sphere.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "plane.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "cylinder_render.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "ramp_render.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "elongated_render.obj").c_str(),
    }, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    GPUHideSeek::Config app_cfg {
        cfg.enableRender,
        cfg.autoReset,
    };

    switch (cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        EpisodeManager *episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

        PhysicsLoader phys_loader(PhysicsLoader::StorageType::CUDA, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();

        HeapArray<WorldInit> world_inits(cfg.numWorlds);

        for (int64_t i = 0; i < (int64_t)cfg.numWorlds; i++) {
            world_inits[i] = WorldInit {
                episode_mgr,
                phys_obj_mgr,
                cfg.minEntitiesPerWorld,
                cfg.maxEntitiesPerWorld,
            };
        }

        MWCudaExecutor mwgpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(WorldInit),
            .userConfigPtr = &app_cfg,
            .numUserConfigBytes = sizeof(GPUHideSeek::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = cfg.numWorlds,
            .maxViewsPerWorld = consts::maxAgents,
            .numExportedBuffers = 16,
            .gpuID = (uint32_t)cfg.gpuID,
            .cameraMode = cfg.enableRender ? 
                render::CameraMode::Perspective :
                render::CameraMode::None,
            .renderWidth = cfg.renderWidth,
            .renderHeight = cfg.renderHeight,
        }, {
            "",
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            cfg.debugCompile ? CompileConfig::OptMode::Debug :
                CompileConfig::OptMode::LTO,
            CompileConfig::Executor::TaskGraph,
        });

        if (cfg.enableRender) {
            mwgpu_exec.loadObjects(render_assets->objects);
        }

        HostEventLogging(HostEvent::initEnd);
        return new CUDAImpl {
            { 
                cfg,
                std::move(phys_loader),
                episode_mgr,
            },
            std::move(mwgpu_exec),
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        EpisodeManager *episode_mgr = new EpisodeManager { 0 };

        PhysicsLoader phys_loader(PhysicsLoader::StorageType::CPU, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();

        HeapArray<WorldInit> world_inits(cfg.numWorlds);

        for (int64_t i = 0; i < (int64_t)cfg.numWorlds; i++) {
            world_inits[i] = WorldInit {
                episode_mgr,
                phys_obj_mgr,
                cfg.minEntitiesPerWorld,
                cfg.maxEntitiesPerWorld,
            };
        }

        auto cpu_impl = new CPUImpl {
            { 
                cfg,
                std::move(phys_loader),
                episode_mgr,
            },
            CPUImpl::TaskGraphT {
                ThreadPoolExecutor::Config {
                    .numWorlds = cfg.numWorlds,
                    .maxViewsPerWorld = consts::maxAgents,
                    .maxInstancesPerWorld = 1024,
                    .renderWidth = cfg.renderWidth,
                    .renderHeight = cfg.renderHeight,
                    .maxObjects = 50,
                    .numExportedBuffers = 16,
                    .cameraMode = cfg.enableRender ?
                        render::CameraMode::Perspective :
                        render::CameraMode::None,
                    .renderGPUID = cfg.gpuID,
                },
                app_cfg,
                world_inits.data()
            },
        };

        if (cfg.enableRender) {
            cpu_impl->cpuExec.loadObjects(render_assets->objects);
        }

        HostEventLogging(HostEvent::initEnd);

        return cpu_impl;
    } break;
    default: __builtin_unreachable();
    }
}

MADRONA_EXPORT Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{}

MADRONA_EXPORT Manager::~Manager() {
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        delete static_cast<CUDAImpl *>(impl_);
#endif
    } break;
    case ExecMode::CPU : {
        delete static_cast<CPUImpl *>(impl_);
    } break;
    }
#ifdef MADRONA_TRACING
    FinalizeLogging("/tmp/");
#endif
}

MADRONA_EXPORT void Manager::step()
{
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        static_cast<CUDAImpl *>(impl_)->mwGPU.run();
#endif
    } break;
    case ExecMode::CPU: {
        static_cast<CPUImpl *>(impl_)->cpuExec.run();
    } break;
    }
}


MADRONA_EXPORT Tensor Manager::resetTensor() const
{
    return exportStateTensor(0, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds, 3});
}

MADRONA_EXPORT Tensor Manager::doneTensor() const
{
    return exportStateTensor(1, Tensor::ElementType::UInt8,
                             {impl_->cfg.numWorlds * consts::maxAgents, 1});
}

MADRONA_EXPORT madrona::py::Tensor Manager::prepCounterTensor() const
{
    return exportStateTensor(2, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds * consts::maxAgents, 1});
}

MADRONA_EXPORT Tensor Manager::actionTensor() const
{
    return exportStateTensor(3, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds * consts::maxAgents, 5});
}

MADRONA_EXPORT Tensor Manager::rewardTensor() const
{
    return exportStateTensor(4, Tensor::ElementType::Float32,
                             {impl_->cfg.numWorlds * consts::maxAgents, 1});
}

MADRONA_EXPORT Tensor Manager::agentTypeTensor() const
{
    return exportStateTensor(5, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds * consts::maxAgents, 1});
}

MADRONA_EXPORT Tensor Manager::agentMaskTensor() const
{
    return exportStateTensor(6, Tensor::ElementType::Float32,
                             {impl_->cfg.numWorlds * consts::maxAgents, 1});
}


MADRONA_EXPORT madrona::py::Tensor Manager::agentDataTensor() const
{
    return exportStateTensor(7, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 consts::maxAgents - 1,
                                 4,
                             });
}

MADRONA_EXPORT madrona::py::Tensor Manager::boxDataTensor() const
{
    return exportStateTensor(8, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 consts::maxBoxes,
                                 7,
                             });
}

MADRONA_EXPORT madrona::py::Tensor Manager::rampDataTensor() const
{
    return exportStateTensor(9, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 consts::maxRamps,
                                 5,
                             });
}

MADRONA_EXPORT madrona::py::Tensor Manager::visibleAgentsMaskTensor() const
{
    return exportStateTensor(10, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 consts::maxAgents - 1,
                                 1,
                             });
}

MADRONA_EXPORT madrona::py::Tensor Manager::visibleBoxesMaskTensor() const
{
    return exportStateTensor(11, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 consts::maxBoxes,
                                 1,
                             });
}

MADRONA_EXPORT madrona::py::Tensor Manager::visibleRampsMaskTensor() const
{
    return exportStateTensor(12, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 consts::maxRamps,
                                 1,
                             });
}

MADRONA_IMPORT madrona::py::Tensor Manager::globalPositionsTensor() const
{
    return exportStateTensor(13, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds,
                                 consts::maxBoxes + consts::maxRamps +
                                     consts::maxAgents,
                                 2,
                             });
}

MADRONA_EXPORT Tensor Manager::depthTensor() const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr = static_cast<CUDAImpl *>(impl_)->mwGPU.
            depthObservations();
        gpu_id = impl_->cfg.gpuID;
#endif
    } else {
        dev_ptr = static_cast<CPUImpl *>(impl_)->cpuExec.
            depthObservations();

#ifdef MADRONA_LINUX
        gpu_id = impl_->cfg.gpuID;
#endif
    }

    return Tensor(dev_ptr, Tensor::ElementType::Float32,
                     {impl_->cfg.numWorlds * consts::maxAgents,
                      impl_->cfg.renderHeight,
                      impl_->cfg.renderWidth, 1}, gpu_id);
}

MADRONA_EXPORT Tensor Manager::rgbTensor() const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr = static_cast<CUDAImpl *>(impl_)->mwGPU.
            rgbObservations();
        gpu_id = impl_->cfg.gpuID;
#endif
    } else {
        dev_ptr = static_cast<CPUImpl *>(impl_)->cpuExec.
            rgbObservations();

#ifdef MADRONA_LINUX
        gpu_id = impl_->cfg.gpuID;
#endif
    }

    return Tensor(dev_ptr, Tensor::ElementType::UInt8,
                  {impl_->cfg.numWorlds * consts::maxAgents,
                   impl_->cfg.renderHeight,
                   impl_->cfg.renderWidth, 4}, gpu_id);
}


MADRONA_EXPORT madrona::py::Tensor Manager::lidarTensor() const
{
    return exportStateTensor(14, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 30,
                             });
}

MADRONA_EXPORT madrona::py::Tensor Manager::seedTensor() const
{
    return exportStateTensor(15, Tensor::ElementType::Int32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 1,
                             });
}

Tensor Manager::exportStateTensor(int64_t slot,
                                  Tensor::ElementType type,
                                  Span<const int64_t> dimensions) const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();
    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr =
            static_cast<CUDAImpl *>(impl_)->mwGPU.getExported(slot);
        gpu_id = impl_->cfg.gpuID;
#endif
    } else {
        dev_ptr = static_cast<CPUImpl *>(impl_)->cpuExec.getExported(slot);
    }

    return Tensor(dev_ptr, type, dimensions, gpu_id);
}


}
