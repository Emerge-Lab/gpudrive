#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_assets.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>

#include <array>
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
    float *rewardsBuffer;
    uint8_t *donesBuffer;

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
    using SourceCollisionObject = PhysicsLoader::SourceCollisionObject;
    using SourceCollisionPrimitive = PhysicsLoader::SourceCollisionPrimitive;

    SourceCollisionPrimitive sphere_prim {
        .type = CollisionPrimitive::Type::Sphere,
        .sphere = CollisionPrimitive::Sphere {
            .radius = 1.f,
        },
    };

    SourceCollisionPrimitive plane_prim {
        .type = CollisionPrimitive::Type::Plane,
    };

    char import_err_buffer[4096];
    auto imported_hulls = imp::ImportedAssets::importFromDisk({
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "cylinder_collision.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "ramp_collision.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "elongated_collision.obj").c_str(),
    }, import_err_buffer, true);

    if (!imported_hulls.has_value()) {
        FATAL("%s", import_err_buffer);
    }

    DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
    HeapArray<SourceCollisionObject> src_objs(imported_hulls->objects.size() + 2);

    // Sphere (0)
    src_objs[0] = {
        .prims = Span<const SourceCollisionPrimitive>(&sphere_prim, 1),
        .invMass = 1.f,
        .friction = {
            .muS = 0.5f,
            .muD = 0.5f,
        },
    };

    // Plane (1)
    src_objs[1] = {
        .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
        .invMass = 0.f,
        .friction = {
            .muS = 0.5f,
            .muD = 0.5f,
        },
    };

    auto setupHull = [&](CountT obj_idx, float inv_mass,
                         RigidBodyFrictionData friction) {
        auto meshes = imported_hulls->objects[obj_idx].meshes;
        DynArray<SourceCollisionPrimitive> prims(meshes.size());

        for (const imp::SourceMesh &mesh : meshes) {
            prims.push_back({
                .type = CollisionPrimitive::Type::Hull,
                .hullInput = {
                    .mesh = &mesh,
                },
            });
        }

        prim_arrays.emplace_back(std::move(prims));

        return SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(prim_arrays.back()),
            .invMass = inv_mass,
            .friction = friction,
        };
    };

    { // Cube (2)
        src_objs[2] = setupHull(0, 1.f, {
            .muS = 0.5f,
            .muD = 0.5f,
        });
    }

    { // Wall (3)
        src_objs[3] = setupHull(1, 0.f, {
            .muS = 0.5f,
            .muD = 0.5f,
        });
    }

    { // Cylinder (4)
        src_objs[4] = setupHull(2, 1.f, {
            .muS = 0.5f,
            .muD = 0.5f,
        });
    }

    { // Ramp (5)
        src_objs[5] = setupHull(3, 1.f, {
            .muS = 0.5f,
            .muD = 0.5f,
        });
    }

    { // Elongated Box (6)
        src_objs[6] = setupHull(4, 1.f, {
            .muS = 0.5f,
            .muD = 0.5f,
        });
    }

    auto phys_objs_res = loader.importRigidBodyData(
        src_objs.data(), src_objs.size(), false);

    if (!phys_objs_res.has_value()) {
        FATAL("Invalid collision hull input");
    }

    auto &phys_objs = *phys_objs_res;

    // HACK:
    phys_objs.metadatas[4].mass.invInertiaTensor.x = 0.f,
    phys_objs.metadatas[4].mass.invInertiaTensor.y = 0.f,

    loader.loadObjects(
        phys_objs.metadatas.data(),
        phys_objs.objectAABBs.data(),
        phys_objs.primOffsets.data(),
        phys_objs.primCounts.data(),
        phys_objs.metadatas.size(),
        phys_objs.collisionPrimitives.data(),
        phys_objs.primitiveAABBs.data(),
        phys_objs.collisionPrimitives.size(),
        phys_objs.hullData.halfEdges.data(),
        phys_objs.hullData.halfEdges.size(),
        phys_objs.hullData.faceBaseHEs.data(),
        phys_objs.hullData.facePlanes.data(),
        phys_objs.hullData.facePlanes.size(),
        phys_objs.hullData.positions.data(),
        phys_objs.hullData.positions.size());
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

        auto done_buffer = (uint8_t *)cu::allocGPU(sizeof(uint8_t) *
            consts::numAgents * cfg.numWorlds);

        auto reward_buffer = (float *)cu::allocGPU(sizeof(float) *
            consts::numAgents * cfg.numWorlds);

        HeapArray<WorldInit> world_inits(cfg.numWorlds);

        for (int64_t i = 0; i < (int64_t)cfg.numWorlds; i++) {
            world_inits[i] = WorldInit {
                episode_mgr,
                reward_buffer,
                done_buffer,
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
            .maxViewsPerWorld = consts::numAgents,
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
                reward_buffer,
                done_buffer,
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

        auto reward_buffer = (float *)malloc(
            sizeof(float) * consts::numAgents * cfg.numWorlds);

        auto done_buffer = (uint8_t *)malloc(
            sizeof(uint8_t) * consts::numAgents * cfg.numWorlds);

        HeapArray<WorldInit> world_inits(cfg.numWorlds);

        for (int64_t i = 0; i < (int64_t)cfg.numWorlds; i++) {
            world_inits[i] = WorldInit {
                episode_mgr,
                reward_buffer + i * consts::numAgents,
                done_buffer + i * consts::numAgents,
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
                reward_buffer,
                done_buffer,
            },
            CPUImpl::TaskGraphT {
                ThreadPoolExecutor::Config {
                    .numWorlds = cfg.numWorlds,
                    .maxViewsPerWorld = consts::numAgents,
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
        auto cpu_impl = static_cast<CPUImpl *>(impl_);
        cpu_impl->cpuExec.run();

        // FIXME: provide some way to do this more cleanly in madrona
        CountT cur_agent_offset = 0;
        float *base_rewards = cpu_impl->rewardsBuffer;
        uint8_t *base_dones = cpu_impl->donesBuffer;

        for (CountT i = 0; i < (CountT)impl_->cfg.numWorlds; i++) {
            const Sim &sim_data = cpu_impl->cpuExec.getWorldData(i);
            CountT num_agents = consts::numAgents;
            float *world_rewards = sim_data.rewardBuffer;
            uint8_t *world_dones = sim_data.doneBuffer;

            memmove(&base_rewards[cur_agent_offset],
                    world_rewards,
                    sizeof(float) * num_agents);

            memmove(&base_dones[cur_agent_offset],
                    world_dones,
                    sizeof(uint8_t) * num_agents);

            cur_agent_offset += num_agents;
        }
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
    Optional<int> gpu_id = Optional<int>::none();
    if (impl_->cfg.execMode == ExecMode::CUDA) {
        gpu_id = impl_->cfg.gpuID;
    }

    return Tensor(impl_->donesBuffer, Tensor::ElementType::UInt8,
                 {impl_->cfg.numWorlds * consts::numAgents, 1}, gpu_id);
}

MADRONA_EXPORT Tensor Manager::actionTensor() const
{
    return exportStateTensor(2, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds * consts::numAgents, 5});
}

MADRONA_EXPORT Tensor Manager::rewardTensor() const
{
    Optional<int> gpu_id = Optional<int>::none();
    if (impl_->cfg.execMode == ExecMode::CUDA) {
        gpu_id = impl_->cfg.gpuID;
    }

    return Tensor(impl_->rewardsBuffer, Tensor::ElementType::Float32,
                 {impl_->cfg.numWorlds * consts::numAgents, 1}, gpu_id);
}

MADRONA_EXPORT Tensor Manager::agentTypeTensor() const
{
    return exportStateTensor(4, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds * consts::numAgents, 1});
}

MADRONA_EXPORT madrona::py::Tensor Manager::relativeAgentObservationsTensor() const
{
    return exportStateTensor(5, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::numAgents,
                                 consts::numAgents - 1,
                                 2, // Polar coordinates
                             });
}

MADRONA_EXPORT madrona::py::Tensor Manager::relativeButtonObservationsTensor() const
{
    return exportStateTensor(6, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::numAgents,
                                 consts::maxRooms,
                                 2, // Polar coordinates
                             });
}

MADRONA_EXPORT madrona::py::Tensor Manager::relativeDestinationObservationsTensor() const
{
    return exportStateTensor(7, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::numAgents,
                                 1,
                                 2, // Polar coordinates
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
                     {impl_->cfg.numWorlds * consts::numAgents,
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
                  {impl_->cfg.numWorlds * consts::numAgents,
                   impl_->cfg.renderHeight,
                   impl_->cfg.renderWidth, 4}, gpu_id);
}


MADRONA_EXPORT madrona::py::Tensor Manager::lidarTensor() const
{
    return exportStateTensor(10, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::numAgents,
                                 30,
                             });
}

MADRONA_EXPORT madrona::py::Tensor Manager::seedTensor() const
{
    return exportStateTensor(11, Tensor::ElementType::Int32,
                             {
                                 impl_->cfg.numWorlds * consts::numAgents,
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
