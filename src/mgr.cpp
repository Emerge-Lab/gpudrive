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
    WorldReset *worldResetBuffer;
    Action *agentActionsBuffer;

    inline Impl(const Manager::Config &mgr_cfg,
                PhysicsLoader &&phys_loader,
                EpisodeManager *ep_mgr,
                WorldReset *reset_buffer,
                Action *action_buffer)
        : cfg(mgr_cfg),
          physicsLoader(std::move(phys_loader)),
          episodeMgr(ep_mgr),
          worldResetBuffer(reset_buffer),
          agentActionsBuffer(action_buffer)
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;

    virtual Tensor exportTensor(ExportID slot,
        Tensor::ElementType type,
        madrona::Span<const int64_t> dimensions) const = 0;

    static inline Impl * init(const Config &cfg,
                              const viz::VizECSBridge *viz_bridge);
};

struct Manager::CPUImpl final : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, Sim::Config, WorldInit>;

    TaskGraphT cpuExec;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   EpisodeManager *ep_mgr,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   TaskGraphT &&cpu_exec)
        : Impl(mgr_cfg, std::move(phys_loader),
               ep_mgr, reset_buffer, action_buffer),
          cpuExec(std::move(cpu_exec))
    {}

    inline virtual ~CPUImpl() final
    {
        delete episodeMgr;
    }

    inline virtual void run()
    {
        cpuExec.run();
    }

    virtual inline Tensor exportTensor(ExportID slot,
        Tensor::ElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   EpisodeManager *ep_mgr,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   MWCudaExecutor &&gpu_exec)
        : Impl(mgr_cfg, std::move(phys_loader),
               ep_mgr, reset_buffer, action_buffer),
          gpuExec(std::move(gpu_exec))
    {}

    inline virtual ~CUDAImpl() final
    {
        REQ_CUDA(cudaFree(episodeMgr));
    }

    inline virtual void run()
    {
        gpuExec.run();
    }

    virtual inline Tensor exportTensor(ExportID slot,
        Tensor::ElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
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
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "cylinder_collision.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "ramp_collision.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "elongated_collision.obj").string().c_str(),
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

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg,
    const viz::VizECSBridge *viz_bridge)
{
    HostEventLogging(HostEvent::initStart);

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk({
        (std::filesystem::path(DATA_DIR) / "sphere.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "plane.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "cylinder_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "ramp_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "elongated_render.obj").string().c_str(),
    }, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    Sim::Config sim_cfg {
        mgr_cfg.enableBatchRender,
        viz_bridge != nullptr,
        mgr_cfg.autoReset,
    };

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        EpisodeManager *episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

        PhysicsLoader phys_loader(ExecMode::CUDA, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();

        HeapArray<WorldInit> world_inits(mgr_cfg.numWorlds);

        for (int64_t i = 0; i < (int64_t)mgr_cfg.numWorlds; i++) {
            world_inits[i] = WorldInit {
                episode_mgr,
                phys_obj_mgr,
                viz_bridge,
            };
        }

        MWCudaExecutor gpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(WorldInit),
            .userConfigPtr = (void *)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = mgr_cfg.numWorlds,
            .numExportedBuffers = (uint32_t)ExportID::NumExports, 
            .gpuID = (uint32_t)mgr_cfg.gpuID,
        }, {
            "",
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
            CompileConfig::Executor::TaskGraph,
        });

        WorldReset *world_reset_buffer = 
            (WorldReset *)gpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)gpu_exec.getExported((uint32_t)ExportID::Action);


        return new CUDAImpl {
            mgr_cfg,
            std::move(phys_loader),
            episode_mgr,
            world_reset_buffer,
            agent_actions_buffer,
            std::move(gpu_exec),
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        EpisodeManager *episode_mgr = new EpisodeManager { 0 };

        PhysicsLoader phys_loader(ExecMode::CPU, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();

        HeapArray<WorldInit> world_inits(mgr_cfg.numWorlds);

        for (int64_t i = 0; i < (int64_t)mgr_cfg.numWorlds; i++) {
            world_inits[i] = WorldInit {
                episode_mgr,
                phys_obj_mgr,
                viz_bridge,
            };
        }

        CPUImpl::TaskGraphT cpu_exec {
            ThreadPoolExecutor::Config {
                .numWorlds = mgr_cfg.numWorlds,
                .numExportedBuffers = (uint32_t)ExportID::NumExports,
            },
            sim_cfg,
            world_inits.data(),
        };

        WorldReset *world_reset_buffer = 
            (WorldReset *)cpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)cpu_exec.getExported((uint32_t)ExportID::Action);

        auto cpu_impl = new CPUImpl {
            mgr_cfg,
            std::move(phys_loader),
            episode_mgr,
            world_reset_buffer,
            agent_actions_buffer,
            std::move(cpu_exec),
        };

        return cpu_impl;
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg,
                                const viz::VizECSBridge *viz_bridge)
    : impl_(Impl::init(cfg, viz_bridge))
{}

Manager::~Manager() {}

void Manager::step()
{
    impl_->run();
}

Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::Reset,
                               Tensor::ElementType::Int32,
                               {impl_->cfg.numWorlds, 1});
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, Tensor::ElementType::Int32,
        {impl_->cfg.numWorlds * consts::numAgents, 3});
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, Tensor::ElementType::Float32,
                               {impl_->cfg.numWorlds * consts::numAgents, 1});
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, Tensor::ElementType::Int32,
                               {impl_->cfg.numWorlds * consts::numAgents, 1});
}

Tensor Manager::positionObservationTensor() const
{
    return impl_->exportTensor(ExportID::PositionObservation,
                               Tensor::ElementType::Float32,
                               {impl_->cfg.numWorlds * consts::numAgents, 2});
}

Tensor Manager::toOtherAgentsTensor() const
{
    return impl_->exportTensor(ExportID::ToOtherAgents,
                               Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   consts::numAgents - 1,
                                   2, // Polar coordinates
                               });
}

Tensor Manager::toButtonsTensor() const
{
    return impl_->exportTensor(ExportID::ToButtons,
                               Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   consts::maxRooms,
                                   2, // Polar coordinates
                               });
}

Tensor Manager::toGoalTensor() const
{
    return impl_->exportTensor(ExportID::ToGoal,
                               Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   2, // Polar coordinates
                               });
}

Tensor Manager::lidarTensor() const
{
    return impl_->exportTensor(ExportID::Lidar, Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   consts::numLidarSamples,
                               });
}

Tensor Manager::seedTensor() const
{
    return impl_->exportTensor(ExportID::Seed, Tensor::ElementType::Int32,
                               {
                                   impl_->cfg.numWorlds * consts::numAgents,
                                   1,
                               });
}

Tensor Manager::depthTensor() const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();

#if 0
    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr = static_cast<CUDAImpl *>(impl_)->gpuExec.
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
#endif

    return Tensor(dev_ptr, Tensor::ElementType::Float32,
                     {impl_->cfg.numWorlds * consts::numAgents,
                      impl_->cfg.renderHeight,
                      impl_->cfg.renderWidth, 1}, gpu_id);
}

Tensor Manager::rgbTensor() const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();

#if 0
    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr = static_cast<CUDAImpl *>(impl_)->gpuExec.
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
#endif

    return Tensor(dev_ptr, Tensor::ElementType::UInt8,
                  {impl_->cfg.numWorlds * consts::numAgents,
                   impl_->cfg.renderHeight,
                   impl_->cfg.renderWidth, 4}, gpu_id);
}

void Manager::triggerReset(int32_t world_idx)
{
    WorldReset reset {
        1,
    };

    auto *reset_ptr = impl_->worldResetBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *reset_ptr = reset;
    }
}

void Manager::setAction(int32_t world_idx, int32_t agent_idx,
                        int32_t x, int32_t y, int32_t r)
{
    Action action { 
        .x = x,
        .y = y,
        .r = r,
    };

    auto *action_ptr = impl_->agentActionsBuffer +
        world_idx * consts::numAgents + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

}
