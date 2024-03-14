#include "mgr.hpp"
#include "MapReader.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <string>
#include <cstdlib>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace gpudrive {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};


static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
    if (mgr_cfg.extRenderDev || !mgr_cfg.enableBatchRenderer) {
        return Optional<RenderGPUState>::none();
    }

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (!mgr_cfg.extRenderDev && !mgr_cfg.enableBatchRenderer) {
        return Optional<render::RenderManager>::none();
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        render_api = mgr_cfg.extRenderAPI;
        render_dev = mgr_cfg.extRenderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = 2, // FIXME?
        .maxInstancesPerWorld = 450,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

struct Manager::Impl {
    Config cfg;
    PhysicsLoader physicsLoader;
    EpisodeManager *episodeMgr;
    WorldReset *worldResetBuffer;
    Action *agentActionsBuffer;
    std::pair<uint32_t, uint32_t> agentRoadCounts;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;

    inline Impl(const Manager::Config &mgr_cfg,
                PhysicsLoader &&phys_loader,
                EpisodeManager *ep_mgr,
                WorldReset *reset_buffer,
                Action *action_buffer,
                std::pair<uint32_t, uint32_t> agentRoadCounts,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr)
        : cfg(mgr_cfg),
          physicsLoader(std::move(phys_loader)),
          episodeMgr(ep_mgr),
          worldResetBuffer(reset_buffer),
          agentActionsBuffer(action_buffer),
          agentRoadCounts(agentRoadCounts),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr))
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;

    virtual Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dimensions) const = 0;

    static inline Impl * init(const Config &cfg);
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
                   TaskGraphT &&cpu_exec,
                   std::pair<uint32_t, uint32_t> agentRoadCounts,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr)
        : Impl(mgr_cfg, std::move(phys_loader),
               ep_mgr, reset_buffer, action_buffer, agentRoadCounts,
               std::move(render_gpu_state), std::move(render_mgr)),
          cpuExec(std::move(cpu_exec))
    {}

    inline virtual ~CPUImpl() final
    {
        delete episodeMgr;
    }

    inline virtual void run()
    {
        cpuExec.runTaskGraph(TaskGraphID::Step);
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph stepGraph;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   EpisodeManager *ep_mgr,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   MWCudaExecutor &&gpu_exec,
                   std::pair<uint32_t, uint32_t> agentRoadCounts,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr)
        : Impl(mgr_cfg, std::move(phys_loader),
               ep_mgr, reset_buffer, action_buffer, agentRoadCounts,
               std::move(render_gpu_state), std::move(render_mgr)),
          gpuExec(std::move(gpu_exec)),
          stepGraph(gpuExec.buildLaunchGraph(TaskGraphID::Step))
    {}

    inline virtual ~CUDAImpl() final
    {
        REQ_CUDA(cudaFree(episodeMgr));
    }

    inline virtual void run()
    {
        gpuExec.run(stepGraph);
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

static void loadRenderObjects(render::RenderManager &render_mgr)
{
    std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;
    render_asset_paths[(size_t)SimObject::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_render.obj").string();
    render_asset_paths[(size_t)SimObject::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();
    render_asset_paths[(size_t)SimObject::StopSign] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::SpeedBump] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();

    std::array<const char *, (size_t)SimObject::NumObjects> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    auto materials = std::to_array<imp::SourceMaterial>({
        { render::rgb8ToFloat(191, 108, 10), -1, 0.8f, 1.0f },
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,},
        { math::Vector4{1.f, 1.f, 1.f, 0.0f}, 1, 0.5f, 1.0f,},
        { render::rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },
        { math::Vector4{0.5f, 0.3f, 0.3f, 0.0f},  0, 0.8f, 0.2f,},
        { render::rgb8ToFloat(230, 20, 20),   -1, 0.8f, 1.0f },
        { render::rgb8ToFloat(230, 230, 20),   -1, 0.8f, 1.0f },
        { render::rgb8ToFloat(255,0,0), -1, 0.8f, 1.0f},
        { render::rgb8ToFloat(0,0,0), -1, 0.8f, 0.2f}
    });

    // Override materials
    render_assets->objects[(CountT)SimObject::Cube].meshes[0].materialIDX = 0;
    render_assets->objects[(CountT)SimObject::Agent].meshes[0].materialIDX = 2;
    render_assets->objects[(CountT)SimObject::Agent].meshes[1].materialIDX = 3;
    render_assets->objects[(CountT)SimObject::Agent].meshes[2].materialIDX = 3;
    render_assets->objects[(CountT)SimObject::Plane].meshes[0].materialIDX = 4;
    render_assets->objects[(CountT)SimObject::StopSign].meshes[0].materialIDX = 7;
    render_assets->objects[(CountT)SimObject::SpeedBump].meshes[0].materialIDX = 8;
    // render_assets->objects[(CountT)SimObject::Cylinder].meshes[0].materialIDX = 7;

    render_mgr.loadObjects(render_assets->objects, materials, {
        { (std::filesystem::path(DATA_DIR) /
           "green_grid.png").string().c_str() },
        { (std::filesystem::path(DATA_DIR) /
           "smile.png").string().c_str() },
    });

    render_mgr.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });
}

static void loadPhysicsObjects(PhysicsLoader &loader)
{
    std::array<std::string, (size_t)SimObject::NumObjects - 1> asset_paths;
    asset_paths[(size_t)SimObject::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    asset_paths[(size_t)SimObject::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_collision_simplified.obj").string();
    asset_paths[(size_t)SimObject::StopSign] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    asset_paths[(size_t)SimObject::SpeedBump] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    // asset_paths[(size_t)SimObject::Cylinder] =
    //     (std::filesystem::path(DATA_DIR) / "cylinder_collision.obj").string();

    std::array<const char *, (size_t)SimObject::NumObjects - 1> asset_cstrs;
    for (size_t i = 0; i < asset_paths.size(); i++) {
        asset_cstrs[i] = asset_paths[i].c_str();
    }

    char import_err_buffer[4096];
    auto imported_src_hulls = imp::ImportedAssets::importFromDisk(
        asset_cstrs, import_err_buffer, true);

    if (!imported_src_hulls.has_value()) {
        FATAL("%s", import_err_buffer);
    }

    DynArray<imp::SourceMesh> src_convex_hulls(
        imported_src_hulls->objects.size());

    DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
    HeapArray<SourceCollisionObject> src_objs(
        (CountT)SimObject::NumObjects);

    auto setupHull = [&](SimObject obj_id,
                         float inv_mass,
                         RigidBodyFrictionData friction) {
        auto meshes = imported_src_hulls->objects[(CountT)obj_id].meshes;
        DynArray<SourceCollisionPrimitive> prims(meshes.size());

        for (const imp::SourceMesh &mesh : meshes) {
            src_convex_hulls.push_back(mesh);
            prims.push_back({
                .type = CollisionPrimitive::Type::Hull,
                .hullInput = {
                    .hullIDX = uint32_t(src_convex_hulls.size() - 1),
                },
            });
        }

        prim_arrays.emplace_back(std::move(prims));

        src_objs[(CountT)obj_id] = SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(prim_arrays.back()),
            .invMass = inv_mass,
            .friction = friction,
        };
    };

    setupHull(SimObject::Cube, 0.075f, {
        .muS = 0.5f,
        .muD = 0.75f,
    });

    setupHull(SimObject::Agent, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObject::StopSign, 1.f, {
    .muS = 0.5f,
    .muD = 0.5f,
        });

    setupHull(SimObject::SpeedBump, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    SourceCollisionPrimitive plane_prim {
        .type = CollisionPrimitive::Type::Plane,
    };

    src_objs[(CountT)SimObject::Plane] = {
        .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
        .invMass = 0.f,
        .friction = {
            .muS = 0.5f,
            .muD = 0.5f,
        },
    };

    StackAlloc tmp_alloc;
    RigidBodyAssets rigid_body_assets;
    CountT num_rigid_body_data_bytes;
    void *rigid_body_data = RigidBodyAssets::processRigidBodyAssets(
        src_convex_hulls,
        src_objs,
        false,
        tmp_alloc,
        &rigid_body_assets,
        &num_rigid_body_data_bytes);

    if (rigid_body_data == nullptr) {
        FATAL("Invalid collision hull input");
    }

    // This is a bit hacky, but in order to make sure the agents
    // remain controllable by the policy, they are only allowed to
    // rotate around the Z axis (infinite inertia in x & y axes)
    rigid_body_assets.metadatas[
        (CountT)SimObject::Agent].mass.invInertiaTensor.x = 0.f;
    rigid_body_assets.metadatas[
        (CountT)SimObject::Agent].mass.invInertiaTensor.y = 0.f;

    loader.loadRigidBodies(rigid_body_assets);
    free(rigid_body_data);
}

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg)
{
    Sim::Config sim_cfg;
    sim_cfg.autoReset = mgr_cfg.autoReset;
    sim_cfg.kMaxAgentCount = 0;
    sim_cfg.kMaxRoadEntityCount = 0;

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

        EpisodeManager *episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

        PhysicsLoader phys_loader(ExecMode::CUDA, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();

        HeapArray<WorldInit> world_inits(mgr_cfg.numWorlds);


        Parameters* paramsDevicePtr = (Parameters*)cu::allocGPU(sizeof(Parameters));
        REQ_CUDA(cudaMemcpy(paramsDevicePtr, &(mgr_cfg.params), sizeof(Parameters), cudaMemcpyHostToDevice));
        
        int64_t worldIdx{0};
    
        for (auto const &mapFile : std::filesystem::directory_iterator(mgr_cfg.jsonPath))
        {
            auto [map_, mapCounts] = MapReader::parseAndWriteOut(mapFile.path(), mgr_cfg.execMode, mgr_cfg.params.polylineReductionThreshold);
            world_inits[worldIdx++] = WorldInit{episode_mgr, phys_obj_mgr,
                                                map_, mgr_cfg.execMode, paramsDevicePtr};
            sim_cfg.kMaxAgentCount = std::max(mapCounts.first, sim_cfg.kMaxAgentCount);
            sim_cfg.kMaxRoadEntityCount = std::max(mapCounts.second, sim_cfg.kMaxRoadEntityCount);
        }

        assert(worldIdx == static_cast<int64_t>(mgr_cfg.numWorlds));
        // Bounds on the maxagent and maxroadentity counts.
        assert(sim_cfg.kMaxAgentCount <= consts::kMaxAgentCount);
        assert(sim_cfg.kMaxRoadEntityCount <= consts::kMaxRoadEntityCount);
        assert(sim_cfg.kMaxAgentCount > 0);
        assert(sim_cfg.kMaxRoadEntityCount > 0);

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(*render_mgr);
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        MWCudaExecutor gpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(WorldInit),
            .userConfigPtr = (void *)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = mgr_cfg.numWorlds,
            .numTaskGraphs = (uint32_t)TaskGraphID::NumTaskGraphs,
            .numExportedBuffers = (uint32_t)ExportID::NumExports, 
        }, {
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, cu_ctx);

        WorldReset *world_reset_buffer = 
            (WorldReset *)gpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)gpu_exec.getExported((uint32_t)ExportID::Action);

        madrona::cu::deallocGPU(paramsDevicePtr);
        for (int64_t i = 0; i < (int64_t)mgr_cfg.numWorlds; i++) {
          auto &init = world_inits[i];
          madrona::cu::deallocGPU(init.map);
        }

        return new CUDAImpl {
            mgr_cfg,
            std::move(phys_loader),
            episode_mgr,
            world_reset_buffer,
            agent_actions_buffer,
            std::move(gpu_exec),
            std::make_pair(sim_cfg.kMaxAgentCount, sim_cfg.kMaxRoadEntityCount),
            std::move(render_gpu_state),
            std::move(render_mgr),
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

        int64_t worldIdx{0};
    
        for (auto const &mapFile : std::filesystem::directory_iterator(mgr_cfg.jsonPath))
        {
            auto [map_, mapCounts] = MapReader::parseAndWriteOut(mapFile.path(), mgr_cfg.execMode, mgr_cfg.params.polylineReductionThreshold);
            world_inits[worldIdx++] = WorldInit{episode_mgr, phys_obj_mgr,
                                                map_, mgr_cfg.execMode, &(mgr_cfg.params)};
            sim_cfg.kMaxAgentCount = std::max(mapCounts.first, sim_cfg.kMaxAgentCount);
            sim_cfg.kMaxRoadEntityCount = std::max(mapCounts.second, sim_cfg.kMaxRoadEntityCount);
        }

        assert(worldIdx == static_cast<int64_t>(mgr_cfg.numWorlds));
        // Bounds on the maxagent and maxroadentity counts.
        assert(sim_cfg.kMaxAgentCount <= consts::kMaxAgentCount);
        assert(sim_cfg.kMaxRoadEntityCount <= consts::kMaxRoadEntityCount);
        assert(sim_cfg.kMaxAgentCount > 0);
        assert(sim_cfg.kMaxRoadEntityCount > 0);


        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(*render_mgr);
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        CPUImpl::TaskGraphT cpu_exec {
            ThreadPoolExecutor::Config {
                .numWorlds = mgr_cfg.numWorlds,
                .numExportedBuffers = (uint32_t)ExportID::NumExports,
            },
            sim_cfg,
            world_inits.data(),
            (uint32_t)TaskGraphID::NumTaskGraphs,
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
            std::make_pair(sim_cfg.kMaxAgentCount, sim_cfg.kMaxRoadEntityCount),
            std::move(render_gpu_state),
            std::move(render_mgr),
        };

        for (int64_t i = 0; i < (int64_t)mgr_cfg.numWorlds; i++) {
          auto &init = world_inits[i];
          delete init.map;
        }

        return cpu_impl;
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{
    // Currently, there is no way to populate the initial set of observations
    // without stepping the simulations in order to execute the taskgraph.
    // Therefore, after setup, we step all the simulations with a forced reset
    // that ensures the first real step will have valid observations at the
    // start of a fresh episode in order to compute actions.
    //
    // This will be improved in the future with support for multiple task
    // graphs, allowing a small task graph to be executed after initialization.
    
    for (int32_t i = 0; i < (int32_t)cfg.numWorlds; i++) {
        triggerReset(i);
    }

    step();
}

Manager::~Manager() {}

void Manager::step()
{
    impl_->run();
}

Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::Reset,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   1,
                               });
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds,
            impl_->agentRoadCounts.first,
            3, // Num_actions
        });
}

Tensor Manager::bicycleModelTensor() const
{
    return impl_->exportTensor(ExportID::BicycleModel, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds,
            impl_->agentRoadCounts.first,
            4, // Number of states for the bicycle model
        });
}


Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   1,
                               });
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   1,
                               });
}

Tensor Manager::selfObservationTensor() const
{
    return impl_->exportTensor(ExportID::SelfObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   6
                               });
}

Tensor Manager::mapObservationTensor() const
{
    return impl_->exportTensor(ExportID::MapObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.second,
                                   4
                               });
}

Tensor Manager::partnerObservationsTensor() const
{
    return impl_->exportTensor(ExportID::PartnerObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   consts::kMaxAgentCount-1,
                                   7,
                               });
}

Tensor Manager::agentMapObservationsTensor() const
{
    return impl_->exportTensor(ExportID::AgentMapObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   consts::kMaxRoadEntityCount,
                                   4,
                               });

}

Tensor Manager::lidarTensor() const
{
    return impl_->exportTensor(ExportID::Lidar, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   consts::numLidarSamples,
                                   2,
                               });
}

Tensor Manager::stepsRemainingTensor() const
{
    return impl_->exportTensor(ExportID::StepsRemaining,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   1,
                               });
}

Tensor Manager::shapeTensor() const {
    return impl_->exportTensor(ExportID::Shape, TensorElementType::Int32,
                               {impl_->cfg.numWorlds, 2});
}

Tensor Manager::controlledStateTensor() const {
    return impl_->exportTensor(ExportID::ControlledState, TensorElementType::Int32,
                               {impl_->cfg.numWorlds,consts::kMaxAgentCount, 1});
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
                        float acceleration, float steering, float headAngle) {
    Action action{.acceleration = acceleration,
                  .steering = steering,
                  .headAngle = headAngle};

    auto *action_ptr =
        impl_->agentActionsBuffer + world_idx * consts::kMaxAgentCount + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action), cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

std::vector<Shape>
Manager::getShapeTensorFromDeviceMemory(madrona::ExecMode mode,
                                        uint32_t numWorlds) {
    const auto &tensor = shapeTensor();

    const std::size_t floatsPerShape{2};
    const std::size_t tensorByteCount{sizeof(float) * floatsPerShape *
                                      numWorlds};

    std::vector<Shape> worldToShape(numWorlds);
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA:
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(worldToShape.data(), tensor.devicePtr(), tensorByteCount,
                   cudaMemcpyDeviceToHost);
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
        break;
    case ExecMode::CPU:
        std::memcpy(worldToShape.data(), tensor.devicePtr(), tensorByteCount);
        break;
    }

    return worldToShape;
}

render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

}
