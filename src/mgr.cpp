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
#include <random>

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
        .numWorlds = static_cast<uint32_t>(mgr_cfg.scenes.size()),
        .maxViewsPerWorld = consts::kMaxAgentCount + 1, // FIXME?
        .maxInstancesPerWorld = 3000,
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
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;
    int64_t numWorlds{0};

    inline Impl(const Manager::Config &mgr_cfg,
                PhysicsLoader &&phys_loader,
                EpisodeManager *ep_mgr,
                WorldReset *reset_buffer,
                Action *action_buffer,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr,
		int64_t numWorlds) 
        : cfg(mgr_cfg),
          physicsLoader(std::move(phys_loader)),
          episodeMgr(ep_mgr),
          worldResetBuffer(reset_buffer),
          agentActionsBuffer(action_buffer),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr)),
          numWorlds(numWorlds) {}

    inline virtual ~Impl() {}

    virtual void step() = 0;
    virtual void reset() = 0;

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
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
		   int64_t numWorlds)
        : Impl(mgr_cfg, std::move(phys_loader), ep_mgr, reset_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr), numWorlds),
          cpuExec(std::move(cpu_exec))
    {}

    inline virtual ~CPUImpl() final
    {
        delete episodeMgr;
    }

    inline virtual void step() { cpuExec.runTaskGraph(TaskGraphID::Step); }

    inline virtual void reset() { cpuExec.runTaskGraph(TaskGraphID::Reset); }

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
    MWCudaLaunchGraph resetGraph;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   EpisodeManager *ep_mgr,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   MWCudaExecutor &&gpu_exec,
                   Optional<RenderGPUState> &&render_gpu_state,
                  Optional<render::RenderManager> &&render_mgr,
                   int64_t numWorlds)
        : Impl(mgr_cfg, std::move(phys_loader),
               ep_mgr, reset_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr), numWorlds),  
          gpuExec(std::move(gpu_exec)),
          stepGraph(gpuExec.buildLaunchGraph(TaskGraphID::Step)),
          resetGraph(gpuExec.buildLaunchGraph(TaskGraphID::Reset)) {}

    inline virtual ~CUDAImpl() final
    {
        REQ_CUDA(cudaFree(episodeMgr));
    }

    inline virtual void step() { gpuExec.run(stepGraph); }

    inline virtual void reset() { gpuExec.run(resetGraph); }

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
        { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{50.0f, 50.0f, 1.0f} }
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

bool isRoadObservationAlgorithmValid(FindRoadObservationsWith algo) {
    madrona::CountT roadObservationsCount =
        sizeof(AgentMapObservations) / sizeof(MapObservation);

    return algo ==
               FindRoadObservationsWith::KNearestEntitiesWithRadiusFiltering ||
           (algo ==
                FindRoadObservationsWith::AllEntitiesWithRadiusFiltering &&
            roadObservationsCount == consts::kMaxAgentMapObservationsCount);
}

Manager::Impl * Manager::Impl::init(const Manager::Config &mgr_cfg) { 
    Sim::Config sim_cfg;
    sim_cfg.enableLidar = mgr_cfg.params.enableLidar;

    assert(isRoadObservationAlgorithmValid(
        mgr_cfg.params.roadObservationAlgorithm));

    const int64_t numWorlds = mgr_cfg.scenes.size();

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

        HeapArray<WorldInit> world_inits(numWorlds);


        Parameters* paramsDevicePtr = (Parameters*)cu::allocGPU(sizeof(Parameters));
        REQ_CUDA(cudaMemcpy(paramsDevicePtr, &(mgr_cfg.params), sizeof(Parameters), cudaMemcpyHostToDevice));
        
        int64_t worldIdx{0};
        for (auto const &scene : mgr_cfg.scenes) {
	    Map *map = (Map *)MapReader::parseAndWriteOut(scene,
							  ExecMode::CUDA, mgr_cfg.params.polylineReductionThreshold);
            world_inits[worldIdx++] = WorldInit{episode_mgr, phys_obj_mgr, map, paramsDevicePtr};
        }
        assert(worldIdx == numWorlds);

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
            .numWorlds = static_cast<uint32_t>(numWorlds),
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
        for (int64_t i = 0; i < numWorlds; i++) {
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
            std::move(render_gpu_state),
            std::move(render_mgr),
	    numWorlds
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

        HeapArray<WorldInit> world_inits(numWorlds);

        int64_t worldIdx{0};
    
        for (auto const &scene : mgr_cfg.scenes)
        {
            Map *map_ = (Map *)MapReader::parseAndWriteOut(scene,
                                                           ExecMode::CPU, mgr_cfg.params.polylineReductionThreshold);
            world_inits[worldIdx++] = WorldInit{episode_mgr, phys_obj_mgr, map_, &(mgr_cfg.params)};
        }
        assert(worldIdx == numWorlds);



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
                .numWorlds = static_cast<uint32_t>(mgr_cfg.scenes.size()),
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
            std::move(render_gpu_state),
            std::move(render_mgr),
	    numWorlds
        };

        for (size_t i = 0; i < mgr_cfg.scenes.size(); i++) {
          auto &init = world_inits[i];
          delete init.map;
        }

        return cpu_impl;
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg) : impl_(Impl::init(cfg)) { reset({}); }

Manager::~Manager() {}

void Manager::step()
{
    impl_->step();

    if (impl_->renderMgr.has_value()) {
        impl_->renderMgr->readECS();
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

void Manager::reset(std::vector<int32_t> worldsToReset) {
    for (const auto &worldIdx : worldsToReset) {
        triggerReset(worldIdx);
    }

    impl_->reset();

    if (impl_->renderMgr.has_value()) {
        impl_->renderMgr->readECS();
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

void Manager::setMaps(const std::vector<std::string> &maps)
{
    assert(impl_->cfg.scenes.size() == maps.size());
    impl_->cfg.scenes = maps;

    ResetMap resetmap{
        1,
    };

    if (impl_->cfg.execMode == madrona::ExecMode::CUDA)
    {
#ifdef MADRONA_CUDA_SUPPORT
        auto &gpu_exec = static_cast<CUDAImpl *>(impl_.get())->gpuExec;
        for (size_t world_idx = 0; world_idx < maps.size(); world_idx++)
        {
            Map *map = static_cast<Map *>(MapReader::parseAndWriteOut(maps[world_idx],
                                                                      ExecMode::CUDA, impl_->cfg.params.polylineReductionThreshold));
            Map *mapDevicePtr = (Map *)gpu_exec.getExported((uint32_t)ExportID::Map) + world_idx;
            REQ_CUDA(cudaMemcpy(mapDevicePtr, map, sizeof(Map), cudaMemcpyHostToDevice));
            madrona::cu::deallocGPU(map);

            auto resetMapPtr = (ResetMap *)gpu_exec.getExported((uint32_t)ExportID::ResetMap) + world_idx;
            REQ_CUDA(cudaMemcpy(resetMapPtr, &resetmap, sizeof(ResetMap), cudaMemcpyHostToDevice));
        }

#else
        // Handle the case where CUDA support is not available
        FATAL("Madrona was not compiled with CUDA support");
#endif
    }
    else
    {

        auto &cpu_exec = static_cast<CPUImpl *>(impl_.get())->cpuExec;

        for (size_t world_idx = 0; world_idx < maps.size(); world_idx++)
        {
            // Parse the map string into your MapData structure
            Map *map = static_cast<Map *>(MapReader::parseAndWriteOut(maps[world_idx],
                                                                      ExecMode::CPU, impl_->cfg.params.polylineReductionThreshold));

            Map *mapDevicePtr = (Map *)cpu_exec.getExported((uint32_t)ExportID::Map) + world_idx;
            memcpy(mapDevicePtr, map, sizeof(Map));
            delete map;

            auto resetMapPtr = (ResetMap *)cpu_exec.getExported((uint32_t)ExportID::ResetMap) + world_idx;
            memcpy(resetMapPtr, &resetmap, sizeof(ResetMap));
        }
    }

    // Vector of range on integers from 0 to the number of worlds
    std::vector<int32_t> worldIndices(maps.size());
    std::iota(worldIndices.begin(), worldIndices.end(), 0);
    reset(worldIndices);
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, TensorElementType::Float32,
        {
            impl_->numWorlds,
            consts::kMaxAgentCount,
            ActionExportSize, // Num_actions
        });
}


Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
                               {
                                   impl_->numWorlds,
                                   consts::kMaxAgentCount,
                                   1,
                               });
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, TensorElementType::Int32,
                               {
                                   impl_->numWorlds,
                                   consts::kMaxAgentCount,
                                   1,
                               });
}

Tensor Manager::infoTensor() const
{
    return impl_->exportTensor(ExportID::Info, TensorElementType::Int32,
                               {
                                   impl_->numWorlds,
                                   consts::kMaxAgentCount,
                                   InfoExportSize
                               });
}

Tensor Manager::selfObservationTensor() const
{
    return impl_->exportTensor(ExportID::SelfObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->numWorlds,
                                   consts::kMaxAgentCount,
                                   SelfObservationExportSize
			       });
}

Tensor Manager::mapObservationTensor() const
{
    return impl_->exportTensor(ExportID::MapObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->numWorlds,
                                   consts::kMaxRoadEntityCount,
                                   MapObservationExportSize
                               });
}

Tensor Manager::partnerObservationsTensor() const
{
    return impl_->exportTensor(ExportID::PartnerObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->numWorlds,
                                   consts::kMaxAgentCount,
                                   consts::kMaxAgentCount - 1,
                                   PartnerObservationExportSize
                               });
}

Tensor Manager::agentMapObservationsTensor() const
{
    return impl_->exportTensor(ExportID::AgentMapObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->numWorlds,
				                    consts::kMaxAgentCount,
                                   consts::kMaxAgentMapObservationsCount,
                                   AgentMapObservationExportSize,
                               });

}

Tensor Manager::lidarTensor() const
{
    return impl_->exportTensor(ExportID::Lidar, TensorElementType::Float32,
                               {
                                   impl_->numWorlds,
                                   consts::kMaxAgentCount,
                                   3, // Trace lidars on 3 planes
                                   consts::numLidarSamples,
                                   LidarExportSize / (3 * consts::numLidarSamples), 
                               });
}

Tensor Manager::stepsRemainingTensor() const
{
    return impl_->exportTensor(ExportID::StepsRemaining,
                               TensorElementType::Int32,
                               {
                                   impl_->numWorlds,
                                   consts::kMaxAgentCount,
                                   1,
                               });
}

Tensor Manager::shapeTensor() const {
    return impl_->exportTensor(ExportID::Shape, TensorElementType::Int32,
                               {impl_->numWorlds, 2});
}

Tensor Manager::controlledStateTensor() const {
    return impl_->exportTensor(ExportID::ControlledState, TensorElementType::Int32,
                               {impl_->numWorlds, consts::kMaxAgentCount, 1});
}

Tensor Manager::responseTypeTensor() const {
    return impl_->exportTensor(ExportID::ResponseType, TensorElementType::Int32,
                               {impl_->numWorlds, consts::kMaxAgentCount, 1});
}

Tensor Manager::absoluteSelfObservationTensor() const {
    return impl_->exportTensor(
        ExportID::AbsoluteSelfObservation, TensorElementType::Float32,
        {impl_->numWorlds, consts::kMaxAgentCount, AbsoluteSelfObservationExportSize});
}

Tensor Manager::validStateTensor() const {
    return impl_->exportTensor(
        ExportID::ValidState, TensorElementType::Int32,
        {impl_->numWorlds, consts::kMaxAgentCount, 1});
}

Tensor Manager::expertTrajectoryTensor() const {
    return impl_->exportTensor(
        ExportID::Trajectory, TensorElementType::Float32,
        {impl_->numWorlds, consts::kMaxAgentCount, TrajectoryExportSize});
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

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    assert(rgb_ptr != nullptr);

    return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->numWorlds,
        consts::kMaxAgentCount,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->numWorlds,
        consts::kMaxAgentCount,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

void Manager::setAction(int32_t world_idx, int32_t agent_idx,
                        float acceleration, float steering, float headAngle) {
    Action action{.classic = {acceleration, steering, headAngle}};

    auto *action_ptr = impl_->agentActionsBuffer + world_idx * consts::kMaxAgentCount + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action), cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

std::vector<Shape>
Manager::getShapeTensorFromDeviceMemory() {
    const uint32_t numWorlds = impl_->numWorlds;
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
