#include "mgr.hpp"
#include "MapReader.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>

#include <nlohmann/json.hpp>

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

struct Manager::Impl {
    Config cfg;
    PhysicsLoader physicsLoader;
    EpisodeManager *episodeMgr;
    WorldReset *worldResetBuffer;
    Action *agentActionsBuffer;
    std::pair<uint32_t, uint32_t> agentRoadCounts;

    inline Impl(const Manager::Config &mgr_cfg,
                PhysicsLoader &&phys_loader,
                EpisodeManager *ep_mgr,
                WorldReset *reset_buffer,
                Action *action_buffer,
                std::pair<uint32_t, uint32_t> agentRoadCounts)
        : cfg(mgr_cfg),
          physicsLoader(std::move(phys_loader)),
          episodeMgr(ep_mgr),
          worldResetBuffer(reset_buffer),
          agentActionsBuffer(action_buffer),
          agentRoadCounts(agentRoadCounts)
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
                   TaskGraphT &&cpu_exec,
                   std::pair<uint32_t, uint32_t> agentRoadCounts)
        : Impl(mgr_cfg, std::move(phys_loader),
               ep_mgr, reset_buffer, action_buffer, agentRoadCounts),
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
                   MWCudaExecutor &&gpu_exec,
                   std::pair<uint32_t, uint32_t> agentRoadCounts)
        : Impl(mgr_cfg, std::move(phys_loader),
               ep_mgr, reset_buffer, action_buffer, agentRoadCounts),
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

static std::vector<std::string> getMapFiles(const Manager::Config &cfg)
{
    std::filesystem::path path(cfg.jsonPath);
    auto validFilesJsonPath = path / "valid_files.json";
    assert(std::filesystem::exists(validFilesJsonPath));
    // check if validFiles.json exists

    std::ifstream validFilesJson(validFilesJsonPath);
    assert(validFilesJson.good());

    nlohmann::json validFiles;
    validFilesJson >> validFiles;

    std::vector<std::string> mapFiles;
    for (auto& [key, value] : validFiles.items()) {
        std::filesystem::path fullPath = path / key;
        mapFiles.emplace_back(fullPath.string());
    }
    assert(mapFiles.size() != 0);
    
    if(cfg.params.datasetInitOptions == DatasetInitOptions::FirstN)
    {
        assert(cfg.numWorlds <= mapFiles.size());
        mapFiles.resize(cfg.numWorlds);
    }
    else if(cfg.params.datasetInitOptions == DatasetInitOptions::RandomN)
    {
        assert(cfg.numWorlds <= mapFiles.size());
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(mapFiles.begin(), mapFiles.end(), g);
        mapFiles.resize(cfg.numWorlds);
    }
    else if(cfg.params.datasetInitOptions == DatasetInitOptions::PadN)
    {
        assert(cfg.numWorlds >= mapFiles.size());
        for(int i = 0; i < cfg.numWorlds; i++)
        {
            mapFiles.push_back(mapFiles[0]);
        }
    }
    else if(cfg.params.datasetInitOptions == DatasetInitOptions::ExactN)
    {
        // Do nothing
    }
    else
    {
        FATAL("Invalid datasetInitOptions");
    }

    return mapFiles;
}

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg,
    const viz::VizECSBridge *viz_bridge)
{
    Sim::Config sim_cfg {
        viz_bridge != nullptr,
        mgr_cfg.autoReset,
        0, // kMaxAgentCount
        0 // kMaxRoadEntityCount
    };

    std::vector<std::string> mapFiles = getMapFiles(mgr_cfg);

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

        Parameters* paramsDevicePtr = (Parameters*)cu::allocGPU(sizeof(Parameters));
        REQ_CUDA(cudaMemcpy(paramsDevicePtr, &(mgr_cfg.params), sizeof(Parameters), cudaMemcpyHostToDevice));
        
        int64_t worldIdx{0};
    
        for (auto const &mapFile : mapFiles)
        {
            auto [map_, mapCounts] = MapReader::parseAndWriteOut(mapFile, mgr_cfg.execMode, mgr_cfg.params.polylineReductionThreshold);
            world_inits[worldIdx++] = WorldInit{episode_mgr, phys_obj_mgr,
                                                viz_bridge, map_, paramsDevicePtr};
            sim_cfg.kMaxAgentCount = std::max(mapCounts.first, sim_cfg.kMaxAgentCount);
            sim_cfg.kMaxRoadEntityCount = std::max(mapCounts.second, sim_cfg.kMaxRoadEntityCount);
        }

        
        assert(worldIdx == static_cast<int64_t>(mgr_cfg.numWorlds));
        // Bounds on the maxagent and maxroadentity counts.
        assert(sim_cfg.kMaxAgentCount <= consts::kMaxAgentCount);
        assert(sim_cfg.kMaxRoadEntityCount <= consts::kMaxRoadEntityCount);
        assert(sim_cfg.kMaxAgentCount > 0);
        assert(sim_cfg.kMaxRoadEntityCount > 0);

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
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        });

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
            std::make_pair(sim_cfg.kMaxAgentCount, sim_cfg.kMaxRoadEntityCount)
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
    
        for (auto const &mapFile : mapFiles)
        {
            auto [map_, mapCounts] = MapReader::parseAndWriteOut(mapFile, mgr_cfg.execMode, mgr_cfg.params.polylineReductionThreshold);
            world_inits[worldIdx++] = WorldInit{episode_mgr, phys_obj_mgr,
                                                viz_bridge, map_, &(mgr_cfg.params)};
            sim_cfg.kMaxAgentCount = std::max(mapCounts.first, sim_cfg.kMaxAgentCount);
            sim_cfg.kMaxRoadEntityCount = std::max(mapCounts.second, sim_cfg.kMaxRoadEntityCount);
        }

        assert(worldIdx == static_cast<int64_t>(mgr_cfg.numWorlds));
        // Bounds on the maxagent and maxroadentity counts.
        assert(sim_cfg.kMaxAgentCount <= consts::kMaxAgentCount);
        assert(sim_cfg.kMaxRoadEntityCount <= consts::kMaxRoadEntityCount);
        assert(sim_cfg.kMaxAgentCount > 0);
        assert(sim_cfg.kMaxRoadEntityCount > 0);

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
            std::make_pair(sim_cfg.kMaxAgentCount, sim_cfg.kMaxRoadEntityCount)
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

Manager::Manager(const Config &cfg,
                 const viz::VizECSBridge *viz_bridge)
    : impl_(Impl::init(cfg, viz_bridge))
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
                               Tensor::ElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   1,
                               });
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, Tensor::ElementType::Float32,
        {
            impl_->cfg.numWorlds,
            impl_->agentRoadCounts.first,
            3, // Num_actions
        });
}

Tensor Manager::bicycleModelTensor() const
{
    return impl_->exportTensor(ExportID::BicycleModel, Tensor::ElementType::Float32,
        {
            impl_->cfg.numWorlds,
            impl_->agentRoadCounts.first,
            BicycleModelExportSize, // Number of states for the bicycle model
        });
}


Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   1,
                               });
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, Tensor::ElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   1,
                               });
}

Tensor Manager::selfObservationTensor() const
{
    return impl_->exportTensor(ExportID::SelfObservation,
                               Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   SelfObservationExportSize 
                               });
}

Tensor Manager::mapObservationTensor() const
{
    return impl_->exportTensor(ExportID::MapObservation,
                               Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.second,
                                   MapObservationExportSize
                               });
}

Tensor Manager::partnerObservationsTensor() const
{
    return impl_->exportTensor(ExportID::PartnerObservations,
                               Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   consts::kMaxAgentCount-1,
                                   PartnerObservationExportSize,
                               });
}

Tensor Manager::agentMapObservationsTensor() const
{
    return impl_->exportTensor(ExportID::AgentMapObservations,
                               Tensor::ElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   consts::kMaxRoadEntityCount,
                                   AgentMapObservationExportSize,
                               });

}

Tensor Manager::lidarTensor() const
{
    return impl_->exportTensor(ExportID::Lidar, Tensor::ElementType::Float32,
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
                               Tensor::ElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->agentRoadCounts.first,
                                   1,
                               });
}

Tensor Manager::shapeTensor() const {
    return impl_->exportTensor(ExportID::Shape, Tensor::ElementType::Int32,
                               {impl_->cfg.numWorlds, 2});
}

Tensor Manager::controlledStateTensor() const {
    return impl_->exportTensor(ExportID::ControlledState, Tensor::ElementType::Int32,
                               {impl_->cfg.numWorlds,impl_->agentRoadCounts.first, 1});
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
        impl_->agentActionsBuffer + world_idx * impl_->agentRoadCounts.first + agent_idx;

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
}
