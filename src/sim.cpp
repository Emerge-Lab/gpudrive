#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "level_gen.hpp"
#include "geo_gen.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace GPUHideSeek {

constexpr inline float wallSpeed = 4.0f;
constexpr inline float deltaT = 0.075;
constexpr inline CountT numPhysicsSubsteps = 4;
constexpr inline CountT numPrepSteps = 96;
constexpr inline CountT episodeLen = 240;

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);
    phys::RigidBodyPhysicsSystem::registerTypes(registry);
    viz::VizRenderingSystem::registerTypes(registry);

    registry.registerComponent<Action>();
    registry.registerComponent<RelativeAgentObservations>();
    registry.registerComponent<RelativeButtonObservations>();
    registry.registerComponent<RelativeDestinationObservations>();
    registry.registerComponent<OpenState>();

    registry.registerComponent<Lidar>();
    registry.registerComponent<Seed>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<GlobalDebugPositions>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<DynamicObject>();
    registry.registerArchetype<WallObject>();
    registry.registerArchetype<ButtonObject>();

    registry.exportSingleton<WorldReset>((uint32_t)ExportIDs::Reset);

    registry.exportColumn<Agent, Action>((uint32_t)ExportIDs::Action);
    registry.exportColumn<Agent, RelativeAgentObservations>(5);
    registry.exportColumn<Agent, RelativeButtonObservations>(6);
    registry.exportColumn<Agent, RelativeDestinationObservations>(7);
    registry.exportColumn<Agent, Lidar>(10);
    registry.exportColumn<Agent, Seed>(11);
}

static inline void resetEnvironment(Engine &ctx)
{
    ctx.data().curEpisodeStep = 0;

    if (ctx.data().enableVizRender) {
        viz::VizRenderingSystem::reset(ctx);
    }

    phys::RigidBodyPhysicsSystem::reset(ctx);

#if 1
    for (CountT i = 0; i < ctx.data().leafCount; ++i) {
        ctx.destroyEntity(ctx.data().rooms[ctx.data().leafs[i]].buttonEntity);
    }
#endif

    // Destroy the wall entities (door entities count as walls)
    for (CountT i = 0; i < ctx.data().numWalls; ++i) {
        Entity e = ctx.data().walls[i];
        ctx.destroyEntity(e);
    }

    for (CountT i = 0; i < ctx.data().numDoors; ++i) {
        Entity e = ctx.data().doors[i];
        ctx.destroyEntity(e);
    }

    ctx.data().numDoors = 0;
    ctx.data().roomCount = 0;
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t level = reset.resetLevel;
    if (ctx.data().autoReset && ctx.data().curEpisodeStep == episodeLen - 1) {
        level = 1;
    }

    if (level != 0) {
        reset.resetLevel = 0;

        resetEnvironment(ctx);
        generateEnvironment(ctx);

        if (ctx.data().enableVizRender) {
            viz::VizRenderingSystem::markEpisode(ctx);
        }
    } else {
        ctx.data().curEpisodeStep += 1;
    }
}

// Provide forces for entities which are controlled by actions (the two agents in this case).
inline void movementSystem(Engine &,
                           Action &action, 
                           Rotation &rot, 
                           ExternalForce &external_force,
                           ExternalTorque &external_torque)
{
    constexpr CountT discrete_action_buckets = 11;
    constexpr CountT half_buckets = discrete_action_buckets / 2;
    constexpr float discrete_move_max =  110;
    constexpr float move_delta_per_bucket = discrete_move_max / half_buckets;
    constexpr float discrete_turn_max = 30;
    constexpr float turn_delta_per_bucket = discrete_turn_max / half_buckets;

    Quat cur_rot = rot;

    float f_x = move_delta_per_bucket * action.x;
    float f_y = move_delta_per_bucket * action.y;
    float t_z = turn_delta_per_bucket * action.r;

    external_force = cur_rot.rotateVec({ f_x, f_y, 0 });
    external_torque = Vector3 { 0, 0, t_z };
}

// Resets doors to closed temporarily
inline void resetDoorStateSystem(Engine &, OpenState &open_state)
{
    open_state.isOpen = false;
}

// Sets door open state given where entities are (loops through entities).
inline void doorControlSystem(Engine &ctx, Position &pos, Action &action)
{
    (void)action;

    Room *room = containedRoom({ pos.x, pos.y }, ctx.data().rooms);

    // If the button is pressed, set the doors of this room to be open
    if (isPressingButton({ pos.x, pos.y }, room)) {
        for (CountT i = 0; i < room->doorCount; ++i) {
            CountT doorIdx = room->doors[i];
            ctx.get<OpenState>(ctx.data().doors[doorIdx]).isOpen = true;
        }
    }
}

inline void setDoorPositionSystem(Engine &, Position &pos, Velocity &vel, OpenState &open_state)
{
    if (open_state.isOpen) {
        // Put underground

        if (pos.z > -3.0f)
            pos.z += -wallSpeed * deltaT;
    }
    else if (pos.z < 0.0f) {
        // Put back on surface
        pos.z += wallSpeed * deltaT;
    }
    
    if (pos.z >= 0.0f) {
        pos.z = 0.0f;
    }
}

inline void agentZeroVelSystem(Engine &,
                               Velocity &vel,
                               viz::VizCamera &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = fminf(vel.linear.z, 0);

    vel.angular = Vector3::zero();
}

inline void collectObservationsSystem(Engine &ctx,
                                      Entity agent_e,
                                      RelativeAgentObservations &agent_obs,
                                      RelativeButtonObservations &button_obs,
                                      RelativeDestinationObservations &des_obs)
{
    Vector2 agentPosV2 = { ctx.get<Position>(agent_e).x, ctx.get<Position>(agent_e).y };

    // Get relative agent observations
    if (agent_e == ctx.data().agents[0]) {
        Vector3 relAgentObsV3 = ctx.get<Position>(ctx.data().agents[1]) - ctx.get<Position>(agent_e);
        Vector2 relAgentObs = { relAgentObsV3.x, relAgentObsV3.y };
        agent_obs.obs[0] = { relAgentObs.length(), atan(relAgentObs.y / relAgentObs.x) };
    }

    // Get relative button observations
    CountT buttonCount = 0;
    for (; buttonCount < ctx.data().leafCount; ++buttonCount) {
        Room &room = ctx.data().rooms[ctx.data().leafs[buttonCount]];

        Vector2 buttonPos = (room.button.start + room.button.end) * 0.5f;
        Vector2 diff = buttonPos - agentPosV2;
        button_obs.obs[buttonCount] = { diff.length(), atan(diff.y / diff.x) };
    }

    for (; buttonCount < consts::maxRooms; ++buttonCount) {
        button_obs.obs[buttonCount] = { FLT_MAX, 0.0f };
    }

    { // Get relative to destination observation
        Room &dst = ctx.data().rooms[ctx.data().dstRoom];
        Vector2 dstButtonPos = (dst.button.start + dst.button.end) * 0.5f;
        Vector2 diff = dstButtonPos - agentPosV2;
        des_obs.obs = { diff.length(), atan(diff.y / diff.x) };
    }
}

inline void lidarSystem(Engine &ctx,
                        Entity e,
                        Lidar &lidar)
{
    Vector3 pos = ctx.get<Position>(e);
    Quat rot = ctx.get<Rotation>(e);
    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 agent_fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);

    auto traceRay = [&](int32_t idx) {
        float theta = 2.f * math::pi * (float(idx) / float(30));
        float x = cosf(theta);
        float y = sinf(theta);

        Vector3 ray_dir = (x * right + y * agent_fwd).normalize();

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(pos , ray_dir, &hit_t, &hit_normal, 200.f);

        if (hit_entity == Entity::none()) {
            lidar.depth[idx] = 0.f;
        } else {
            lidar.depth[idx] = hit_t;
        }
    };


#ifdef MADRONA_GPU_MODE
    int32_t idx = threadIdx.x % 32;

    if (idx < 30) {
        traceRay(idx);
    }
#else
    for (int32_t i = 0; i < 30; i++) {
        traceRay(i);
    }
#endif
}

inline void rewardSystem(Engine &ctx,
                         Entity e,
                         Action &)
{
    Loc l = ctx.loc(e);

    Vector2 agentPosV2 = { ctx.get<Position>(e).x, ctx.get<Position>(e).y };

    Room &dst = ctx.data().rooms[ctx.data().dstRoom];
    Vector2 dstButtonPos = (dst.button.start + dst.button.end) * 0.5f;
    Vector2 diff = dstButtonPos - agentPosV2;


    CountT cur_step = ctx.data().curEpisodeStep;

    if (cur_step == 0) {
        ctx.data().doneBuffer[l.row] = 0;
        ctx.data().rewardBuffer[l.row] = 0.f;
    }

    float reward = 0.0f;
    if (diff.length2() < BUTTON_WIDTH*BUTTON_WIDTH) {
        reward = 100.0f;
    }
    else {
        reward = -0.05f;
    }

    ctx.data().rewardBuffer[l.row] += reward;

    if (cur_step == episodeLen - 1) {
        ctx.data().doneBuffer[l.row] = 1;
    }
}

#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif


void Sim::setupTasks(TaskGraph::Builder &builder, const Config &cfg)
{
    auto move_sys = builder.addToGraph<ParallelForNode<Engine, movementSystem,
        Action, Rotation, ExternalForce, ExternalTorque>>({});

#if 1
    auto reset_door_sys = builder.addToGraph<ParallelForNode<Engine, resetDoorStateSystem,
        OpenState>>({move_sys});

    auto door_control_sys = builder.addToGraph<ParallelForNode<Engine, doorControlSystem,
        Position, Action>>({reset_door_sys});

    auto set_door_pos_sys = builder.addToGraph<ParallelForNode<Engine, setDoorPositionSystem,
        Position, Velocity, OpenState>>({door_control_sys});

    auto broadphase_setup_sys = phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(builder,
        {set_door_pos_sys});

#else
    auto broadphase_setup_sys = phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(builder,
        {move_sys});
#endif

    auto substep_sys = phys::RigidBodyPhysicsSystem::setupSubstepTasks(builder,
        {broadphase_setup_sys}, numPhysicsSubsteps);

    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, viz::VizCamera>>(
            {substep_sys});

    auto sim_done = agent_zero_vel;

    sim_done = phys::RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {sim_done});

    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
         rewardSystem, Entity, Action>>({sim_done});

    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem, WorldReset>>({reward_sys});

    auto clearTmp = builder.addToGraph<ResetTmpAllocNode>({reset_sys});

#ifdef MADRONA_GPU_MODE
    // FIXME: these 3 need to be compacted, but sorting is unnecessary
    auto sort_dyn_agent = queueSortByWorld<DynAgent>(builder, {sort_cam_agent});
    auto sort_objects = queueSortByWorld<DynamicObject>(builder, {sort_dyn_agent});
    auto sort_agent_iface =
        queueSortByWorld<AgentInterface>(builder, {sort_objects});
    auto reset_finish = sort_agent_iface;
#else
    auto reset_finish = clearTmp;
#endif

    if (cfg.enableViewer) {
        viz::VizRenderingSystem::setupTasks(builder, {reset_finish});
    }

#ifdef MADRONA_GPU_MODE
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({reset_finish});
    (void)recycle_sys;
#endif

    auto collect_observations = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Entity,
            RelativeAgentObservations,
            RelativeButtonObservations,
            RelativeDestinationObservations
        >>({reset_finish});

#ifdef MADRONA_GPU_MODE
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 32, 1,
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine,
        lidarSystem,
#endif
            Entity,
            Lidar
        >>({reset_finish});

    (void)lidar;
    (void)collect_observations;
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      rewardBuffer(init.rewardBuffer),
      doneBuffer(init.doneBuffer)
{
    CountT max_total_entities = init.maxEntitiesPerWorld + 100;

    phys::RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT,
         numPhysicsSubsteps, -9.8 * math::up, max_total_entities,
         50 * 20, 10);

    minEpisodeEntities = init.minEntitiesPerWorld;
    maxEpisodeEntities = init.maxEntitiesPerWorld;

    curEpisodeStep = 0;

    enableBatchRender = cfg.enableBatchRender;
    enableVizRender = cfg.enableViewer;

    if (enableVizRender) {
        viz::VizRenderingSystem::init(ctx, init.vizBridge);
    }

    autoReset = cfg.autoReset;

    rooms = (Room *)rawAlloc(sizeof(Room) * consts::maxRooms);
    leafs = (CountT *)rawAlloc(sizeof(CountT) * consts::maxRooms);
    walls = (Entity *)rawAlloc(sizeof(Entity) * consts::maxRooms * consts::maxDoorsPerRoom);
    doors = (Entity *)rawAlloc(sizeof(Entity) * consts::maxRooms * consts::maxDoorsPerRoom);

    ctx.data().numDoors = 0;
    ctx.data().numWalls = 0;
    ctx.data().roomCount = 0;

    createAgents(ctx);
    createFloor(ctx);

    // Creates the wall entities and placess the agents into the source room
    resetEnvironment(ctx);
    generateEnvironment(ctx);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Config, WorldInit);

}
