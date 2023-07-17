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
constexpr inline CountT episodeLen = 240;

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);
    phys::RigidBodyPhysicsSystem::registerTypes(registry);
    viz::VizRenderingSystem::registerTypes(registry);

    registry.registerComponent<Action>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<OtherAgents>();
    registry.registerComponent<ToOtherAgents>();
    registry.registerComponent<ToButtons>();
    registry.registerComponent<ToGoal>();
    registry.registerComponent<OpenState>();

    registry.registerComponent<Lidar>();
    registry.registerComponent<Seed>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<GlobalDebugPositions>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<DynamicObject>();
    registry.registerArchetype<WallObject>();
    registry.registerArchetype<ButtonObject>();

    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<Agent, ToOtherAgents>(
        (uint32_t)ExportID::ToOtherAgents);
    registry.exportColumn<Agent, ToButtons>(
        (uint32_t)ExportID::ToButtons);
    registry.exportColumn<Agent, ToGoal>(
        (uint32_t)ExportID::ToGoal);
    registry.exportColumn<Agent, Lidar>(
        (uint32_t)ExportID::Lidar);
    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);
    registry.exportColumn<Agent, Seed>(
        (uint32_t)ExportID::Seed);
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
    int32_t level = reset.reset;
    if (ctx.data().autoReset && ctx.data().curEpisodeStep == episodeLen - 1) {
        level = 1;
    }

    if (level != 0) {
        reset.reset = 0;

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

static inline PolarCoord xyToPolar(Vector3 v)
{
    Vector2 xy { v.x, v.y };
    return PolarCoord {
        .r = xy.length(),
        .theta = atan2f(xy.y, xy.x),
    };
}

inline void collectObservationsSystem(Engine &ctx,
                                      Position pos,
                                      const OtherAgents &other_agents,
                                      ToOtherAgents &to_other_agents,
                                      ToButtons &to_buttons,
                                      ToGoal &to_goal)
{
#pragma unroll
    for (CountT i = 0; i < consts::numAgents - 1; i++) {
        Entity other = other_agents.e[i];

        Vector3 other_pos = ctx.get<Position>(other);
        to_other_agents.obs[i] = xyToPolar(other_pos - pos);
    }

    // Compute polar coords to buttons
    CountT button_idx = 0;
    for (; button_idx < ctx.data().leafCount; button_idx++) {
        Room &room = ctx.data().rooms[ctx.data().leafs[button_idx]];

        Vector2 button_pos = (room.button.start + room.button.end) * 0.5f;
        to_buttons.obs[button_idx] = xyToPolar(
            Vector3 { button_pos.x, button_pos.y, 0 } - pos);
    }

    for (; button_idx < consts::maxRooms; button_idx++) {
        // FIXME: is this a good invalid output?
        to_buttons.obs[button_idx] = { 0.f, 2.f * math::pi };
    }

    { // Compute polar coords to goal
        Room &dst = ctx.data().rooms[ctx.data().dstRoom];
        Vector2 dst_button_pos = (dst.button.start + dst.button.end) * 0.5f;
        to_goal.obs = xyToPolar(
            Vector3 { dst_button_pos.x, dst_button_pos.y, 0 }  - pos);
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
        float theta = 2.f * math::pi * (
            float(idx) / float(consts::numLidarSamples));
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

    if (idx < consts::numLidarSamples) {
        traceRay(idx);
    }
#else
    for (CountT i = 0; i < consts::numLidarSamples; i++) {
        traceRay(i);
    }
#endif
}

inline void rewardSystem(Engine &ctx,
                         Position pos,
                         Reward &out_reward,
                         Done &done)
{
    Room &dst = ctx.data().rooms[ctx.data().dstRoom];
    Vector2 dstButtonPos = (dst.button.start + dst.button.end) * 0.5f;
    Vector2 diff = dstButtonPos - Vector2 {pos .x, pos.y };

    CountT cur_step = ctx.data().curEpisodeStep;

    if (cur_step == 0) {
        out_reward.v = 0.f;
        done.v = 0.f;
    }

    float reward = 0.0f;
    if (diff.length2() < BUTTON_WIDTH*BUTTON_WIDTH) {
        reward = 100.0f;
    }
    else {
        reward = -0.05f;
    }

    out_reward.v = reward;

    if (cur_step == episodeLen - 1) {
        done.v = 1;
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

    auto reset_door_sys = builder.addToGraph<ParallelForNode<Engine, resetDoorStateSystem,
        OpenState>>({move_sys});

    auto door_control_sys = builder.addToGraph<ParallelForNode<Engine, doorControlSystem,
        Position, Action>>({reset_door_sys});

    auto set_door_pos_sys = builder.addToGraph<ParallelForNode<Engine, setDoorPositionSystem,
        Position, Velocity, OpenState>>({door_control_sys});

    auto broadphase_setup_sys = phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(builder,
        {set_door_pos_sys});

    auto substep_sys = phys::RigidBodyPhysicsSystem::setupSubstepTasks(builder,
        {broadphase_setup_sys}, numPhysicsSubsteps);

    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, viz::VizCamera>>(
            {substep_sys});

    auto sim_done = agent_zero_vel;

    sim_done = phys::RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {sim_done});

    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
         rewardSystem, Position, Reward, Done>>({sim_done});

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
            Position,
            OtherAgents,
            ToOtherAgents,
            ToButtons,
            ToGoal
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
      episodeMgr(init.episodeMgr)
{
    CountT max_total_entities = 100;

    phys::RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT,
         numPhysicsSubsteps, -9.8 * math::up, max_total_entities,
         50 * 20, 10);

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
    ctx.data().leafCount = 0;

    createAgents(ctx);
    createFloor(ctx);

    // Creates the wall entities and placess the agents into the source room
    resetEnvironment(ctx);
    generateEnvironment(ctx);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Config, WorldInit);

}
