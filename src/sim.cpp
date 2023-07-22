#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "level_gen.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace GPUHideSeek {

constexpr inline float wallSpeed = 12.0f;
constexpr inline float deltaT = 0.075;
constexpr inline CountT numPhysicsSubsteps = 4;
constexpr inline int32_t episodeLen = 100;

// Register all the ECS components and archetypes that will be
// use in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);
    phys::RigidBodyPhysicsSystem::registerTypes(registry);
    viz::VizRenderingSystem::registerTypes(registry);

    registry.registerComponent<Action>();
    registry.registerComponent<PositionObservation>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<Progress>();
    registry.registerComponent<OtherAgents>();
    registry.registerComponent<ToOtherAgents>();
    registry.registerComponent<ToRoomEntities>();
    registry.registerComponent<ButtonState>();
    registry.registerComponent<OpenState>();
    registry.registerComponent<DoorProperties>();

    registry.registerComponent<Lidar>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<LevelState>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();
    registry.registerArchetype<DoorEntity>();
    registry.registerArchetype<ButtonEntity>();

    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<Agent, PositionObservation>(
        (uint32_t)ExportID::PositionObservation);
    registry.exportColumn<Agent, ToOtherAgents>(
        (uint32_t)ExportID::ToOtherAgents);
    registry.exportColumn<Agent, ToRoomEntities>(
        (uint32_t)ExportID::ToRoomEntities);
    registry.exportColumn<Agent, Lidar>(
        (uint32_t)ExportID::Lidar);
    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);
}

static inline void cleanupWorld(Engine &ctx)
{
    // Destroy current level entities
    LevelState &level = ctx.singleton<LevelState>();
    for (CountT i = 0; i < consts::numRooms; i++) {
        Room &room = level.rooms[i];
        for (CountT j = 0; j < consts::maxEntitiesPerRoom; j++) {
            if (room.entities[j].type != DynEntityType::None) {
                ctx.destroyEntity(room.entities[j].e);
            }
        }

        ctx.destroyEntity(room.walls[0]);
        ctx.destroyEntity(room.walls[1]);
        ctx.destroyEntity(room.door);
    }
}

static inline void initWorld(Engine &ctx)
{
    ctx.data().curEpisodeStep = 0;

    if (ctx.data().enableVizRender) {
        viz::VizRenderingSystem::reset(ctx);
    }

    phys::RigidBodyPhysicsSystem::reset(ctx);

    // Assign a new episode ID
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    int32_t episode_idx = episode_mgr.curEpisode.fetch_add<sync::relaxed>(1);
    ctx.data().rng = RNG::make(episode_idx);
    ctx.data().curEpisodeIdx = episode_idx;

    generateWorld(ctx);
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t should_reset = reset.reset;
    if (ctx.data().autoReset) {
        for (CountT i = 0; i < consts::numAgents; i++) {
            Entity agent = ctx.data().agents[i];
            Done done = ctx.get<Done>(agent);
            if (done.v) {
                should_reset = 1;
            }
        }
    }

    if (should_reset != 0) {
        reset.reset = 0;

        cleanupWorld(ctx);
        initWorld(ctx);

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
    constexpr CountT discrete_action_buckets = consts::numMoveBuckets;
    constexpr CountT half_buckets = discrete_action_buckets / 2;
    constexpr float discrete_move_max = 300;
    constexpr float move_delta_per_bucket = discrete_move_max / half_buckets;
    constexpr float discrete_turn_max = 40;
    constexpr float turn_delta_per_bucket = discrete_turn_max / half_buckets;

    Quat cur_rot = rot;

    float f_x = move_delta_per_bucket * (action.x - (int32_t)half_buckets);
    float f_y = move_delta_per_bucket * (action.y - (int32_t)half_buckets);
    float t_z = turn_delta_per_bucket * (action.r - (int32_t)half_buckets);

    external_force = cur_rot.rotateVec({ f_x, f_y, 0 });
    external_torque = Vector3 { 0, 0, t_z };
}

static bool isPressingButton(madrona::math::Vector3 agent_pos,
                             madrona::math::Vector3 button_pos)
{
    constexpr float press_radius = 1.5f;
    constexpr float button_dim = consts::buttonWidth + 0.05f;

    float circle_dist_x = fabsf(agent_pos.x - button_pos.x);
    float circle_dist_y = fabsf(agent_pos.y - button_pos.y);

    if (circle_dist_x > (button_dim / 2 + press_radius) ||
            circle_dist_y > (button_dim / 2 + press_radius)) {
        return false;
    }
    

    if (circle_dist_x <= (button_dim / 2) ||
            circle_dist_y <= (button_dim / 2)) {
        return true;
    }

    float corner_dist_x = circle_dist_x - button_dim;
    float corner_dist_y = circle_dist_y - button_dim;

    float corner_dist2 = corner_dist_x * corner_dist_x +
        corner_dist_y * corner_dist_y;

    return corner_dist2 <= press_radius * press_radius;
}

inline void setDoorPositionSystem(Engine &,
                                  Position &pos,
                                  OpenState &open_state)
{
    if (open_state.isOpen) {
        // Put underground

        if (pos.z > -4.5f)
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


// Checks if button is pressed and update state accordingly
inline void buttonSystem(Engine &ctx,
                         Position pos,
                         ButtonState &state)
{
    bool button_pressed = false;
#pragma unroll
    for (CountT i = 0; i < consts::numAgents; i++) {
        Entity agent = ctx.data().agents[i];
        Vector3 agent_pos = ctx.get<Position>(agent);

        if (isPressingButton(agent_pos, pos)) {
            button_pressed = true;
        }
    }
    state.isPressed = button_pressed;
}

inline void doorOpenSystem(Engine &ctx,
                           OpenState &open_state,
                           const DoorProperties &props)
{
    bool all_pressed = true;
    for (int32_t i = 0; i < props.numButtons; i++) {
        Entity button = props.buttons[i];
        all_pressed = all_pressed  && ctx.get<ButtonState>(button).isPressed;
    }

    if (all_pressed) {
        open_state.isOpen = true;
    } else if (!props.isPersistent) {
        open_state.isOpen = false;
    }
}

inline void agentZeroVelSystem(Engine &,
                               Velocity &vel,
                               Action &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = fminf(vel.linear.z, 0);

    vel.angular = Vector3::zero();
}

static inline float distObs(float v)
{
    return v / consts::worldLength;
}

static inline float globalPosObs(float v)
{
    return v / consts::worldLength;
}

static inline float angleObs(float v)
{
    return v / math::pi;
}

static inline PolarObservation xyToPolar(Vector3 v)
{
    Vector2 xy { v.x, v.y };

    float r = xy.length();

    // Note that this is angle off y-forward
    float theta = atan2f(xy.x, xy.y);

    return PolarObservation {
        .r = distObs(r),
        .theta = angleObs(theta),
    };
}

static inline float encodeDynType(DynEntityType type)
{
    return (float)type / (float)DynEntityType::NumTypes;
}

static inline float computeZAngle(Quat q)
{
    float siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
    return atan2f(siny_cosp, cosy_cosp);
}

inline void collectObservationsSystem(Engine &ctx,
                                      Position pos,
                                      Rotation rot,
                                      PositionObservation &pos_obs,
                                      const OtherAgents &other_agents,
                                      ToOtherAgents &to_other_agents,
                                      ToRoomEntities &to_room_ents)
{
    CountT cur_room_idx = CountT(pos.y / consts::roomLength);
    cur_room_idx = std::max(CountT(0), 
        std::min(consts::numRooms - 1, cur_room_idx));

    pos_obs.roomX = pos.x / (consts::worldWidth / 2.f);
    pos_obs.roomY = (pos.y - cur_room_idx * consts::roomLength) /
        consts::roomLength;
    pos_obs.globalX = globalPosObs(pos.x);
    pos_obs.globalY = globalPosObs(pos.y);
    pos_obs.globalZ = globalPosObs(pos.z);
    pos_obs.theta = angleObs(computeZAngle(rot));

    Quat to_view = rot.inv();

#pragma unroll
    for (CountT i = 0; i < consts::numAgents - 1; i++) {
        Entity other = other_agents.e[i];

        Vector3 other_pos = ctx.get<Position>(other);
        Vector3 to_other = other_pos - pos;

        to_other_agents.obs[i] = xyToPolar(to_view.rotateVec(to_other));
    }

    const LevelState &level = ctx.singleton<LevelState>();
    const Room &room = level.rooms[cur_room_idx];

    for (CountT i = 0; i < consts::maxEntitiesPerRoom; i++) {
        DynEntityState entity_info = room.entities[i];
        EntityObservation ob;
        ob.encodedType = encodeDynType(entity_info.type);

        if (entity_info.type == DynEntityType::None) {
            ob.polar = { 0.f, 1.f };
        } else {
            Vector3 entity_pos = ctx.get<Position>(entity_info.e);
            Vector3 to_entity = entity_pos - pos;
            ob.polar = xyToPolar(to_view.rotateVec(to_entity));
        }

        to_room_ents.obs[i] = ob;
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
            float(idx) / float(consts::numLidarSamples)) + math::pi / 2.f;
        float x = cosf(theta);
        float y = sinf(theta);

        Vector3 ray_dir = (x * right + y * agent_fwd).normalize();

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(pos + 0.5f * math::up, ray_dir, &hit_t,
                         &hit_normal, 200.f);

        if (hit_entity == Entity::none()) {
            lidar.depth[idx] = 0.f;
        } else {
            lidar.depth[idx] = distObs(hit_t);
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

inline void rewardSystem(Engine &,
                         Position pos,
                         Progress &progress,
                         Reward &out_reward)
{
    constexpr float progress_reward = 0.1f;
    constexpr float slack_reward = -0.01f;

    int32_t new_progress = int32_t(pos.y / consts::distancePerProgress);

    float reward;
    if (new_progress > progress.numProgressIncrements) {
        reward = progress_reward *
            (new_progress - progress.numProgressIncrements);
        progress.numProgressIncrements = new_progress;
    } else {
        reward = slack_reward;
    }

    out_reward.v = reward;
}

// Each agent gets half the reward of the other agent in
// order to encourage them to cooperate. Need to cache
// the rewards in this system to avoid a race condition
// when reading the rewards in the next system
inline void gatherPartnerRewardSystem(Engine &ctx,
                                      OtherAgents &others)
{
    for (CountT i = 0; i < consts::numAgents - 1; i++) {
        Entity other = others.e[i];
        others.rewards[i] = ctx.get<Reward>(other).v / 2.f;
    }
}

inline void assignPartnerRewardSystem(Engine &,
                                      OtherAgents &others,
                                      Reward &reward)
{
    float other_rewards = 0.f;
    for (CountT i = 0; i < consts::numAgents - 1; i++) {
        other_rewards += others.rewards[i];
    }

    reward.v += other_rewards / float(consts::numAgents - 1);
}

// Notify training that an episode has completed by
// setting done = 1 on the final step of the episode
inline void doneSystem(Engine &ctx,
                       Done &done)
{
    int32_t cur_step = ctx.data().curEpisodeStep;
    if (cur_step == 0) {
        done.v = 0;
    } else if (cur_step == episodeLen -1) {
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
    // Turn policy actions into movement
    auto move_sys = builder.addToGraph<ParallelForNode<Engine,
        movementSystem,
            Action,
            Rotation,
            ExternalForce,
            ExternalTorque
        >>({});

    // Scripted door behavior
    auto set_door_pos_sys = builder.addToGraph<ParallelForNode<Engine,
        setDoorPositionSystem,
            Position,
            OpenState
        >>({move_sys});

    // Physics systems
    auto broadphase_setup_sys = phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(
        builder, {set_door_pos_sys});

    auto substep_sys = phys::RigidBodyPhysicsSystem::setupSubstepTasks(builder,
        {broadphase_setup_sys}, numPhysicsSubsteps);

    // Improve controllability of agents by setting their velocity to 0
    // after physics is done.
    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, Action>>(
            {substep_sys});

    // Finalize physics subsystem work
    auto phys_done = phys::RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {agent_zero_vel});

    // Check buttons
    auto button_sys = builder.addToGraph<ParallelForNode<Engine,
        buttonSystem,
            Position,
            ButtonState
        >>({phys_done});

    // Set door to start opening if button conditions are met
    auto door_open_sys = builder.addToGraph<ParallelForNode<Engine,
        doorOpenSystem,
            OpenState,
            DoorProperties
        >>({button_sys});

    // Compute initial reward now that physics has updated the world state
    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
         rewardSystem,
            Position,
            Progress,
            Reward
        >>({door_open_sys});

    // Read partner's reward
    auto gather_partner_reward_sys = builder.addToGraph<ParallelForNode<Engine,
         gatherPartnerRewardSystem,
            OtherAgents
        >>({reward_sys});

    // Assign partner's reward
    auto assign_partner_reward_sys = builder.addToGraph<ParallelForNode<Engine,
         assignPartnerRewardSystem,
            OtherAgents,
            Reward
        >>({gather_partner_reward_sys});

    // Check if the episode is over
    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        doneSystem,
            Done
        >>({assign_partner_reward_sys});

    // Conditionally reset the world if the episode is over
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem,
            WorldReset
        >>({done_sys});

    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>({reset_sys});
    (void)clear_tmp;

#ifdef MADRONA_GPU_MODE
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({reset_sys});
    (void)recycle_sys;
#endif

    // This second BVH build is a limitation of the current taskgraph API.
    // It's only necessary if the world was reset, but we don't have a way
    // to conditionally queue taskgraph nodes yet.
    auto post_reset_broadphase = phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(
        builder, {reset_sys});

    // Finally, collect observations for the next step.
    auto collect_obs = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Position,
            Rotation,
            PositionObservation,
            OtherAgents,
            ToOtherAgents,
            ToRoomEntities
        >>({post_reset_broadphase});

#ifdef MADRONA_GPU_MODE
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 32, 1,
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine,
        lidarSystem,
#endif
            Entity,
            Lidar
        >>({post_reset_broadphase});

    if (cfg.enableViewer) {
        viz::VizRenderingSystem::setupTasks(builder, {reset_sys});
    }

#ifdef MADRONA_GPU_MODE
    // Sort entities, again, this could be conditional on reset.
    auto sort_agents = queueSortByWorld<Agent>(
        builder, {lidar, collect_obs});
    auto sort_phys_objects = queueSortByWorld<PhysicsEntity>(
        builder, {sort_agents});
    auto sort_buttons = queueSortByWorld<ButtonEntity>(
        builder, {sort_phys_objects});
    auto sort_walls = queueSortByWorld<DoorEntity>(
        builder, {sort_buttons});
    (void)sort_walls;
#else
    (void)lidar;
    (void)collect_obs;
#endif
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr)
{
    // Currently the physics system needs an upper bound on the number of
    // entities that will be stored in the BVH. We plan to fix this in
    // a future release.
    constexpr CountT max_total_entities = consts::numAgents +
        consts::numRooms * (consts::maxEntitiesPerRoom + 3) +
        4; // side walls + floor

    phys::RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT,
         numPhysicsSubsteps, -9.8 * math::up, max_total_entities,
         max_total_entities * max_total_entities / 2, consts::numAgents);

    curEpisodeStep = 0;

    enableVizRender = cfg.enableViewer;

    if (enableVizRender) {
        viz::VizRenderingSystem::init(ctx, init.vizBridge);
    }

    autoReset = cfg.autoReset;

    // Creates agents, walls, etc.
    createPersistentEntities(ctx);

    // Generate initial world state
    initWorld(ctx);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
