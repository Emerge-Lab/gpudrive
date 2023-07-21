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
    registry.registerComponent<ToDynamicEntities>();
    registry.registerComponent<ButtonState>();
    registry.registerComponent<OpenState>();

    registry.registerComponent<Lidar>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsObject>();
    registry.registerArchetype<WallObject>();
    registry.registerArchetype<ButtonObject>();

    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<Agent, PositionObservation>(
        (uint32_t)ExportID::PositionObservation);
    registry.exportColumn<Agent, ToOtherAgents>(
        (uint32_t)ExportID::ToOtherAgents);
    registry.exportColumn<Agent, ToDynamicEntities>(
        (uint32_t)ExportID::ToDynamicEntities);
    registry.exportColumn<Agent, Lidar>(
        (uint32_t)ExportID::Lidar);
    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);
}

static inline void resetWorld(Engine &ctx)
{
    ctx.data().curEpisodeStep = 0;

    // Destroy old entities
    Entity *dyn_entities = ctx.data().dynamicEntities;
    int32_t num_old_entities= ctx.data().numDynamicEntities;
    for (int32_t i = 0; i < num_old_entities; i++) {
        ctx.destroyEntity(dyn_entities[i]);
    }

    if (ctx.data().enableVizRender) {
        viz::VizRenderingSystem::reset(ctx);
    }

    phys::RigidBodyPhysicsSystem::reset(ctx);

    // Assign a new episode ID
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    int32_t episode_idx = episode_mgr.curEpisode.fetch_add<sync::relaxed>(1);
    ctx.data().rng = RNG::make(0 /*episode_idx*/);
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

        resetWorld(ctx);

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
    constexpr float discrete_turn_max = 90;
    constexpr float turn_delta_per_bucket = discrete_turn_max / half_buckets;

    Quat cur_rot = rot;

    float f_x = move_delta_per_bucket * (action.x - (int32_t)half_buckets);
    float f_y = move_delta_per_bucket * (action.y - (int32_t)half_buckets);
    float t_z = turn_delta_per_bucket * (action.r - (int32_t)half_buckets);

    external_force = cur_rot.rotateVec({ f_x, f_y, 0 });
    external_torque = Vector3 { 0, 0, t_z };
}

// Resets doors to closed temporarily
inline void resetDoorStateSystem(Engine &, OpenState &open_state)
{
    open_state.isOpen = false;
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

// Sets door open state given where entities are (loops through entities).
inline void buttonSystem(Engine &ctx, Position &pos, Action &)
{
#if 0
    Room *room = containedRoom({ pos.x, pos.y }, ctx.data().rooms);
    ButtonInfo button = room->button;

    // If the button is pressed, set the doors of this room to be open
    if (isPressingButton({ pos.x, pos.y }, button)) {
        for (CountT i = 0; i < room->doorCount; ++i) {
            CountT doorIdx = room->doors[i];
            ctx.get<OpenState>(ctx.data().doors[doorIdx]).isOpen = true;
        }

        uint32_t prev_visited =
            AtomicU32Ref(room->button.visited).exchange<sync::relaxed>(1);

        if (!prev_visited) {
            auto &reward_tracker = ctx.singleton<RewardTracker>();
            AtomicU32Ref(
                reward_tracker.numNewButtonsVisited).fetch_add<sync::relaxed>(1);
        }
    }
#endif
}

inline void setDoorPositionSystem(Engine &, Position &pos, Velocity &, OpenState &open_state)
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

static inline float posObs(float v)
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

static inline float convertDynamicEntityType(
    DynamicEntityType type)
{
    return (float)type / (float)DynamicEntityType::NumTypes;
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
                                      ToDynamicEntities &to_buttons)
{
    Quat to_view = rot.inv();

    pos_obs.x = posObs(pos.x);
    pos_obs.y = posObs(pos.y);
    pos_obs.z = posObs(pos.z);
    pos_obs.theta = angleObs(computeZAngle(rot));

#pragma unroll
    for (CountT i = 0; i < consts::numAgents - 1; i++) {
        Entity other = other_agents.e[i];

        Vector3 other_pos = ctx.get<Position>(other);
        Vector3 to_other = other_pos - pos;

        to_other_agents.obs[i] = xyToPolar(to_view.rotateVec(to_other));
    }

#if 0
    // Compute polar coords to buttons
    CountT button_idx = 0;
    for (; button_idx < ctx.data().leafCount; button_idx++) {
        Room &room = ctx.data().rooms[ctx.data().leafs[button_idx]];

        Vector2 button_pos = room.button.pos;
        Vector3 to_button = Vector3 { button_pos.x, button_pos.y, 0 } - pos;

        to_buttons.obs[button_idx] = xyToPolar(to_view.rotateVec(to_button));
    }

    for (; button_idx < consts::maxRooms; button_idx++) {
        // FIXME: is this a good invalid output?
        to_buttons.obs[button_idx] = { 0.f, 1.f };
    }
#endif
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
            bvh.traceRay(pos , ray_dir, &hit_t, &hit_normal, 200.f);

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

inline void rewardSystem(Engine &ctx,
                         Position pos,
                         Progress &progress,
                         Reward &out_reward)
{
    constexpr float dist_per_progress = 2.f;
    constexpr float progress_reward = 0.2f;
    constexpr float slack_reward = -0.01f;

    int32_t new_progress = int32_t(pos.x / dist_per_progress);

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
// order to encourage them to cooperate.
inline void partnerRewardSystem(Engine &ctx,
                                const OtherAgents &others,
                                Reward &out_reward)
{
    for (CountT i = 0; i < consts::numAgents; i++) {
        Entity other = others.e[i];
        out_reward.v += ctx.get<Reward>(other).v / 2.f;
    }
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

    // Scripted door behaviors
    auto reset_door_sys = builder.addToGraph<ParallelForNode<Engine,
        resetDoorStateSystem,
            OpenState
        >>({move_sys});

    auto button_sys = builder.addToGraph<ParallelForNode<Engine,
        buttonSystem,
            Position,
            Action
        >>({reset_door_sys});

    auto set_door_pos_sys = builder.addToGraph<ParallelForNode<Engine,
        setDoorPositionSystem,
            Position,
            Velocity,
            OpenState
        >>({button_sys});

    // Physics systems
    auto broadphase_setup_sys = phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(
        builder, {set_door_pos_sys});

    auto substep_sys = phys::RigidBodyPhysicsSystem::setupSubstepTasks(builder,
        {broadphase_setup_sys}, numPhysicsSubsteps);

    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, Action>>(
            {substep_sys});

    // Finalize physics subsystem work
    auto phys_done = phys::RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {agent_zero_vel});

    // Compute initial reward now that physics has updated the world state
    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
         rewardSystem,
            Position,
            Progress,
            Reward
        >>({phys_done});

    // Read partner's reward and apply
    auto partner_reward_sys = builder.addToGraph<ParallelForNode<Engine,
         partnerRewardSystem,
            OtherAgents,
            Reward
        >>({reward_sys});

    // Check if the episode is over
    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        doneSystem,
            Done
        >>({partner_reward_sys});

    // Conditionally reset the world if the episode is over
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem,
            WorldReset
        >>({done_sys});

    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>({reset_sys});

    // This second BVH build is a limitation of the current taskgraph API.
    // It's only necessary if the world was reset, but we don't have a way
    // to conditionally queue taskgraph nodes yet.
    auto post_reset_broadphase = phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(
        builder, {clear_tmp});

#ifdef MADRONA_GPU_MODE
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({reset_sys});
    (void)recycle_sys;
#endif

    // Finally, collect observations for the next step.
    auto collect_obs = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Position,
            Rotation,
            PositionObservation,
            OtherAgents,
            ToOtherAgents,
            ToDynamicEntities
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
    auto sort_phys_objects = queueSortByWorld<PhysicsObject>(
        builder, {sort_agents});
    auto sort_buttons = queueSortByWorld<ButtonObject>(
        builder, {sort_phys_objects});
    auto sort_walls = queueSortByWorld<WallObject>(
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
    constexpr CountT max_total_entities =
        consts::numAgents +
        consts::numChallenges * consts::maxEntitiesPerChallenge +
        4; // border walls + floor

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
    resetWorld(ctx);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
