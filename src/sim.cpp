#include <algorithm>
#include <limits>
#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "level_gen.hpp"
#include "utils.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace gpudrive {

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);
    phys::RigidBodyPhysicsSystem::registerTypes(registry);
    viz::VizRenderingSystem::registerTypes(registry);

    registry.registerComponent<Action>();
    registry.registerComponent<SelfObservation>();
    registry.registerComponent<MapObservation>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<Progress>();
    registry.registerComponent<OtherAgents>();
    registry.registerComponent<PartnerObservations>();
    registry.registerComponent<Lidar>();
    registry.registerComponent<StepsRemaining>();
    registry.registerComponent<EntityType>();
    registry.registerComponent<BicycleModel>();
    registry.registerComponent<VehicleSize>();
    registry.registerComponent<Goal>();
    registry.registerComponent<Trajectory>();
    registry.registerComponent<ValidState>();
    registry.registerComponent<ControlledState>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<LevelState>();
    registry.registerSingleton<Shape>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();

    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);
    registry.exportSingleton<Shape>((uint32_t)ExportID::Shape);
    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<Agent, SelfObservation>(
        (uint32_t)ExportID::SelfObservation);
    registry.exportColumn<PhysicsEntity, MapObservation>(
        (uint32_t)ExportID::MapObservation);

    registry.exportColumn<Agent, PartnerObservations>(
        (uint32_t)ExportID::PartnerObservations);
    registry.exportColumn<Agent, Lidar>(
        (uint32_t)ExportID::Lidar);
    registry.exportColumn<Agent, StepsRemaining>(
        (uint32_t)ExportID::StepsRemaining);
    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);
    registry.exportColumn<Agent, BicycleModel>(
        (uint32_t) ExportID::BicycleModel);
    registry.exportColumn<Agent, ValidState>(
        (uint32_t) ExportID::ValidState);
    registry.exportColumn<Agent, ControlledState>(
        (uint32_t) ExportID::ControlledState);
}

static inline void cleanupWorld(Engine &ctx) {}

static inline void initWorld(Engine &ctx)
{
    if (ctx.data().enableVizRender) {
        viz::VizRenderingSystem::reset(ctx);
    }

    phys::RigidBodyPhysicsSystem::reset(ctx);

    // Assign a new episode ID
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    int32_t episode_idx = episode_mgr.curEpisode.fetch_add<sync::relaxed>(1);
    ctx.data().rng = RNG::make(episode_idx);
    ctx.data().curEpisodeIdx = episode_idx;

    // Defined in src/level_gen.hpp / src/level_gen.cpp
    generateWorld(ctx);
}

// This system runs each frame and checks if the current episode is complete
// or if code external to the application has forced a reset by writing to the
// WorldReset singleton.
//
// If a reset is needed, cleanup the existing world and generate a new one.
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
    }
}


float quatToYaw(Rotation q) {
    // From https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_(in_3-2-1_sequence)_conversion
    return atan2(2.0f * (q.w * q.z + q.x * q.y), 1.0f - 2.0f * (q.y * q.y + q.z * q.z)); 
}


// This system packages all the egocentric observations together 
// for the policy inputs.
inline void collectObservationsSystem(Engine &ctx,
                                      const BicycleModel &model,
                                      const VehicleSize &size,
                                      const Position &pos,
                                      const Rotation &rot,
                                      const Velocity &vel,
                                      const Goal &goal,
                                      const Progress &progress,
                                      const OtherAgents &other_agents,
                                      SelfObservation &self_obs,
                                      PartnerObservations &partner_obs)
{
    self_obs.bicycle_model = model;
    self_obs.vehicle_size = size; 
    self_obs.goal.position = Vector2{goal.position.x - pos.x, goal.position.y - pos.y};

#pragma unroll
    for (CountT i = 0; i < ctx.data().numAgents - 1; i++) {
        Entity other = other_agents.e[i];

        BicycleModel other_bicycle_model = ctx.get<BicycleModel>(other);
        Rotation other_rot = ctx.get<Rotation>(other);

        Vector2 relative_pos = other_bicycle_model.position - model.position;
        float relative_speed = other_bicycle_model.speed - model.speed;

        Rotation relative_orientation = rot.inv() * other_rot;

        float relative_heading = quatToYaw(relative_orientation);

        partner_obs.obs[i] = {
            .speed = relative_speed,
            .position = relative_pos,
            .heading = relative_heading
        };
    }
}

inline void movementSystem(Engine &e,
			   Action &action,
			   BicycleModel& model,
			   VehicleSize& size,
			   Rotation &rotation,
			   Position& position,
			   Velocity& velocity) 
{

  //TODO: We are not storing previous action for the agent. Is it the ideal behaviour? Tehnically the actions 
  // need to be iterative. If we dont do this, there could be jumps in the acceleration. For eg, acc can go from
  // 4m/s^2 to -4m/s^2 in one step. This is not ideal. We need to store the previous action and then use it to change 
  // gradually.

  // TODO(samk): The following constants are configurable in Nocturne but look to
  // always use the same hard-coded value in practice. Use in-line constants
  // until the configuration is built out. - These values are correct. They are relative and hence are hardcoded.
  const float maxSpeed{std::numeric_limits<float>::max()};
  const float dt{0.1};

  auto clipSpeed = [maxSpeed](float speed) {
    return std::max(std::min(speed, maxSpeed), -maxSpeed);
  };
  // TODO(samk): hoist into Vector2::PolarToVector2D
  auto polarToVector2D = [](float r, float theta) {
    return math::Vector2{r * cosf(theta), r * sinf(theta)};
  };

  // Average speed
  const float v{clipSpeed(model.speed + 0.5f * action.acceleration * dt)};
  const float tanDelta{tanf(action.steering)};
  // Assume center of mass lies at the middle of length, then l / L == 0.5.
  const float beta{std::atan(0.5f * tanDelta)};
  const math::Vector2 d{polarToVector2D(v, model.heading + beta)};
  const float w{v * std::cos(beta) * tanDelta / size.length};

  model.position += d * dt;
  model.heading = utils::AngleAdd(model.heading, w * dt);
  model.speed = clipSpeed(model.speed + action.acceleration * dt);

  // The BVH machinery requires the components rotation, position, and velocity
  // to perform calculations. Thus, to reuse the BVH machinery, we need to also
  // updates these components.

  // TODO(samk): factor out z-dimension constant and reuse when scaling cubes
  position = madrona::base::Position({ .x = model.position.x, .y = model.position.y, .z = 1 });
  rotation = Quat::angleAxis(model.heading, madrona::math::up);
//   velocity.linear = Vector3::zero();
  velocity.linear.x = model.speed * cosf(model.heading);
  velocity.linear.y = model.speed * sinf(model.heading);
  velocity.linear.z = 0;
  velocity.angular = Vector3::zero();
  velocity.angular.z = w;
}

// Make the agents easier to control by zeroing out their velocity
// after each step.
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

static inline float encodeType(EntityType type)
{
    return (float)type / (float)EntityType::NumTypes;
}

// Launches consts::numLidarSamples per agent.
// This system is specially optimized in the GPU version:
// a warp of threads is dispatched for each invocation of the system
// and each thread in the warp traces one lidar ray for the agent.
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
            lidar.samples[idx] = {
                .depth = 0.f,
                .encodedType = encodeType(EntityType::None),
            };
        } else {
            EntityType entity_type = ctx.get<EntityType>(hit_entity);

            lidar.samples[idx] = {
                .depth = distObs(hit_t),
                .encodedType = encodeType(entity_type),
            };
        }
    };


    // MADRONA_GPU_MODE guards GPU specific logic
#ifdef MADRONA_GPU_MODE
    // Can use standard cuda variables like threadIdx for 
    // warp level programming
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

// Computes reward for each agent and keeps track of the max distance achieved
// so far through the challenge. Continuous reward is provided for any new
// distance achieved.
inline void rewardSystem(Engine &,
                         Position pos,
                         Progress &progress,
                         Reward &out_reward)
{
    // Just in case agents do something crazy, clamp total reward
    float reward_pos = fminf(pos.y, consts::worldLength * 2);

    float old_max_y = progress.maxY;

    float new_progress = reward_pos - old_max_y;

    float reward;
    if (new_progress > 0) {
        reward = new_progress * consts::rewardPerDist;
        progress.maxY = reward_pos;
    } else {
        reward = consts::slackReward;
    }

    out_reward.v = reward;
}

// Each agent gets a small bonus to it's reward if the other agent has
// progressed a similar distance, to encourage them to cooperate.
// This system reads the values of the Progress component written by
// rewardSystem for other agents, so it must run after.
inline void bonusRewardSystem(Engine &ctx,
                              OtherAgents &others,
                              Progress &progress,
                              Reward &reward)
{
    bool partners_close = true;
    for (CountT i = 0; i < consts::numAgents - 1; i++) {
        Entity other = others.e[i];
        Progress other_progress = ctx.get<Progress>(other);

        if (fabsf(other_progress.maxY - progress.maxY) > 2.f) {
            partners_close = false;
        }
    }

    if (partners_close && reward.v > 0.f) {
        reward.v *= 1.25f;
    }
}

// Keep track of the number of steps remaining in the episode and
// notify training that an episode has completed by
// setting done = 1 on the final step of the episode
inline void stepTrackerSystem(Engine &,
                              StepsRemaining &steps_remaining,
                              Done &done)
{
    int32_t num_remaining = --steps_remaining.t;
    if (num_remaining == consts::episodeLen - 1) {
        done.v = 0;
    } else if (num_remaining == 0) {
        done.v = 1;
    }

}

// Helper function for sorting nodes in the taskgraph.
// Sorting is only supported / required on the GPU backend,
// since the CPU backend currently keeps separate tables for each world.
// This will likely change in the future with sorting required for both
// environments
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



// Build the task graph
void Sim::setupTasks(TaskGraphBuilder &builder, const Config &cfg)
{
    // Turn policy actions into movement
    auto moveSystem = builder.addToGraph<ParallelForNode<Engine,
        movementSystem,
            Action,
            BicycleModel,
            VehicleSize,
            Rotation,
            Position,
            Velocity
        >>({});
    // setupBroadphaseTasks consists of the following sub-tasks:
    // 1. updateLeafPositionsEntry
    // 2. broadphase::updateBVHEntry
    // 3. broadphase::refitEntry
    auto broadphase_setup_sys =
        phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(builder,
                                                           {moveSystem});

    // Physics collision detection and solver
    // setupSubstepTasks consists of the following sub-tasks:
    // 1. broadphase::findOverlapping
    // 2. collectConstraintSystem
    // 3. substepRigidBodies
    // 4. narrowPhase
    // 5. solvePositions
    // 6. setVelocities
    // 7. solveVelocities
    // 8. updateLeafPositionsEntry
    // 9. refitEntry
    // Tasks 2-7 are executed in a loop where the iteration count is determined
    // by consts::numPhysicsSubsteps. Subtasks 2-7 can be skipped by setting
    // this constant to 0.
    auto substep_sys = phys::RigidBodyPhysicsSystem::setupSubstepTasks(builder,
        {broadphase_setup_sys}, consts::numPhysicsSubsteps);

    // Improve controllability of agents by setting their velocity to 0
    // after physics is done.
    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, Action>>({substep_sys});

    // Finalize physics subsystem work
    auto phys_done = phys::RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {agent_zero_vel});

    // Check if the episode is over
    auto done_sys = builder.addToGraph<
        ParallelForNode<Engine, stepTrackerSystem, StepsRemaining, Done>>(
        {phys_done});

    // Conditionally reset the world if the episode is over
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem,
            WorldReset
        >>({done_sys});

    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>({reset_sys});
    (void)clear_tmp;


#ifdef MADRONA_GPU_MODE
    // RecycleEntitiesNode is required on the GPU backend in order to reclaim
    // deleted entity IDs.
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({reset_sys});
    (void)recycle_sys;
#endif

    // This second BVH build is a limitation of the current taskgraph API.
    // It's only necessary if the world was reset, but we don't have a way
    // to conditionally queue taskgraph nodes yet.
    auto post_reset_broadphase =
        phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(builder,
                                                           {reset_sys});

    // Finally, collect observations for the next step.
    auto collect_obs = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            BicycleModel,
            VehicleSize,
            Position,
            Rotation,
            Velocity,
            Goal,
            Progress,
            OtherAgents,
            SelfObservation,
            PartnerObservations
        >>({post_reset_broadphase});


    // The lidar system
#ifdef MADRONA_GPU_MODE
    // Note the use of CustomParallelForNode to create a taskgraph node
    // that launches a warp of threads (32) for each invocation (1).
    // The 32, 1 parameters could be changed to 32, 32 to create a system
    // that cooperatively processes 32 entities within a warp.
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
    // Sort entities, this could be conditional on reset like the second
    // BVH build above.
    auto sort_agents =
        queueSortByWorld<Agent>(builder, {lidar, collect_obs});
    auto sort_phys_objects = queueSortByWorld<PhysicsEntity>(
        builder, {sort_agents});
    (void)sort_phys_objects;
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

    //Below checks are used to ensure that the map is not empty due to incorrect WorldInit copy to GPU
    if(init.map->numObjects == 0)
        printf("Map numObjects is 0\n");
    assert(init.map->numObjects != 0);

    auto max_total_entities = consts::numAgents + consts::numRoadSegments;

    phys::RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -9.8f * math::up,
        max_total_entities, max_total_entities * max_total_entities / 2,
        consts::numAgents);

    enableVizRender = cfg.enableViewer;

    if (enableVizRender) {
        viz::VizRenderingSystem::init(ctx, init.vizBridge);
    }

    autoReset = cfg.autoReset;

    // Creates agents, walls, etc.
    createPersistentEntities(ctx, init.map);

    // Generate initial world state
    initWorld(ctx);
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
