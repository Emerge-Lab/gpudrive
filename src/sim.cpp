#include <algorithm>
#include <limits>
#include <madrona/mw_gpu_entry.hpp>
#include <madrona/physics.hpp>

#include "level_gen.hpp"
#include "obb.hpp"
#include "sim.hpp"
#include "utils.hpp"
#include "knn.hpp"
#include "dynamics.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace RenderingSystem = madrona::render::RenderingSystem;

namespace gpudrive {

CountT getCurrentStep(const StepsRemaining &stepsRemaining) {
  return consts::episodeLen - stepsRemaining.t;
}

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);
    phys::PhysicsSystem::registerTypes(registry);

    RenderingSystem::registerTypes(registry, cfg.renderBridge);

    registry.registerComponent<Action>();
    registry.registerComponent<SelfObservation>();
    registry.registerComponent<MapObservation>();
    registry.registerComponent<AgentMapObservations>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<Progress>();
    registry.registerComponent<OtherAgents>();
    registry.registerComponent<PartnerObservations>();
    registry.registerComponent<Lidar>();
    registry.registerComponent<StepsRemaining>();
    registry.registerComponent<EntityType>();
    registry.registerComponent<VehicleSize>();
    registry.registerComponent<Goal>();
    registry.registerComponent<Trajectory>();
    registry.registerComponent<ControlledState>();
    registry.registerComponent<CollisionDetectionEvent>();
    registry.registerComponent<AbsoluteSelfObservation>();
    registry.registerComponent<Info>();
    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<Shape>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();
    registry.registerArchetype<CameraAgent>();

    registry.exportSingleton<WorldReset>((uint32_t)ExportID::Reset);
    registry.exportSingleton<Shape>((uint32_t)ExportID::Shape);
    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<Agent, SelfObservation>(
        (uint32_t)ExportID::SelfObservation);
    registry.exportColumn<Agent, AgentMapObservations>(
        (uint32_t)ExportID::AgentMapObservations);
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
    registry.exportColumn<Agent, ControlledState>(
        (uint32_t) ExportID::ControlledState);
    registry.exportColumn<Agent, AbsoluteSelfObservation>(
        (uint32_t)ExportID::AbsoluteSelfObservation);
    registry.exportColumn<Agent, Info>(
        (uint32_t)ExportID::Info);
    registry.exportColumn<Agent, ResponseType>(
        (uint32_t)ExportID::ResponseType);
    registry.exportColumn<Agent, Trajectory>(
        (uint32_t)ExportID::Trajectory);
}

static inline void cleanupWorld(Engine &ctx) {}

static inline void initWorld(Engine &ctx)
{
    phys::PhysicsSystem::reset(ctx);

    // Assign a new episode ID
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    int32_t episode_idx = episode_mgr.curEpisode.fetch_add<sync::relaxed>(1);
    ctx.data().rng = RNG::make(episode_idx);
    ctx.data().curEpisodeIdx = episode_idx;

    // Defined in src/level_gen.hpp / src/level_gen.cpp
    generateWorld(ctx);
}

// This system runs in TaskGraphID::Reset and checks if the code external to the
// application has forced a reset by writing to the WorldReset singleton. If a
// reset is needed, cleanup the existing world and generate a new one.
inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    if (reset.reset == 0) {
      return;
    }

    reset.reset = 0;
  
    cleanupWorld(ctx);
    initWorld(ctx);
}

// This system packages all the egocentric observations together
// for the policy inputs.
inline void collectObservationsSystem(Engine &ctx,
                                      const VehicleSize &size,
                                      const Position &pos,
                                      const Rotation &rot,
                                      const Velocity &vel,
                                      const Goal &goal,
                                      const Progress &progress,
                                      const OtherAgents &other_agents,
                                      SelfObservation &self_obs,
                                      PartnerObservations &partner_obs,
                                      AgentMapObservations &map_obs,
				                      const EntityType& entityType,
				                      const CollisionDetectionEvent& collisionEvent)
{
    if (entityType == EntityType::Padding) {
        return;
    }

    self_obs.speed = vel.linear.length();
    self_obs.vehicle_size = size;
    auto goalPos = goal.position - pos.xy();
    self_obs.goal.position = rot.inv().rotateVec({goalPos.x, goalPos.y, 0}).xy();

    auto hasCollided = collisionEvent.hasCollided.load_relaxed();
    self_obs.collisionState = hasCollided ? 1.f : 0.f;

    if(ctx.data().params.disableClassicalObs)
        return;

    CountT arrIndex = 0; CountT agentIdx = 0;
    while(agentIdx < ctx.data().numAgents - 1)
    {
        Entity other = other_agents.e[agentIdx++];

        const Position &other_position = ctx.get<Position>(other);
        const Velocity &other_velocity = ctx.get<Velocity>(other);
        const Rotation &other_rot = ctx.get<Rotation>(other);
        const VehicleSize &other_size = ctx.get<VehicleSize>(other);

        Vector2 relative_pos = (other_position - pos).xy();
        relative_pos = rot.inv().rotateVec({relative_pos.x, relative_pos.y, 0}).xy();
        float relative_speed = other_velocity.linear.length(); // Design decision: return the speed of the other agent directly

        Rotation relative_orientation = rot.inv() * other_rot;

        float relative_heading = utils::quatToYaw(relative_orientation);

        if(relative_pos.length() > ctx.data().params.observationRadius || ctx.get<EntityType>(other) == EntityType::Padding)
        {
            continue;
        }
        partner_obs.obs[arrIndex++] = {
            .speed = relative_speed,
            .position = relative_pos,
            .heading = relative_heading,
            .vehicle_size = other_size,
            .type = (float)ctx.get<EntityType>(other)
        };
    }
    while(arrIndex < consts::kMaxAgentCount - 1) {
        partner_obs.obs[arrIndex++] = PartnerObservation::zero();
    }

    const auto alg = ctx.data().params.roadObservationAlgorithm;
    if (alg == FindRoadObservationsWith::KNearestEntitiesWithRadiusFiltering) {
        selectKNearestRoadEntities<consts::kMaxAgentMapObservationsCount>(
            ctx, rot, pos.xy(), map_obs.obs);
        return;
    }

    assert(alg == FindRoadObservationsWith::AllEntitiesWithRadiusFiltering);

    utils::ReferenceFrame referenceFrame(pos.xy(), rot);
    arrIndex = 0; CountT roadIdx = 0;
    while(roadIdx < ctx.data().numRoads && arrIndex < consts::kMaxAgentMapObservationsCount) {
        Entity road = ctx.data().roads[roadIdx++];
        auto roadPos = ctx.get<Position>(road);
        auto roadRot = ctx.get<Rotation>(road);

        auto dist = referenceFrame.distanceTo(roadPos);
        if (dist > ctx.data().params.observationRadius) {
            continue;
        }

        map_obs.obs[arrIndex] = referenceFrame.observationOf(
            roadPos, roadRot, ctx.get<Scale>(road), ctx.get<EntityType>(road));
        arrIndex++;
    }

    while (arrIndex < consts::kMaxAgentMapObservationsCount) {
        map_obs.obs[arrIndex++] = MapObservation::zero();
    }
}


// Make the agents easier to control by zeroing out their velocity
// after each step.
inline void agentZeroVelSystem(Engine &,
                               Velocity &vel,
                               Action &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = 0;
    vel.angular = Vector3::zero();
}


inline void movementSystem(Engine &e,
                           Action &action,
                           VehicleSize &size,
                           Rotation &rotation,
                           Position &position,
                           Velocity &velocity,
                           const ControlledState &controlledState,
                           const EntityType &type,
                           const StepsRemaining &stepsRemaining,
                           const Trajectory &trajectory,
                           const CollisionDetectionEvent &collisionEvent,
                           const ResponseType &responseType,
                           Done& done) {
    if (type == EntityType::Padding) {
        return;
    }

    if (collisionEvent.hasCollided.load_relaxed()) {
        switch (e.data().params.collisionBehaviour) {
            case CollisionBehaviour::AgentStop:
                done.v = 1;
                agentZeroVelSystem(e, velocity, action);
                break;

            case CollisionBehaviour::AgentRemoved:
                done.v = 1;
                position = consts::kPaddingPosition;
                agentZeroVelSystem(e, velocity, action);
                break;

            case CollisionBehaviour::Ignore:
                // Do nothing.
                break;
        }
    }

    if(responseType == ResponseType::Static)
    {
        // Do nothing. The agent is static.
        // Agent can only be static if isStaticAgentControlled is set to true.
        return;
    }

    if(done.v)
    {
        // Case: Agent has not collided but is done. 
        // This can only happen if the agent has reached goal or the episode has ended.
        // In that case we teleport the agent. The agent will not collide with anything.
        position = consts::kPaddingPosition;
        velocity.linear.x = 0;
        velocity.linear.y = 0;
        velocity.linear.z = 0;
        velocity.angular = Vector3::zero();
        return;
    }

    if (type == EntityType::Vehicle && controlledState.controlledState == ControlMode::BICYCLE)
    {
        if(e.data().params.useWayMaxModel)
        {
            forwardWaymaxModel(action, rotation, position, velocity);
        }
        else 
        {
            forwardKinematics(action, size, rotation, position, velocity);
        }
        // TODO(samk): factor out z-dimension constant and reuse when scaling cubes
    }
    else
    {
        // Follow expert trajectory
        CountT curStepIdx = getCurrentStep(stepsRemaining);
        position.x = trajectory.positions[curStepIdx].x;
        position.y = trajectory.positions[curStepIdx].y;
        position.z = 1;
        velocity.linear.x = trajectory.velocities[curStepIdx].x;
        velocity.linear.y = trajectory.velocities[curStepIdx].y;
        velocity.linear.z = 0;
        velocity.angular = Vector3::zero();
        rotation = Quat::angleAxis(trajectory.headings[curStepIdx], madrona::math::up);
    }
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
inline void lidarSystem(Engine &ctx, Entity e, Lidar &lidar,
                        EntityType &entityType) {
    assert(entityType != EntityType::None);
    if (entityType == EntityType::Padding) {
        return;
    }
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
inline void rewardSystem(Engine &ctx,
                         const Position &position,
                         const Trajectory &trajectory,
                         const Goal &goal,
                         Progress &progress,
                         Reward &out_reward)
{

    const auto &rewardType = ctx.data().params.rewardParams.rewardType;
    if(rewardType == RewardType::DistanceBased)
    {
        float dist = (position.xy() - goal.position).length();
        float reward = -dist;
        out_reward.v = reward;
    }
    else if(rewardType == RewardType::OnGoalAchieved)
    {
        float dist = (position.xy() - goal.position).length();
        float reward = (dist < ctx.data().params.rewardParams.distanceToGoalThreshold) ? 1.f : 0.f;
        out_reward.v = reward;
    }
    else if(rewardType == RewardType::Dense)
    {
        // TODO: Implement full trajectory reward
        assert(false);
    }

    // Just in case agents do something crazy, clamp total reward
    // out_reward.v = fmaxf(fminf(out_reward.v, 1.f), 0.f);
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
    for (CountT i = 0; i < ctx.data().numAgents - 1; i++) {
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
inline void stepTrackerSystem(Engine &ctx,
                              const Position &position,
                              const Goal &goal,
                              StepsRemaining &steps_remaining,
                              Done &done,
                              Info &info)
{
    // Absolute done is 90 steps.
    int32_t num_remaining = --steps_remaining.t;
    if (num_remaining == consts::episodeLen - 1 && done.v != 1)
    { // Make sure to not reset an agent's done flag
        done.v = 0;
    }
    else if (num_remaining == 0)
    {
        done.v = 1;
    }

    // An agent can be done early if it reaches the goal
    if (done.v != 1 || info.reachedGoal != 1)
    {
        float dist = (position.xy() - goal.position).length();
        if (dist < ctx.data().params.rewardParams.distanceToGoalThreshold)
        {
            done.v = 1;
            info.reachedGoal = 1;
        }
    }
}

void collisionDetectionSystem(Engine &ctx,
                              const CandidateCollision &candidateCollision) {

    auto isInvalidExpertOrDone = [&](const Loc &candidate) -> bool
    {
        auto controlledState = ctx.getCheck<ControlledState>(candidate);
        if (controlledState.valid())
        {
            if( controlledState.value().controlledState == ControlMode::EXPERT)
            {
                // Case: If an expert agent is in an invalid state, we need to ignore the collision detection for it.
                auto currStep = getCurrentStep(ctx.get<StepsRemaining>(candidate));
                auto validState = ctx.get<Trajectory>(candidate).valids[currStep];
                if (!validState)
                {
                    return true;
                }
            }
            else if (controlledState.value().controlledState == ControlMode::BICYCLE)
            {
                // Case: If a controlled agent gets done, we teleport it to the padding position
                // Hence we need to ignore the collision detection for it.
                // The agent can also be done because it collided. 
                // In that case, we dont want to ignore collision. Especially if AgentStop is set.
                auto done = ctx.get<Done>(candidate);
                auto collisionEvent = ctx.getCheck<CollisionDetectionEvent>(candidate);
                if(done.v && collisionEvent.valid() && !collisionEvent.value().hasCollided.load_relaxed())
                {
                    return true;
                }
            }
        }
        return false;
    };

    if (isInvalidExpertOrDone(candidateCollision.a) || 
        isInvalidExpertOrDone(candidateCollision.b)) {

        return;
    }

    const CountT PositionColumn{2};
    const CountT RotationColumn{3};
    const CountT ScaleColumn{4};

    const Loc locationA{candidateCollision.a};
    const Position positionA{
        ctx.getDirect<Position>(PositionColumn, locationA)};
    const Rotation rotationA{
        ctx.getDirect<Rotation>(RotationColumn, locationA)};
    const Scale scaleA{ctx.getDirect<Scale>(ScaleColumn, locationA)};

    const Loc locationB{candidateCollision.b};
    const Position positionB{
        ctx.getDirect<Position>(PositionColumn, locationB)};
    const Rotation rotationB{
        ctx.getDirect<Rotation>(RotationColumn, locationB)};
    const Scale scaleB{ctx.getDirect<Scale>(ScaleColumn, locationB)};

    auto obbA = OrientedBoundingBox2D::from(positionA, rotationA, scaleA);
    auto obbB = OrientedBoundingBox2D::from(positionB, rotationB, scaleB);

    bool hasCollided = OrientedBoundingBox2D::hasCollided(obbA, obbB);
    if (not hasCollided) {
        return;
    }

    EntityType aEntityType = ctx.get<EntityType>(candidateCollision.a);
    EntityType bEntityType = ctx.get<EntityType>(candidateCollision.b);

    // Ignore collisions between certain entity types
    if(aEntityType == EntityType::Padding || bEntityType == EntityType::Padding)
    {
       return;
    }

    for(auto &pair : ctx.data().collisionPairs)
    {
        if((pair.first == aEntityType && pair.second == bEntityType) ||
           (pair.first == bEntityType && pair.second == aEntityType))
        {
            return;
        }
    }

    auto maybeCollisionDetectionEventA =
        ctx.getCheck<CollisionDetectionEvent>(candidateCollision.a);
    if (maybeCollisionDetectionEventA.valid()) {
        maybeCollisionDetectionEventA.value().hasCollided.store_relaxed(1);
        if(bEntityType > EntityType::None && bEntityType <= EntityType::StopSign)
        {
            ctx.get<Info>(candidateCollision.a).collidedWithRoad = 1;
        }
        else if(bEntityType == EntityType::Vehicle)
        {
            ctx.get<Info>(candidateCollision.a).collidedWithVehicle = 1;
        }
        else if(bEntityType <= EntityType::Cyclist)
        {
            ctx.get<Info>(candidateCollision.a).collidedWithNonVehicle = 1;
        }
    }

    auto maybeCollisionDetectionEventB =
        ctx.getCheck<CollisionDetectionEvent>(candidateCollision.b);
    if (maybeCollisionDetectionEventB.valid()) {
        maybeCollisionDetectionEventB.value().hasCollided.store_relaxed(1);
        if(aEntityType > EntityType::None && aEntityType <= EntityType::StopSign)
        {
            ctx.get<Info>(candidateCollision.b).collidedWithRoad = 1;
        }
        else if(aEntityType == EntityType::Vehicle)
        {
            ctx.get<Info>(candidateCollision.b).collidedWithVehicle = 1;
        }
        else if(aEntityType <= EntityType::Cyclist)
        {
            ctx.get<Info>(candidateCollision.b).collidedWithNonVehicle = 1;
        }
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

inline void collectAbsoluteObservationsSystem(Engine &ctx,
                                              const Position &position,
                                              const Rotation &rotation,
                                              const Goal &goal,
                                              const EntityType &entityType,
                                              const VehicleSize &vehicleSize,
                                              AbsoluteSelfObservation &out) {
    if (entityType == EntityType::Padding) {
        return;
    }

    out.position = position;
    out.rotation.rotationAsQuat = rotation;
    out.rotation.rotationFromAxis = utils::quatToYaw(rotation);
    out.goal = goal;
    out.vehicle_size = vehicleSize;
}

void setupRestOfTasks(TaskGraphBuilder &builder,
		      const Sim::Config &cfg,
		      Span<const TaskGraphNodeID> dependencies) {
    // setupBroadphaseTasks consists of the following sub-tasks:
    // 1. updateLeafPositionsEntry
    // 2. broadphase::updateBVHEntry
    // 3. broadphase::refitEntry
    auto broadphase_setup_sys = phys::PhysicsSystem::setupBroadphaseTasks(
        builder, dependencies);

    auto findOverlappingEntities =
        phys::PhysicsSystem::setupStandaloneBroadphaseOverlapTasks(
            builder, {broadphase_setup_sys});

    auto detectCollisions = builder.addToGraph<
        ParallelForNode<Engine, collisionDetectionSystem, CandidateCollision>>(
        {findOverlappingEntities});

    // Finalize physics subsystem work
    auto phys_done = phys::PhysicsSystem::setupStandaloneBroadphaseCleanupTasks(
        builder, {detectCollisions});

    phys_done = phys::PhysicsSystem::setupCleanupTasks(
        builder, {detectCollisions});

    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
         rewardSystem,
            Position,
            Trajectory,
            Goal,
            Progress,
            Reward
        >>({phys_done});


    // Check if the episode is over
    auto done_sys = builder.addToGraph<
        ParallelForNode<Engine, stepTrackerSystem, Position, Goal, StepsRemaining, Done, Info>>(
        {reward_sys});

    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>({done_sys});
    (void)clear_tmp;


#ifdef MADRONA_GPU_MODE
    // RecycleEntitiesNode is required on the GPU backend in order to reclaim
    // deleted entity IDs.
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({done_sys});
    (void)recycle_sys;
#endif

    // Finally, collect observations for the next step.
    auto collect_obs = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            VehicleSize,
            Position,
            Rotation,
            Velocity,
            Goal,
            Progress,
            OtherAgents,
            SelfObservation,
	        PartnerObservations,
            AgentMapObservations,
            EntityType,
            CollisionDetectionEvent
        >>({clear_tmp});

    auto collectAbsoluteSelfObservations = builder.addToGraph<
        ParallelForNode<Engine, collectAbsoluteObservationsSystem, Position,
                        Rotation, Goal, EntityType, VehicleSize, AbsoluteSelfObservation>>(
        {collect_obs});

    if (cfg.renderBridge) {
        RenderingSystem::setupTasks(builder, {done_sys});
    }

    TaskGraphNodeID lidar;
    if(cfg.enableLidar) {
        // The lidar system
#ifdef MADRONA_GPU_MODE
    // Note the use of CustomParallelForNode to create a taskgraph node
    // that launches a warp of threads (32) for each invocation (1).
    // The 32, 1 parameters could be changed to 32, 32 to create a system
    // that cooperatively processes 32 entities within a warp.
    lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 32, 1,
#else
    lidar = builder.addToGraph<ParallelForNode<Engine,
        lidarSystem,
#endif
            Entity,
            Lidar,
            EntityType
        >>({collectAbsoluteSelfObservations});
    }

#ifdef MADRONA_GPU_MODE
    TaskGraphNodeID sort_agents;
    if(cfg.enableLidar)
    {
        sort_agents = queueSortByWorld<Agent>(builder, {lidar});
    } else {
        sort_agents = queueSortByWorld<Agent>(builder, {collectAbsoluteSelfObservations});
    }
    // Sort entities, this could be conditional on reset like the second
    // BVH build above.
        
    auto sort_phys_objects = queueSortByWorld<PhysicsEntity>(
        builder, {sort_agents});
    (void)sort_phys_objects;
#else
    (void)lidar;
    (void)collectAbsoluteSelfObservations;
#endif
}

static void setupStepTasks(TaskGraphBuilder &builder, const Sim::Config &cfg) {
    auto moveSystem = builder.addToGraph<ParallelForNode<Engine,
        movementSystem,
            Action,
            VehicleSize,
            Rotation,
            Position,
            Velocity,
            ControlledState,
            EntityType,
            StepsRemaining,
            Trajectory,
            CollisionDetectionEvent,
            ResponseType,
            Done
        >>({});

    setupRestOfTasks(builder, cfg, {moveSystem});
}

static void setupResetTasks(TaskGraphBuilder &builder, const Sim::Config &cfg) {
    auto reset = builder.addToGraph<ParallelForNode<Engine, resetSystem,  WorldReset>>({});

    setupRestOfTasks(builder, cfg, {reset});
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg) {
    setupResetTasks(taskgraph_mgr.init(TaskGraphID::Reset), cfg);
    setupStepTasks(taskgraph_mgr.init(TaskGraphID::Step), cfg);
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      params(*init.params)
{
    // Below check is used to ensure that the map is not empty due to incorrect WorldInit copy to GPU
    assert(init.map->numObjects);
    assert(init.map->numRoadSegments <= consts::kMaxRoadEntityCount);

    // Currently the physics system needs an upper bound on the number of
    // entities that will be stored in the BVH. We plan to fix this in
    // a future release.
    auto max_total_entities = consts::kMaxAgentCount + consts::kMaxRoadEntityCount;

    phys::PhysicsSystem::init(ctx, init.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -9.8f * math::up,
        max_total_entities);

    enableRender = cfg.renderBridge != nullptr;

    if (enableRender) {
        RenderingSystem::init(ctx, cfg.renderBridge);
    }

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
