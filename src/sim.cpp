#include <algorithm>
#include <limits>
#include <madrona/mw_gpu_entry.hpp>
#include <madrona/physics.hpp>

#include "level_gen.hpp"
#include "obb.hpp"
#include "sim.hpp"
#include "utils.hpp"
#include "knn.hpp"

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
    registry.registerComponent<BicycleModel>();
    registry.registerComponent<VehicleSize>();
    registry.registerComponent<Goal>();
    registry.registerComponent<Trajectory>();
    registry.registerComponent<ControlledState>();
    registry.registerComponent<CollisionDetectionEvent>();
    registry.registerComponent<AbsoluteSelfObservation>();
    registry.registerComponent<Info>();
    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<Shape>();
    registry.registerSingleton<ValidState>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();

    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);
    registry.exportSingleton<Shape>((uint32_t)ExportID::Shape);
    registry.exportSingleton<ValidState>((uint32_t)ExportID::ValidState);
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
    registry.exportColumn<Agent, BicycleModel>(
        (uint32_t) ExportID::BicycleModel);
    registry.exportColumn<Agent, ControlledState>(
        (uint32_t) ExportID::ControlledState);
    registry.exportColumn<Agent, AbsoluteSelfObservation>(
        (uint32_t)ExportID::AbsoluteSelfObservation);
    registry.exportColumn<Agent, Info>(
        (uint32_t)ExportID::Info);
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

// This system runs each frame and checks if the current episode is complete
// or if code external to the application has forced a reset by writing to the
// WorldReset singleton.
//
// If a reset is needed, cleanup the existing world and generate a new one.
inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t should_reset = reset.reset;
    if (ctx.data().autoReset && ctx.data().numControlledVehicles > 0) {
        int32_t areAllControlledAgentsDone = 1;
        for (CountT i = 0; i < ctx.data().numAgents; i++) {
            Entity agent = ctx.data().agents[i];
            Done done = ctx.get<Done>(agent);
            ControlledState controlledState = ctx.get<ControlledState>(agent);
            if (controlledState.controlledState == ControlMode::BICYCLE && !done.v) {
                areAllControlledAgentsDone = 0;
            }
        }
        should_reset = areAllControlledAgentsDone;
    }
    else if (ctx.data().autoReset && ctx.data().numControlledVehicles == 0) {
        should_reset = 1;
        for(CountT i = 0; i < ctx.data().numAgents; i++) {
            Entity agent = ctx.data().agents[i];
            if(ctx.get<EntityType>(agent) == EntityType::Padding) {
                continue;
            }
            if(ctx.get<Done>(agent).v == 0) {
                should_reset = 0;
                break;
            }
        }
    }

    if (should_reset != 0) {
        reset.reset = 0;

        cleanupWorld(ctx);
        initWorld(ctx);
    }
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
                                      PartnerObservations &partner_obs,
                                      AgentMapObservations &map_obs,
				                      const EntityType& entityType,
				                      const CollisionDetectionEvent& collisionEvent)
{
    if (entityType == EntityType::Padding) {
        return;
    }

    self_obs.speed = model.speed;
    self_obs.vehicle_size = size;
    auto goalPos = goal.position - model.position;
    self_obs.goal.position = rot.inv().rotateVec({goalPos.x, goalPos.y, 0}).xy();

    auto hasCollided = collisionEvent.hasCollided.load_relaxed();
    self_obs.collisionState = hasCollided ? 1.f : 0.f;


    CountT arrIndex = 0; CountT agentIdx = 0;
    while(agentIdx < ctx.data().numAgents - 1)
    {
        Entity other = other_agents.e[agentIdx++];

        BicycleModel other_bicycle_model = ctx.get<BicycleModel>(other);
        Rotation other_rot = ctx.get<Rotation>(other);
        VehicleSize other_size = ctx.get<VehicleSize>(other);

        Vector2 relative_pos = other_bicycle_model.position - model.position;
        relative_pos = rot.inv().rotateVec({relative_pos.x, relative_pos.y, 0}).xy();
        float relative_speed = other_bicycle_model.speed; // Design decision: return the speed of the other agent directly

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
    while(arrIndex < consts::kMaxAgentCount - 1)
    {
        partner_obs.obs[arrIndex].type = (float)EntityType::None;
        arrIndex++;
    }

    const auto alg = ctx.data().params.roadObservationAlgorithm;
    if (alg == FindRoadObservationsWith::KNearestEntitiesWithRadiusFiltering) {
        selectKNearestRoadEntities<consts::kMaxAgentMapObservationsCount>(
            ctx, rot, model.position, map_obs.obs);
        return;
    }

    assert(alg == FindRoadObservationsWith::AllEntitiesWithRadiusFiltering);
    arrIndex = 0; CountT roadIdx = 0;
    while(roadIdx < ctx.data().numRoads) {
        Entity road = ctx.data().roads[roadIdx++];
        Vector2 relative_pos = Vector2{ctx.get<Position>(road).x, ctx.get<Position>(road).y} - model.position;
        relative_pos = rot.inv().rotateVec({relative_pos.x, relative_pos.y, 0}).xy();
        if(relative_pos.length() > ctx.data().params.observationRadius)
        {
            continue;
        }
        map_obs.obs[arrIndex] = ctx.get<MapObservation>(road);
        map_obs.obs[arrIndex].position = relative_pos;
        map_obs.obs[arrIndex].heading = utils::quatToYaw(rot.inv() * ctx.get<Rotation>(road));
        arrIndex++;
    }
    while (arrIndex < consts::kMaxRoadEntityCount)
    {
        map_obs.obs[arrIndex].position = Vector2{0.f, 0.f};
        map_obs.obs[arrIndex].heading = 0.f;
        map_obs.obs[arrIndex].type = (float)EntityType::None;
        arrIndex++;
    }
}

inline void movementSystem(Engine &e,
                           Action &action,
                           BicycleModel &model,
                           VehicleSize &size,
                           Rotation &rotation,
                           Position &position,
                           Velocity &velocity,
                           const ControlledState &controlledState,
                           const EntityType &type,
                           const StepsRemaining &stepsRemaining,
                           const Trajectory &trajectory,
                           const CollisionDetectionEvent &collisionEvent,
                           Done& done) {
    if (type == EntityType::Padding) {
        return;
    }

    if (collisionEvent.hasCollided.load_relaxed())
    {
        if(e.data().params.collisionBehaviour == CollisionBehaviour::AgentStop) {
            velocity.linear.x = 0;
            velocity.linear.y = 0;
            velocity.linear.z = 0;
            velocity.angular = Vector3::zero();
            done.v = 1;
            return;
       }  else if(e.data().params.collisionBehaviour == CollisionBehaviour::AgentRemoved)
        {
            done.v = 1;
            position = consts::kPaddingPosition;
            velocity.linear.x = 0;
            velocity.linear.y = 0;
            velocity.linear.z = 0;
            velocity.angular = Vector3::zero();
            return;
        }
        else if(e.data().params.collisionBehaviour == CollisionBehaviour::Ignore)
        {
            // Do nothing.
        }
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
        // TODO: Handle the case when the agent is not valid. Currently, we are not doing anything.

        // TODO: We are not storing previous action for the agent. Is it the ideal behaviour? Tehnically the actions
        // need to be iterative. If we dont do this, there could be jumps in the acceleration. For eg, acc can go from
        // 4m/s^2 to -4m/s^2 in one step. This is not ideal. We need to store the previous action and then use it to change
        // gradually.

        // TODO(samk): The following constants are configurable in Nocturne but look to
        // always use the same hard-coded value in practice. Use in-line constants
        // until the configuration is built out. - These values are correct. They are relative and hence are hardcoded.
        const float maxSpeed{std::numeric_limits<float>::max()};
        const float dt{0.1};

        auto clipSpeed = [maxSpeed](float speed)
        {
            return std::max(std::min(speed, maxSpeed), -maxSpeed);
        };
        // TODO(samk): hoist into Vector2::PolarToVector2D
        auto polarToVector2D = [](float r, float theta)
        {
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
        position = madrona::base::Position({.x = model.position.x, .y = model.position.y, .z = 1});
        rotation = Quat::angleAxis(model.heading, madrona::math::up);
        velocity.linear.x = model.speed * cosf(model.heading);
        velocity.linear.y = model.speed * sinf(model.heading);
        velocity.linear.z = 0;
        velocity.angular = Vector3::zero();
        velocity.angular.z = w;
    }
    else
    {
        // Follow expert trajectory
        CountT curStepIdx = getCurrentStep(stepsRemaining);
        model.position= trajectory.positions[curStepIdx];
        model.heading = trajectory.headings[curStepIdx];
        model.speed = trajectory.velocities[curStepIdx].length();
        position.x = trajectory.positions[curStepIdx].x;
        position.y = trajectory.positions[curStepIdx].y;
        velocity.linear.x = trajectory.velocities[curStepIdx].x;
        velocity.linear.y = trajectory.velocities[curStepIdx].y;
        rotation = Quat::angleAxis(trajectory.headings[curStepIdx], madrona::math::up);
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
                         const BicycleModel &model,
                         const Trajectory &trajectory,
                         const Goal &goal,
                         Progress &progress,
                         Reward &out_reward)
{

    const auto &rewardType = ctx.data().params.rewardParams.rewardType;
    if(rewardType == RewardType::DistanceBased)
    {
        float dist = (model.position - goal.position).length();
        float reward = -dist;
        out_reward.v = reward;
    }
    else if(rewardType == RewardType::OnGoalAchieved)
    {
        float dist = (model.position - goal.position).length();
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
                              const BicycleModel &model,
                              const Goal &goal,
                              StepsRemaining &steps_remaining,
                              Done &done,
                              Info &info)
{
    // Absolute done is 90 steps.
    int32_t num_remaining = --steps_remaining.t;
    if (num_remaining == consts::episodeLen - 1 && done.v != 1) { // Make sure to not reset an agent's done flag
        done.v = 0;
    } else if (num_remaining == 0) {
        done.v = 1;
    }

    // An agent can be done early if it reaches the goal
    if(done.v != 1 || info.reachedGoal != 1)
    {
        float dist = (model.position - goal.position).length();
        if(dist < ctx.data().params.rewardParams.distanceToGoalThreshold)
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

inline void validStateSystem(Engine &ctx, ValidState &validOut) {
    // This assumes StepsRemaining is constant across all agents
    Entity anyAgent = ctx.data().agents[0];
    const auto stepsRemaining = ctx.get<StepsRemaining>(anyAgent);
    const CountT currentStep = getCurrentStep(stepsRemaining);

    for (CountT agentIdx = 0; agentIdx < ctx.data().numAgents; ++agentIdx) {
        auto agentHandle = ctx.data().agents[agentIdx];
        const auto &trajectory = ctx.get<Trajectory>(agentHandle);

        assert(currentStep < consts::kTrajectoryLength);
        int32_t isValid = trajectory.valids[currentStep];

        validOut.valid[agentIdx] = static_cast<Validity>(isValid);
    }

    for (CountT agentIdx = ctx.data().numAgents;
         agentIdx < consts::kMaxAgentCount; ++agentIdx) {
        auto agentHandle = ctx.data().agents[agentIdx];
        validOut.valid[agentIdx] = Validity::Invalid;
    }
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(TaskGraphID::Step);

    // Turn policy actions into movement
    auto moveSystem = builder.addToGraph<ParallelForNode<Engine,
        movementSystem,
            Action,
            BicycleModel,
            VehicleSize,
            Rotation,
            Position,
            Velocity,
            ControlledState,
            EntityType,
            StepsRemaining,
            Trajectory,
            CollisionDetectionEvent,
            Done
        >>({});

    auto updateValidStateSystem =
        builder
            .addToGraph<ParallelForNode<Engine, validStateSystem, ValidState>>(
                {moveSystem});

    // setupBroadphaseTasks consists of the following sub-tasks:
    // 1. updateLeafPositionsEntry
    // 2. broadphase::updateBVHEntry
    // 3. broadphase::refitEntry
    auto broadphase_setup_sys = phys::PhysicsSystem::setupBroadphaseTasks(
        builder, {updateValidStateSystem});

    auto findOverlappingEntities =
        phys::PhysicsSystem::setupBroadphaseOverlapTasks(
            builder, {broadphase_setup_sys});

    auto detectCollisions = builder.addToGraph<
        ParallelForNode<Engine, collisionDetectionSystem, CandidateCollision>>(
        {findOverlappingEntities});

    // Improve controllability of agents by setting their velocity to 0
    // after physics is done.
    auto agent_zero_vel = builder.addToGraph<
        ParallelForNode<Engine, agentZeroVelSystem, Velocity, Action>>(
        {detectCollisions});

    // Finalize physics subsystem work
    auto phys_done = phys::PhysicsSystem::setupCleanupTasks(
        builder, {agent_zero_vel});

    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
         rewardSystem,
            BicycleModel,
            Trajectory,
            Goal,
            Progress,
            Reward
        >>({phys_done});


    // Check if the episode is over
    auto done_sys = builder.addToGraph<
        ParallelForNode<Engine, stepTrackerSystem, BicycleModel, Goal, StepsRemaining, Done, Info>>(
        {reward_sys});

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
        phys::PhysicsSystem::setupBroadphaseTasks(builder,
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
	        PartnerObservations,
            AgentMapObservations,
            EntityType,
            CollisionDetectionEvent
        >>({post_reset_broadphase});

    auto collectAbsoluteSelfObservations = builder.addToGraph<
        ParallelForNode<Engine, collectAbsoluteObservationsSystem, Position,
                        Rotation, Goal, EntityType, VehicleSize, AbsoluteSelfObservation>>(
        {collect_obs});

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
            Lidar,
            EntityType
        >>({collectAbsoluteSelfObservations});

    if (cfg.renderBridge) {
        RenderingSystem::setupTasks(builder, {reset_sys});
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
      episodeMgr(init.episodeMgr),
      params(*init.params)
{
    // Below check is used to ensure that the map is not empty due to incorrect WorldInit copy to GPU
    assert(init.map->numObjects);
    assert(init.map->numObjects <= consts::kMaxAgentCount);
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
