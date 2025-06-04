#include <iostream>
#include <algorithm>
#include <limits>
#include <madrona/mw_gpu_entry.hpp>
#include <madrona/physics.hpp>

#include "level_gen.hpp"
#include "obb.hpp"
#include "sim.hpp"
#include "utils.hpp"
#include "rasterizer.hpp"
#include "knn.hpp"
#include "dynamics.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace RenderingSystem = madrona::render::RenderingSystem;

namespace madrona_gpudrive {

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
    registry.registerComponent<BevObservations>();
    registry.registerComponent<StepsRemaining>();
    registry.registerComponent<EntityType>();
    registry.registerComponent<VehicleSize>();
    registry.registerComponent<Goal>();
    registry.registerComponent<Trajectory>();
    registry.registerComponent<ControlledState>();
    registry.registerComponent<CollisionDetectionEvent>();
    registry.registerComponent<AbsoluteSelfObservation>();
    registry.registerComponent<Info>();
    registry.registerComponent<AgentInterfaceEntity>();
    registry.registerComponent<RoadInterfaceEntity>();
    registry.registerComponent<AgentID>();
    registry.registerComponent<RoadMapId>();
    registry.registerComponent<MapType>();
    registry.registerComponent<MetaData>();
    registry.registerComponent<VBDTrajectory>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<Shape>();
    registry.registerSingleton<Map>();
    registry.registerSingleton<ResetMap>();
    registry.registerSingleton<WorldMeans>();
    registry.registerSingleton<DeletedAgents>();
    registry.registerSingleton<MapName>();
    registry.registerSingleton<ScenarioId>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();
    registry.registerArchetype<CameraAgent>();
    registry.registerArchetype<AgentInterface>();
    registry.registerArchetype<RoadInterface>();

    registry.exportSingleton<WorldReset>((uint32_t)ExportID::Reset);
    registry.exportSingleton<Shape>((uint32_t)ExportID::Shape);
    registry.exportSingleton<Map>((uint32_t)ExportID::Map);
    registry.exportSingleton<ResetMap>((uint32_t)ExportID::ResetMap);
    registry.exportSingleton<WorldMeans>((uint32_t)ExportID::WorldMeans);
    registry.exportSingleton<DeletedAgents>((uint32_t)ExportID::DeletedAgents);
    registry.exportSingleton<MapName>((uint32_t)ExportID::MapName);
    registry.exportSingleton<ScenarioId>((uint32_t)ExportID::ScenarioId);

    registry.exportColumn<AgentInterface, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<AgentInterface, SelfObservation>(
        (uint32_t)ExportID::SelfObservation);
    registry.exportColumn<AgentInterface, AgentMapObservations>(
        (uint32_t)ExportID::AgentMapObservations);
    registry.exportColumn<RoadInterface, MapObservation>(
        (uint32_t)ExportID::MapObservation);

    registry.exportColumn<AgentInterface, PartnerObservations>(
        (uint32_t)ExportID::PartnerObservations);
    registry.exportColumn<AgentInterface, Lidar>(
        (uint32_t)ExportID::Lidar);
    registry.exportColumn<AgentInterface, BevObservations>(
        (uint32_t)ExportID::BevObservations);
    registry.exportColumn<AgentInterface, StepsRemaining>(
        (uint32_t)ExportID::StepsRemaining);
    registry.exportColumn<AgentInterface, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<AgentInterface, Done>(
        (uint32_t)ExportID::Done);
    registry.exportColumn<AgentInterface, ControlledState>(
        (uint32_t) ExportID::ControlledState);
    registry.exportColumn<AgentInterface, AbsoluteSelfObservation>(
        (uint32_t)ExportID::AbsoluteSelfObservation);
    registry.exportColumn<AgentInterface, Info>(
        (uint32_t)ExportID::Info);
    registry.exportColumn<AgentInterface, ResponseType>(
        (uint32_t)ExportID::ResponseType);
    registry.exportColumn<AgentInterface, Trajectory>(
        (uint32_t)ExportID::Trajectory);
    registry.exportColumn<AgentInterface, MetaData>(
        (uint32_t)ExportID::MetaData);
    registry.exportColumn<AgentInterface, VBDTrajectory>(
        (uint32_t)ExportID::VBDTrajectory);
}

static inline void cleanupWorld(Engine &ctx) {
    destroyWorld(ctx);
}

static inline void initWorld(Engine &ctx)
{
    phys::PhysicsSystem::reset(ctx);

    // Assign a new episode ID
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    int32_t episode_idx = episode_mgr.curEpisode.fetch_add<sync::relaxed>(1);
    ctx.data().rng = RNG::make(episode_idx);
    ctx.data().curEpisodeIdx = episode_idx;

    if(ctx.singleton<ResetMap>().reset == 1)
    {
        createPersistentEntities(ctx);
        ctx.singleton<ResetMap>().reset = 0;
        phys::PhysicsSystem::reset(ctx);
    }

    // Defined in src/level_gen.hpp / src/level_gen.cpp
    resetWorld(ctx);
}

// This system runs in TaskGraphID::Reset and checks if the code external to the
// application has forced a reset by writing to the WorldReset singleton. If a
// reset is needed, cleanup the existing world and generate a new one.
inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    if (reset.reset == 0)
    {
        return;
    }

    reset.reset = 0;

    auto resetMap = ctx.singleton<ResetMap>();

    if (resetMap.reset == 1)
    {
        cleanupWorld(ctx);
    }
    initWorld(ctx);
}

inline void collectSelfObsSystem(Engine &ctx,
    const VehicleSize &size,
    const Position &pos,
    const Rotation &rot,
    const Velocity &vel,
    const Goal &goal,
    const CollisionDetectionEvent& collisionEvent,
    const AgentInterfaceEntity &agent_iface)
{
    auto &self_obs = ctx.get<SelfObservation>(agent_iface.e);
    self_obs.speed = vel.linear.length();
    self_obs.vehicle_size = size;

    // Get the relative goal position
    auto goalPos = goal.position - pos.xy();

    // For the agent's observation, convert the goal position to the agent's local frame
    self_obs.goal.position = rot.inv().rotateVec({goalPos.x, goalPos.y, 0}).xy();

    auto hasCollided = collisionEvent.hasCollided.load_relaxed();
    self_obs.collisionState = hasCollided ? 1.f : 0.f;
    self_obs.id = ctx.get<AgentID>(agent_iface.e).id;

    const Info& info = ctx.get<Info>(agent_iface.e);
    self_obs.goalState = info.reachedGoal ? 1.f : 0.f;
    self_obs.steerAngle = vel.angular.z;
}

struct ViewField {
    const Position& position;
    const float heading;
    const float viewRadius;
    const float viewAngle;

    ViewField(const Position& position, const float heading, const float viewRadius, const float viewAngle)
        : position(position), heading(heading), viewRadius(viewRadius), viewAngle(viewAngle) {}
};

// Broad-phase filtering: Get dynamic objects that are potentially visible within the view field
inline std::vector<Entity> visibleCandidatesDynamic(Engine &ctx, 
                                                    const ViewField &vf,
                                                    const OtherAgents &other_agents,
                                                    Entity self_entity) {
    std::vector<Entity> candidates;
    
    // Simple radius-based broad phase filtering for now
    for (CountT agentIdx = 0; agentIdx < ctx.data().numAgents - 1; ++agentIdx) {
        Entity other = other_agents.e[agentIdx];
        if (other == self_entity) continue; // Don't include self
        
        const Position &other_pos = ctx.get<Position>(other);
        Vector2 relative_pos = (other_pos - vf.position).xy();
        
        // Check if within view radius
        if (relative_pos.length() <= vf.viewRadius) {
            // Check if within view cone
            Vector2 forward_dir = {cosf(vf.heading), sinf(vf.heading)};
            Vector2 normalized_relative_pos = relative_pos / relative_pos.length();
            float dot_product = normalized_relative_pos.dot(forward_dir);
            float angle = acosf(fmaxf(-1.0f, fminf(1.0f, dot_product))); // clamp to [-1, 1]
            
            if (angle <= vf.viewAngle * 0.5f) {
                candidates.push_back(other);
            }
        }
    }
    
    return candidates;
}

// Broad-phase filtering: Get static road objects that are potentially visible within the view field
inline std::vector<Entity> visibleCandidatesStatic(Engine &ctx,
                                                   const ViewField &vf) {
    std::vector<Entity> candidates;
    
    // Simple radius-based broad phase filtering for roads
    for (CountT roadIdx = 0; roadIdx < ctx.data().numRoads; ++roadIdx) {
        Entity road = ctx.data().roads[roadIdx];
        const Position &road_pos = ctx.get<Position>(road);
        Vector2 relative_pos = (road_pos - vf.position).xy();
        
        // Check if within view radius
        if (relative_pos.length() <= vf.viewRadius) {
            // Check if within view cone
            Vector2 forward_dir = {cosf(vf.heading), sinf(vf.heading)};
            Vector2 normalized_relative_pos = relative_pos / relative_pos.length();
            float dot_product = normalized_relative_pos.dot(forward_dir);
            float angle = acosf(fmaxf(-1.0f, fminf(1.0f, dot_product))); // clamp to [-1, 1]
            
            if (angle <= vf.viewAngle * 0.5f) {
                candidates.push_back(road);
            }
        }
    }
    
    return candidates;
}

// Narrow-phase filtering: Check for occlusion using simple ray casting
inline void filterByOcclusion(Engine &ctx,
                              const ViewField &vf,
                              std::vector<Entity> &objects) {
    auto &bvh = ctx.singleton<broadphase::BVH>();
    Vector3 observer_pos = {vf.position.x, vf.position.y, vf.position.z};
    
    // Filter out occluded objects
    objects.erase(std::remove_if(objects.begin(), objects.end(), 
        [&](Entity target) {
            const Position &target_pos = ctx.get<Position>(target);
            Vector3 target_pos_3d = {target_pos.x, target_pos.y, target_pos.z};
            
            Vector3 ray_dir = (target_pos_3d - observer_pos).normalize();
            float target_distance = (target_pos_3d - observer_pos).length();
            
            // Trace ray from observer to target
            float hit_t;
            Vector3 hit_normal;
            Entity hit_entity = bvh.traceRay(observer_pos, ray_dir, &hit_t, &hit_normal, target_distance - 0.1f);
            
            // If we hit something before reaching the target, it's occluded
            return (hit_entity != Entity::none() && hit_entity != target);
        }), objects.end()); // erase-remove idiom
}

inline void filterByOcclusionAll(Engine &ctx,
                                 const ViewField &vf, // Contains observer's position
                                 std::vector<Entity> &objects_to_filter // List of potential targets
                                 )
{
    using namespace madrona::math;

    Vector3 observer_pos_3d = {vf.position.x, vf.position.y, vf.position.z};

    std::vector<Entity> visible_entities;
    visible_entities.reserve(objects_to_filter.size());


    for (Entity target_entity : objects_to_filter) {
        const Position& target_pos_comp = ctx.get<Position>(target_entity);
        // Assuming Position component includes Z, or it's implicitly handled (e.g., all agents at a fixed Z for this check if needed)
        Vector3 target_pos_3d = {target_pos_comp.x, target_pos_comp.y, target_pos_comp.z}; 

        Vector3 vec_observer_to_target = target_pos_3d - observer_pos_3d;
        float dist_to_target = vec_observer_to_target.length();
        
        if (dist_to_target < 0.1f) { // add to visible if really close
            visible_entities.push_back(target_entity);
            continue;
        }
        Vector3 normalized_ray_dir_to_target = vec_observer_to_target / dist_to_target;

        bool is_target_occluded = false;

        for (Entity potential_occluder_entity : objects_to_filter) {
            if (potential_occluder_entity == target_entity) {
                continue; // Don't check occlusion against self
            }

            const Position& occluder_pos_comp = ctx.get<Position>(potential_occluder_entity);
            const Rotation& occluder_rot_comp = ctx.get<Rotation>(potential_occluder_entity);
            const VehicleSize& occluder_vehicle_size = ctx.get<VehicleSize>(potential_occluder_entity);

            Diag3x3 occluder_dims_from_vehicle_size = {
                occluder_vehicle_size.width,
                occluder_vehicle_size.length,
                occluder_vehicle_size.height 
            };


            // unit aabb
            AABB local_aabb = {
                {-0.5f, -0.5f, -0.5f},
                { 0.5f,  0.5f,  0.5f}
            };
            
            // transform unit aabb to world space
            AABB world_occluder_aabb = local_aabb.applyTRS(
                occluder_pos_comp, 
                occluder_rot_comp,
                occluder_dims_from_vehicle_size
            );

            // inverse ray direction
            Diag3x3 inv_ray_d = {
                (normalized_ray_dir_to_target.x == 0.f) ? copysignf(FLT_MAX, normalized_ray_dir_to_target.x) : 1.f / normalized_ray_dir_to_target.x,
                (normalized_ray_dir_to_target.y == 0.f) ? copysignf(FLT_MAX, normalized_ray_dir_to_target.y) : 1.f / normalized_ray_dir_to_target.y,
                (normalized_ray_dir_to_target.z == 0.f) ? copysignf(FLT_MAX, normalized_ray_dir_to_target.z) : 1.f / normalized_ray_dir_to_target.z
            };

            float hit_t_occluder = 0.f; 
            // Check intersection from a small epsilon away from observer up to just before the target
            if (world_occluder_aabb.rayIntersects(observer_pos_3d, inv_ray_d, 0.01f, dist_to_target - 0.01f, hit_t_occluder)) {
                is_target_occluded = true;
                break; 
            }
        }

        if (!is_target_occluded) {
            visible_entities.push_back(target_entity);
        }
    }

    objects_to_filter = visible_entities;
}

inline void collectPartnerObsSystem(Engine &ctx,
                              const Position &pos,
                              const Rotation &rot,
                              const OtherAgents &other_agents,
                              const AgentInterfaceEntity &agent_iface)
{
    if(ctx.data().params.disableClassicalObs)
        return;

    // User config
    // viewRadius (float), viewConeHalfAngle (float), viewOccludeObjects (bool; False = no filtering), 

    // to write partner features into array
    auto &partner_obs = ctx.get<PartnerObservations>(agent_iface.e);

    // Get agent's heading from rotation
    float heading = utils::quatToYaw(rot);
    float viewRadius = ctx.data().params.observationRadius;
    float viewAngle = ctx.data().params.viewConeHalfAngle * 2.0f; // Convert half-angle to full angle
    
    ViewField vf(pos, heading, viewRadius, viewAngle);

    // broad phase: remove if not in view field + remove self
    auto objectsDynamic = visibleCandidatesDynamic(ctx, vf, other_agents, agent_iface.e);
        
    // narrow phase: raycast to get occlusion by other dynamic objects
    if (ctx.data().params.viewOccludeObjects) {
        //filterByOcclusion(ctx, vf, objectsDynamic);
        filterByOcclusionAll(ctx, vf, objectsDynamic);
    }

    // add observations to array partner_obs.obs based on filtered list
    CountT arrIndex = 0;
    for (Entity other : objectsDynamic) {
        // future optimization: k nearest maybe in case of many agents in view
        if (arrIndex >= consts::kMaxAgentCount - 1) break;
        
        const Position &other_position = ctx.get<Position>(other);
        const Velocity &other_velocity = ctx.get<Velocity>(other);
        const Rotation &other_rot = ctx.get<Rotation>(other);
        const VehicleSize &other_size = ctx.get<VehicleSize>(other);

        Vector2 relative_pos = (other_position - pos).xy();
        relative_pos = rot.inv().rotateVec({relative_pos.x, relative_pos.y, 0}).xy();
        float relative_speed = other_velocity.linear.length(); // Design decision: return the speed of the other agent directly

        Rotation relative_orientation = rot.inv() * other_rot;
        float relative_heading = utils::quatToYaw(relative_orientation);

        partner_obs.obs[arrIndex++] = {
            .speed = relative_speed,
            .position = relative_pos,
            .heading = relative_heading,
            .vehicle_size = other_size,
            .type = (float)ctx.get<EntityType>(other),
            .id = (float)ctx.get<AgentID>(ctx.get<AgentInterfaceEntity>(other).e).id
        };
    }
    
    // Zero out remaining entries
    while(arrIndex < consts::kMaxAgentCount - 1) {
        partner_obs.obs[arrIndex++] = PartnerObservation::zero();
    }
}

inline void collectMapObservationsSystem(Engine &ctx,
                                        const Position &pos,
                                        const Rotation &rot,
                                        const AgentInterfaceEntity &agent_iface)
{
    if(ctx.data().params.disableClassicalObs)
        return;

    auto &map_obs = ctx.get<AgentMapObservations>(agent_iface.e);

    // User config (some of it not implemented yet, but assume it will be): 
    // viewRadius (float), viewConeHalfAngle (float), viewOccludeObjects (bool; False = no filtering), 

    // Get agent's heading from rotation
    float heading = utils::quatToYaw(rot);
    float view_radius = ctx.data().params.observationRadius;
    float view_angle = ctx.data().params.viewConeHalfAngle * 2.0f;
    
    ViewField vf(pos, heading, view_radius, view_angle);
    auto objectsStatic = visibleCandidatesStatic(ctx, vf);

    const auto alg = ctx.data().params.roadObservationAlgorithm;
    if (alg == FindRoadObservationsWith::KNearestEntitiesWithRadiusFiltering) {
        selectKNearestRoadEntities<consts::kMaxAgentMapObservationsCount>(
            ctx, rot, pos.xy(), map_obs.obs);
        return;
    }

    assert(alg == FindRoadObservationsWith::AllEntitiesWithRadiusFiltering);

    utils::ReferenceFrame referenceFrame(pos.xy(), rot);
    CountT arrIndex = 0;
    
    for (Entity road : objectsStatic) {
        if (arrIndex >= consts::kMaxAgentMapObservationsCount) break;
        
        auto roadPos = ctx.get<Position>(road);
        auto roadRot = ctx.get<Rotation>(road);

        auto dist = referenceFrame.distanceTo(roadPos);
        if (dist > ctx.data().params.observationRadius) {
            continue;
        }

        map_obs.obs[arrIndex] = referenceFrame.observationOf(
            roadPos, roadRot, ctx.get<Scale>(road), ctx.get<EntityType>(road), 
            static_cast<float>(ctx.get<RoadMapId>(road).id), ctx.get<MapType>(road));
        arrIndex++;
    }
    
    // Zero out remaining entries
    while (arrIndex < consts::kMaxAgentMapObservationsCount) {
        map_obs.obs[arrIndex++] = MapObservation::zero();
    }
}

inline void zeroVelocity(Velocity &velocity) {
    velocity.linear = Vector3::zero();
    velocity.angular = Vector3::zero();
}

inline void movementSystem(Engine &e,
                           const AgentInterfaceEntity &agent_iface,
                           VehicleSize &size,
                           Rotation &rotation,
                           Position &position,
                           Velocity &velocity,
                           CollisionDetectionEvent& collisionEvent,
                           const ResponseType &responseType) {

    if (responseType == ResponseType::Static) {
        // Do nothing. The agent is static.
        // Agent can only be static if isStaticAgentControlled is set to true.
        // If agent is static, we dont want to teleport it away in case of collision or goal completion
        return;
    }

    if (collisionEvent.hasCollided.load_relaxed())
    {
        switch (e.data().params.collisionBehaviour){
            case CollisionBehaviour::AgentStop:
            {
                e.get<Done>(agent_iface.e).v = 1;
                zeroVelocity(velocity);
                return;
            }

            case CollisionBehaviour::AgentRemoved:
            {
                e.get<Done>(agent_iface.e).v = 1;
                position = consts::kPaddingPosition;
                zeroVelocity(velocity);
                return;
            }

            case CollisionBehaviour::Ignore:
            { // Reset collision state at the start of each timestep.
                // This ensures the collision state is only true if the agent collided in the current timestep.
                collisionEvent.hasCollided.store_relaxed(0); // Reset the collision state.
                Info &info = e.get<Info>(agent_iface.e);
                info.collidedWithRoad = info.collidedWithVehicle = info.collidedWithNonVehicle = 0;
                break;
            }
        }
    }

    // Check if the agent is done
    if (e.get<Done>(agent_iface.e).v) {
        // This case only happens if the agent is done for reaching goal or episode end or ignoring collisions
        if (e.get<Info>(agent_iface.e).reachedGoal) {
            switch (e.data().params.goalBehaviour){
                case GoalBehaviour::Remove:
                {
                position = consts::kPaddingPosition;
                zeroVelocity(velocity);
                return;
            }

            case GoalBehaviour::Stop:
            {
                zeroVelocity(velocity);
                return;
            }

            case GoalBehaviour::Ignore:
                {
                    break; // We dont want to return here because the agent is still controlled
                }
            }
        }
    }

    const auto &controlledState = e.get<ControlledState>(agent_iface.e);
    if (controlledState.controlled) {
        Action &action = e.get<Action>(agent_iface.e);
        switch (e.data().params.dynamicsModel) {

            case DynamicsModel::InvertibleBicycle:
            {
                forwardBicycleModel(action, rotation, position, velocity);
                break;
            }
            case DynamicsModel::DeltaLocal:
            {
                forwardDeltaModel(action, rotation, position, velocity);
                break;
            }
            case DynamicsModel::Classic:
            {
                forwardKinematics(action, size, rotation, position, velocity);
                break;
            }
            case DynamicsModel::State:
            {
                forwardStateModel(action, rotation, position, velocity);
                break;
            }
        }
    } else {
        // Follow expert trajectory
        const Trajectory &trajectory = e.get<Trajectory>(agent_iface.e);
        CountT curStepIdx = getCurrentStep(e.get<StepsRemaining>(agent_iface.e));
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

static inline float encodeType(EntityType type)
{
    return (float)type;
}

// Launches consts::numLidarSamples per agent.
// This system is specially optimized in the GPU version:
// a warp of threads is dispatched for each invocation of the system
// and each thread in the warp traces one lidar ray for the agent.
inline void lidarSystem(Engine &ctx, Entity e, const AgentInterfaceEntity &agent_iface,
                        EntityType &entityType) {
    Lidar &lidar = ctx.get<Lidar>(agent_iface.e);
    const Action &action = ctx.get<Action>(agent_iface.e);

    Vector3 pos = ctx.get<Position>(e);
    Quat rot = ctx.get<Rotation>(e);
    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 agent_fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);

    auto traceRay = [&](int32_t idx, float offset, LidarSample *samples) {
        // float theta = 2.f * math::pi * (
        //     float(idx) / float(consts::numLidarSamples));
        float head_angle = ctx.get<ControlledState>(agent_iface.e).controlled ? action.classic.headAngle : 0.f;
        float theta = consts::lidarAngle * (2 * float(idx) / float(consts::numLidarSamples) - 1) + head_angle;
        float x = cosf(theta);
        float y = sinf(theta);

        Vector3 ray_dir = (x * right + y * agent_fwd).normalize();

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(pos + offset * math::up, ray_dir, &hit_t,
                         &hit_normal, consts::lidarDistance);

        if (hit_entity == Entity::none()) {
            samples[idx] = {
                .depth = 0.f,
                .encodedType = encodeType(EntityType::None),
                .position = {0.f, 0.f},
            };
        } else {
            EntityType entity_type = ctx.get<EntityType>(hit_entity);

            samples[idx] = {
                .depth = hit_t,
                .encodedType = encodeType(entity_type),
                .position = {hit_t * x,
                             hit_t * y},
            };
        }
    };


    // MADRONA_GPU_MODE guards GPU specific logic
#ifdef MADRONA_GPU_MODE
    // Can use standard cuda variables like threadIdx for
    // warp level programming
    int32_t idx = threadIdx.x % 32;

    while (idx < consts::numLidarSamples) {
        traceRay(idx, consts::lidarCarOffset, lidar.samplesCars);
        traceRay(idx, consts::lidarRoadEdgeOffset, lidar.samplesRoadEdges);
        traceRay(idx, consts::lidarRoadLineOffset, lidar.samplesRoadLines);
        idx += 32;
    }
#else
    for (CountT i = 0; i < consts::numLidarSamples; i++) {
        traceRay(i, consts::lidarCarOffset, lidar.samplesCars);
        traceRay(i, consts::lidarRoadEdgeOffset, lidar.samplesRoadEdges);
        traceRay(i, consts::lidarRoadLineOffset, lidar.samplesRoadLines);
    }
#endif
}

inline void collectBevObservationsSystem(Engine &ctx,
                        const Position &pos,
                        const Rotation &rot,
                        const OtherAgents &other_agents,
                        const AgentInterfaceEntity &agent_iface)
{
    if(ctx.data().params.disableClassicalObs)
        return;

    auto &bev_obs = ctx.get<BevObservations>(agent_iface.e);

    for (size_t i = 0; i < consts::bev_rasterization_resolution; i++)
    {
        for (size_t j = 0; j < consts::bev_rasterization_resolution; j++)
        {
            bev_obs.obs[i][j].type = 0;
        }
    }

    utils::ReferenceFrame referenceFrame(pos.xy(), rot);

    // Roads
    CountT roadIdx = 0;
    CountT arrIndex = 0;
    while (roadIdx < ctx.data().numRoads && arrIndex < consts::kMaxAgentMapObservationsCount)
    {
        Entity road = ctx.data().roads[roadIdx++];
        auto roadPos = ctx.get<Position>(road);
        auto roadRot = ctx.get<Rotation>(road);
        const MapObservation &map_obs = referenceFrame.observationOf(
            roadPos,
            roadRot,
            ctx.get<Scale>(road),
            ctx.get<EntityType>(road),
            static_cast<float>(ctx.get<RoadMapId>(road).id),
            ctx.get<MapType>(road)
        );

        auto dist = referenceFrame.distanceTo(roadPos);
        if (dist > ctx.data().params.observationRadius)
            continue;

        madrona::math::Vector2 rel_pos = map_obs.position;
        float rel_yaw = map_obs.heading;
        auto new_scale = map_obs.scale;
        new_scale.d1 = std::max( //Ensure minimum segment width
            map_obs.scale.d1,
            (2 * ctx.data().params.observationRadius / consts::bev_rasterization_resolution)
        );

        rasterizer::rasterizeRotatedRectangle(
            bev_obs,
            rel_pos,
            rel_yaw,
            new_scale.d0,
            new_scale.d1,
            map_obs.type,
            ctx.data().params.observationRadius,
            consts::bev_rasterization_resolution
        );
        arrIndex++;
    }

    // Other agents
    CountT agentIdx = 0;
    while (agentIdx < ctx.data().numAgents - 1)
    {
        Entity other = other_agents.e[agentIdx++];

        const Position &other_position = ctx.get<Position>(other);
        const Rotation &other_rot = ctx.get<Rotation>(other);
        const VehicleSize &other_size = ctx.get<VehicleSize>(other);
        const auto type = static_cast<size_t>(ctx.get<EntityType>(other));
        Vector2 relative_pos = (other_position - pos).xy();
        relative_pos = rot.inv().rotateVec({relative_pos.x, relative_pos.y, 0}).xy();
        Rotation relative_orientation = rot.inv() * other_rot;

        float relative_heading = utils::quatToYaw(relative_orientation);

        if(relative_pos.length() > ctx.data().params.observationRadius)
            continue;

        rasterizer::rasterizeRotatedRectangle(
            bev_obs,
            relative_pos,
            relative_heading,
            other_size.length,
            other_size.width,
            type,
            ctx.data().params.observationRadius,
            consts::bev_rasterization_resolution
        );
    }
}

// Computes reward for each agent and keeps track of the max distance achieved
// so far through the challenge. Continuous reward is provided for any new
// distance achieved.
inline void rewardSystem(Engine &ctx,
                         const Position &position,
                         const Goal &goal,
                         const AgentInterfaceEntity &agent_iface)
{
    Reward &out_reward = ctx.get<Reward>(agent_iface.e);
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

inline void stepTrackerSystem(Engine &ctx, const AgentInterfaceEntity &agent_iface) {
    StepsRemaining &stepsRemaining = ctx.get<StepsRemaining>(agent_iface.e);
    --stepsRemaining.t;
}

// Keep track of the number of steps remaining in the episode and
// notify training that an episode has completed by
// setting done = 1 on the final step of the episode
inline void doneSystem(Engine &ctx,
                      const Position &position,
                      const Goal &goal,
                      AgentInterfaceEntity &agent_iface)
{
    StepsRemaining &steps_remaining = ctx.get<StepsRemaining>(agent_iface.e);
    Done &done = ctx.get<Done>(agent_iface.e);
    Info &info = ctx.get<Info>(agent_iface.e);
    int32_t num_remaining = steps_remaining.t;
    if (num_remaining == consts::episodeLen && done.v != 1)
    { // Make sure to not reset an agent's done flag
        done.v = 0;
        return;
    }
    else if (num_remaining == 0)
    {
        done.v = 1;
    }

    // // An agent can be done early if it reaches the goal
    // if (done.v != 1 || info.reachedGoal != 1)
    // {
    //     float dist = (position.xy() - goal.position).length();
    //     if (dist < ctx.data().params.rewardParams.distanceToGoalThreshold)
    //     {
    //         done.v = 1;
    //         info.reachedGoal = 1;
    //     }
    // }
}


void collisionDetectionSystem(Engine &ctx,
                              const CandidateCollision &candidateCollision) {

    auto isInvalidExpertOrDone = [&](const Loc &candidate) -> bool
    {
        auto agent_iface = ctx.getCheck<AgentInterfaceEntity>(candidate);
        if (agent_iface.valid())
        {
            auto controlledState = ctx.get<ControlledState>(agent_iface.value().e).controlled;
            // Case: If an expert agent is in an invalid state, we need to ignore the collision detection for it.
            if (controlledState == false)
            {
                auto currStep = getCurrentStep(ctx.get<StepsRemaining>(agent_iface.value().e));
                auto &validState = ctx.get<Trajectory>(agent_iface.value().e).valids[currStep];
                if (!validState)
                {
                    return true;
                }
            }
            else
            {
                // Case: If a controlled agent gets done, we teleport it to the padding position
                // Hence we need to ignore the collision detection for it.
                // The agent can also be done because it collided.
                // In that case, we dont want to ignore collision. Especially if AgentStop is set.
                auto &done = ctx.get<Done>(agent_iface.value().e);
                auto &collisionEvent = ctx.get<CollisionDetectionEvent>(candidate);
                if (done.v && !collisionEvent.hasCollided.load_relaxed())
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
        auto agent_iface = ctx.get<AgentInterfaceEntity>(candidateCollision.a).e;
        if(bEntityType > EntityType::None && bEntityType <= EntityType::StopSign)
        {
            ctx.get<Info>(agent_iface).collidedWithRoad = 1;
        }
        else if(bEntityType == EntityType::Vehicle)
        {
            ctx.get<Info>(agent_iface).collidedWithVehicle = 1;
        }
        else if(bEntityType <= EntityType::Cyclist)
        {
            ctx.get<Info>(agent_iface).collidedWithNonVehicle = 1;
        }
    }

    auto maybeCollisionDetectionEventB =
        ctx.getCheck<CollisionDetectionEvent>(candidateCollision.b);
    if (maybeCollisionDetectionEventB.valid()) {
        maybeCollisionDetectionEventB.value().hasCollided.store_relaxed(1);
        auto agent_iface = ctx.get<AgentInterfaceEntity>(candidateCollision.b).e;
        if(aEntityType > EntityType::None && aEntityType <= EntityType::StopSign)
        {
            ctx.get<Info>(agent_iface).collidedWithRoad = 1;
        }
        else if(aEntityType == EntityType::Vehicle)
        {
            ctx.get<Info>(agent_iface).collidedWithVehicle = 1;
        }
        else if(aEntityType <= EntityType::Cyclist)
        {
            ctx.get<Info>(agent_iface).collidedWithNonVehicle = 1;
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
                                              const VehicleSize &vehicleSize,
                                              AgentInterfaceEntity &agent_iface) {

    auto &out = ctx.get<AbsoluteSelfObservation>(agent_iface.e);
    out.position = position;
    out.rotation.rotationAsQuat = rotation;
    out.rotation.rotationFromAxis = utils::quatToYaw(rotation);
    out.goal = goal;
    out.vehicle_size = vehicleSize;
    out.id = ctx.get<AgentID>(agent_iface.e).id;
}

void setupRestOfTasks(TaskGraphBuilder &builder, const Sim::Config &cfg,
                      Span<const TaskGraphNodeID> dependencies,
                      bool decrementStep) {
    // setupBroadphaseTasks consists of the following sub-tasks:
    // 1. updateLeafPositionsEntry
    // 2. broadphase::updateBVHEntry
    // 3. broadphase::refitEntry
    auto broadphase_setup_sys =
        phys::PhysicsSystem::setupBroadphaseTasks(builder, dependencies);

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
            Goal,
            AgentInterfaceEntity
        >>({phys_done});

    auto previousSystem = reward_sys;
    if (decrementStep) {
        previousSystem = builder.addToGraph<
            ParallelForNode<Engine, stepTrackerSystem, AgentInterfaceEntity>>(
            {reward_sys});
    }

    // Check if the episode is over
    auto done_sys =
        builder.addToGraph<ParallelForNode<Engine, doneSystem, Position, Goal,
                                           AgentInterfaceEntity>>(
            {previousSystem});

    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>({done_sys});
    (void)clear_tmp;


#ifdef MADRONA_GPU_MODE
    // RecycleEntitiesNode is required on the GPU backend in order to reclaim
    // deleted entity IDs.
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({done_sys});
    (void)recycle_sys;
#endif

    // Finally, collect observations for the next step.
    // auto collect_obs = builder.addToGraph<ParallelForNode<Engine,
    //     collectObservationsSystem,
    //         VehicleSize,
    //         Position,
    //         Rotation,
    //         Velocity,
    //         Goal,
    //         Progress,
    //         OtherAgents,
    //         EntityType,
    //         CollisionDetectionEvent,
    //         AgentInterfaceEntity
    //     >>({clear_tmp});

    auto collect_self_obs = builder.addToGraph<ParallelForNode<Engine,
        collectSelfObsSystem,
        VehicleSize,
        Position,
        Rotation,
        Velocity,
        Goal,
        CollisionDetectionEvent,
        AgentInterfaceEntity>>({clear_tmp});

    auto collect_partner_obs = builder.addToGraph<ParallelForNode<Engine,
        collectPartnerObsSystem,
        Position,
        Rotation,
        OtherAgents,
        AgentInterfaceEntity>>({clear_tmp});

    auto collect_map_obs = builder.addToGraph<ParallelForNode<Engine,
        collectMapObservationsSystem,
        Position,
        Rotation,
        AgentInterfaceEntity>>({clear_tmp});

    auto collect_bev_obs = builder.addToGraph<ParallelForNode<Engine,
        collectBevObservationsSystem,
        Position,
        Rotation,
        OtherAgents,
        AgentInterfaceEntity>>({clear_tmp});

    auto collectAbsoluteSelfObservations = builder.addToGraph<
        ParallelForNode<Engine, collectAbsoluteObservationsSystem, Position,
                        Rotation, Goal, VehicleSize, AgentInterfaceEntity>>(
        {clear_tmp});

    if (cfg.renderBridge) {
        RenderingSystem::setupTasks(builder, dependencies);
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
            AgentInterfaceEntity,
            EntityType
        >>({clear_tmp});
    }

#ifdef MADRONA_GPU_MODE
    TaskGraphNodeID sort_agents;
    if(cfg.enableLidar)
    {
        sort_agents = queueSortByWorld<Agent>(builder, {lidar, collect_self_obs, collect_partner_obs, collect_map_obs, collectAbsoluteSelfObservations});
    } else {
        sort_agents = queueSortByWorld<Agent>(builder, {collect_self_obs, collect_partner_obs, collect_map_obs, collectAbsoluteSelfObservations});
    }
    // Sort entities, this could be conditional on reset like the second
    // BVH build above.

    auto sort_phys_objects = queueSortByWorld<PhysicsEntity>(
        builder, {sort_agents});

    auto sort_agent_ifaces = queueSortByWorld<AgentInterface>(
        builder, {sort_phys_objects});

    auto sort_road_ifaces = queueSortByWorld<RoadInterface>(
        builder, {sort_agent_ifaces});
    (void)sort_road_ifaces;
#else
    (void)lidar;
    (void)collect_self_obs;
    (void)collect_partner_obs;
    (void)collect_map_obs;
    (void)collectAbsoluteSelfObservations;
    (void)collect_bev_obs;
#endif
}

static void setupStepTasks(TaskGraphBuilder &builder, const Sim::Config &cfg) {
    auto moveSystem = builder.addToGraph<ParallelForNode<Engine,
        movementSystem,
            AgentInterfaceEntity,
            VehicleSize,
            Rotation,
            Position,
            Velocity,
            CollisionDetectionEvent,
            ResponseType
        >>({});

    setupRestOfTasks(builder, cfg, {moveSystem}, true);
}

static void setupResetTasks(TaskGraphBuilder &builder, const Sim::Config &cfg) {
    auto reset =
        builder.addToGraph<ParallelForNode<Engine, resetSystem, WorldReset>>(
            {});

    setupRestOfTasks(builder, cfg, {reset}, false);
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
    // auto max_total_entities = init.map->numObjects + init.map->numRoadSegments;
    auto max_total_entities = consts::kMaxAgentCount + consts::kMaxRoadEntityCount;

    phys::PhysicsSystem::init(ctx, init.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -9.8f * math::up,
        max_total_entities);

    enableRender = cfg.renderBridge != nullptr;

    if (enableRender) {
        RenderingSystem::init(ctx, cfg.renderBridge);
    }

    auto& map = ctx.singleton<Map>();
    map = *(init.map);

    auto& deletedAgents = ctx.singleton<DeletedAgents>();
    for (auto i = 0; i < consts::kMaxAgentCount; i++) {
        deletedAgents.deletedAgents[i] = -1;
    }
    // Creates agents, walls, etc.
    createPersistentEntities(ctx);

    // Generate initial world state
    initWorld(ctx);
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
