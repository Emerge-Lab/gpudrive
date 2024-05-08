#include "level_gen.hpp"
#include "utils.hpp"

namespace gpudrive {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

// Register the entity with the broadphase system
// This is needed for every entity with all the physics components.
// Not registering an entity will cause a crash because the broadphase
// systems will still execute over entities with the physics components.
static void registerRigidBodyEntity(
    Engine &ctx,
    Entity e,
    SimObject sim_obj)
{
    ObjectID obj_id { (int32_t)sim_obj };
    ctx.get<broadphase::LeafID>(e) = PhysicsSystem::registerEntity(ctx, e, obj_id);
}

static inline void resetAgent(Engine &ctx, Entity agent) {
    auto agent_iface = ctx.get<InterfaceEntity>(agent).e;
    auto xCoord = ctx.get<Trajectory>(agent).positions[0].x;
    auto yCoord = ctx.get<Trajectory>(agent).positions[0].y;
    auto xVelocity = ctx.get<Trajectory>(agent).velocities[0].x;
    auto yVelocity = ctx.get<Trajectory>(agent).velocities[0].y;
    auto speed = ctx.get<Trajectory>(agent).velocities[0].length();
    auto heading = ctx.get<Trajectory>(agent).headings[0];

    ctx.get<BicycleModel>(agent) = {
        .position = {.x = xCoord, .y = yCoord}, .heading = heading, .speed = speed};
    ctx.get<Position>(agent) = Vector3{.x = xCoord, .y = yCoord, .z = 1};
    ctx.get<Rotation>(agent) = Quat::angleAxis(heading, madrona::math::up);
    ctx.get<Velocity>(agent) = {
        Vector3{.x = xVelocity, .y = yVelocity, .z = 0}, Vector3::zero()};
    ctx.get<Action>(agent_iface) =
        Action{.acceleration = 0, .steering = 0, .headAngle = 0};
    ctx.get<StepsRemaining>(agent_iface).t = consts::episodeLen;
    ctx.get<Done>(agent_iface).v = 0;
    ctx.get<Reward>(agent_iface).v = 0;
    ctx.get<Info>(agent_iface) = Info{};
    ctx.get<Info>(agent_iface).type = (int32_t)ctx.get<EntityType>(agent);

#ifndef GPUDRIVE_DISABLE_NARROW_PHASE
    ctx.get<CollisionDetectionEvent>(agent).hasCollided.store_release(0);
#endif
}


static inline Entity createAgent(Engine &ctx, const MapObject &agentInit) {
    auto agent = ctx.makeRenderableEntity<Agent>();
    
    // The following components do not vary within an episode and so need only
    // be set once
    ctx.get<VehicleSize>(agent) = {.length = agentInit.length, .width = agentInit.width};
    ctx.get<Scale>(agent) = Diag3x3{.d0 = agentInit.length/2, .d1 = agentInit.width/2, .d2 = 1};
    ctx.get<ObjectID>(agent) = ObjectID{(int32_t)SimObject::Agent};
    ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
    assert(agentInit.type >= EntityType::Vehicle || agentInit.type == EntityType::None);
    ctx.get<EntityType>(agent) = agentInit.type;

    auto agent_iface = ctx.get<InterfaceEntity>(agent).e = ctx.makeEntity<AgentInterface>();

    ctx.get<Goal>(agent)= Goal{.position = Vector2{.x = agentInit.goalPosition.x - ctx.data().mean.x, .y = agentInit.goalPosition.y - ctx.data().mean.y}};
    if(ctx.data().numControlledVehicles < ctx.data().params.maxNumControlledVehicles && agentInit.type == EntityType::Vehicle && agentInit.valid[0])
    {
        ctx.get<ControlledState>(agent_iface) = ControlledState{.controlledState = ControlMode::BICYCLE};
        ctx.data().numControlledVehicles++;
    }
    else
    {
        ctx.get<ControlledState>(agent_iface) = ControlledState{.controlledState = ControlMode::EXPERT};
    }

    // Since position, heading, and speed may vary within an episode, their
    // values are retained so that on an episode reset they can be restored to
    // their initial values.
    auto &trajectory = ctx.get<Trajectory>(agent);
    for(CountT i = 0; i < agentInit.numPositions; i++)
    {
        trajectory.positions[i] = Vector2{.x = agentInit.position[i].x - ctx.data().mean.x, .y = agentInit.position[i].y - ctx.data().mean.y};
        trajectory.velocities[i] = Vector2{.x = agentInit.velocity[i].x, .y = agentInit.velocity[i].y};
        trajectory.headings[i] = toRadians(agentInit.heading[i]);
        trajectory.valids[i] = agentInit.valid[i];
    }

    // This is not stricly necessary since , but is kept here for consistency
    resetAgent(ctx, agent);

    if (ctx.data().enableRender) {
        render::RenderingSystem::attachEntityToView(ctx,
                agent,
                90.f, 0.001f,
                1.5f * math::up);
    }

    return agent;
}

static Entity makeRoadEdge(Engine &ctx, const MapVector2 &p1,
                           const MapVector2 &p2, const EntityType &type) {
    float x1 = p1.x;
    float y1 = p1.y;
    float x2 = p2.x;
    float y2 = p2.y;

    Vector3 start{.x = x1 - ctx.data().mean.x, .y = y1 - ctx.data().mean.y, .z = 1};
    Vector3 end{.x = x2 - ctx.data().mean.x, .y = y2 - ctx.data().mean.y, .z = 1};
    float distance = end.distance(start);
    auto road_edge = ctx.makeRenderableEntity<PhysicsEntity>();
    ctx.get<Position>(road_edge) = Vector3{.x = (start.x + end.x)/2, .y = (start.y + end.y)/2, .z = 1};
    ctx.get<Rotation>(road_edge) = Quat::angleAxis(atan2(end.y - start.y, end.x - start.x), madrona::math::up);
    ctx.get<Scale>(road_edge) = Diag3x3{.d0 = distance/2, .d1 = 0.1, .d2 = 0.1};
    ctx.get<EntityType>(road_edge) = type;
    ctx.get<ObjectID>(road_edge) = ObjectID{(int32_t)SimObject::Cube};
    registerRigidBodyEntity(ctx, road_edge, SimObject::Cube);
    ctx.get<ResponseType>(road_edge) = ResponseType::Static;
    auto road_iface = ctx.get<RoadInterfaceEntity>(road_edge).e = ctx.makeEntity<RoadInterface>();
    ctx.get<MapObservation>(road_iface) = MapObservation{.position = ctx.get<Position>(road_edge).xy(),
                                                        .scale = ctx.get<Scale>(road_edge), 
                                                        .heading = utils::quatToYaw(ctx.get<Rotation>(road_edge)), 
                                                        .type = (float)type};
    return road_edge;
}

float calculateDistance(float x1, float y1, float x2, float y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

static Entity makeCube(Engine &ctx, const MapVector2 &p1, const MapVector2 &p2, const MapVector2 &p3,
                            const MapVector2 &p4, const EntityType &type) {
    float x1 = p1.x;
    float y1 = p1.y;
    float x2 = p2.x;
    float y2 = p2.y;
    float x3 = p3.x;
    float y3 = p3.y;
    float x4 = p4.x;
    float y4 = p4.y;

    // Calculate distances (sides and diagonals)
    float d12 = calculateDistance(x1, y1, x2, y2); // Side 1-2
    float d23 = calculateDistance(x2, y2, x3, y3); // Side 2-3
    float d34 = calculateDistance(x3, y3, x4, y4); // Side 3-4
    float d41 = calculateDistance(x4, y4, x1, y1); // Side 4-1

    float lengths[] = {d12, d23, d34, d41};
    int maxLength_i = 0;
    int minLength_i = 0;

    #pragma unroll
    for (int i = 1; i < 4; ++i) {
        if (lengths[i] > lengths[maxLength_i])
            maxLength_i = i;
        if (lengths[i] < lengths[minLength_i]) 
            minLength_i = i;
    }

    float coords[] = {0, 0, 0, 0};
    switch(maxLength_i)
    {
        case 0:
            coords[0] = x1; coords[1] = y1;
            coords[2] = x2; coords[3] = y2;
            break;
        case 1:
            coords[0] = x2; coords[1] = y2;
            coords[2] = x3; coords[3] = y3;
            break;
        case 2:
            coords[0] = x3; coords[1] = y3;
            coords[2] = x4; coords[3] = y4;
            break;
        case 3:
            coords[0] = x4; coords[1] = y4;
            coords[2] = x1; coords[3] = y1;
            break;
        default:
            break;
    }
    // Calculate rotation angle (assuming longer side is used to calculate angle)
    float angle = atan2(coords[3] - coords[1], coords[2] - coords[0]);

    auto speed_bump = ctx.makeRenderableEntity<PhysicsEntity>();
    ctx.get<Position>(speed_bump) = Vector3{.x = (x1 + x2 + x3 + x4)/4 - ctx.data().mean.x, .y = (y1 + y2 + y3 + y4)/4 - ctx.data().mean.y, .z = 1};
    ctx.get<Rotation>(speed_bump) = Quat::angleAxis(angle, madrona::math::up);
    ctx.get<Scale>(speed_bump) = Diag3x3{.d0 = lengths[maxLength_i]/2, .d1 = lengths[minLength_i]/2, .d2 = 0.1};
    ctx.get<EntityType>(speed_bump) = type;
    ctx.get<ObjectID>(speed_bump) = ObjectID{(int32_t)SimObject::SpeedBump};
    registerRigidBodyEntity(ctx, speed_bump, SimObject::SpeedBump);
    ctx.get<ResponseType>(speed_bump) = ResponseType::Static;
    auto road_iface = ctx.get<RoadInterfaceEntity>(speed_bump).e = ctx.makeEntity<RoadInterface>();
    ctx.get<MapObservation>(road_iface) = MapObservation{.position = ctx.get<Position>(speed_bump).xy(),
                                                         .scale = ctx.get<Scale>(speed_bump), 
                                                         .heading = utils::quatToYaw(ctx.get<Rotation>(speed_bump)), 
                                                         .type = (float)type};
    return speed_bump;
}

static Entity makeStopSign(Engine &ctx, const MapVector2 &p1) {
    float x1 = p1.x;
    float y1 = p1.y;

    auto stop_sign = ctx.makeRenderableEntity<PhysicsEntity>();
    ctx.get<Position>(stop_sign) = Vector3{.x = x1 - ctx.data().mean.x, .y = y1 - ctx.data().mean.y, .z = 0.5};
    ctx.get<Rotation>(stop_sign) = Quat::angleAxis(0, madrona::math::up);
    ctx.get<Scale>(stop_sign) = Diag3x3{.d0 = 0.2, .d1 = 0.2, .d2 = 0.5};
    ctx.get<EntityType>(stop_sign) = EntityType::StopSign;
    ctx.get<ObjectID>(stop_sign) = ObjectID{(int32_t)SimObject::StopSign};
    registerRigidBodyEntity(ctx, stop_sign, SimObject::StopSign);
    ctx.get<ResponseType>(stop_sign) = ResponseType::Static;
    auto road_iface = ctx.get<RoadInterfaceEntity>(stop_sign).e = ctx.makeEntity<RoadInterface>();
    ctx.get<MapObservation>(road_iface) = MapObservation{.position = ctx.get<Position>(stop_sign).xy(),
                                                        .scale = ctx.get<Scale>(stop_sign), 
                                                        .heading = utils::quatToYaw(ctx.get<Rotation>(stop_sign)), 
                                                        .type = (float)EntityType::StopSign};
    return stop_sign;
}

static inline void createRoadEntities(Engine &ctx, const MapRoad &roadInit, CountT &idx) {
    if (roadInit.type == EntityType::RoadEdge || roadInit.type == EntityType::RoadLine || roadInit.type == EntityType::RoadLane)
    {
        size_t numPoints = roadInit.numPoints;
        for (size_t j = 1; j <= numPoints - 1; j++)
        {
            if (idx >= consts::kMaxRoadEntityCount)
                return;
            auto road = ctx.data().roads[idx] = makeRoadEdge(ctx, roadInit.geometry[j - 1], roadInit.geometry[j], roadInit.type);
            ctx.data().road_ifaces[idx++] = ctx.get<RoadInterfaceEntity>(road).e;
        }
    }
    else if (roadInit.type == EntityType::SpeedBump || roadInit.type == EntityType::CrossWalk)
    {
        assert(roadInit.numPoints >= 4);
        // TODO: Speed Bump are not guranteed to have 4 points. Need to handle this case.
        if (idx >= consts::kMaxRoadEntityCount)
            return;
        auto road = ctx.data().roads[idx] = makeCube(ctx, roadInit.geometry[0], roadInit.geometry[1], roadInit.geometry[2], roadInit.geometry[3], roadInit.type);
        ctx.data().road_ifaces[idx++] = ctx.get<RoadInterfaceEntity>(road).e;
    }
    else if (roadInit.type == EntityType::StopSign)
    {
        assert(roadInit.numPoints >= 1);
        // TODO: Stop Sign are not guranteed to have 1 point. Need to handle this case.
        if (idx >= consts::kMaxRoadEntityCount)
            return;
        auto road = ctx.data().roads[idx] = makeStopSign(ctx, roadInit.geometry[0]);
        ctx.data().road_ifaces[idx++] = ctx.get<RoadInterfaceEntity>(road).e;
    }
    else
    {
        // TODO: Need to handle Cross Walk.
        //   assert(false);
        return;
    }
}

static void createFloorPlane(Engine &ctx)
{
    ctx.data().floorPlane = ctx.makeRenderableEntity<PhysicsEntity>();
    ctx.get<Position>(ctx.data().floorPlane) = Vector3{.x = 0, .y = 0, .z = 0};
    ctx.get<Rotation>(ctx.data().floorPlane) = Quat { 1, 0, 0, 0 };
    ctx.get<Scale>(ctx.data().floorPlane) = Diag3x3{1, 1, 1};
    ctx.get<ObjectID>(ctx.data().floorPlane) = ObjectID{(int32_t)SimObject::Plane};
    ctx.get<Velocity>(ctx.data().floorPlane) = {Vector3::zero(), Vector3::zero()};
    ctx.get<ResponseType>(ctx.data().floorPlane) = ResponseType::Static;
    ctx.get<EntityType>(ctx.data().floorPlane) = EntityType::None;
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);
}

static inline Entity createPhysicsEntityPadding(Engine &ctx) {
    auto physicsEntity = ctx.makeRenderableEntity<PhysicsEntity>();

    ctx.get<Position>(physicsEntity) = consts::kPaddingPosition;
    ctx.get<Rotation>(physicsEntity) = Quat::angleAxis(0, madrona::math::up);
    ctx.get<Scale>(physicsEntity) = Diag3x3{.d0 = 0, .d1 = 0, .d2 = 0};
    ctx.get<Velocity>(physicsEntity) = {Vector3::zero(), Vector3::zero()};
    ctx.get<ObjectID>(physicsEntity) = ObjectID{(int32_t)SimObject::Cube};
    ctx.get<ResponseType>(physicsEntity) = ResponseType::Static;
    ctx.get<MapObservation>(physicsEntity) = MapObservation{.position = ctx.get<Position>(physicsEntity).xy(), 
                                                            .scale = ctx.get<Scale>(physicsEntity),
                                                            .heading = utils::quatToYaw(ctx.get<Rotation>(physicsEntity)), 
                                                            .type = 0};
    ctx.get<EntityType>(physicsEntity) = EntityType::Padding;

    return physicsEntity;
}

void createPaddingEntities(Engine &ctx) {
    for (CountT agentIdx = ctx.data().numAgents;
         agentIdx < consts::kMaxAgentCount; ++agentIdx) {
        ctx.data().agent_ifaces[agentIdx] = ctx.makeEntity<AgentInterface>();
    }

    for (CountT roadIdx = ctx.data().numRoads;
         roadIdx < consts::kMaxRoadEntityCount; ++roadIdx) {
        // ctx.data().roads[roadIdx] = createPhysicsEntityPadding(ctx);
        ctx.data().road_ifaces[roadIdx] = ctx.makeEntity<RoadInterface>();
    }
}

void createPersistentEntities(Engine &ctx, Map *map) {
    // createFloorPlane(ctx);
    ctx.data().mean = {0, 0};
    ctx.data().mean.x = map->mean.x;
    ctx.data().mean.y = map->mean.y;
    ctx.data().numControlledVehicles = 0;

    CountT agentIdx = 0;
    for (CountT agentCtr = 0; agentCtr < map->numObjects; ++agentCtr) {
        if(agentIdx >= consts::kMaxAgentCount)
            break;
        if (ctx.data().params.IgnoreNonVehicles)
        {
            if (map->objects[agentCtr].type == EntityType::Pedestrian || map->objects[agentCtr].type == EntityType::Cyclist)
            {
                continue;
            }
        }
        const auto &agentInit = map->objects[agentCtr];
        auto agent = createAgent(
            ctx, agentInit);
        ctx.data().agent_ifaces[agentIdx] = ctx.get<InterfaceEntity>(agent).e;
        ctx.data().agents[agentIdx++] = agent;
    } 
    
    ctx.data().numAgents = agentIdx; 

    CountT roadIdx = 0;
    for(CountT roadCtr = 0; roadCtr < map->numRoads; roadCtr++)
    {
        if(roadIdx >= consts::kMaxRoadEntityCount)
            break;
        const auto &roadInit = map->roads[roadCtr];
        createRoadEntities(ctx, roadInit, roadIdx);
    }
    ctx.data().numRoads = roadIdx;

    auto &shape = ctx.singleton<Shape>();
    shape.agentEntityCount = ctx.data().numAgents;
    shape.roadEntityCount = ctx.data().numRoads;

    createPaddingEntities(ctx);
}

static void resetPersistentEntities(Engine &ctx)
{

    for (CountT idx = 0; idx < ctx.data().numAgents; ++idx)
    {
        Entity agent = ctx.data().agents[idx];

        resetAgent(ctx, agent);

        registerRigidBodyEntity(ctx, agent, SimObject::Agent);
    }

    for (CountT idx = 0; idx < ctx.data().numRoads; idx++)
    {
      Entity road = ctx.data().roads[idx];
      if(road == Entity::none()) break;
      if(ctx.get<ObjectID>(road).idx == (int32_t)SimObject::Cube){
        registerRigidBodyEntity(ctx, road, SimObject::Cube);
      }
      else if (ctx.get<ObjectID>(road).idx == (int32_t)SimObject::StopSign){
        registerRigidBodyEntity(ctx, road, SimObject::StopSign);
      }
      else if (ctx.get<ObjectID>(road).idx == (int32_t)SimObject::SpeedBump){
        registerRigidBodyEntity(ctx, road, SimObject::SpeedBump);
      }
    }
  
    for (CountT i = 0; i < ctx.data().numAgents; i++) {
        Entity cur_agent = ctx.data().agents[i];
        OtherAgents &other_agents = ctx.get<OtherAgents>(cur_agent);
        CountT out_idx = 0;
        for (CountT j = 0; j < ctx.data().numAgents; j++)
        {
            if (i == j)
            {
                continue;
            }

            Entity other_agent = ctx.data().agents[j];
            other_agents.e[out_idx++] = other_agent;
        }
    }
}

void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
}

}
