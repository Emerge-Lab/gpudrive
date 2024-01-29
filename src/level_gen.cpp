#include "level_gen.hpp"


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
    ctx.get<broadphase::LeafID>(e) =
        RigidBodyPhysicsSystem::registerEntity(ctx, e, obj_id);
}

static inline void resetVehicle(Engine &ctx, Entity vehicle) {
    auto xCoord = ctx.get<Trajectory>(vehicle).positions[0].x;
    auto yCoord = ctx.get<Trajectory>(vehicle).positions[0].y;
    auto xVelocity = ctx.get<Trajectory>(vehicle).velocities[0].x;
    auto yVelocity = ctx.get<Trajectory>(vehicle).velocities[0].y;
    auto speed = ctx.get<Trajectory>(vehicle).velocities[0].length();
    auto heading = ctx.get<Trajectory>(vehicle).initialHeading;

    ctx.get<BicycleModel>(vehicle) = {
        .position = {.x = xCoord, .y = yCoord}, .heading = heading, .speed = speed};
    ctx.get<Position>(vehicle) = Vector3{.x = xCoord, .y = yCoord, .z = 1};
    ctx.get<Rotation>(vehicle) = Quat::angleAxis(heading, madrona::math::up);
    ctx.get<Velocity>(vehicle) = {
        Vector3{.x = xVelocity, .y = yVelocity, .z = 0}, Vector3::zero()};
    ctx.get<ExternalForce>(vehicle) = Vector3::zero();
    ctx.get<ExternalTorque>(vehicle) = Vector3::zero();
    ctx.get<Action>(vehicle) =
        Action{.acceleration = 0, .steering = 0, .headAngle = 0};
    ctx.get<StepsRemaining>(vehicle).t = consts::episodeLen;
}

static inline Entity createVehicle(Engine &ctx, const MapObject &agentInit) {
    auto vehicle = ctx.makeEntity<Agent>();
    
    // The following components do not vary within an episode and so need only
    // be set once
    ctx.get<VehicleSize>(vehicle) = {.length = agentInit.length, .width = agentInit.width};
    ctx.get<Scale>(vehicle) = Diag3x3{.d0 = agentInit.length/2, .d1 = agentInit.width/2, .d2 = 1};
    ctx.get<ObjectID>(vehicle) = ObjectID{(int32_t)SimObject::Agent};
    ctx.get<ResponseType>(vehicle) = ResponseType::Dynamic;
    ctx.get<EntityType>(vehicle) = EntityType::Agent;
    ctx.get<Goal>(vehicle)= Goal{.position = Vector2{.x = agentInit.goalPosition.x - ctx.data().mean.x, .y = agentInit.goalPosition.y - ctx.data().mean.y}};
    // Since position, heading, and speed may vary within an episode, their
    // values are retained so that on an episode reset they can be restored to
    // their initial values.
    ctx.get<Trajectory>(vehicle).positions[0] =
        Vector2{.x = agentInit.position[0].x - ctx.data().mean.x, .y = agentInit.position[0].y - ctx.data().mean.y};
    ctx.get<Trajectory>(vehicle).initialHeading = toRadians(agentInit.heading[0]);
    ctx.get<Trajectory>(vehicle).velocities[0] =
        Vector2{.x = agentInit.velocity[0].x, .y = agentInit.velocity[0].y};
    ctx.get<ValidState>(vehicle) = ValidState{.isValid = agentInit.valid[0]};
    // This is not stricly necessary since , but is kept here for consistency
    resetVehicle(ctx, vehicle);

    return vehicle;
}

static Entity makeRoadEdge(Engine &ctx, const MapVector2 &p1,
                           const MapVector2 &p2, const MapRoadType &type) {
    float x1 = p1.x;
    float y1 = p1.y;
    float x2 = p2.x;
    float y2 = p2.y;

    Vector3 start{.x = x1 - ctx.data().mean.x, .y = y1 - ctx.data().mean.y, .z = 0};
    Vector3 end{.x = x2 - ctx.data().mean.x, .y = y2 - ctx.data().mean.y, .z = 0};
    float distance = end.distance(start);
    auto road_edge = ctx.makeEntity<PhysicsEntity>();
    ctx.get<Position>(road_edge) = Vector3{.x = (start.x + end.x)/2, .y = (start.y + end.y)/2, .z = 0};
    ctx.get<Rotation>(road_edge) = Quat::angleAxis(atan2(end.y - start.y, end.x - start.x), madrona::math::up);
    ctx.get<Scale>(road_edge) = Diag3x3{.d0 = distance/2, .d1 = 0.1, .d2 = 0.1};
    ctx.get<EntityType>(road_edge) = EntityType::Cube;
    ctx.get<ObjectID>(road_edge) = ObjectID{(int32_t)SimObject::Cube};
    registerRigidBodyEntity(ctx, road_edge, SimObject::Cube);
    ctx.get<ResponseType>(road_edge) = ResponseType::Static;
    ctx.get<MapObservation>(road_edge) = MapObservation{.position = Vector2{.x = (start.x + end.x)/2, .y = (start.y + end.y)/2}, .heading = atan2(end.y - start.y, end.x - start.x), .type = (float)type};
    return road_edge;
}

float calculateDistance(float x1, float y1, float x2, float y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

static Entity makeSpeedBump(Engine &ctx, const MapVector2 &p1, const MapVector2 &p2, const MapVector2 &p3,
                            const MapVector2 &p4) {
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

    auto speed_bump = ctx.makeEntity<PhysicsEntity>();
    ctx.get<Position>(speed_bump) = Vector3{.x = (x1 + x2 + x3 + x4)/4 - ctx.data().mean.x, .y = (y1 + y2 + y3 + y4)/4 - ctx.data().mean.y, .z = 1};
    ctx.get<Rotation>(speed_bump) = Quat::angleAxis(angle, madrona::math::up);
    ctx.get<Scale>(speed_bump) = Diag3x3{.d0 = lengths[maxLength_i]/2, .d1 = lengths[minLength_i]/2, .d2 = 0.1};
    ctx.get<EntityType>(speed_bump) = EntityType::Cube;
    ctx.get<ObjectID>(speed_bump) = ObjectID{(int32_t)SimObject::SpeedBump};
    registerRigidBodyEntity(ctx, speed_bump, SimObject::SpeedBump);
    ctx.get<ResponseType>(speed_bump) = ResponseType::Static;
    ctx.get<MapObservation>(speed_bump) = MapObservation{.position = Vector2{.x = (x1 + x2 + x3 + x4)/4 - ctx.data().mean.x, .y =  (y1 + y2 + y3 + y4)/4 - ctx.data().mean.y}, .heading = angle, .type = (float)MapRoadType::SpeedBump};
    return speed_bump;
}

static Entity makeStopSign(Engine &ctx, const MapVector2 &p1) {
    float x1 = p1.x;
    float y1 = p1.y;

    auto stop_sign = ctx.makeEntity<PhysicsEntity>();
    ctx.get<Position>(stop_sign) = Vector3{.x = x1 - ctx.data().mean.x, .y = y1 - ctx.data().mean.y, .z = 0.5};
    ctx.get<Rotation>(stop_sign) = Quat::angleAxis(0, madrona::math::up);
    ctx.get<Scale>(stop_sign) = Diag3x3{.d0 = 0.2, .d1 = 0.2, .d2 = 0.5};
    ctx.get<EntityType>(stop_sign) = EntityType::Cube;
    ctx.get<ObjectID>(stop_sign) = ObjectID{(int32_t)SimObject::StopSign};
    registerRigidBodyEntity(ctx, stop_sign, SimObject::StopSign);
    ctx.get<ResponseType>(stop_sign) = ResponseType::Static;
    ctx.get<MapObservation>(stop_sign) = MapObservation{.position = Vector2{.x = x1 - ctx.data().mean.x, .y = y1 - ctx.data().mean.y}, .heading = 0, .type = (float)MapRoadType::StopSign};
    return stop_sign;
}

static inline void createRoadEntities(Engine &ctx, const MapRoad &roadInit, CountT &idx) {
    if (roadInit.type == MapRoadType::RoadEdge || roadInit.type == MapRoadType::RoadLine || roadInit.type == MapRoadType::Lane)
    {
        size_t numPoints = roadInit.numPoints;
        for(size_t j = 1; j <= numPoints - 1; j++)
        {
            if(idx >= consts::kMaxRoadEntityCount)
                 return;
            ctx.data().roads[idx++] = makeRoadEdge(ctx, roadInit.geometry[j-1], roadInit.geometry[j], roadInit.type);
        }
    } else if (roadInit.type == MapRoadType::SpeedBump) {
      assert(roadInit.numPoints == 4);
      // TODO: Speed Bump are not guranteed to have 4 points. Need to handle this case.
      if(idx >= consts::kMaxRoadEntityCount)
        return;
      ctx.data().roads[idx++] = makeSpeedBump(ctx, roadInit.geometry[0], roadInit.geometry[1], roadInit.geometry[2], roadInit.geometry[3]);
    } else if (roadInit.type == MapRoadType::StopSign) {
      assert(roadInit.numPoints == 1);
      // TODO: Stop Sign are not guranteed to have 1 point. Need to handle this case.
      if(idx >= consts::kMaxRoadEntityCount)
        return;
      ctx.data().roads[idx++] = makeStopSign(ctx, roadInit.geometry[0]);
    } else {
      // TODO: Need to handle Cross Walk.
    //   assert(false);
        return;
    }
}

static void createFloorPlane(Engine &ctx)
{
    ctx.data().floorPlane = ctx.makeEntity<PhysicsEntity>();
    ctx.get<Position>(ctx.data().floorPlane) = Vector3{.x = 0, .y = 0, .z = 0};
    ctx.get<Rotation>(ctx.data().floorPlane) = Quat { 1, 0, 0, 0 };
    ctx.get<Scale>(ctx.data().floorPlane) = Diag3x3{1, 1, 1};
    ctx.get<ObjectID>(ctx.data().floorPlane) = ObjectID{(int32_t)SimObject::Plane};
    ctx.get<Velocity>(ctx.data().floorPlane) = {Vector3::zero(), Vector3::zero()};
    ctx.get<ExternalForce>(ctx.data().floorPlane) = Vector3::zero();
    ctx.get<ExternalTorque>(ctx.data().floorPlane) = Vector3::zero();
    ctx.get<ResponseType>(ctx.data().floorPlane) = ResponseType::Static;
    ctx.get<EntityType>(ctx.data().floorPlane) = EntityType::None;
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);
}

static inline Entity createAgentPadding(Engine &ctx) {
    auto agent = ctx.makeEntity<Agent>();

    ctx.get<Position>(agent) = consts::kPaddingPosition;
    ctx.get<Rotation>(agent) = Quat::angleAxis(0, madrona::math::up);
    ctx.get<Scale>(agent) = Diag3x3{.d0 = 0, .d1 = 0, .d2 = 0};
    ctx.get<Velocity>(agent) = {Vector3::zero(), Vector3::zero()};
    ctx.get<ObjectID>(agent) = ObjectID{(int32_t)SimObject::Agent};
    ctx.get<ResponseType>(agent) = ResponseType::Static;
    ctx.get<ExternalForce>(ctx.data().floorPlane) = Vector3::zero();
    ctx.get<ExternalTorque>(ctx.data().floorPlane) = Vector3::zero();
    ctx.get<EntityType>(agent) = EntityType::Padding;

    return agent;
}

static inline Entity createPhysicsEntityPadding(Engine &ctx) {
    auto physicsEntity = ctx.makeEntity<PhysicsEntity>();

    ctx.get<Position>(physicsEntity) = consts::kPaddingPosition;
    ctx.get<Rotation>(physicsEntity) = Quat::angleAxis(0, madrona::math::up);
    ctx.get<Scale>(physicsEntity) = Diag3x3{.d0 = 0, .d1 = 0, .d2 = 0};
    ctx.get<Velocity>(physicsEntity) = {Vector3::zero(), Vector3::zero()};
    ctx.get<ObjectID>(physicsEntity) = ObjectID{(int32_t)SimObject::Cube};
    ctx.get<ResponseType>(physicsEntity) = ResponseType::Static;
    ctx.get<ExternalForce>(ctx.data().floorPlane) = Vector3::zero();
    ctx.get<ExternalTorque>(ctx.data().floorPlane) = Vector3::zero();
    ctx.get<MapObservation>(physicsEntity) = MapObservation{
        .position = Vector2{.x = 0, .y = 0}, .heading = 0, .type = 0};
    ctx.get<EntityType>(physicsEntity) = EntityType::Padding;

    return physicsEntity;
}

void createPaddingEntities(Engine &ctx) {
    for (CountT agentIdx = ctx.data().numAgents;
         agentIdx < consts::kMaxAgentCount; ++agentIdx) {
        ctx.data().agents[agentIdx] = createAgentPadding(ctx);
    }

    for (CountT roadIdx = ctx.data().numRoads;
         roadIdx < consts::kMaxRoadEntityCount; ++roadIdx) {
        ctx.data().roads[roadIdx] = createPhysicsEntityPadding(ctx);
    }
}

void createPersistentEntities(Engine &ctx, Map *map) {

    ctx.data().mean = {0, 0};
    ctx.data().mean.x = map->mean.x;
    ctx.data().mean.y = map->mean.y;

    createFloorPlane(ctx);
    CountT agentIdx;
    for (agentIdx = 0; agentIdx < map->numObjects; ++agentIdx) {
        if(agentIdx >= consts::kMaxAgentCount)
            break;
        const auto &agentInit = map->objects[agentIdx];
        if(agentInit.type != MapObjectType::Vehicle)
            continue;
        auto vehicle = createVehicle(
            ctx, agentInit);
        ctx.data().agents[agentIdx] = vehicle;
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

static void resetPaddingEntities(Engine &ctx) {
    for (CountT agentIdx = ctx.data().numAgents;
         agentIdx < consts::kMaxAgentCount; ++agentIdx) {
        Entity agent = ctx.data().agents[agentIdx];
        registerRigidBodyEntity(ctx, agent, SimObject::Agent);
    }

    for (CountT roadIdx = ctx.data().numRoads;
         roadIdx < consts::kMaxRoadEntityCount; ++roadIdx) {
        Entity road = ctx.data().roads[roadIdx];
        registerRigidBodyEntity(ctx, road, SimObject::Cube);
    }
}

static void resetPersistentEntities(Engine &ctx)
{

    for (CountT idx = 0; idx < ctx.data().numAgents; ++idx)
    {
        Entity vehicle = ctx.data().agents[idx];

        resetVehicle(ctx, vehicle);

        registerRigidBodyEntity(ctx, vehicle, SimObject::Agent);


        ctx.get<viz::VizCamera>(vehicle) = viz::VizRenderingSystem::setupView(
            ctx, 90.f, 0.001f, 1.5f * math::up, (int32_t)idx);
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
    resetPaddingEntities(ctx);
}

}
