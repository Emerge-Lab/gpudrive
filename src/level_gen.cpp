#include "level_gen.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <cmath>
#include <vector>
#include <iostream>

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


float degreesToRadians(float degrees) { return degrees * M_PI / 180.0; }

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

static inline Entity createVehicle(Engine &ctx, float xCoord, float yCoord,
                                   float length, float width, float heading,
                                   float speedX, float speedY, float goalX, float goalY) {
    auto vehicle = ctx.makeEntity<Agent>();

    // The following components do not vary within an episode and so need only
    // be set once
    ctx.get<VehicleSize>(vehicle) = {.length = length, .width = width};
    ctx.get<Scale>(vehicle) = Diag3x3{.d0 = length/2, .d1 = width/2, .d2 = 1};
    ctx.get<ObjectID>(vehicle) = ObjectID{(int32_t)SimObject::Agent};
    ctx.get<ResponseType>(vehicle) = ResponseType::Dynamic;
    ctx.get<EntityType>(vehicle) = EntityType::Agent;
    ctx.get<Goal>(vehicle)= Goal{.position = Vector2{.x = goalX, .y = goalY}};
    // Since position, heading, and speed may vary within an episode, their
    // values are retained so that on an episode reset they can be restored to
    // their initial values.
    ctx.get<Trajectory>(vehicle).positions[0] =
        Vector2{.x = xCoord - ctx.data().mean.first, .y = yCoord - ctx.data().mean.second};
    ctx.get<Trajectory>(vehicle).initialHeading = degreesToRadians(heading);
    ctx.get<Trajectory>(vehicle).velocities[0] =
        Vector2{.x = speedX, .y = speedY};

    // This is not stricly necessary since , but is kept here for consistency
    resetVehicle(ctx, vehicle);

    return vehicle;
}

static Entity makeRoadEdge(Engine &ctx,
                         const nlohmann::json& p1,
                         const nlohmann::json& p2)
{
    float x1 = p1["x"]; float y1 = p1["y"];
    float x2 = p2["x"]; float y2 = p2["y"] ;
    Vector3 start{.x = x1 - ctx.data().mean.first, .y = y1 - ctx.data().mean.second, .z = 0};
    Vector3 end{.x = x2 - ctx.data().mean.first, .y = y2 - ctx.data().mean.second, .z = 0};
    float distance = end.distance(start);
    auto road_edge = ctx.makeEntity<PhysicsEntity>();
    ctx.get<Position>(road_edge) = Vector3{.x = (start.x + end.x)/2, .y = (start.y + end.y)/2, .z = 0};
    ctx.get<Rotation>(road_edge) = Quat::angleAxis(atan2(end.y - start.y, end.x - start.x), madrona::math::up);
    ctx.get<Scale>(road_edge) = Diag3x3{.d0 = distance/2, .d1 = 0.1, .d2 = 0.1};
    ctx.get<EntityType>(road_edge) = EntityType::Cube;
    ctx.get<ObjectID>(road_edge) = ObjectID{(int32_t)SimObject::Cube};
    registerRigidBodyEntity(ctx, road_edge, SimObject::Cube);
    ctx.get<ResponseType>(road_edge) = ResponseType::Static;
    ctx.get<MapObservation>(road_edge) = MapObservation{.position = Vector2{.x = (start.x + end.x)/2, .y = (start.y + end.y)/2}, .heading = atan2(end.y - start.y, end.x - start.x), .type = 0};
    return road_edge;
}

float calculateDistance(float x1, float y1, float x2, float y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

static Entity makeSpeedBump(Engine &ctx, const nlohmann::json& geometryList)
{    // Extract coordinates
    float x1 = geometryList[0]["x"]; float y1 = geometryList[0]["y"];
    float x2 = geometryList[1]["x"]; float y2 = geometryList[1]["y"];
    float x3 = geometryList[2]["x"]; float y3 = geometryList[2]["y"];
    float x4 = geometryList[3]["x"]; float y4 = geometryList[3]["y"];

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
    ctx.get<Position>(speed_bump) = Vector3{.x = (x1 + x2 + x3 + x4)/4 - ctx.data().mean.first, .y = (y1 + y2 + y3 + y4)/4 - ctx.data().mean.second, .z = 1};
    ctx.get<Rotation>(speed_bump) = Quat::angleAxis(angle, madrona::math::up);
    ctx.get<Scale>(speed_bump) = Diag3x3{.d0 = lengths[maxLength_i]/2, .d1 = lengths[minLength_i]/2, .d2 = 0.1};
    ctx.get<EntityType>(speed_bump) = EntityType::Cube;
    ctx.get<ObjectID>(speed_bump) = ObjectID{(int32_t)SimObject::SpeedBump};
    registerRigidBodyEntity(ctx, speed_bump, SimObject::SpeedBump);
    ctx.get<ResponseType>(speed_bump) = ResponseType::Static;
    ctx.get<MapObservation>(speed_bump) = MapObservation{.position = Vector2{.x = (x1 + x2 + x3 + x4)/4 - ctx.data().mean.first, .y =  (y1 + y2 + y3 + y4)/4 - ctx.data().mean.second}, .heading = angle, .type = 1};
    return speed_bump;
}

static Entity makeStopSign(Engine &ctx, const nlohmann::json& geomeryList)
{
    float x1 = geomeryList[0]["x"]; float y1 = geomeryList[0]["y"];
    auto stop_sign = ctx.makeEntity<PhysicsEntity>();
    ctx.get<Position>(stop_sign) = Vector3{.x = x1 - ctx.data().mean.first, .y = y1 - ctx.data().mean.second, .z = 0.5};
    ctx.get<Rotation>(stop_sign) = Quat::angleAxis(0, madrona::math::up);
    ctx.get<Scale>(stop_sign) = Diag3x3{.d0 = 0.2, .d1 = 0.2, .d2 = 0.5};
    ctx.get<EntityType>(stop_sign) = EntityType::Cube;
    ctx.get<ObjectID>(stop_sign) = ObjectID{(int32_t)SimObject::StopSign};
    registerRigidBodyEntity(ctx, stop_sign, SimObject::StopSign);
    ctx.get<ResponseType>(stop_sign) = ResponseType::Static;
    ctx.get<MapObservation>(stop_sign) = MapObservation{.position = Vector2{.x = x1 - ctx.data().mean.first, .y = y1 - ctx.data().mean.second}, .heading = 0, .type = 2};
    return stop_sign;
}

static inline size_t createRoadEntities(Engine &ctx, const nlohmann::json& geometryList,const std::string type, size_t &idx) {
    if (type == "road_edge" || type == "lane" || type == "road_line")
    {
        size_t numPoints = geometryList.size();

        size_t start = 0;
        size_t j = 0;
        while (j < numPoints - 2)
        {
            float x1 = geometryList[j]["x"]; float y1 = geometryList[j]["y"];
            float x2 = geometryList[j + 1]["x"]; float y2 = geometryList[j + 1]["y"];
            float x3 = geometryList[j + 2]["x"]; float y3 = geometryList[j + 2]["y"];
            float shoelace_area = std::abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1)); // https://en.wikipedia.org/wiki/Shoelace_formula#Triangle_form,_determinant_form
            if (shoelace_area < ctx.data().polylineReductionThreshold)
                j++; // Skip over points that are too close together
            else
            {
                if (j != start)
                {
                    if(idx == ctx.data().max_num_roads) return idx; // TODO: Handle this better. Will be solved by the GPU version
                    ctx.data().roads[idx++] = makeRoadEdge(ctx, geometryList[start], geometryList[j]);
                    start = j;
                }
                else
                {
                    if(idx == ctx.data().max_num_roads) return idx; // TODO: Handle this better. Will be solved by the GPU version
                    ctx.data().roads[idx++] = makeRoadEdge(ctx, geometryList[j], geometryList[j + 1]);
                    start = ++j;
                }
            }
        }

        //Handle last point
        if(idx == ctx.data().max_num_roads) return idx; // TODO: Handle this better. Will be solved by the GPU version
        ctx.data().roads[idx++] = j!=start ? makeRoadEdge(ctx, geometryList[start], geometryList[j]) : makeRoadEdge(ctx, geometryList[j], geometryList[j + 1]);
    }
    else if(type == "speed_bump")
    {
        assert(geometryList.size() == 4);
        if(idx == ctx.data().max_num_roads) return idx; // TODO: Not necessary here since we only make one speedbump. Kept for consistency.
        ctx.data().roads[idx++] = makeSpeedBump(ctx, geometryList);
    }
    else if(type == "stop_sign")
    {
        assert(geometryList.size() == 1);
        if(idx == ctx.data().max_num_roads) return idx; // TODO: Not necessary here since we only make one speedbump. Kept for consistency.
        ctx.data().roads[idx++] = makeStopSign(ctx, geometryList);
    }

    return idx;
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

void createPersistentEntities(Engine &ctx, const std::string &pathToScenario) {

    createFloorPlane(ctx);
    std::cout<<"CTEST Path to scenario: "<<pathToScenario<<std::endl;
    std::ifstream data(pathToScenario);
    assert(data.is_open());

    using nlohmann::json;

    json rawJson;
    data >> rawJson;

    ctx.data().mean = std::make_pair(0, 0);
    size_t numEntities = 0;
    for (const auto &obj : rawJson["objects"])
    {
        if (obj["type"] != "vehicle")
        {
            continue;
        }
        numEntities++;
        float newX = obj["position"][0]["x"];
        float newY = obj["position"][0]["y"];

        // Update mean incrementally
        ctx.data().mean.first += (newX - ctx.data().mean.first) / numEntities;
        ctx.data().mean.second += (newY - ctx.data().mean.second) / numEntities;
    }
    for (const auto &obj: rawJson["roads"])
    {
        for (const auto &point: obj["geometry"])
        {   
            numEntities++;
            float newX = point["x"];
            float newY = point["y"];

            // Update mean incrementally
            ctx.data().mean.first += (newX - ctx.data().mean.first) / numEntities;
            ctx.data().mean.second += (newY - ctx.data().mean.second) / numEntities;
        }
    }

    std::cout<<"Mean: "<<ctx.data().mean.first<<", "<<ctx.data().mean.second<<"\n";

    // TODO(samk): handle keys not existing
    size_t agentCount{0};
    for (const auto &obj : rawJson["objects"]) {
      if (agentCount == ctx.data().max_num_agents) {
        break;
      }
      if (obj["type"] != "vehicle") {
        continue;
      }
      auto vehicle = createVehicle(
          ctx,
          // TODO(samk): Nocturne allows for configuring the initial position
          // but in practice it looks to always be set to 0.
          obj["position"][0]["x"], obj["position"][0]["y"], obj["length"],
          obj["width"], obj["heading"][0], obj["velocity"][0]["x"],
          obj["velocity"][0]["y"], obj["goalPosition"]["x"], obj["goalPosition"]["y"]);

      ctx.data().agents[agentCount++] = vehicle;
    }
    ctx.data().num_agents = agentCount;

    size_t roadCount{0};
    for (const auto &obj : rawJson["roads"]) {
      if (roadCount >= ctx.data().max_num_roads) break;
      auto geometrylist = obj["geometry"];
      std::string type = obj["type"];
      createRoadEntities(
          ctx, geometrylist, type, roadCount);
    }
    ctx.data().num_roads = roadCount;
}


static void generateLevel(Engine &) {}

static void resetPersistentEntities(Engine &ctx)
{
    for (CountT idx = 0; idx < ctx.data().num_agents; ++idx)
    {
        Entity vehicle = ctx.data().agents[idx];

        resetVehicle(ctx, vehicle);

        registerRigidBodyEntity(ctx, vehicle, SimObject::Agent);


        ctx.get<viz::VizCamera>(vehicle) = viz::VizRenderingSystem::setupView(
            ctx, 90.f, 0.001f, 1.5f * math::up, (int32_t)idx);
    }

    for (CountT idx = 0; idx < ctx.data().num_roads; idx++) {
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
  
    for (CountT i = 0; i < ctx.data().num_agents; i++)
    {
        Entity cur_agent = ctx.data().agents[i];
        OtherAgents &other_agents = ctx.get<OtherAgents>(cur_agent);
        CountT out_idx = 0;
        for (CountT j = 0; j < ctx.data().num_agents; j++)
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
    generateLevel(ctx);
}

}
