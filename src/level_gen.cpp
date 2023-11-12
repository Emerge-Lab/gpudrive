#include "level_gen.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <vector>

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
                                   float speedX, float speedY, float goalX, float goalY, int32_t idx) {
    auto vehicle = ctx.makeEntity<Agent>();

    // The following components do not vary within an episode and so need only
    // be set once
    ctx.get<VehicleSize>(vehicle) = {.length = length, .width = width};
    ctx.get<Scale>(vehicle) = Diag3x3{.d0 = width, .d1 = length, .d2 = 1};
    ctx.get<ObjectID>(vehicle) = ObjectID{(int32_t)SimObject::Cube};
    ctx.get<ResponseType>(vehicle) = ResponseType::Dynamic;
    ctx.get<EntityType>(vehicle) = EntityType::Agent;
    ctx.get<Goal>(vehicle)= Goal{.position = Vector2{.x = goalX, .y = goalY}};
    // Since position, heading, and speed may vary within an episode, their
    // values are retained so that on an episode reset they can be restored to
    // their initial values.
    ctx.get<Trajectory>(vehicle).positions[0] =
        Vector2{.x = xCoord, .y = yCoord};
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
    Vector3 start = Vector3{.x = x1 + 452, .y = y1 + 10353, .z = 0};
    Vector3 end = Vector3{.x = x2 + 452, .y = y2 + 10353, .z = 0};
    float distance = end.distance(start);
    auto road_edge = ctx.makeEntity<PhysicsEntity>();
    ctx.get<Position>(road_edge) = Vector3{.x = (start.x + end.x)/2, .y = (start.y + end.y)/2, .z = 0};
    ctx.get<Rotation>(road_edge) = Quat::angleAxis(atan2(end.y - start.y, end.x - start.x), madrona::math::up);
    ctx.get<Scale>(road_edge) = Diag3x3{.d0 = distance/2, .d1 = 0.1, .d2 = 0.1};
    ctx.get<EntityType>(road_edge) = EntityType::Cube;
    ctx.get<ObjectID>(road_edge) = ObjectID{(int32_t)SimObject::Cube};
    registerRigidBodyEntity(ctx, road_edge, SimObject::Cube);
    ctx.get<ResponseType>(road_edge) = ResponseType::Static;
    return road_edge;
}

static inline int32_t createRoadEntities(Engine &ctx, const nlohmann::json& geometryList, std::string type, int32_t idx) {
    // Access elements in geometryList
    bool reduceRoad = true;
    if (type == "road_edge" || type == "lane")
    {    
        if (reduceRoad)
        {
                    size_t numPoints = geometryList.size();

                    int32_t start = 0;
                    int32_t j = 0;
                    while (j < numPoints - 2)
                    {
                        float x1 = geometryList[j]["x"] ; float y1 = geometryList[j]["y"] ;
                        float x2 = geometryList[j+1]["x"] ; float y2 = geometryList[j+1]["y"] ;
                        float x3 = geometryList[j+2]["x"] ; float y3 = geometryList[j+2]["y"] ;
                        float shoelace_area = std::abs((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1));
                        if(shoelace_area < 0.01)
                        {
                            j++;
                        }
                        else
                        {
                            if(j != start)
                            {
                                auto road_edge = makeRoadEdge(ctx, geometryList[start], geometryList[j]);
                                ctx.data().roads[idx++] = road_edge;
                                start = j;
                            }
                            else
                            {
                                auto road_edge = makeRoadEdge(ctx, geometryList[j], geometryList[j+1]);
                                ctx.data().roads[idx++] = road_edge;
                                start = j + 1;
                                j += 1;
                            }
                        }
                    }
                    if (j != start)
                    {
                        auto road_edge = makeRoadEdge(ctx, geometryList[start], geometryList[j]);
                        ctx.data().roads[idx++] = road_edge;
                    }
                    if(j == start)
                    {
                        auto road_edge = makeRoadEdge(ctx, geometryList[j], geometryList[j+1]);
                        ctx.data().roads[idx++] = road_edge;
                    }

        }
        else
        {
            size_t numPoints = geometryList.size();
            for(size_t i = 0; i < numPoints - 1; i++)
            {
                Entity road_edge = makeRoadEdge(ctx, geometryList[i], geometryList[i+1]);
                ctx.data().roads[idx++] = road_edge;
            }
        }

        // int32_t ctr = idx;
        // int32_t start_idx = 0;
        // int32_t end_idx = -1;
        // for(size_t i = 0; i < numPoints - 2; )
        // {   std::cout<<"i: "<<i<<std::endl;
        //     // x1, y1 = geometryList[i]["x"], geometryList[i]["y"]
        //     float x1 = geometryList[i]["x"] ; float y1 = geometryList[i]["y"] ;
        //     float x2 = geometryList[i+1]["x"] ; float y2 = geometryList[i+1]["y"] ;
        //     float x3 = geometryList[i+2]["x"] ; float y3 = geometryList[i+2]["y"] ;
        //     // https://en.wikipedia.org/wiki/Shoelace_formula#Triangle_form,_determinant_form
        //     // Checking for collinearity using area of triangle formed by 3 points
        //     // Naively, we can using slopes, but that has division which is expensive
        //     float shoelace_area = std::abs((x1*y2 + x2*y3 + x3*y1) - (x2*y1 + x3*y2 + x1*y3));
        //     if(shoelace_area < 0.001){
        //         end_idx = i+2;
        //         i += 2;
        //     }
        //     else
        //     {
        //         if(end_idx != -1)
        //         {
        //             Entity road_edge = makeRoadEdge(ctx, geometryList[start_idx], geometryList[end_idx]);
        //             ctx.data().roads[ctr++] = road_edge;
        //             road_edge = makeRoadEdge(ctx, geometryList[end_idx], geometryList[i]);
        //             ctx.data().roads[ctr++] = road_edge;
        //             end_idx = -1;
        //         }
        //         Entity road_edge = makeRoadEdge(ctx, geometryList[i], geometryList[i+1]);
        //         ctx.data().roads[ctr++] = road_edge;
        //         start_idx = i+1;
        //         i++;
        //     }

        // } 
        // return ctr;
    }
    return idx;
}

void createPersistentEntities(Engine &ctx, const std::string &pathToScenario) {
    std::ifstream data(pathToScenario);
    assert(data.is_open());

    using nlohmann::json;

    json rawJson;
    data >> rawJson;
    
    // TODO(samk): handle keys not existing
    size_t agentCount{0};
    for (const auto &obj : rawJson["objects"]) {
      if (agentCount == consts::numAgents) {
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
          obj["velocity"][0]["y"], obj["goalPosition"]["x"], obj["goalPosition"]["y"], agentCount);

      ctx.data().agents[agentCount++] = vehicle;
    }

    std::cout<<"Agent count: "<<agentCount<<std::endl;
    size_t roadCount{0};
    for (const auto &obj : rawJson["roads"]) {
      if (roadCount >= consts::numRoadSegments) break;
      auto geometrylist = obj["geometry"];
      std::string type = obj["type"];
      roadCount = createRoadEntities(
          ctx, geometrylist, type, roadCount);
    }
    ctx.data().num_roads = roadCount;
    std::cout<<"Road count: "<<roadCount<<std::endl;
}

 
static void generateLevel(Engine &ctx) {
       for (CountT i = 0; i < consts::numAgents; i++) {
        Entity cur_agent = ctx.data().agents[i];
        OtherAgents &other_agents = ctx.get<OtherAgents>(cur_agent);
        CountT out_idx = 0;
        for (CountT j = 0; j < consts::numAgents; j++) {
            if (i == j) {
                continue;
            }

            Entity other_agent = ctx.data().agents[j];
            other_agents.e[out_idx++] = other_agent;
        }
    }
    std::cout<<"Level Generated."<<std::endl;
}

static void resetPersistentEntities(Engine &ctx) {
    for (CountT idx = 0; idx < consts::numAgents; ++idx) {
      Entity vehicle = ctx.data().agents[idx];

      resetVehicle(ctx, vehicle);

      registerRigidBodyEntity(ctx, vehicle, SimObject::Cube);

      ctx.get<viz::VizCamera>(vehicle) = viz::VizRenderingSystem::setupView(
          ctx, 90.f, 0.001f, 1.5f * math::up, (int32_t)idx);
    }

    std::cout<<"Resetting roads"<<std::endl;
    for (CountT idx = 0; idx < ctx.data().num_roads; idx++) {
      Entity road = ctx.data().roads[idx];
      if(road == Entity::none()) break;
      registerRigidBodyEntity(ctx, road, SimObject::Cube);
    }
    std::cout<<"Resetting roads done"<<std::endl;
}

void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
    generateLevel(ctx);
}

}
