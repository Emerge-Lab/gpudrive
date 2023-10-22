#include "level_gen.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

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

static inline Entity createVehicle(Engine &ctx, float xCoord, float yCoord,
                                   float length, float width, float heading,
                                   float speed, int32_t idx) {
    heading = degreesToRadians(heading);
    auto vehicle = ctx.makeEntity<Agent>();

    ctx.get<VehicleSize>(vehicle) = {.length = length, .width = width};
    ctx.get<BicycleModel>(vehicle) = {
        .position = {.x = xCoord, .y = yCoord}, .heading = heading, .speed = 0};
    ctx.get<Position>(vehicle) = Vector3{.x = xCoord, .y = yCoord, .z = 1};
    ctx.get<Rotation>(vehicle) = Quat::angleAxis(heading, madrona::math::up);
    ctx.get<Scale>(vehicle) = Diag3x3{.d0 = width, .d1 = length, .d2 = 1};
    ctx.get<ObjectID>(vehicle) = ObjectID{(int32_t)SimObject::Cube};
    ctx.get<Velocity>(vehicle) = {Vector3::zero(), Vector3::zero()};

    // TODO(samk): look into what this value controls. Should it be set to
    // ResponseType::Kinematic?
    ctx.get<ResponseType>(vehicle) = ResponseType::Dynamic;
    ctx.get<ExternalForce>(vehicle) = Vector3::zero();
    ctx.get<ExternalTorque>(vehicle) = Vector3::zero();
    ctx.get<EntityType>(vehicle) = EntityType::Agent;
    ctx.get<Action>(vehicle) =
        Action{.acceleration = 0, .steering = 0, .headAngle = 0};

    registerRigidBodyEntity(ctx, vehicle, SimObject::Cube);
    ctx.get<viz::VizCamera>(vehicle) = viz::VizRenderingSystem::setupView(
        ctx, 90.f, 0.001f, 1.5f * math::up, idx);

    return vehicle;
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
          obj["width"], obj["heading"][0], obj["velocity"][0]["x"], agentCount);

      ctx.data().agents[agentCount] = vehicle;

      ctx.data().agentToInitialLocation[agentCount].x = obj["position"][0]["x"];
      ctx.data().agentToInitialLocation[agentCount].y = obj["position"][0]["y"];
      ctx.data().agentToInitialLocation[agentCount].heading = obj["heading"][0];

      ++agentCount;
    }
}

static void resetPersistentEntities(Engine &ctx) {
    for (CountT idx = 0; idx < consts::numAgents; ++idx) {
      Entity vehicle = ctx.data().agents[idx];
      registerRigidBodyEntity(ctx, vehicle, SimObject::Cube);

      ctx.get<viz::VizCamera>(vehicle) = viz::VizRenderingSystem::setupView(
          ctx, 90.f, 0.001f, 1.5f * math::up, (int32_t)idx);

      auto &initialLocation = ctx.data().agentToInitialLocation[idx];
      ctx.get<BicycleModel>(vehicle) = {
          .position = {.x = initialLocation.x, .y = initialLocation.y},
          .heading = initialLocation.heading,
          .speed = 0};
      ctx.get<Position>(vehicle) =
          Vector3{.x = initialLocation.x, .y = initialLocation.y, .z = 1};

      ctx.get<Rotation>(vehicle) =
          Quat::angleAxis(initialLocation.heading, madrona::math::up);

      ctx.get<Velocity>(vehicle) = {Vector3::zero(), Vector3::zero()};
      ctx.get<Action>(vehicle) = Action{0, 0, 0};
      ctx.get<StepsRemaining>(vehicle).t = consts::episodeLen;
    }
}

static void generateLevel(Engine &ctx) {}

void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
    generateLevel(ctx);
}

}
