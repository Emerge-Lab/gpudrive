#include "level_gen.hpp"
#include <cassert>
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

static inline Entity createVehicle(Engine &ctx, float xCoord, float yCoord,
                                   float length, float width, float heading,
                                   float speed, int32_t idx) {
    auto vehicle = ctx.makeEntity<Agent>();

    ctx.get<VehicleSize>(vehicle) = {.length = length, .width = width};
    ctx.get<BicycleModel>(vehicle) = {.position = {.x = xCoord, .y = yCoord},
                                      .heading = heading,
                                      .speed = speed};
    ctx.get<Position>(vehicle) = Vector3{.x = xCoord, .y = yCoord, .z = 0};
    ctx.get<Rotation>(vehicle) = Quat::angleAxis(heading, madrona::math::up);
    ctx.get<Scale>(vehicle) =
        Diag3x3{.d0 = width, .d1 = length, .d2 = consts::zDimensionScale};
    ctx.get<ObjectID>(vehicle) = ObjectID{(int32_t)SimObject::Cube};

    // TODO(samk): compute velocity

    // TODO(samk): look into what this value controls. Should it be set to
    // ResponseType::Kinematic?
    ctx.get<ResponseType>(vehicle) = ResponseType::Dynamic;
    ctx.get<ExternalForce>(vehicle) = Vector3::zero();
    ctx.get<ExternalTorque>(vehicle) = Vector3::zero();
    ctx.get<EntityType>(vehicle) = EntityType::Agent;

    registerRigidBodyEntity(ctx, vehicle, SimObject::Cube);
    ctx.get<viz::VizCamera>(vehicle) = viz::VizRenderingSystem::setupView(
        ctx, 90.f, 0.001f, 1.5f * math::up, idx);

    return vehicle;
}

void createPersistentEntities(Engine &ctx) {}

static void resetPersistentEntities(Engine &ctx) {}

static void generateLevel(Engine &ctx) {
    std::ifstream data(
        "/Users/samk/src/nocturne/data/nocturne_mini/"
        "formatted_json_v2_no_tl_valid/tfrecord-00004-of-00150_246.json");
    assert(data.is_open());

    using nlohmann::json;

    json rawJson;
    data >> rawJson;

    // TODO(samk): handle keys not existing
    size_t agentCount{0};
    for (const auto &obj : rawJson["objects"]) {
      assert(agentCount < consts::numAgents);

      auto vehicle = createVehicle(
          ctx,
          // TODO(samk): Nocturne allows for configuring the initial position
          // but in practice it looks to always be set to 0.
          obj["position"][0]["x"], obj["position"][0]["y"], obj["length"],
          obj["width"], obj["heading"][0], obj["velocity"][0]["x"], agentCount);

      ctx.data().agents[agentCount++] = vehicle;
    }
}

void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
    generateLevel(ctx);
}

}
