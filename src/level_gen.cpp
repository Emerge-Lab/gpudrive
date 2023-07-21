#include "level_gen.hpp"

namespace GPUHideSeek {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

static inline Entity makePhysicsObject(
    Engine &ctx,
    Vector3 pos,
    Quat rot,
    SimObject sim_obj,
    ResponseType response_type = ResponseType::Dynamic,
    Diag3x3 scale = {1, 1, 1})
{
    ObjectID obj_id { (int32_t)sim_obj };

    Entity e = ctx.makeEntity<PhysicsObject>();
    ctx.get<Position>(e) = pos;
    ctx.get<Rotation>(e) = rot;
    ctx.get<Scale>(e) = scale;
    ctx.get<ObjectID>(e) = obj_id;
    ctx.get<Velocity>(e) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.get<ResponseType>(e) = response_type;
    ctx.get<ExternalForce>(e) = Vector3::zero();
    ctx.get<ExternalTorque>(e) = Vector3::zero();

    return e;
}

#if 0
static Entity makeButtonEntity(Engine &ctx, Vector2 pos, Vector2 scale)
{
    Entity e = ctx.makeEntity<ButtonObject>();
    ctx.get<Position>(e) = Vector3 { pos.x, pos.y, 0.f };
    ctx.get<Rotation>(e) = Quat::angleAxis(0, {1, 0, 0});
    ctx.get<Scale>(e) = Diag3x3 {
        scale.x * BUTTON_WIDTH,
        scale.y * BUTTON_WIDTH,
        0.2f,
    };
    ctx.get<ObjectID>(e) = ObjectID { 2 };

    return e;
}
#endif

// Creates floor, outer walls, and agent entities.
// All these entities persist across all episodes.
void createPersistentEntities(Engine &ctx)
{
    // Create the floor entity, just a simple static plane.
    ctx.data().floorPlane = makePhysicsObject(
        ctx,
        Vector3 { 0, 0, 0 },
        Quat { 1, 0, 0, 0 },
        SimObject::Plane,
        ResponseType::Static);

    // Create the outer wall entities
    // Left
    ctx.data().borders[0] = makePhysicsObject(
        ctx,
        Vector3 { -consts::wallWidth / 2.f, 0, 0 },
        Quat::angleAxis(math::pi / 2.f, math::up),
        SimObject::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::worldWidth,
            consts::wallWidth,
            2.f,
        });

    // Top
    ctx.data().borders[1] = makePhysicsObject(
        ctx,
        Vector3 {
            consts::worldLength / 2.f,
            consts::worldWidth / 2.f + consts::wallWidth / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::worldLength,
            consts::wallWidth,
            2.f,
        });

    // Bottom
    ctx.data().borders[2] = makePhysicsObject(
        ctx,
        Vector3 {
            consts::worldLength / 2.f,
            -consts::worldWidth / 2.f - consts::wallWidth / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::worldLength,
            consts::wallWidth,
            2.f,
        });

    // Create agent entities. Note that this leaves a lot of components
    // uninitialized, these will be set during world generation, which is
    // called for every episode.
    for (CountT i = 0; i < consts::numAgents; ++i) {
        Entity agent = ctx.data().agents[i] = ctx.makeEntity<Agent>();

        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.get<ObjectID>(agent) = ObjectID { (int32_t)SimObject::Agent };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
    }

    // Populate OtherAgents component, which maintains a reference to the
    // other agents in the world for each agent.
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
}

static inline float randInRangeCentered(Engine &ctx, float range)
{
    return ctx.data().rng.rand() * range - range / 2.f;
}

static inline float randInRange(Engine &ctx, float range)
{
    return ctx.data().rng.rand() * range;
}

static void resetPersistentEntities(Engine &ctx)
{
    {
        Entity floor_entity = ctx.data().floorPlane;
        ctx.get<broadphase::LeafID>(floor_entity) = 
            phys::RigidBodyPhysicsSystem::registerEntity(ctx, floor_entity,
                ctx.get<ObjectID>(floor_entity));
    }

     for (CountT i = 0; i < 3; i++) {
         Entity wall_entity = ctx.data().borders[i];
         ctx.get<broadphase::LeafID>(wall_entity) =
             phys::RigidBodyPhysicsSystem::registerEntity(
                ctx, wall_entity, ctx.get<ObjectID>(wall_entity));
     }

     for (CountT i = 0; i < consts::numAgents; i++) {
         Entity agent_entity = ctx.data().agents[i];
         ctx.get<broadphase::LeafID>(agent_entity) =
             phys::RigidBodyPhysicsSystem::registerEntity(
                ctx, agent_entity, ctx.get<ObjectID>(agent_entity));
         ctx.get<viz::VizCamera>(agent_entity) =
             viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                 1.5f * math::up, (int32_t)i);

         // Place the agents near the starting wall
         Vector3 pos {
             randInRange(ctx, consts::distancePerProgress / 2.f) +
                 1.1f * consts::agentRadius,
             randInRangeCentered(ctx, 
                 consts::worldWidth / 2.f - 2.f * consts::agentRadius),
             0.f,
         };

         if (i % 2 == 0) {
             pos.y += consts::worldWidth / 4.f;
         } else {
             pos.y -= consts::worldWidth / 4.f;
         }

         ctx.get<Position>(agent_entity) = pos;
         ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
             -math::pi / 2.f - randInRangeCentered(ctx, math::pi / 2.f),
             math::up);

         ctx.get<Progress>(agent_entity).numProgressIncrements = 0;

         ctx.get<Velocity>(agent_entity) = {
             Vector3::zero(),
             Vector3::zero(),
         };
         ctx.get<ExternalForce>(agent_entity) = Vector3::zero();
         ctx.get<ExternalTorque>(agent_entity) = Vector3::zero();
         ctx.get<Action>(agent_entity) = Action {
             .x = consts::numMoveBuckets / 2,
             .y = consts::numMoveBuckets / 2,
             .r = consts::numMoveBuckets / 2,
         };

         ctx.get<Done>(agent_entity).v = 0;
     }
}

// Randomly generate a new world for a training episode
void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
}

}
