#include "level_gen.hpp"

namespace GPUHideSeek {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace consts {

inline constexpr float wallWidth = 0.3f;

}

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
            -consts::worldLength / 2.f,
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
            -consts::worldLength / 2.f,
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
    // uninitialized, these will be set during world generation, which is called
    // for every episode.
    for (CountT i = 0; i < consts::numAgents; ++i) {
        Entity agent = ctx.data().agents[i] = ctx.makeEntity<Agent>();

        ctx.get<ObjectID>(agent) = ObjectID { (int32_t)SimObject::Agent };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;

        if (ctx.data().enableVizRender) {
            ctx.get<viz::VizCamera>(agent) =
                viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                        Vector3 { 0, 0, 0.8 }, (int32_t)agentIdx);
        }
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

#if 0
static void generateTrainingEnvironment(Engine &ctx)
{
    const Vector2 bounds { -consts::worldBounds, consts::worldBounds };

    // After this function, all the entities for the walls have been created
    populateStaticGeometry(ctx, ctx.data().rng, {bounds.y, bounds.y}, ctx.data().srcRoom, ctx.data().dstRoom);

    Room &room = ctx.data().rooms[ctx.data().srcRoom];

    // Need to create the entities themselves
    for (CountT i = 0; i < consts::numAgents; ++i) {
        float xStart = room.offset.x + 1.0f;
        float yStart = room.offset.y + 1.0f;
        float xEnd = room.offset.x+room.extent.x - 1.0f;
        float yEnd = room.offset.y+room.extent.y - 1.0f;

        float x = xStart + ctx.data().rng.rand() * (xEnd - xStart);
        float y = yStart + ctx.data().rng.rand() * (yEnd - yStart);

        Vector3 pos {
            x, y, 1.5f,
        };

        const auto rot = Quat::angleAxis(ctx.data().rng.rand() * math::pi, {0, 0, 1});

        Entity agent = ctx.data().agents[i];

        // Reset state for the agent
        ctx.get<phys::broadphase::LeafID>(agent) =
            phys::RigidBodyPhysicsSystem::registerEntity(ctx, agent,
                    ctx.get<ObjectID>(agent));
        ctx.get<Action>(agent) = {
            .x = consts::numMoveBuckets / 2,
            .y = consts::numMoveBuckets / 2,
            .r = consts::numMoveBuckets / 2,
        };
        ctx.get<Position>(agent) = pos;
        ctx.get<Rotation>(agent) = rot;
        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.get<Velocity>(agent) = { Vector3::zero(), Vector3::zero() };
        ctx.get<ExternalForce>(agent) = Vector3::zero();
        ctx.get<ExternalTorque>(agent) = Vector3::zero();

        if (ctx.data().enableVizRender) {
            ctx.get<viz::VizCamera>(agent) =
                viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                    Vector3 { 0, 0, 0.8 }, (int32_t)i);
        }
    }

    // Register the plane object again to the physics system
    ctx.get<phys::broadphase::LeafID>(ctx.data().floorPlane) =
        phys::RigidBodyPhysicsSystem::registerEntity(ctx, ctx.data().floorPlane, 
            ctx.get<ObjectID>(ctx.data().floorPlane));
}
#endif

// Randomly generate a new world for a training episode
// First, destroys any non-persistent state for the current world and then
// generates a new play area.
void generateWorld(Engine &ctx)
{
    // Destroy old entities
    Entity *dyn_entities = ctx.data().dynamicObjects;
    int32_t num_old_objects = ctx.data().numDynamicObjects;
    for (int32_t i = 0; i < num_old_objects; i++) {
        ctx.destroyEntity(dyn_entities[i]);
    }

    // Assign a new episode ID
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    int32_t episode_idx = episode_mgr.curEpisode.fetch_add<sync::relaxed>(1);
    ctx.data().rng = RNG::make(0 /*episode_idx*/);
    ctx.data().curEpisodeIdx = episode_idx;


}

}
