#include "level_gen.hpp"

#include "geo_gen.hpp"

namespace GPUHideSeek {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

static Entity makeAgent(Engine &ctx, uint32_t agentIdx, Vector3 pos, Quat rot)
{
    Entity agent = ctx.data().agents[agentIdx] = ctx.makeEntity<Agent>();

    ctx.get<Seed>(agent).seed = ctx.data().curEpisodeSeed;

    // Zero out actions
    ctx.get<Action>(agent) = { .x = 0, .y = 0, .r = 0, };

    ctx.get<Position>(agent) = pos;
    ctx.get<Rotation>(agent) = rot;
    ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };

    if (ctx.data().enableRender) {
        ctx.get<render::ViewSettings>(agent) =
            render::RenderingSystem::setupView(ctx, 90.f, 0.001f,
                    Vector3 { 0, 0, 0.8 }, { (int32_t)agentIdx });
    }

    ObjectID agent_obj_id = ObjectID { 4 };
    ctx.get<ObjectID>(agent) = agent_obj_id;

    ctx.get<phys::broadphase::LeafID>(agent) =
        phys::RigidBodyPhysicsSystem::registerEntity(ctx, agent,
                agent_obj_id);

    ctx.get<Velocity>(agent) = {
        Vector3::zero(),
        Vector3::zero(),
    };

    ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
    ctx.get<ExternalForce>(agent) = Vector3::zero();
    ctx.get<ExternalTorque>(agent) = Vector3::zero();

    return agent;
}

static Entity makePlane(Engine &ctx, Vector3 offset, Quat rot) {
    return makeDynObject(ctx, offset, rot, 1, ResponseType::Static);
}

static void generateTrainingEnvironment(Engine &ctx)
{
    const Vector2 bounds { -18.f, 18.f };

    // After this function, all the entities for the walls have been created
    populateStaticGeometry(ctx, ctx.data().rng, {bounds.y, bounds.y}, ctx.data().srcRoom, ctx.data().dstRoom);

    Room &room = ctx.data().rooms[ctx.data().srcRoom];

    const ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;

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
        Diag3x3 scale = {1.0f, 1.0f, 1.0f};

        AABB aabb = obj_mgr.rigidBodyAABBs[4];
        aabb = aabb.applyTRS(pos, rot, scale);

        Entity agent = ctx.data().agents[i];

        // Reset state for the agent
        ctx.get<phys::broadphase::LeafID>(agent) =
            phys::RigidBodyPhysicsSystem::registerEntity(ctx, agent,
                    ctx.get<ObjectID>(agent));
        ctx.get<Action>(agent) = { .x = 0, .y = 0, .r = 0, };
        ctx.get<Position>(agent) = pos;
        ctx.get<Rotation>(agent) = rot;
        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.get<Velocity>(agent) = { Vector3::zero(), Vector3::zero() };
        ctx.get<ExternalForce>(agent) = Vector3::zero();
        ctx.get<ExternalTorque>(agent) = Vector3::zero();

        if (ctx.data().enableRender) {
            ctx.get<render::ViewSettings>(agent) =
                render::RenderingSystem::setupView(ctx, 90.f, 0.001f,
                        Vector3 { 0, 0, 0.8 }, { (int32_t)i });
        }
    }

    // Register the plane object again to the physics system
    ctx.get<phys::broadphase::LeafID>(ctx.data().floorPlane) =
        phys::RigidBodyPhysicsSystem::registerEntity(ctx, ctx.data().floorPlane, 
            ctx.get<ObjectID>(ctx.data().floorPlane));
}

void generateEnvironment(Engine &ctx)
{
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx =
        episode_mgr.curEpisode.fetch_add<sync::relaxed>(1);
    ctx.data().rng = RNG::make(episode_idx);

    ctx.data().curEpisodeSeed = episode_idx;

    generateTrainingEnvironment(ctx);
}

void createAgents(Engine &ctx)
{
    const ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;

    // Need to create the entities themselves
    for (CountT i = 0; i < consts::numAgents; ++i) {
        Vector3 pos {
            0, 0, 1.5f,
        };

        const auto rot = Quat::angleAxis(ctx.data().rng.rand() * math::pi, {0, 0, 1});
        Diag3x3 scale = {1.0f, 1.0f, 1.0f};

        AABB aabb = obj_mgr.rigidBodyAABBs[4];
        aabb = aabb.applyTRS(pos, rot, scale);

        ctx.data().agents[i] = makeAgent(ctx, i, pos, rot);
    }
}

void createFloor(Engine &ctx)
{
    ctx.data().floorPlane = makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
}

}

