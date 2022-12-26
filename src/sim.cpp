#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

namespace GPUHideSeek {

constexpr inline float deltaT = 1.f / 30.f;

void Sim::registerTypes(ECSRegistry &registry)
{
    base::registerTypes(registry);
    phys::RigidBodyPhysicsSystem::registerTypes(registry);
    render::RenderingSystem::registerTypes(registry);

    registry.registerComponent<Action>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<DynamicObject>();
    registry.registerArchetype<StaticObject>();
    registry.registerArchetype<Agent>();

    registry.exportSingleton<WorldReset>(0);
    registry.exportColumn<Agent, Action>(1);
}

static void resetWorld(Engine &ctx)
{
    phys::RigidBodyPhysicsSystem::reset(ctx);

    Entity *all_entities = ctx.data().allEntities;
    for (CountT i = 0; i < ctx.data().numEntities; i++) {
        Entity e = all_entities[i];
        ctx.destroyEntityNow(e);
    }

    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx =
        episode_mgr.curEpisode.fetch_add(1, std::memory_order_relaxed);
    ctx.data().rng = RNG::make(episode_idx);

    CountT num_entities_range =
        ctx.data().maxEpisodeEntities - ctx.data().minEpisodeEntities;

    CountT num_dyn_entities =
        CountT(ctx.data().rng.rand() * num_entities_range) +
        ctx.data().minEpisodeEntities;

    const math::Vector2 bounds { -10.f, 10.f };
    float bounds_diff = bounds.y - bounds.x;

    for (CountT i = 0; i < num_dyn_entities; i++) {
        Entity e = ctx.makeEntityNow<DynamicObject>();
        ctx.getUnsafe<phys::broadphase::LeafID>(e) =
            phys::RigidBodyPhysicsSystem::registerEntity(ctx, e);

        math::Vector3 pos {
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            1.f,
        };

        const auto rot = math::Quat::angleAxis(0, {0, 0, 1});

        ctx.getUnsafe<Position>(e) = pos;
        ctx.getUnsafe<Rotation>(e) = rot;
        ctx.getUnsafe<Scale>(e) = math::Vector3 {1, 1, 1};
        ctx.getUnsafe<ObjectID>(e).idx = 2;

        all_entities[i] = e;
    }

    CountT total_entities = num_dyn_entities;

    auto makePlane = [&](Vector3 offset, Quat rot) {
        Entity plane = ctx.makeEntityNow<StaticObject>();
        ctx.getUnsafe<Position>(plane) = offset;
        ctx.getUnsafe<Rotation>(plane) = rot;
        ctx.getUnsafe<Scale>(plane) = Vector3 { 1, 1, 1 };
        ctx.getUnsafe<ObjectID>(plane) = ObjectID { 1 };
        ctx.getUnsafe<phys::broadphase::LeafID>(plane) =
            phys::RigidBodyPhysicsSystem::registerEntity(ctx, plane);

        all_entities[total_entities++] = plane;

        return plane;
    };

    //makePlane({0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    //makePlane({0, 0, 40}, Quat::angleAxis(math::pi, {1, 0, 0}));
    //makePlane({-20, 0, 0}, Quat::angleAxis(math::pi_d2, {0, 1, 0}));
    //makePlane({20, 0, 0}, Quat::angleAxis(-math::pi_d2, {0, 1, 0}));
    //makePlane({0, -20, 0}, Quat::angleAxis(-math::pi_d2, {1, 0, 0}));
    //makePlane({0, 20, 0}, Quat::angleAxis(math::pi_d2, {1, 0, 0}));

    const math::Quat agent_rot =
        math::Quat::angleAxis(-math::pi_d2, {1, 0, 0});

    Entity agent = ctx.data().agent;
    ctx.getUnsafe<Position>(agent) = math::Vector3 { 0, 0, 14 };
    ctx.getUnsafe<Rotation>(agent) = agent_rot;

    printf("World reset %d\n", total_entities);
    ctx.data().numEntities = total_entities;
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    if (!reset.resetNow) {
        return;
    }
    reset.resetNow = false;

    resetWorld(ctx);
}

inline void actionSystem(Engine &, Action &action,
                         Position &pos, Rotation &rot)
{
    constexpr float turn_angle = helpers::toRadians(10.f);

    switch(action.action) {
    case 0: {
        // Do nothing
    } break;
    case 1: {
        Vector3 fwd = rot.rotateDir(math::fwd);
        pos += fwd;
    } break;
    case 2: {
        const Quat left_rot = Quat::angleAxis(turn_angle, math::up);
        rot = (rot * left_rot).normalize();
    } break;
    case 3: {
        const Quat right_rot = Quat::angleAxis(-turn_angle, math::up);
        rot = (rot * right_rot).normalize();
    } break;
    case 4: {
        Vector3 fwd = rot.rotateDir(math::fwd);
        pos -= fwd;
    } break;
    case 5: {
        Vector3 up = rot.rotateDir(math::up);
        pos += up;
    } break;
    case 6: {
        Vector3 up = rot.rotateDir(math::up);
        pos -= up;
    } break;
    default:
        break;
    }

    // "Consume" the action
    action.action = 0;
}

void Sim::setupTasks(TaskGraph::Builder &builder)
{
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem, WorldReset>>({});

    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
        Action, Position, Rotation>>({reset_sys});

    auto phys_sys = phys::RigidBodyPhysicsSystem::setupTasks(builder,
                                                             {action_sys}, 4);

    auto sim_done = phys_sys;

    auto phys_cleanup_sys = phys::RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {sim_done});

    auto renderer_sys = render::RenderingSystem::setupTasks(builder,
        {sim_done});

    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({sim_done});

    (void)phys_cleanup_sys;
    (void)renderer_sys;
    (void)recycle_sys;

    printf("Setup done\n");
}


Sim::Sim(Engine &ctx, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr)
{
    CountT max_total_entities = init.maxEntitiesPerWorld + 10;

    phys::RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT, 4,
                                       max_total_entities, 100 * 50);

    render::RenderingSystem::init(ctx);

    allEntities =
        (Entity *)rawAlloc(sizeof(Entity) * size_t(max_total_entities));

    numEntities = 0;
    minEpisodeEntities = init.minEntitiesPerWorld;
    maxEpisodeEntities = init.maxEntitiesPerWorld;

    agent = ctx.makeEntityNow<Agent>();
    ctx.getUnsafe<render::ActiveView>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f, math::up * 0.5f);

    resetWorld(ctx);
    ctx.getSingleton<WorldReset>().resetNow = false;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, WorldInit);

}
