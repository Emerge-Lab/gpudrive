#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

namespace GPUHideSeek {

constexpr inline CountT max_instances = 45;
constexpr inline float deltaT = 1.f / 30.f;

void Sim::registerTypes(ECSRegistry &registry)
{
    base::registerTypes(registry);
    phys::RigidBodyPhysicsSystem::registerTypes(registry);
    render::RenderingSystem::registerTypes(registry);

    registry.registerComponent<Action>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<DynamicObject>();
    registry.registerArchetype<Agent>();

    registry.exportSingleton<WorldReset>(0);
    registry.exportColumn<Agent, Action>(1);
}

static void resetWorld(Engine &ctx)
{
    phys::RigidBodyPhysicsSystem::reset(ctx);

    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx =
        episode_mgr.curEpisode.fetch_add(1, std::memory_order_relaxed);
    ctx.data().rng = RNG::make(episode_idx);

    auto reinit_entity = [&](Entity e,
                             Position pos,
                             Rotation rot,
                             Optional<Scale> scale) {
        ctx.getUnsafe<Position>(e) = pos;
        ctx.getUnsafe<Rotation>(e) = rot;

        if (scale.has_value()) {
            ctx.getUnsafe<Scale>(e) = *scale;
        }
    };

    const math::Vector2 bounds { -10.f, 10.f };
    float bounds_diff = bounds.y - bounds.x;

    Entity *dyn_entities = ctx.data().dynObjects;
    for (CountT i = 0; i < max_instances; i++) {
        Entity dyn_entity = dyn_entities[i];

        math::Vector3 pos {
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            1.f,
        };

        const auto rot = math::Quat::angleAxis(0, {0, 0, 1});

        reinit_entity(dyn_entity, pos, rot, Scale(math::Vector3 {1, 1, 1}));

        ctx.getUnsafe<ObjectID>(dyn_entity).idx = 0;
    }

    Entity agent_entity = ctx.data().agent;

    const math::Quat agent_rot =
        math::Quat::angleAxis(-math::pi_d2, {1, 0, 0});

    reinit_entity(agent_entity, math::Vector3 { 0, 0, 14 },
                  agent_rot, Optional<Scale>::none());
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

    printf("(%f %f %f) (%f %f %f %f)\n",
           pos.x, pos.y, pos.z,
           rot.w, rot.x, rot.y, rot.z);
}

void Sim::setupTasks(TaskGraph::Builder &builder)
{
    auto reset_sys =
        builder.parallelForNode<Engine, resetSystem, WorldReset>({});

    auto action_sys = builder.parallelForNode<Engine, actionSystem,
        Action, Position, Rotation>({reset_sys});

    auto phys_sys = phys::RigidBodyPhysicsSystem::setupTasks(builder,
                                                             {action_sys}, 4);

    auto sim_done = phys_sys;

    auto phys_cleanup_sys = phys::RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {sim_done});

    auto renderer_sys = render::RenderingSystem::setupTasks(builder,
        {sim_done});

    (void)phys_cleanup_sys;
    (void)renderer_sys;

    printf("Setup done\n");
}


Sim::Sim(Engine &ctx, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr)
{
    phys::RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT, 4,
                                       max_instances + 1, 100 * 50);

    render::RenderingSystem::init(ctx);

    agent = ctx.makeEntityNow<Agent>();
    ctx.getUnsafe<render::ActiveView>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f, math::up * 0.5f);

    dynObjects = (Entity *)rawAlloc(sizeof(Entity) * size_t(max_instances));

    for (CountT i = 0; i < max_instances; i++) {
        dynObjects[i] = ctx.makeEntityNow<DynamicObject>();
        ctx.getUnsafe<phys::broadphase::LeafID>(dynObjects[i]) =
            phys::RigidBodyPhysicsSystem::registerObject(ctx);
    }

    resetWorld(ctx);
    ctx.getSingleton<WorldReset>().resetNow = false;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, WorldInit);

}
