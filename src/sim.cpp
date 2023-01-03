#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

namespace GPUHideSeek {

constexpr inline float deltaT = 1.f / 30.f;
constexpr inline float numPhysicsSubsteps = 4;

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

static Entity makeDynObject(Engine &ctx, Vector3 pos, Quat rot,
                            int32_t obj_id)
{
    Entity e = ctx.makeEntityNow<DynamicObject>();
    ctx.getUnsafe<Position>(e) = pos;
    ctx.getUnsafe<Rotation>(e) = rot;
    ctx.getUnsafe<Scale>(e) = Vector3 { 1, 1, 1 };
    ctx.getUnsafe<ObjectID>(e) = ObjectID { obj_id };
    ctx.getUnsafe<phys::broadphase::LeafID>(e) =
        phys::RigidBodyPhysicsSystem::registerEntity(ctx, e);
    ctx.getUnsafe<Velocity>(e) = {
        Vector3 { 0, 0, 0 },
        Vector3 { 0, 0, 0 },
    };

    return e;
}

static Entity makePlane(Engine &ctx, Vector3 offset, Quat rot) {
    return makeDynObject(ctx, offset, rot, 1);
}

static void level1(Engine &ctx)
{
    Entity *all_entities = ctx.data().allEntities;
    CountT num_entities_range =
        ctx.data().maxEpisodeEntities - ctx.data().minEpisodeEntities;

    CountT num_dyn_entities =
        CountT(ctx.data().rng.rand() * num_entities_range) +
        ctx.data().minEpisodeEntities;

    const math::Vector2 bounds { -10.f, 10.f };
    float bounds_diff = bounds.y - bounds.x;

    for (CountT i = 0; i < num_dyn_entities; i++) {
        math::Vector3 pos {
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            1.f,
        };

        const auto rot = math::Quat::angleAxis(0, {0, 0, 1});

        all_entities[i] = makeDynObject(ctx, pos, rot, 2);
    }

    CountT total_entities = num_dyn_entities;

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 40}, Quat::angleAxis(math::pi, {1, 0, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {-20, 0, 0}, Quat::angleAxis(math::pi_d2, {0, 1, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {20, 0, 0}, Quat::angleAxis(-math::pi_d2, {0, 1, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {0, -20, 0}, Quat::angleAxis(-math::pi_d2, {1, 0, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {0, 20, 0}, Quat::angleAxis(math::pi_d2, {1, 0, 0}));

    const math::Quat agent_rot =
        math::Quat::angleAxis(-math::pi_d2, {1, 0, 0});

    Entity agent = ctx.data().agent;
    ctx.getUnsafe<Position>(agent) = math::Vector3 { 0, 0, 35 };
    ctx.getUnsafe<Rotation>(agent) = agent_rot;

    ctx.data().numEntities = total_entities;
}

static void singleCubeLevel(Engine &ctx, Vector3 pos, Quat rot)
{
    Entity *all_entities = ctx.data().allEntities;

    CountT total_entities = 0;

    Entity test_cube = makeDynObject(ctx, pos, rot, 2);
    all_entities[total_entities++] = test_cube;

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));

    const math::Quat agent_rot =
        math::Quat::angleAxis(helpers::toRadians(-45), {0, 0, 1});

    Entity agent = ctx.data().agent;
    ctx.getUnsafe<Position>(agent) = math::Vector3 { -5, -5, 0 };
    ctx.getUnsafe<Rotation>(agent) = agent_rot;

    ctx.data().numEntities = total_entities;
}

static void level2(Engine &ctx)
{
    Quat cube_rotation = (Quat::angleAxis(atanf(1.f/sqrtf(2.f)), {0, 1, 0}) *
        Quat::angleAxis(helpers::toRadians(45), {1, 0, 0})).normalize().normalize();
    singleCubeLevel(ctx, { 0, 0, 5 }, cube_rotation);
}

static void level3(Engine &ctx)
{
    singleCubeLevel(ctx, { 0, 0, 5 }, Quat::angleAxis(0, {0, 0, 1}));
}

static void level4(Engine &ctx)
{
    Quat cube_rotation = (
        Quat::angleAxis(helpers::toRadians(45), {0, 1, 0}) *
        Quat::angleAxis(helpers::toRadians(40), {1, 0, 0})).normalize();
    singleCubeLevel(ctx, { 0, 0, 5 }, cube_rotation);
}

static void resetWorld(Engine &ctx, int32_t level)
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

    switch (level) {
    case 1: {
        level1(ctx);
    } break;
    case 2: {
        level2(ctx);
    } break;
    case 3: {
        level3(ctx);
    } break;
    case 4: {
        level4(ctx);
    } break;
    }
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t level = reset.resetLevel;
    if (level == 0) {
        return;
    }
    reset.resetLevel = 0;

    resetWorld(ctx, level);
}

#if 0
inline void sortDebugSystem(Engine &ctx, WorldReset &)
{
    if (ctx.worldID().idx != 0) {
        return;
    }

    auto state_mgr = mwGPU::getStateManager();

    int32_t num_rows = state_mgr->numArchetypeRows(
        TypeTracker::typeID<DynamicObject>());

    auto col = (WorldID *)state_mgr->getArchetypeComponent(
        TypeTracker::typeID<DynamicObject>(),
        TypeTracker::typeID<WorldID>());

    for (int i = 0; i < num_rows; i++) {
        printf("%d\n", col[i].idx);
    }
}
#endif

inline void actionSystem(Engine &, Action &action,
                         Position &pos, Rotation &rot)
{
    constexpr float turn_angle = helpers::toRadians(10.f);

    switch(action.action) {
    case 0: {
        // Do nothing
    } break;
    case 1: {
        Vector3 fwd = rot.rotateVec(math::fwd);
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
        Vector3 fwd = rot.rotateVec(math::fwd);
        pos -= fwd;
    } break;
    case 5: {
        Vector3 up = rot.rotateVec(math::up);
        pos += up;
    } break;
    case 6: {
        Vector3 up = rot.rotateVec(math::up);
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

    auto sort_agent_sys = builder.addToGraph<SortArchetypeNode<Agent, WorldID>>({reset_sys});
    auto post_sort_agent_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_agent_sys});

    auto sort_dyn_sys = builder.addToGraph<SortArchetypeNode<DynamicObject, WorldID>>(
        {post_sort_agent_reset_tmp});
    auto post_sort_dyn_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_dyn_sys});

    auto sort_static_sys = builder.addToGraph<SortArchetypeNode<StaticObject, WorldID>>(
        {post_sort_dyn_reset_tmp});
    auto post_sort_static_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_static_sys});

    auto prep_finish = post_sort_static_reset_tmp;

#if 0
    prep_finish = builder.addToGraph<ParallelForNode<Engine,
        sortDebugSystem, WorldReset>>({prep_finish});
#endif

    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
        Action, Position, Rotation>>({prep_finish});

    auto phys_sys = phys::RigidBodyPhysicsSystem::setupTasks(builder,
        {action_sys}, numPhysicsSubsteps);

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

    phys::RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT,
         numPhysicsSubsteps, -9.8 * math::up, max_total_entities, 100 * 50);

    render::RenderingSystem::init(ctx);

    allEntities =
        (Entity *)rawAlloc(sizeof(Entity) * size_t(max_total_entities));

    numEntities = 0;
    minEpisodeEntities = init.minEntitiesPerWorld;
    maxEpisodeEntities = init.maxEntitiesPerWorld;

    agent = ctx.makeEntityNow<Agent>();
    ctx.getUnsafe<render::ActiveView>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f, math::up * 0.5f);

    resetWorld(ctx, 1);
    ctx.getSingleton<WorldReset>().resetLevel = 0;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, WorldInit);

}
