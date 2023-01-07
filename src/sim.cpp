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
    registry.registerComponent<AgentImpl>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<DynamicObject>();
    registry.registerArchetype<StaticObject>();
    registry.registerArchetype<AgentInterface>();
    registry.registerArchetype<CameraAgent>();
    registry.registerArchetype<DynAgent>();

    registry.exportSingleton<WorldReset>(0);
    registry.exportColumn<AgentInterface, Action>(1);
}

static Entity makeDynObject(Engine &ctx, Vector3 pos, Quat rot,
                            int32_t obj_id, Vector3 scale = {1, 1, 1})
{
    Entity e = ctx.makeEntityNow<DynamicObject>();
    ctx.getUnsafe<Position>(e) = pos;
    ctx.getUnsafe<Rotation>(e) = rot;
    ctx.getUnsafe<Scale>(e) = scale;
    ctx.getUnsafe<ObjectID>(e) = ObjectID { obj_id };
    ctx.getUnsafe<phys::broadphase::LeafID>(e) =
        phys::RigidBodyPhysicsSystem::registerEntity(ctx, e, ObjectID {obj_id});
    ctx.getUnsafe<Velocity>(e) = {
        Vector3 { 0, 0, 0 },
        Vector3 { 0, 0, 0 },
    };

    return e;
}

static Entity makePlane(Engine &ctx, Vector3 offset, Quat rot) {
    return makeDynObject(ctx, offset, rot, 1);
}

template <typename T>
static Entity makeAgent(Engine &ctx)
{
    Entity agent_iface = ctx.data().agents[ctx.data().numAgents++] =
        ctx.makeEntityNow<AgentInterface>();
    Entity agent = ctx.makeEntityNow<T>();
    ctx.getUnsafe<AgentImpl>(agent_iface).implEntity = agent;

    return agent;
}

// Emergent tool use configuration:
// 1 - 3 Hiders
// 1 - 3 Seekers
// 3 - 9 Movable boxes (at least 3 elongated)
// 2 movable ramps

static void level1(Engine &ctx)
{
    Entity *all_entities = ctx.data().allEntities;
    auto &rng = ctx.data().rng;

    CountT total_num_boxes = CountT(rng.rand() * 6) + 3;
    CountT num_elongated = 
        CountT(ctx.data().rng.rand() * (total_num_boxes  - 3)) + 3;

    CountT num_hiders = CountT(ctx.data().rng.rand() * 2) + 1;
    CountT num_seekers = CountT(ctx.data().rng.rand() * 2) + 1;

    CountT num_cubes = total_num_boxes - num_elongated;

    const Vector2 bounds { -18.f, 18.f };
    float bounds_diff = bounds.y - bounds.x;

    CountT num_entities = 0;
    for (CountT i = 0; i < num_elongated; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            1.0f,
        };

        const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});

        all_entities[num_entities++] =
            makeDynObject(ctx, pos, rot, 6);
    }

    for (CountT i = 0; i < num_cubes; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            1.0f,
        };

        const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});
        all_entities[num_entities++] = makeDynObject(ctx, pos, rot, 2);
    }

    const CountT num_ramps = 2;
    for (CountT i = 0; i < num_ramps; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            2.f / 3.f,
        };

        const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});

        all_entities[num_entities++] = makeDynObject(ctx, pos, rot, 5);
    }

    all_entities[num_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    all_entities[num_entities++] =
        makePlane(ctx, {0, 0, 100}, Quat::angleAxis(pi, {1, 0, 0}));
    all_entities[num_entities++] =
        makePlane(ctx, {-100, 0, 0}, Quat::angleAxis(pi_d2, {0, 1, 0}));
    all_entities[num_entities++] =
        makePlane(ctx, {100, 0, 0}, Quat::angleAxis(-pi_d2, {0, 1, 0}));
    all_entities[num_entities++] =
        makePlane(ctx, {0, -100, 0}, Quat::angleAxis(-pi_d2, {1, 0, 0}));
    all_entities[num_entities++] =
        makePlane(ctx, {0, 100, 0}, Quat::angleAxis(pi_d2, {1, 0, 0}));

    auto makeHider = [&](Vector3 pos, Quat rot) {
        Entity agent = makeAgent<DynAgent>(ctx);
        ctx.getUnsafe<Position>(agent) = pos;
        ctx.getUnsafe<Rotation>(agent) = rot;
        ctx.getUnsafe<Scale>(agent) = Vector3 { 1, 1, 1 };
        ctx.getUnsafe<render::ViewSettings>(agent) =
            render::RenderingSystem::setupView(ctx, 90.f, Vector3 { 0, 0, 0.8 });

        ObjectID agent_obj_id = ObjectID { 4 };
        ctx.getUnsafe<ObjectID>(agent) = agent_obj_id;
        ctx.getUnsafe<phys::broadphase::LeafID>(agent) =
            phys::RigidBodyPhysicsSystem::registerEntity(ctx, agent,
                                                         agent_obj_id);

        ctx.getUnsafe<Velocity>(agent) = {
            Vector3::zero(),
            Vector3::zero(),
        };
    };

    makeHider({ -15, -15, 1.5 },
        Quat::angleAxis(helpers::toRadians(-45), {0, 0, 1}));

    for (CountT i = 1; i < num_hiders; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            1.5f,
        };

        const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});
        makeHider(pos, rot);
    }

    for (CountT i = 0; i < num_seekers; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            1.5f,
        };

        const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});

        makeHider(pos, rot);
    }

    ctx.data().numEntities = num_entities;
}

static void singleCubeLevel(Engine &ctx, Vector3 pos, Quat rot)
{
    Entity *all_entities = ctx.data().allEntities;

    CountT total_entities = 0;

    Entity test_cube = makeDynObject(ctx, pos, rot, 2);
    all_entities[total_entities++] = test_cube;

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));

    const Quat agent_rot =
        Quat::angleAxis(helpers::toRadians(-45), {0, 0, 1});

    ctx.data().numEntities = total_entities;

    Entity agent = makeAgent<CameraAgent>(ctx);
    ctx.getUnsafe<render::ViewSettings>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f, up * 0.5f);
    ctx.getUnsafe<Position>(agent) = Vector3 { -5, -5, 0 };
    ctx.getUnsafe<Rotation>(agent) = agent_rot;
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

static void level5(Engine &ctx)
{
    Entity *all_entities = ctx.data().allEntities;
    CountT num_entities_range =
        ctx.data().maxEpisodeEntities - ctx.data().minEpisodeEntities;

    CountT num_dyn_entities =
        CountT(ctx.data().rng.rand() * num_entities_range) +
        ctx.data().minEpisodeEntities;

    const Vector2 bounds { -10.f, 10.f };
    float bounds_diff = bounds.y - bounds.x;

    for (CountT i = 0; i < num_dyn_entities; i++) {
        Vector3 pos {
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            1.f,
        };

        const auto rot = Quat::angleAxis(0, {0, 0, 1});

        all_entities[i] = makeDynObject(ctx, pos, rot, 2);
    }

    CountT total_entities = num_dyn_entities;

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 40}, Quat::angleAxis(pi, {1, 0, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {-20, 0, 0}, Quat::angleAxis(pi_d2, {0, 1, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {20, 0, 0}, Quat::angleAxis(-pi_d2, {0, 1, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {0, -20, 0}, Quat::angleAxis(-pi_d2, {1, 0, 0}));
    all_entities[total_entities++] =
        makePlane(ctx, {0, 20, 0}, Quat::angleAxis(pi_d2, {1, 0, 0}));

    const Quat agent_rot =
        Quat::angleAxis(-pi_d2, {1, 0, 0});

    Entity agent = makeAgent<CameraAgent>(ctx);
    ctx.getUnsafe<render::ViewSettings>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f, up * 0.5f);
    ctx.getUnsafe<Position>(agent) = Vector3 { 0, 0, 35 };
    ctx.getUnsafe<Rotation>(agent) = agent_rot;

    ctx.data().numEntities = total_entities;
}

static void resetWorld(Engine &ctx, int32_t level)
{
    phys::RigidBodyPhysicsSystem::reset(ctx);

    Entity *all_entities = ctx.data().allEntities;
    for (CountT i = 0; i < ctx.data().numEntities; i++) {
        Entity e = all_entities[i];
        ctx.destroyEntityNow(e);
    }
    ctx.data().numEntities = 0;

    for (CountT i = 0; i < ctx.data().numAgents; i++) {
        ctx.destroyEntityNow(
            ctx.getUnsafe<AgentImpl>(ctx.data().agents[i]).implEntity);
        ctx.destroyEntityNow(ctx.data().agents[i]);
    }
    ctx.data().numAgents = 0;

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
    case 5: {
        level5(ctx);
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

    {
        int32_t num_rows = state_mgr->numArchetypeRows(
            TypeTracker::typeID<AgentInterface>());
        
        printf("AgentInterface num rows: %u %d\n",
               TypeTracker::typeID<AgentInterface>(),
               num_rows);

        auto col = (WorldID *)state_mgr->getArchetypeComponent(
            TypeTracker::typeID<AgentInterface>(),
            TypeTracker::typeID<WorldID>());

        for (int i = 0; i < num_rows; i++) {
            printf("%d\n", col[i].idx);
        }
    }

    {
        int32_t num_rows = state_mgr->numArchetypeRows(
            TypeTracker::typeID<CameraAgent>());
        
        printf("CameraAgent num rows: %u %d\n",
               TypeTracker::typeID<CameraAgent>(),
               num_rows);

        auto col = (WorldID *)state_mgr->getArchetypeComponent(
            TypeTracker::typeID<CameraAgent>(),
            TypeTracker::typeID<WorldID>());

        for (int i = 0; i < num_rows; i++) {
            printf("%d\n", col[i].idx);
        }
    }

    {
        int32_t num_rows = state_mgr->numArchetypeRows(
            TypeTracker::typeID<DynAgent>());
        
        printf("DynAgent num rows: %u %d\n",
               TypeTracker::typeID<DynAgent>(),
               num_rows);

        auto col = (WorldID *)state_mgr->getArchetypeComponent(
            TypeTracker::typeID<DynAgent>(),
            TypeTracker::typeID<WorldID>());

        for (int i = 0; i < num_rows; i++) {
            printf("%d\n", col[i].idx);
        }
    }
}
#endif

inline void actionSystem(Engine &ctx, Action &action, AgentImpl impl)
{
    constexpr float turn_angle = helpers::toRadians(10.f);


    Position &pos = ctx.getUnsafe<Position>(impl.implEntity);
    Rotation &rot = ctx.getUnsafe<Rotation>(impl.implEntity);

    switch(action.action) {
    case 0: {
        // Do nothing
    } break;
    case 1: {
        Vector3 fwd = rot.rotateVec(math::fwd);
        pos += 0.5f * fwd;
    } break;
    case 2: {
        const Quat left_rot = Quat::angleAxis(turn_angle, math::up);
        rot = (left_rot * rot).normalize();
    } break;
    case 3: {
        const Quat right_rot = Quat::angleAxis(-turn_angle, math::up);
        rot = (right_rot * rot).normalize();
    } break;
    case 4: {
        Vector3 fwd = rot.rotateVec(math::fwd);
        pos -= 0.5f * fwd;
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

inline void agentZeroVelSystem(Engine &,
                               Velocity &vel,
                               render::ViewID &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = fminf(vel.linear.z, 0);

    vel.angular = Vector3::zero();
}

void Sim::setupTasks(TaskGraph::Builder &builder)
{
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem, WorldReset>>({});

    auto sort_agent_iface_sys =
        builder.addToGraph<SortArchetypeNode<AgentInterface, WorldID>>(
            {reset_sys});
    auto post_sort_agent_iface_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_agent_iface_sys});

    auto sort_cam_agent_sys =
        builder.addToGraph<SortArchetypeNode<CameraAgent, WorldID>>(
            {post_sort_agent_iface_reset_tmp});
    auto post_sort_cam_agent_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_cam_agent_sys});

    auto sort_dyn_agent_sys =
        builder.addToGraph<SortArchetypeNode<DynAgent, WorldID>>(
            {post_sort_cam_agent_reset_tmp});
    auto post_sort_dyn_agent_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_dyn_agent_sys});

    auto sort_dyn_sys = 
        builder.addToGraph<SortArchetypeNode<DynamicObject, WorldID>>(
            {post_sort_dyn_agent_reset_tmp});
    auto post_sort_dyn_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_dyn_sys});

    auto sort_static_sys =
        builder.addToGraph<SortArchetypeNode<StaticObject, WorldID>>(
            {post_sort_dyn_reset_tmp});
    auto post_sort_static_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_static_sys});

    auto prep_finish = post_sort_static_reset_tmp;

#if 0
    prep_finish = builder.addToGraph<ParallelForNode<Engine,
        sortDebugSystem, WorldReset>>({prep_finish});
#endif

    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
        Action, AgentImpl>>({prep_finish});

    auto phys_sys = phys::RigidBodyPhysicsSystem::setupTasks(builder,
        {action_sys}, numPhysicsSubsteps);

    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, render::ViewID>>(
            {phys_sys});

    auto sim_done = agent_zero_vel;

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
    CountT max_total_entities =
        std::max(init.maxEntitiesPerWorld, uint32_t(3 + 3 + 9 + 2 + 6)) + 10;

    phys::RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT,
         numPhysicsSubsteps, -9.8 * math::up, max_total_entities, 100 * 50);

    render::RenderingSystem::init(ctx);

    allEntities =
        (Entity *)rawAlloc(sizeof(Entity) * size_t(max_total_entities));

    numEntities = 0;
    minEpisodeEntities = init.minEntitiesPerWorld;
    maxEpisodeEntities = init.maxEntitiesPerWorld;

    numAgents = 0;
    resetWorld(ctx, 1);
    ctx.getSingleton<WorldReset>().resetLevel = 0;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, WorldInit);

}
