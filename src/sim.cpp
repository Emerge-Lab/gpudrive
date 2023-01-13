#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>
#include <madrona/mw_gpu/host_print.hpp>

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace GPUHideSeek {

constexpr inline float deltaT = 0.075;
constexpr inline CountT numPhysicsSubsteps = 4;

void Sim::registerTypes(ECSRegistry &registry)
{
    base::registerTypes(registry);
    phys::RigidBodyPhysicsSystem::registerTypes(registry);
    render::RenderingSystem::registerTypes(registry);

    registry.registerComponent<Action>();
    registry.registerComponent<Reward>();
    registry.registerComponent<OwnerTeam>();
    registry.registerComponent<AgentType>();
    registry.registerComponent<GrabData>();

    registry.registerComponent<SimEntity>();

    registry.registerComponent<AgentActiveMask>();
    registry.registerComponent<RelativeAgentObservations>();
    registry.registerComponent<RelativeBoxObservations>();
    registry.registerComponent<RelativeRampObservations>();
    registry.registerComponent<AgentVisibilityMasks>();
    registry.registerComponent<BoxVisibilityMasks>();
    registry.registerComponent<RampVisibilityMasks>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<WorldDone>();
    registry.registerSingleton<PrepPhaseCounter>();

    registry.registerArchetype<DynamicObject>();
    registry.registerArchetype<AgentInterface>();
    registry.registerArchetype<CameraAgent>();
    registry.registerArchetype<DynAgent>();

    registry.exportSingleton<WorldReset>(0);
    registry.exportSingleton<WorldDone>(1);
    registry.exportSingleton<PrepPhaseCounter>(2);
    registry.exportColumn<AgentInterface, Action>(3);
    registry.exportColumn<AgentInterface, Reward>(4);
    registry.exportColumn<AgentInterface, AgentType>(5);
    registry.exportColumn<AgentInterface, AgentActiveMask>(6);
    registry.exportColumn<AgentInterface, RelativeAgentObservations>(7);
    registry.exportColumn<AgentInterface, RelativeBoxObservations>(8);
    registry.exportColumn<AgentInterface, RelativeRampObservations>(9);
    registry.exportColumn<AgentInterface, AgentVisibilityMasks>(10);
    registry.exportColumn<AgentInterface, BoxVisibilityMasks>(11);
    registry.exportColumn<AgentInterface, RampVisibilityMasks>(12);
}

static Entity makeDynObject(Engine &ctx,
                            Vector3 pos,
                            Quat rot,
                            int32_t obj_id,
                            ResponseType response_type = ResponseType::Dynamic,
                            OwnerTeam owner_team = OwnerTeam::None,
                            Vector3 scale = {1, 1, 1})
{
    Entity e = ctx.makeEntityNow<DynamicObject>();
    ctx.getUnsafe<Position>(e) = pos;
    ctx.getUnsafe<Rotation>(e) = rot;
    ctx.getUnsafe<Scale>(e) = scale;
    ctx.getUnsafe<ObjectID>(e) = ObjectID { obj_id };
    ctx.getUnsafe<phys::broadphase::LeafID>(e) =
        phys::RigidBodyPhysicsSystem::registerEntity(ctx, e, ObjectID {obj_id});
    ctx.getUnsafe<Velocity>(e) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.getUnsafe<ResponseType>(e) = response_type;
    ctx.getUnsafe<OwnerTeam>(e) = owner_team;
    ctx.getUnsafe<ExternalForce>(e) = Vector3::zero();
    ctx.getUnsafe<ExternalTorque>(e) = Vector3::zero();

    return e;
}

static Entity makePlane(Engine &ctx, Vector3 offset, Quat rot) {
    return makeDynObject(ctx, offset, rot, 1, ResponseType::Static,
                         OwnerTeam::Unownable);
}

template <typename T>
static Entity makeAgent(Engine &ctx, AgentType agent_type)
{
    Entity agent_iface =
        ctx.data().agentInterfaces[ctx.data().numActiveAgents++];

    Entity agent = ctx.makeEntityNow<T>();
    ctx.getUnsafe<SimEntity>(agent_iface).e = agent;
    ctx.getUnsafe<AgentActiveMask>(agent_iface).mask = 1.f;

    ctx.getUnsafe<AgentType>(agent_iface) = agent_type;

    if (agent_type == AgentType::Seeker) {
        ctx.data().seekers[ctx.data().numSeekers++] = agent;
    } else {
        ctx.data().hiders[ctx.data().numHiders++] = agent;
    }

    return agent;
}

// Emergent tool use configuration:
// 1 - 3 Hiders
// 1 - 3 Seekers
// 3 - 9 Movable boxes (at least 3 elongated)
// 2 movable ramps

static void level1(Engine &ctx, CountT num_hiders, CountT num_seekers)
{
    Entity *all_entities = ctx.data().obstacles;
    auto &rng = ctx.data().rng;

    CountT total_num_boxes = CountT(rng.rand() * 6) + 3;
    assert(total_num_boxes < consts::maxBoxes);

    CountT num_elongated = 
        CountT(ctx.data().rng.rand() * (total_num_boxes - 3)) + 3;

    CountT num_cubes = total_num_boxes - num_elongated;

    assert(num_elongated + num_cubes == total_num_boxes);

    const Vector2 bounds { -18.f, 18.f };
    float bounds_diff = bounds.y - bounds.x;

    CountT num_entities = 0;
    for (CountT i = 0; i < num_elongated; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            1.0f,
        };

        float box_rotation = rng.rand() * math::pi;
        const auto rot = Quat::angleAxis(box_rotation, {0, 0, 1});

        ctx.data().boxes[i] = all_entities[num_entities++] =
                makeDynObject(ctx, pos, rot, 6);

        ctx.data().boxSizes[i] = { 8, 1.5 };
        ctx.data().boxRotations[i] = box_rotation;
    }

    for (CountT i = 0; i < num_cubes; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            1.0f,
        };

        float box_rotation = rng.rand() * math::pi;
        const auto rot = Quat::angleAxis(box_rotation, {0, 0, 1});

        CountT box_idx = i + num_elongated;

        ctx.data().boxes[box_idx] = all_entities[num_entities++] =
            makeDynObject(ctx, pos, rot, 2);

        ctx.data().boxSizes[box_idx] = { 2, 2 };
        ctx.data().boxRotations[box_idx] = box_rotation;
    }

    ctx.data().numActiveBoxes = total_num_boxes;

    const CountT num_ramps = consts::maxRamps;
    for (CountT i = 0; i < num_ramps; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            2.f / 3.f,
        };

        float ramp_rotation = rng.rand() * math::pi;
        const auto rot = Quat::angleAxis(ramp_rotation, {0, 0, 1});

        ctx.data().ramps[i] = all_entities[num_entities++] =
            makeDynObject(ctx, pos, rot, 5);
        ctx.data().rampRotations[i] = ramp_rotation;
    }
    ctx.data().numActiveRamps = num_ramps;

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

    auto makeDynAgent = [&](Vector3 pos, Quat rot, bool is_hider,
                            int32_t view_idx) {
        Entity agent = makeAgent<DynAgent>(ctx,
            is_hider ? AgentType::Hider : AgentType::Seeker);
        ctx.getUnsafe<Position>(agent) = pos;
        ctx.getUnsafe<Rotation>(agent) = rot;
        ctx.getUnsafe<Scale>(agent) = Vector3 { 1, 1, 1 };
        ctx.getUnsafe<render::ViewSettings>(agent) =
            render::RenderingSystem::setupView(ctx, 90.f,
                Vector3 { 0, 0, 0.8 }, { view_idx });

        ObjectID agent_obj_id = ObjectID { 4 };
        ctx.getUnsafe<ObjectID>(agent) = agent_obj_id;
        ctx.getUnsafe<phys::broadphase::LeafID>(agent) =
            phys::RigidBodyPhysicsSystem::registerEntity(ctx, agent,
                                                         agent_obj_id);

        ctx.getUnsafe<Velocity>(agent) = {
            Vector3::zero(),
            Vector3::zero(),
        };
        ctx.getUnsafe<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.getUnsafe<OwnerTeam>(agent) = OwnerTeam::Unownable;
        ctx.getUnsafe<ExternalForce>(agent) = Vector3::zero();
        ctx.getUnsafe<ExternalTorque>(agent) = Vector3::zero();
        ctx.getUnsafe<GrabData>(agent).constraintEntity = Entity::none();

        return agent;
    };

    for (CountT i = 0; i < num_hiders; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            1.5f,
        };

        const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});
        makeDynAgent(pos, rot, true, i);
    }

    for (CountT i = 0; i < num_seekers; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            1.5f,
        };

        const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});

        makeDynAgent(pos, rot, false, num_hiders + i);
    }

    ctx.data().numObstacles = num_entities;
}

static void singleCubeLevel(Engine &ctx, Vector3 pos, Quat rot)
{
    Entity *all_entities = ctx.data().obstacles;

    CountT total_entities = 0;

    Entity test_cube = makeDynObject(ctx, pos, rot, 2);
    all_entities[total_entities++] = test_cube;

    all_entities[total_entities++] =
        makePlane(ctx, {0, 0, 0}, Quat::angleAxis(0, {1, 0, 0}));

    const Quat agent_rot =
        Quat::angleAxis(helpers::toRadians(-45), {0, 0, 1});

    ctx.data().numObstacles = total_entities;

    Entity agent = makeAgent<CameraAgent>(ctx, AgentType::Camera);
    ctx.getUnsafe<render::ViewSettings>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f, up * 0.5f, { 0 });
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
    Entity *all_entities = ctx.data().obstacles;
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

    Entity agent = makeAgent<CameraAgent>(ctx, AgentType::Camera);
    ctx.getUnsafe<render::ViewSettings>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f, up * 0.5f, { 0 });
    ctx.getUnsafe<Position>(agent) = Vector3 { 0, 0, 35 };
    ctx.getUnsafe<Rotation>(agent) = agent_rot;

    ctx.data().numObstacles = total_entities;
}

static void level6(Engine &ctx)
{
    Entity *all_entities = ctx.data().obstacles;
    auto &rng = ctx.data().rng;

    CountT num_entities = 0;
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

    auto makeDynAgent = [&](Vector3 pos, Quat rot, bool is_hider,
                            int32_t view_idx) {
        Entity agent = makeAgent<DynAgent>(ctx,
            is_hider ? AgentType::Hider : AgentType::Seeker);
        printf("Made %d\n", agent.id);
        ctx.getUnsafe<Position>(agent) = pos;
        ctx.getUnsafe<Rotation>(agent) = rot;
        ctx.getUnsafe<Scale>(agent) = Vector3 { 1, 1, 1 };
        ctx.getUnsafe<render::ViewSettings>(agent) =
            render::RenderingSystem::setupView(ctx, 90.f,
                Vector3 { 0, 0, 0.8 }, { view_idx });

        ObjectID agent_obj_id = ObjectID { 4 };
        ctx.getUnsafe<ObjectID>(agent) = agent_obj_id;
        ctx.getUnsafe<phys::broadphase::LeafID>(agent) =
            phys::RigidBodyPhysicsSystem::registerEntity(ctx, agent,
                                                         agent_obj_id);

        ctx.getUnsafe<Velocity>(agent) = {
            Vector3::zero(),
            Vector3::zero(),
        };
        ctx.getUnsafe<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.getUnsafe<OwnerTeam>(agent) = OwnerTeam::Unownable;
        ctx.getUnsafe<ExternalForce>(agent) = Vector3::zero();
        ctx.getUnsafe<ExternalTorque>(agent) = Vector3::zero();
        ctx.getUnsafe<GrabData>(agent).constraintEntity = Entity::none();

        return agent;
    };

    makeDynAgent({ -15, -15, 1.5 },
        Quat::angleAxis(helpers::toRadians(-45), {0, 0, 1}), true, 0);

    makeDynAgent({ -15, -10, 1.5 },
        Quat::angleAxis(helpers::toRadians(45), {0, 0, 1}), false, 1);

    ctx.data().numObstacles = num_entities;
}

static void resetWorld(Engine &ctx,
                       CountT level,
                       CountT num_hiders,
                       CountT num_seekers)
{
    ctx.data().curEpisodeStep = 0;

    ctx.getSingleton<WorldDone>().done = 0;
    ctx.getSingleton<PrepPhaseCounter>().numPrepStepsLeft = 96;

    phys::RigidBodyPhysicsSystem::reset(ctx);

    Entity *all_entities = ctx.data().obstacles;
    for (CountT i = 0; i < ctx.data().numObstacles; i++) {
        Entity e = all_entities[i];
        ctx.destroyEntityNow(e);
    }
    ctx.data().numObstacles = 0;
    ctx.data().numActiveBoxes = 0;
    ctx.data().numActiveRamps = 0;

    auto destroyAgent = [&](Entity e) {
        auto grab_data = ctx.get<GrabData>(e);

        if (grab_data.valid()) {
            auto constraint_entity = grab_data.value().constraintEntity;
            if (constraint_entity != Entity::none()) {
                ctx.destroyEntityNow(constraint_entity);
            }
        }

        ctx.destroyEntityNow(e);
    };

    for (CountT i = 0; i < ctx.data().numHiders; i++) {
        destroyAgent(ctx.data().hiders[i]);
    }
    ctx.data().numHiders = 0;

    for (CountT i = 0; i < ctx.data().numSeekers; i++) {
        destroyAgent(ctx.data().seekers[i]);
    }
    ctx.data().numSeekers = 0;

    for (int32_t i = 0; i < consts::maxAgents; i++) {
        Entity e = ctx.data().agentInterfaces[i];
        ctx.getUnsafe<SimEntity>(e).e = Entity::none();
        ctx.getUnsafe<AgentActiveMask>(e).mask = 0.f;
    }
    ctx.data().numActiveAgents = 0;

    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx =
        episode_mgr.curEpisode.fetch_add(1, std::memory_order_relaxed);
    ctx.data().rng = RNG::make(episode_idx);

    switch (level) {
    case 1: {
        level1(ctx, num_hiders, num_seekers);
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
    case 6: {
        level6(ctx);
    } break;
    }
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t level = reset.resetLevel;

    if (level != 0) {
        reset.resetLevel = 0;

        int32_t num_hiders = reset.numHiders;
        int32_t num_seekers = reset.numSeekers;

        resetWorld(ctx, level, num_hiders, num_seekers);
    }

    // These don't belong here but this runs once per world every frame, so...
    ctx.data().hiderTeamReward.store(1.f, std::memory_order_relaxed);
    CountT step_idx = ++ctx.data().curEpisodeStep;
    if (step_idx >= 239) {
        ctx.getSingleton<WorldDone>().done = 1;
    }

    if (step_idx <= 96) {
        --ctx.getSingleton<PrepPhaseCounter>().numPrepStepsLeft;
    }
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

inline void actionSystem(Engine &ctx, Action &action, SimEntity sim_e,
                         AgentType agent_type)
{
    if (sim_e.e == Entity::none()) return;

    if (agent_type == AgentType::Seeker) {
        int32_t num_prep_left =
            ctx.getSingleton<PrepPhaseCounter>().numPrepStepsLeft;

        if (num_prep_left > 0) {
            return;
        }
    }

    constexpr CountT discrete_action_buckets = 11;
    constexpr CountT half_buckets = discrete_action_buckets / 2;
    constexpr float discrete_action_max = 0.9 * 125;
    constexpr float delta_per_bucket = discrete_action_max / half_buckets;

    Vector3 cur_pos = ctx.getUnsafe<Position>(sim_e.e);
    Quat cur_rot = ctx.getUnsafe<Rotation>(sim_e.e);

    float f_x = delta_per_bucket * action.x;
    float f_y = delta_per_bucket * action.y;
    float t_z = delta_per_bucket * action.r;

    if (agent_type == AgentType::Camera) {
        ctx.getUnsafe<Position>(sim_e.e) = cur_pos + 0.001f * cur_rot.rotateVec({f_x, f_y, 0});

        Quat delta_rot = Quat::angleAxis(t_z * 0.001f, math::up);
        ctx.getUnsafe<Rotation>(sim_e.e) = (delta_rot * cur_rot).normalize();

        return;
    }

    ctx.getUnsafe<ExternalForce>(sim_e.e) = cur_rot.rotateVec({ f_x, f_y, 0 });
    ctx.getUnsafe<ExternalTorque>(sim_e.e) = Vector3 { 0, 0, t_z };

    if (action.l == 1) {
        auto &bvh = ctx.getSingleton<broadphase::BVH>();
        float hit_t;
        Vector3 hit_normal;
        Entity lock_entity = bvh.traceRay(cur_pos - 0.5f * math::up,
            cur_rot.rotateVec(math::fwd), &hit_t, &hit_normal, 1.5f);

        if (lock_entity != Entity::none()) {
            auto &owner = ctx.getUnsafe<OwnerTeam>(lock_entity);
            auto &response_type = ctx.getUnsafe<ResponseType>(lock_entity);

            if (response_type == ResponseType::Static) {
                if ((agent_type == AgentType::Seeker &&
                        owner == OwnerTeam::Seeker) ||
                        (agent_type == AgentType::Hider &&
                         owner == OwnerTeam::Hider)) {
                    response_type = ResponseType::Dynamic;
                    owner = OwnerTeam::None;
                }
            } else {
                if (owner == OwnerTeam::None) {
                    response_type = ResponseType::Static;
                    owner = agent_type == AgentType::Hider ?
                        OwnerTeam::Hider : OwnerTeam::Seeker;
                }
            }
        }
    }

    if (action.g == 1) {
        auto &grab_data = ctx.getUnsafe<GrabData>(sim_e.e);

        if (grab_data.constraintEntity != Entity::none()) {
            ctx.destroyEntityNow(grab_data.constraintEntity);
            grab_data.constraintEntity = Entity::none();
        } else {
            auto &bvh = ctx.getSingleton<broadphase::BVH>();
            float hit_t;
            Vector3 hit_normal;

            Vector3 ray_o = cur_pos - 0.5f * math::up;
            Vector3 ray_d = cur_rot.rotateVec(math::fwd);

            Entity grab_entity =
                bvh.traceRay(ray_o, ray_d, &hit_t, &hit_normal, 1.5f);

            if (grab_entity != Entity::none()) {
                auto &owner = ctx.getUnsafe<OwnerTeam>(grab_entity);
                auto &response_type = ctx.getUnsafe<ResponseType>(grab_entity);

                if (owner == OwnerTeam::None &&
                    response_type == ResponseType::Dynamic) {

                    grab_data.constraintEntity =
                        ctx.makeEntityNow<ConstraintData>();

                    Vector3 other_pos = ctx.getUnsafe<Position>(grab_entity);
                    Quat other_rot = ctx.getUnsafe<Rotation>(grab_entity);

                    auto &joint_constraint = ctx.getUnsafe<JointConstraint>(
                        grab_data.constraintEntity);

                    joint_constraint.e1 = sim_e.e;
                    joint_constraint.e2 = grab_entity;

                    Vector3 r1 = 0.25f * math::fwd;

                    Vector3 hit_pos = ray_o + ray_d * hit_t;
                    Vector3 r2 = other_rot.inv().rotateVec(hit_pos - other_pos);

                    joint_constraint.r1 = r1;
                    joint_constraint.r2 = r2;

                    Vector3 ray_dir_other_local =
                        other_rot.inv().rotateVec(ray_d);

                    // joint_constraint.axes2 needs to map from
                    // (0, 0, 1) to ray_dir_other_local
                    // and (1, 0, 0) to the agent's right vector in the grabbed
                    // object's local space

                    Vector3 right_vec_other_local = other_rot.inv().rotateVec(
                        cur_rot.rotateVec(math::right));

                    Vector3 up_vec_other_local =
                        cross(right_vec_other_local, ray_dir_other_local);

                    joint_constraint.axes1 = { 1, 0, 0, 0 };
                    joint_constraint.axes2 = Quat::fromBasis(
                        right_vec_other_local, up_vec_other_local,
                        ray_dir_other_local);
                    joint_constraint.separation = hit_t - 0.25f;
                }
            }
        }
    }

    // "Consume" the actions. This isn't strictly necessary but
    // allows step to be called without every agent having acted
    action.x = 0;
    action.y = 0;
    action.r = 0;
    action.g = 0;
    action.l = 0;
}

inline void agentZeroVelSystem(Engine &,
                               Velocity &vel,
                               render::ViewSettings &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = fminf(vel.linear.z, 0);

    vel.angular = Vector3::zero();
}

inline void collectObservationsSystem(Engine &ctx,
                                      Entity agent_e,
                                      SimEntity sim_e,
                                      AgentType agent_type,
                                      RelativeAgentObservations &agent_obs,
                                      RelativeBoxObservations &box_obs,
                                      RelativeRampObservations &ramp_obs)
{
    if (sim_e.e == Entity::none() || agent_type == AgentType::Camera) {
        return;
    }

    Vector3 agent_pos = ctx.getUnsafe<Position>(sim_e.e);
    Quat agent_rot = ctx.getUnsafe<Rotation>(sim_e.e);

    CountT num_boxes = ctx.data().numActiveBoxes;
    for (CountT box_idx = 0; box_idx < consts::maxBoxes; box_idx++) {
        auto &obs = box_obs.obs[box_idx];

        if (box_idx >= num_boxes) {
            obs= {};
            continue;
        }

        Entity box_e = ctx.data().boxes[box_idx];

        Vector3 box_pos = ctx.getUnsafe<Position>(box_e);
        Vector3 box_vel = ctx.getUnsafe<Velocity>(box_e).linear;
        Quat box_rot = ctx.getUnsafe<Rotation>(box_e);

        Vector3 box_relative_pos =
            agent_rot.inv().rotateVec(box_pos - agent_pos);
        Vector3 box_relative_vel =
            agent_rot.inv().rotateVec(box_vel);

        obs.pos = { box_relative_pos.x, box_relative_pos.y };
        obs.vel = { box_relative_vel.x, box_relative_vel.y };
        obs.boxSize = ctx.data().boxSizes[box_idx];

        Quat relative_rot = agent_rot * box_rot.inv();
        obs.boxRotation = atan2f(
            2.f * (relative_rot.w * relative_rot.z +
                   relative_rot.x * relative_rot.y),
            1.f - 2.f * (relative_rot.y * relative_rot.y +
                         relative_rot.z * relative_rot.z));
    }

    CountT num_ramps = ctx.data().numActiveRamps;
    for (CountT ramp_idx = 0; ramp_idx < consts::maxRamps; ramp_idx++) {
        auto &obs = ramp_obs.obs[ramp_idx];

        if (ramp_idx >= num_ramps) {
            obs = {};
            continue;
        }

        Entity ramp_e = ctx.data().ramps[ramp_idx];

        Vector3 ramp_pos = ctx.getUnsafe<Position>(ramp_e);
        Vector3 ramp_vel = ctx.getUnsafe<Velocity>(ramp_e).linear;
        Quat ramp_rot = ctx.getUnsafe<Rotation>(ramp_e);

        Vector3 ramp_relative_pos =
            agent_rot.inv().rotateVec(ramp_pos - agent_pos);
        Vector3 ramp_relative_vel =
            agent_rot.inv().rotateVec(ramp_vel);

        obs.pos = { ramp_relative_pos.x, ramp_relative_pos.y };
        obs.vel = { ramp_relative_vel.x, ramp_relative_vel.y };

        Quat relative_rot = agent_rot * ramp_rot.inv();
        obs.rampRotation = atan2f(
            2.f * (relative_rot.w * relative_rot.z +
                   relative_rot.x * relative_rot.y),
            1.f - 2.f * (relative_rot.y * relative_rot.y +
                         relative_rot.z * relative_rot.z));
    }

    CountT num_agents = ctx.data().numActiveAgents;
    CountT num_other_agents = 0;
    for (CountT agent_idx = 0; agent_idx < consts::maxAgents; agent_idx++) {
        if (agent_idx >= num_agents) {
            agent_obs.obs[num_other_agents++] = {};
            continue;
        }

        Entity other_agent_e = ctx.data().agentInterfaces[agent_idx];
        if (agent_e == other_agent_e) {
            continue;
        }

        Entity other_agent_sim_e = ctx.getUnsafe<SimEntity>(other_agent_e).e;

        auto &obs = agent_obs.obs[num_other_agents++];

        Vector3 other_agent_pos =
            ctx.getUnsafe<Position>(other_agent_sim_e);
        Vector3 other_agent_vel =
            ctx.getUnsafe<Velocity>(other_agent_sim_e).linear;
        Quat other_agent_rot =
            ctx.getUnsafe<Rotation>(other_agent_sim_e);

        Vector3 other_agent_relative_pos =
            agent_rot.inv().rotateVec(other_agent_pos - agent_pos);
        Vector3 other_agent_relative_vel =
            agent_rot.inv().rotateVec(other_agent_vel);

        obs.pos = { other_agent_relative_pos.x, other_agent_relative_pos.y };
        obs.vel = { other_agent_relative_vel.x, other_agent_relative_vel.y };
    }
}

inline void computeVisibilitySystem(Engine &ctx,
                                    Entity agent_e,
                                    SimEntity sim_e,
                                    AgentType agent_type,
                                    AgentVisibilityMasks &agent_vis,
                                    BoxVisibilityMasks &box_vis,
                                    RampVisibilityMasks &ramp_vis)
{
    if (sim_e.e == Entity::none() || agent_type == AgentType::Camera) {
        return;
    }

    Vector3 agent_pos = ctx.getUnsafe<Position>(sim_e.e);
    Quat agent_rot = ctx.getUnsafe<Rotation>(sim_e.e);
    Vector3 fwd = agent_rot.rotateVec(math::fwd);
    const float cos_angle_threshold = cosf(helpers::toRadians(135.f / 2.f));

    auto &bvh = ctx.getSingleton<broadphase::BVH>();

    auto checkVisibility = [&](Entity other_e) {
        Vector3 other_pos = ctx.getUnsafe<Position>(other_e);

        Vector3 to_other = other_pos - agent_pos;

        Vector3 to_other_norm = to_other.normalize();

        float cos_angle = dot(to_other_norm, fwd);

        if (cos_angle < cos_angle_threshold) {
            return 0.f;
        }

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(agent_pos, to_other, &hit_t, &hit_normal, 1.f);

        return hit_entity == other_e ? 1.f : 0.f;
    };

    CountT num_boxes = ctx.data().numActiveBoxes;
    for (CountT box_idx = 0; box_idx < consts::maxBoxes; box_idx++) {
        if (box_idx < num_boxes) {
            Entity box_e = ctx.data().boxes[box_idx];
            box_vis.visible[box_idx] = checkVisibility(box_e);
        } else {
            box_vis.visible[box_idx] = 0.f;
        }
    }

    CountT num_ramps = ctx.data().numActiveRamps;
    for (CountT ramp_idx = 0; ramp_idx < consts::maxRamps; ramp_idx++) {
        if (ramp_idx < num_ramps) {
            Entity ramp_e = ctx.data().ramps[ramp_idx];
            ramp_vis.visible[ramp_idx] = checkVisibility(ramp_e);
        } else {
            ramp_vis.visible[ramp_idx] = 0.f;
        }
    }

    CountT num_agents = ctx.data().numActiveAgents;
    CountT num_other_agents = 0;
    for (CountT agent_idx = 0; agent_idx < consts::maxAgents; agent_idx++) {
        if (agent_idx >= num_agents) {
            agent_vis.visible[num_other_agents++] = 0.f;
            continue;
        }

        Entity other_agent_e = ctx.data().agentInterfaces[agent_idx];
        if (agent_e == other_agent_e) {
            continue;
        }

        Entity other_agent_sim_e = ctx.getUnsafe<SimEntity>(other_agent_e).e;

        bool is_visible = checkVisibility(other_agent_sim_e);

        if (agent_type == AgentType::Seeker && is_visible) {
            AgentType other_type = ctx.getUnsafe<AgentType>(other_agent_e);
            if (other_type == AgentType::Hider) {
                ctx.data().hiderTeamReward.store(-1.f,
                    std::memory_order_relaxed);
            }
        }

        agent_vis.visible[num_other_agents++] = is_visible;
    }
}

inline void agentRewardsSystem(Engine &ctx,
                               SimEntity sim_e,
                               AgentType agent_type,
                               Reward &reward)
{
    if (sim_e.e == Entity::none()) {
        return;
    }

    int32_t num_prep_left =
        ctx.getSingleton<PrepPhaseCounter>().numPrepStepsLeft;

    if (num_prep_left > 0) {
        reward.reward = 0.f;
        return;
    }

    float reward_val = ctx.data().hiderTeamReward.load(
        std::memory_order_relaxed);
    if (agent_type == AgentType::Seeker) {
        reward_val *= -1.f;
    }

    Vector3 pos = ctx.getUnsafe<Position>(sim_e.e);

    if (fabsf(pos.x) >= 18.f || fabsf(pos.y) >= 18.f) {
        reward_val -= 10.f;
    }

    reward.reward = reward_val;
}

template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}

void Sim::setupTasks(TaskGraph::Builder &builder)
{
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem, WorldReset>>({});

    // FIXME: these 3 need to be compacted, but sorting is unnecessary
    auto sort_cam_agent = queueSortByWorld<CameraAgent>(builder, {reset_sys});
    auto sort_dyn_agent = queueSortByWorld<DynAgent>(builder, {sort_cam_agent});
    auto sort_objects = queueSortByWorld<DynamicObject>(builder, {sort_dyn_agent});

    // FIXME: These don't need to be sorted after initialization (purely static)
    auto sort_agent_iface =
        queueSortByWorld<AgentInterface>(builder, {sort_objects});
    auto prep_finish = sort_agent_iface;

#if 0
    prep_finish = builder.addToGraph<ParallelForNode<Engine,
        sortDebugSystem, WorldReset>>({prep_finish});
#endif

    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
        Action, SimEntity, AgentType>>({prep_finish});

    auto phys_sys = phys::RigidBodyPhysicsSystem::setupTasks(builder,
        {action_sys}, numPhysicsSubsteps);

    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, render::ViewSettings>>(
            {phys_sys});

    auto sim_done = agent_zero_vel;

    auto phys_cleanup_sys = phys::RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {sim_done});

    auto renderer_sys = render::RenderingSystem::setupTasks(builder,
        {sim_done});

    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({sim_done});

    auto collect_observations = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Entity,
            SimEntity,
            AgentType,
            RelativeAgentObservations,
            RelativeBoxObservations,
            RelativeRampObservations
        >>({sim_done});


    auto compute_visibility = builder.addToGraph<ParallelForNode<Engine,
        computeVisibilitySystem,
            Entity,
            SimEntity,
            AgentType,
            AgentVisibilityMasks,
            BoxVisibilityMasks,
            RampVisibilityMasks
        >>({sim_done});

    auto agent_rewards = builder.addToGraph<ParallelForNode<Engine,
        agentRewardsSystem,
            SimEntity,
            AgentType,
            Reward
        >>({compute_visibility});

    (void)phys_cleanup_sys;
    (void)renderer_sys;
    (void)recycle_sys;
    (void)collect_observations;
    (void)agent_rewards;

    printf("Setup done\n");
}

Sim::Sim(Engine &ctx, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr)
{
    CountT max_total_entities =
        std::max(init.maxEntitiesPerWorld, uint32_t(3 + 3 + 9 + 2 + 6)) + 10;

    phys::RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT,
         numPhysicsSubsteps, -9.8 * math::up, max_total_entities,
         100 * 50, 10);

    render::RenderingSystem::init(ctx);

    obstacles =
        (Entity *)rawAlloc(sizeof(Entity) * size_t(max_total_entities));

    numObstacles = 0;
    minEpisodeEntities = init.minEntitiesPerWorld;
    maxEpisodeEntities = init.maxEntitiesPerWorld;

    numHiders = 0;
    numSeekers = 0;

    for (int32_t i = 0; i < consts::maxAgents; i++) {
        agentInterfaces[i] = ctx.makeEntityNow<AgentInterface>();
    }

    resetWorld(ctx, 1, 3, 3);
    ctx.getSingleton<WorldReset>().resetLevel = 0;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, WorldInit);

}
