#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>
#include <madrona/mw_gpu/host_print.hpp>

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace GPUHideSeek {

constexpr inline float deltaT = 0.075;
constexpr inline float numPhysicsSubsteps = 4;

void Sim::registerTypes(ECSRegistry &registry)
{
    base::registerTypes(registry);
    phys::RigidBodyPhysicsSystem::registerTypes(registry);
    render::RenderingSystem::registerTypes(registry);

    registry.registerComponent<Action>();
    registry.registerComponent<Reward>();
    registry.registerComponent<OwnerTeam>();
    registry.registerComponent<AgentType>();

    registry.registerComponent<SimEntity>();
    registry.registerComponent<PositionObservation>();
    registry.registerComponent<VelocityObservation>();
    registry.registerComponent<ObservationMask>();
    registry.registerComponent<VisibilityMasks>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<WorldDone>();

    registry.registerArchetype<DynamicObject>();
    registry.registerArchetype<AgentInterface>();
    registry.registerArchetype<CameraAgent>();
    registry.registerArchetype<DynAgent>();
    registry.registerArchetype<BoxObservation>();
    registry.registerArchetype<RampObservation>();

    registry.exportSingleton<WorldReset>(0);
    registry.exportSingleton<WorldDone>(1);
    registry.exportColumn<AgentInterface, Action>(2);
    registry.exportColumn<AgentInterface, Reward>(3);
    registry.exportColumn<AgentInterface, AgentType>(4);
    registry.exportColumn<AgentInterface, PositionObservation>(5);
    registry.exportColumn<AgentInterface, VelocityObservation>(6);
    registry.exportColumn<AgentInterface, ObservationMask>(7);
    registry.exportColumn<AgentInterface, VisibilityMasks>(8);

    registry.exportColumn<BoxObservation, PositionObservation>(9);
    registry.exportColumn<BoxObservation, VelocityObservation>(10);
    registry.exportColumn<BoxObservation, ObservationMask>(11);

    registry.exportColumn<RampObservation, PositionObservation>(12);
    registry.exportColumn<RampObservation, VelocityObservation>(13);
    registry.exportColumn<RampObservation, ObservationMask>(14);
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
static Entity makeAgent(Engine &ctx, bool is_hider = true)
{
    Entity agent_iface =
        ctx.data().agentObservations[ctx.data().numActiveAgents++];

    Entity agent = ctx.makeEntityNow<T>();
    ctx.getUnsafe<SimEntity>(agent_iface).e = agent;
    ctx.getUnsafe<ObservationMask>(agent_iface).mask = 1.f;

    ctx.getUnsafe<AgentType>(agent_iface) =
        is_hider ? AgentType::Hider : AgentType::Seeker;

    if (is_hider) {
        ctx.data().hiders[ctx.data().numHiders++] = agent;
    } else {
        ctx.data().seekers[ctx.data().numSeekers++] = agent;
    }

    return agent;
}

// Emergent tool use configuration:
// 1 - 3 Hiders
// 1 - 3 Seekers
// 3 - 9 Movable boxes (at least 3 elongated)
// 2 movable ramps

static void level1(Engine &ctx)
{
    Entity *all_entities = ctx.data().obstacles;
    auto &rng = ctx.data().rng;

    CountT total_num_boxes = CountT(rng.rand() * 6) + 3;
    assert(total_num_boxes < consts::maxBoxes);

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

        Entity box = all_entities[num_entities++] =
            makeDynObject(ctx, pos, rot, 6);
        
        Entity obs_e = ctx.data().boxObservations[i];
        ctx.getUnsafe<SimEntity>(obs_e).e = box;
        ctx.getUnsafe<ObservationMask>(obs_e).mask = 1.f;
    }

    for (CountT i = 0; i < num_cubes; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            1.0f,
        };

        const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});
        Entity box = all_entities[num_entities++] =
            makeDynObject(ctx, pos, rot, 2);

        Entity obs_e = ctx.data().boxObservations[num_elongated + i];
        ctx.getUnsafe<SimEntity>(obs_e).e = box;
        ctx.getUnsafe<ObservationMask>(obs_e).mask = 1.f;
    }

    const CountT num_ramps = consts::maxRamps;
    for (CountT i = 0; i < num_ramps; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            2.f / 3.f,
        };

        const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});

        Entity ramp = all_entities[num_entities++] =
            makeDynObject(ctx, pos, rot, 5);

        Entity obs_e = ctx.data().rampObservations[i];
        ctx.getUnsafe<SimEntity>(obs_e).e = ramp;
        ctx.getUnsafe<ObservationMask>(obs_e).mask = 1.f;
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

    auto makeDynAgent = [&](Vector3 pos, Quat rot, bool is_hider) {
        Entity agent = makeAgent<DynAgent>(ctx, is_hider);
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
        ctx.getUnsafe<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.getUnsafe<OwnerTeam>(agent) = OwnerTeam::Unownable;
        ctx.getUnsafe<ExternalForce>(agent) = Vector3::zero();
        ctx.getUnsafe<ExternalTorque>(agent) = Vector3::zero();

        return agent;
    };

    makeDynAgent({ -15, -15, 1.5 },
        Quat::angleAxis(helpers::toRadians(-45), {0, 0, 1}), true);

    for (CountT i = 1; i < num_hiders; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            1.5f,
        };

        const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});
        makeDynAgent(pos, rot, true);
    }

    for (CountT i = 0; i < num_seekers; i++) {
        Vector3 pos {
            bounds.x + rng.rand() * bounds_diff,
            bounds.x + rng.rand() * bounds_diff,
            1.5f,
        };

        const auto rot = Quat::angleAxis(rng.rand() * math::pi, {0, 0, 1});

        makeDynAgent(pos, rot, false);
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

    Entity agent = makeAgent<CameraAgent>(ctx);
    ctx.getUnsafe<render::ViewSettings>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f, up * 0.5f);
    ctx.getUnsafe<Position>(agent) = Vector3 { 0, 0, 35 };
    ctx.getUnsafe<Rotation>(agent) = agent_rot;

    ctx.data().numObstacles = total_entities;
}

static void resetWorld(Engine &ctx, int32_t level)
{
    ctx.getSingleton<WorldDone>().done = 0;
    phys::RigidBodyPhysicsSystem::reset(ctx);

    Entity *all_entities = ctx.data().obstacles;
    for (CountT i = 0; i < ctx.data().numObstacles; i++) {
        Entity e = all_entities[i];
        ctx.destroyEntityNow(e);
    }
    ctx.data().numObstacles = 0;

    for (CountT i = 0; i < ctx.data().numHiders; i++) {
        ctx.destroyEntityNow(ctx.data().hiders[i]);
    }
    ctx.data().numHiders = 0;

    for (CountT i = 0; i < ctx.data().numSeekers; i++) {
        ctx.destroyEntityNow(ctx.data().seekers[i]);
    }
    ctx.data().numSeekers = 0;

    auto clearObservationEntity = [&](Entity e) {
        ctx.getUnsafe<SimEntity>(e).e = Entity::none();
        ctx.getUnsafe<ObservationMask>(e).mask = 0.f;
    };

    for (int32_t i = 0; i < consts::maxBoxes; i++) {
        clearObservationEntity(ctx.data().boxObservations[i]);
    }

    for (int32_t i = 0; i < consts::maxRamps; i++) {
        clearObservationEntity(ctx.data().rampObservations[i]);
    }

    for (int32_t i = 0; i < consts::maxAgents; i++) {
        clearObservationEntity(ctx.data().agentObservations[i]);
    }
    ctx.data().numActiveAgents = 0;

    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx =
        episode_mgr.curEpisode.fetch_add(1, std::memory_order_relaxed);
    ctx.data().rng = RNG::make(episode_idx);

    ctx.data().curEpisodeStep = 0;

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
    // These don't belong here but this runs once per world every frame, so...
    ctx.data().hiderTeamReward.store(1.f, std::memory_order_relaxed);
    CountT step_idx = ++ctx.data().curEpisodeStep;
    if (step_idx >= 239) {
        ctx.getSingleton<WorldDone>().done = 1;
    }

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

inline void actionSystem(Engine &ctx, Action &action, SimEntity sim_e,
                         AgentType agent_type)
{
    if (sim_e.e == Entity::none()) return;

    constexpr CountT discrete_action_buckets = 11;
    constexpr CountT half_buckets = discrete_action_buckets / 2;
    constexpr float discrete_action_max = 0.9 * 10;
    constexpr float delta_per_bucket = discrete_action_max / half_buckets;

    Vector3 cur_pos = ctx.getUnsafe<Position>(sim_e.e);
    Quat cur_rot = ctx.getUnsafe<Rotation>(sim_e.e);

    float f_x = delta_per_bucket * action.x;
    float f_y = delta_per_bucket * action.y;
    float t_z = delta_per_bucket * action.r;

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
                               render::ViewID &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = fminf(vel.linear.z, 0);

    vel.angular = Vector3::zero();
}

inline void collectObservationsSystem(Engine &ctx,
                                      SimEntity sim_e,
                                      PositionObservation &pos_obs,
                                      VelocityObservation &vel_obs)
{
    if (sim_e.e == Entity::none()) {
        return;
    }

    Vector3 pos = ctx.getUnsafe<Position>(sim_e.e);
    Vector3 linear_vel = ctx.getUnsafe<Velocity>(sim_e.e).linear;

    pos_obs.x = pos;
    vel_obs.v = linear_vel;
}

inline void computeVisibilitySystem(Engine &ctx,
                                    Entity agent_e,
                                    SimEntity sim_e,
                                    AgentType agent_type,
                                    PositionObservation pos,
                                    VisibilityMasks &vis)
{
    if (sim_e.e == Entity::none()) {
        return;
    }

    Quat agent_rot = ctx.getUnsafe<Rotation>(sim_e.e);
    Vector3 fwd = agent_rot.rotateVec(math::fwd);
    const float cos_angle_threshold = cosf(helpers::toRadians(135.f / 2.f));

    auto &bvh = ctx.getSingleton<broadphase::BVH>();

    auto checkVisibility = [&](Entity other_e, Entity other_sim_e) {
        Vector3 other_pos = ctx.getUnsafe<PositionObservation>(other_e).x;

        Vector3 to_other = other_pos - pos.x;

        Vector3 to_other_norm = to_other.normalize();

        float cos_angle = dot(to_other_norm, fwd);

        if (cos_angle < cos_angle_threshold) {
            return 0.f;
        }

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(pos.x, to_other, &hit_t, &hit_normal, 1.f);

        return hit_entity == other_sim_e ? 1.f : 0.f;
    };

    CountT box_idx;
    for (box_idx = 0; box_idx  < consts::maxBoxes; box_idx++) {
        Entity box_e = ctx.data().boxObservations[box_idx];
        Entity box_sim_e = ctx.getUnsafe<SimEntity>(box_e).e;
        if (box_sim_e == Entity::none()) {
            break;
        }
        vis.visible[box_idx] = checkVisibility(box_e, box_sim_e);
    }

    for (; box_idx < consts::maxBoxes; box_idx++) {
        vis.visible[box_idx] = 0.f;
    }

    CountT ramp_idx;
    for (ramp_idx = 0; ramp_idx < consts::maxRamps; ramp_idx++) {
        Entity ramp_e = ctx.data().rampObservations[ramp_idx];
        Entity ramp_sim_e = ctx.getUnsafe<SimEntity>(ramp_e).e;
        if (ramp_sim_e == Entity::none()) {
            break;
        }

        vis.visible[consts::maxBoxes + ramp_idx] =
            checkVisibility(ramp_e, ramp_sim_e);
    }

    for (; ramp_idx < consts::maxRamps; ramp_idx++) {
        vis.visible[consts::maxBoxes + ramp_idx] = 0.f;
    }

    CountT agent_idx;
    CountT num_other_agents = 0;
    for (agent_idx = 0; agent_idx < consts::maxAgents; agent_idx++) {
        Entity other_agent_e = ctx.data().agentObservations[agent_idx];
        if (agent_e == other_agent_e) {
            continue;
        }

        Entity other_agent_sim_e = ctx.getUnsafe<SimEntity>(other_agent_e).e;
        if (other_agent_sim_e == Entity::none()) {
            break;
        }

        bool is_visible =
            checkVisibility(other_agent_e, other_agent_sim_e);

        if (agent_type == AgentType::Seeker && is_visible) {
            AgentType other_type = ctx.getUnsafe<AgentType>(other_agent_e);
            if (other_type == AgentType::Hider) {
                ctx.data().hiderTeamReward.store(-1.f,
                    std::memory_order_relaxed);
            }
        }

        vis.visible[consts::maxBoxes + consts::maxRamps + num_other_agents] =
            is_visible;

        num_other_agents++;
    }

    for (; num_other_agents < consts::maxAgents - 1; num_other_agents++) {
        vis.visible[consts::maxBoxes + consts::maxRamps + num_other_agents] =
            0.f;
    }
}

inline void agentRewardsSystem(Engine &ctx,
                               SimEntity sim_e,
                               const PositionObservation &pos,
                               AgentType agent_type,
                               Reward &reward)
{
    if (sim_e.e == Entity::none()) {
        return;
    }

    if (ctx.data().curEpisodeStep < 96) {
        reward.reward = 0.f;
        return;
    }

    float reward_val = ctx.data().hiderTeamReward.load(
        std::memory_order_relaxed);
    if (agent_type == AgentType::Seeker) {
        reward_val *= -1.f;
    }

    if (fabsf(pos.x.x) >= 18.f || fabsf(pos.x.y) >= 18.f) {
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

    // FIXME: these 3 really shouldn't need to be sorted. They only need
    // to be sorted after initialization (purely static afterwards)
    auto sort_agent_iface = queueSortByWorld<AgentInterface>(builder, {sort_objects});
    auto sort_box_obs = queueSortByWorld<BoxObservation>(builder, {sort_agent_iface});
    auto sort_ramp_obs = queueSortByWorld<BoxObservation>(builder, {sort_box_obs});

    auto prep_finish = sort_ramp_obs;

#if 0
    prep_finish = builder.addToGraph<ParallelForNode<Engine,
        sortDebugSystem, WorldReset>>({prep_finish});
#endif

    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
        Action, SimEntity, AgentType>>({prep_finish});

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

    auto collect_observations = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            SimEntity,
            PositionObservation,
            VelocityObservation>>({sim_done});

    auto compute_visibility = builder.addToGraph<ParallelForNode<Engine,
        computeVisibilitySystem,
            Entity,
            SimEntity,
            AgentType,
            PositionObservation,
            VisibilityMasks>>({collect_observations});

    auto agent_rewards = builder.addToGraph<ParallelForNode<Engine,
        agentRewardsSystem, SimEntity, PositionObservation, AgentType, Reward>>(
            {compute_visibility});

    (void)phys_cleanup_sys;
    (void)renderer_sys;
    (void)recycle_sys;
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
         numPhysicsSubsteps, -9.8 * math::up, max_total_entities, 100 * 50);

    render::RenderingSystem::init(ctx);

    obstacles =
        (Entity *)rawAlloc(sizeof(Entity) * size_t(max_total_entities));

    numObstacles = 0;
    minEpisodeEntities = init.minEntitiesPerWorld;
    maxEpisodeEntities = init.maxEntitiesPerWorld;

    numHiders = 0;
    numSeekers = 0;

    for (int32_t i = 0; i < consts::maxBoxes; i++) {
        boxObservations[i] = ctx.makeEntityNow<BoxObservation>();
    }

    for (int32_t i = 0; i < consts::maxRamps; i++) {
        rampObservations[i] = ctx.makeEntityNow<RampObservation>();
    }

    for (int32_t i = 0; i < consts::maxAgents; i++) {
        agentObservations[i] = ctx.makeEntityNow<AgentInterface>();
    }

    resetWorld(ctx, 1);
    ctx.getSingleton<WorldReset>().resetLevel = 0;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, WorldInit);

}
