#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "level_gen.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace GPUHideSeek {

constexpr inline float deltaT = 0.075;
constexpr inline CountT numPhysicsSubsteps = 4;

void Sim::registerTypes(ECSRegistry &registry, const Config &)
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
    registry.registerComponent<Lidar>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<WorldDone>();
    registry.registerSingleton<PrepPhaseCounter>();
    registry.registerSingleton<GlobalDebugPositions>();

    registry.registerArchetype<DynamicObject>();
    registry.registerFixedSizeArchetype<AgentInterface>(consts::maxAgents);
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
    registry.exportColumn<AgentInterface, Lidar>(14);
    registry.exportSingleton<GlobalDebugPositions>(13);
}

static inline void resetEnvironment(Engine &ctx)
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
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t level = reset.resetLevel;

    if (level != 0) {
        resetEnvironment(ctx);

        reset.resetLevel = 0;

        int32_t num_hiders = reset.numHiders;
        int32_t num_seekers = reset.numSeekers;

        generateEnvironment(ctx, level, num_hiders, num_seekers);
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

                    joint_constraint.e1 = ctx.getLoc(sim_e.e);
                    joint_constraint.e2 = ctx.getLoc(grab_entity);

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

#ifdef MADRONA_GPU_MODE
    constexpr int32_t num_total_vis =
        consts::maxBoxes + consts::maxRamps + consts::maxAgents;
    const int32_t lane_id = threadIdx.x % 32;
    for (int32_t global_offset = 0; global_offset < num_total_vis;
         global_offset += 32) {
        int32_t cur_idx = global_offset + lane_id;

        Entity check_e = Entity::none();
        float *vis_out = nullptr;

        bool checking_agent = cur_idx < consts::maxAgents;
        uint32_t agent_mask = __ballot_sync(mwGPU::allActive, checking_agent);
        bool seeker_checking_hider = false;
        if (checking_agent) {
            bool valid_check = true;
            if (cur_idx < ctx.data().numActiveAgents) {
                Entity other_agent_e = ctx.data().agentInterfaces[cur_idx];
                valid_check = other_agent_e != agent_e;

                if (valid_check) {
                    check_e = ctx.getUnsafe<SimEntity>(other_agent_e).e;
                    seeker_checking_hider = 
                        agent_type == AgentType::Seeker &&
                        ctx.getUnsafe<AgentType>(other_agent_e) ==
                            AgentType::Hider;
                }
            }

            uint32_t valid_mask = __ballot_sync(agent_mask, valid_check);
            valid_mask <<= (32 - lane_id);
            uint32_t num_lower_valid = __popc(valid_mask);

            if (valid_check) {
                vis_out = &agent_vis.visible[num_lower_valid];
            }
        } else if (int32_t box_idx = cur_idx - consts::maxAgents;
                   box_idx < consts::maxBoxes) {
            if (cur_idx < ctx.data().numActiveBoxes) {
                check_e = ctx.data().boxes[cur_idx];
            }
            vis_out = &box_vis.visible[cur_idx];
        } else if (int32_t ramp_idx =
                       cur_idx - consts::maxAgents - consts::maxBoxes;
                   ramp_idx < consts::maxRamps) {
            if (ramp_idx < ctx.data().numActiveRamps) {
                check_e = ctx.data().ramps[ramp_idx];
            }
            vis_out = &ramp_vis.visible[ramp_idx];
        } 

        if (check_e == Entity::none()) {
           if (vis_out != nullptr) {
               *vis_out = 0.f;
           }
        } else {
            bool is_visible = checkVisibility(check_e);
            *vis_out = is_visible ? 1.f : 0.f;

            if (seeker_checking_hider && is_visible) {
                ctx.data().hiderTeamReward.store(-1.f,
                    std::memory_order_relaxed);
            }
        }
    }
#else
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
#endif
}

inline void lidarSystem(Engine &ctx,
                        SimEntity sim_e,
                        Lidar &lidar)
{
    if (sim_e.e == Entity::none()) {
        return;
    }

    Vector3 pos = ctx.getUnsafe<Position>(sim_e.e);
    Quat rot = ctx.getUnsafe<Rotation>(sim_e.e);
    auto &bvh = ctx.getSingleton<broadphase::BVH>();

    Vector3 fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);

    auto traceRay = [&](int32_t idx) {
        float theta = 2.f * math::pi * (float(idx) / float(30));
        float x = cosf(theta);
        float y = sinf(theta);

        Vector3 ray_dir = (x * right + y * fwd).normalize();

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(pos , ray_dir, &hit_t, &hit_normal, 200.f);

        if (hit_entity == Entity::none()) {
            lidar.depth[idx] = 0.f;
        } else {
            lidar.depth[idx] = hit_t;
        }
    };


#ifdef MADRONA_GPU_MODE
    int32_t idx = threadIdx.x % 32;

    if (idx < 30) {
        traceRay(idx);
    }
#else
    for (int32_t i = 0; i < 30; i++) {
        traceRay(i);
    }
#endif
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

inline void globalPositionsDebugSystem(Engine &ctx,
                                       GlobalDebugPositions &global_positions)
{
    auto getXY = [](Vector3 v) {
        return Vector2 {
            v.x,
            v.y,
        };
    };

    for (CountT i = 0; i < consts::maxBoxes; i++) {
        if (i >= ctx.data().numActiveBoxes) {
            global_positions.boxPositions[i] = Vector2 {0, 0};
            continue;
        }

        global_positions.boxPositions[i] =
            getXY(ctx.getUnsafe<Position>(ctx.data().boxes[i]));
    }

    for (CountT i = 0; i < consts::maxRamps; i++) {
        if (i >= ctx.data().numActiveRamps) {
            global_positions.rampPositions[i] = Vector2 {0, 0};
            continue;
        }

        global_positions.rampPositions[i] =
            getXY(ctx.getUnsafe<Position>(ctx.data().ramps[i]));
    }

    {
        CountT out_offset = 0;
        for (CountT i = 0; i < ctx.data().numHiders; i++) {
            global_positions.agentPositions[out_offset++] = 
                getXY(ctx.getUnsafe<Position>(ctx.data().hiders[i]));
        }

        for (CountT i = 0; i < ctx.data().numSeekers; i++) {
            global_positions.agentPositions[out_offset++] = 
                getXY(ctx.getUnsafe<Position>(ctx.data().seekers[i]));
        }

        for (; out_offset < consts::maxAgents; out_offset++) {
            global_positions.agentPositions[out_offset++] = Vector2 {0, 0};
        }
    }
}

#ifdef MADRONA_GPU_MODE
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
#endif

void Sim::setupTasks(TaskGraph::Builder &builder, const Config &cfg)
{
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem, WorldReset>>({});

    auto clearTmp = builder.addToGraph<ResetTmpAllocNode>({reset_sys});

#ifdef MADRONA_GPU_MODE
    // FIXME: these 3 need to be compacted, but sorting is unnecessary
    auto sort_cam_agent = queueSortByWorld<CameraAgent>(builder, {clearTmp});
    auto sort_dyn_agent = queueSortByWorld<DynAgent>(builder, {sort_cam_agent});
    auto sort_objects = queueSortByWorld<DynamicObject>(builder, {sort_dyn_agent});

    // FIXME: These don't need to be sorted after initialization (purely static)
    auto sort_agent_iface =
        queueSortByWorld<AgentInterface>(builder, {sort_objects});
    auto prep_finish = sort_agent_iface;
#else
    auto prep_finish = clearTmp;
#endif

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

    if (cfg.enableRender) {
        render::RenderingSystem::setupTasks(builder,
            {sim_done});
    }

#ifdef MADRONA_GPU_MODE
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({sim_done});
    (void)recycle_sys;
#endif

    auto collect_observations = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Entity,
            SimEntity,
            AgentType,
            RelativeAgentObservations,
            RelativeBoxObservations,
            RelativeRampObservations
        >>({sim_done});


#ifdef MADRONA_GPU_MODE
    auto compute_visibility = builder.addToGraph<CustomParallelForNode<Engine,
        computeVisibilitySystem, 32, 1,
#else
    auto compute_visibility = builder.addToGraph<ParallelForNode<Engine,
        computeVisibilitySystem,
#endif
            Entity,
            SimEntity,
            AgentType,
            AgentVisibilityMasks,
            BoxVisibilityMasks,
            RampVisibilityMasks
        >>({sim_done});

#ifdef MADRONA_GPU_MODE
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 32, 1,
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine,
        lidarSystem,
#endif
            SimEntity,
            Lidar
        >>({sim_done});

    auto agent_rewards = builder.addToGraph<ParallelForNode<Engine,
        agentRewardsSystem,
            SimEntity,
            AgentType,
            Reward
        >>({compute_visibility});

    auto global_positions_debug = builder.addToGraph<ParallelForNode<Engine,
        globalPositionsDebugSystem,
            GlobalDebugPositions
        >>({sim_done});

    (void)lidar;
    (void)phys_cleanup_sys;
    (void)collect_observations;
    (void)agent_rewards;
    (void)global_positions_debug;
}

Sim::Sim(Engine &ctx,
         const Config &,
         const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr)
{
    CountT max_total_entities =
        std::max(init.maxEntitiesPerWorld, uint32_t(3 + 3 + 9 + 2 + 6)) + 100;

    phys::RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT,
         numPhysicsSubsteps, -9.8 * math::up, max_total_entities,
         50 * 20, 10);


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

    resetEnvironment(ctx);
    generateEnvironment(ctx, 1, 3, 3);
    ctx.getSingleton<WorldReset>().resetLevel = 0;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Config, WorldInit);

}
