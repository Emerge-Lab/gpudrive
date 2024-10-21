#include "level_gen.hpp"
#include "dynamics.hpp"
#include "init.hpp"

namespace gpudrive {
using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

// Register the entity with the broadphase system
// This is needed for every entity with all the physics components.
// Not registering an entity will cause a crash because the broadphase
// systems will still execute over entities with the physics components.
static void registerRigidBodyEntity(
    Engine &ctx,
    Entity e,
    SimObject sim_obj)
{
    ObjectID obj_id { (int32_t)sim_obj };
    ctx.get<broadphase::LeafID>(e) = PhysicsSystem::registerEntity(ctx, e, obj_id);
}

static inline void resetAgentInterface(Engine &ctx, Entity agent_iface, EntityType type, ResponseType resp_type, int32_t steps_remaining= consts::episodeLen, int32_t done = 0) {
    ctx.get<StepsRemaining>(agent_iface).t = steps_remaining;
    ctx.get<Done>(agent_iface).v = done;
    ctx.get<Reward>(agent_iface).v = 0;
    ctx.get<Info>(agent_iface) = Info{};
    ctx.get<Info>(agent_iface).type = (int32_t)type;
    ctx.get<ResponseType>(agent_iface) = resp_type;
}

static inline void resetAgent(Engine &ctx, Entity agent) {
    auto agent_iface = ctx.get<AgentInterfaceEntity>(agent).e;
    auto xCoord = ctx.get<Trajectory>(agent_iface).positions[0].x;
    auto yCoord = ctx.get<Trajectory>(agent_iface).positions[0].y;
    auto xVelocity = ctx.get<Trajectory>(agent_iface).velocities[0].x;
    auto yVelocity = ctx.get<Trajectory>(agent_iface).velocities[0].y;
    auto heading = ctx.get<Trajectory>(agent_iface).headings[0];

    ctx.get<Position>(agent) = Vector3{.x = xCoord, .y = yCoord, .z = 1};
    ctx.get<Rotation>(agent) = Quat::angleAxis(heading, madrona::math::up);
    if (ctx.get<ResponseType>(agent) == ResponseType::Static) {
        ctx.get<Velocity>(agent) = Velocity{Vector3::zero(), Vector3::zero()};
    } else {
        ctx.get<Velocity>(agent) = Velocity{Vector3{.x = xVelocity, .y = yVelocity, .z = 0}, Vector3::zero()};
    }
    ctx.get<Action>(agent_iface) = getZeroAction(ctx.data().params.dynamicsModel);
    
    resetAgentInterface(ctx, agent_iface, ctx.get<EntityType>(agent), ctx.get<ResponseType>(agent));

#ifndef GPUDRIVE_DISABLE_NARROW_PHASE
    ctx.get<CollisionDetectionEvent>(agent).hasCollided.store_release(0);
#endif
}

static inline void populateExpertTrajectory(Engine &ctx, const Entity &agent, const MapObject &agentInit) {
    const auto &agent_iface = ctx.get<AgentInterfaceEntity>(agent).e;
    auto &trajectory = ctx.get<Trajectory>(agent_iface);
    for(CountT i = 0; i < agentInit.numPositions; i++)
    {
        trajectory.positions[i] = Vector2{.x = agentInit.position[i].x - ctx.data().mean.x, .y = agentInit.position[i].y - ctx.data().mean.y};
        trajectory.velocities[i] = Vector2{.x = agentInit.velocity[i].x, .y = agentInit.velocity[i].y};
        trajectory.headings[i] = toRadians(agentInit.heading[i]);
        trajectory.valids[i] = (float)agentInit.valid[i];
        trajectory.inverseActions[i] = getZeroAction(ctx.data().params.dynamicsModel);
    }
    if (ctx.data().params.dynamicsModel == DynamicsModel::Classic || ctx.data().params.dynamicsModel == DynamicsModel::State){
        return;
    }
    for(CountT i = agentInit.numPositions - 2; i >=0; i--)
    {
        if(!trajectory.valids[i] || !trajectory.valids[i+1])
        {
            trajectory.inverseActions[i] = getZeroAction(ctx.data().params.dynamicsModel);
        }

        Rotation rot = Quat::angleAxis(trajectory.headings[i], madrona::math::up);
        Position pos = Vector3{.x = trajectory.positions[i].x, .y = trajectory.positions[i].y, .z = 1};
        Velocity vel = {Vector3{.x = trajectory.velocities[i].x, .y = trajectory.velocities[i].y, .z = 0}, Vector3::zero()};
        Rotation targetRot = Quat::angleAxis(trajectory.headings[i+1], madrona::math::up);
        switch (ctx.data().params.dynamicsModel) {
            case DynamicsModel::Classic:
            case DynamicsModel::State:
                // No inverse action model for classic model
                break;

            case DynamicsModel::InvertibleBicycle: {
                Velocity targetVel = {Vector3{.x = trajectory.velocities[i+1].x, .y = trajectory.velocities[i+1].y, .z = 0}, Vector3::zero()};
                trajectory.inverseActions[i] = inverseBicycleModel(rot, vel, targetRot, targetVel);
                break;
            }

            case DynamicsModel::DeltaLocal: {
                Position targetPos = Vector3{.x = trajectory.positions[i+1].x, .y = trajectory.positions[i+1].y, .z = 1};
                trajectory.inverseActions[i] = inverseDeltaModel(rot, pos, targetRot, targetPos);
                break;
            }
        }
    }
}

static inline bool isAgentStatic(Engine &ctx, Entity agent, bool markAsStatic = false) {
    auto agent_iface = ctx.get<AgentInterfaceEntity>(agent).e;
    bool isStatic = (ctx.get<Goal>(agent).position - ctx.get<Trajectory>(agent_iface).positions[0]).length() < consts::staticThreshold or markAsStatic;
    return !ctx.data().params.isStaticAgentControlled and isStatic;
}

static inline bool isAgentControllable(Engine &ctx, Entity agent) {
    auto agent_iface = ctx.get<AgentInterfaceEntity>(agent).e;
    return ctx.data().numControlledAgents < ctx.data().params.maxNumControlledAgents &&
           ctx.get<Trajectory>(agent_iface).valids[0] &&
           ctx.get<ResponseType>(agent) == ResponseType::Dynamic;
}

static inline Entity createAgent(Engine &ctx, const MapObject &agentInit) {
    assert(agentInit.type >= EntityType::Vehicle && agentInit.type <= EntityType::Cyclist);

    // The following components do not vary within an episode and so need only
    // be set once
    auto agent = ctx.makeRenderableEntity<Agent>();
    auto agent_iface = ctx.get<AgentInterfaceEntity>(agent).e = ctx.makeEntity<AgentInterface>();

    ctx.get<VehicleSize>(agent) = {.length = agentInit.length, .width = agentInit.width};
    ctx.get<Scale>(agent) = Diag3x3{.d0 = agentInit.length/2, .d1 = agentInit.width/2, .d2 = 1};
    ctx.get<Scale>(agent) *= consts::vehicleLengthScale;
    ctx.get<ObjectID>(agent) = ObjectID{(int32_t)SimObject::Agent};
    ctx.get<EntityType>(agent) = agentInit.type;
    ctx.get<Goal>(agent)= Goal{.position = Vector2{.x = agentInit.goalPosition.x - ctx.data().mean.x, .y = agentInit.goalPosition.y - ctx.data().mean.y}};

    populateExpertTrajectory(ctx, agent, agentInit);

    //Applying custom rules
    ctx.get<ResponseType>(agent) = isAgentStatic(ctx, agent, agentInit.markAsStatic) ? ResponseType::Static : ResponseType::Dynamic;
    ctx.get<ControlledState>(agent_iface) = ControlledState{.controlled = isAgentControllable(ctx, agent)};
    ctx.data().numControlledAgents += ctx.get<ControlledState>(agent_iface).controlled;

    if (ctx.data().enableRender) {
        render::RenderingSystem::attachEntityToView(ctx,
                agent,
                90.f, 0.001f,
                1.5f * math::up);
    }

    return agent;
}

static Entity makeRoadEdge(Engine &ctx, const MapVector2 &p1,
                           const MapVector2 &p2, const EntityType &type) {                    
    float z = 1 + (type == EntityType::RoadEdge ? consts::lidarRoadEdgeOffset : consts::lidarRoadLineOffset);

    Vector3 start{.x = p1.x - ctx.data().mean.x, .y = p1.y - ctx.data().mean.y, .z = z};
    Vector3 end{.x = p2.x - ctx.data().mean.x, .y = p2.y - ctx.data().mean.y, .z = z};

    auto road_edge = ctx.makeRenderableEntity<PhysicsEntity>();
    ctx.get<RoadInterfaceEntity>(road_edge).e = ctx.makeEntity<RoadInterface>();

    auto pos = Vector3{.x = (start.x + end.x)/2, .y = (start.y + end.y)/2, .z = z};
    auto rot = Quat::angleAxis(atan2(end.y - start.y, end.x - start.x), madrona::math::up);
    auto scale = Diag3x3{.d0 = start.distance(end)/2, .d1 = 0.1, .d2 = 0.1};
    setRoadEntitiesProps(ctx, road_edge, pos, rot, scale, type, ObjectID{(int32_t)SimObject::Cube}, ResponseType::Static);
    registerRigidBodyEntity(ctx, road_edge, SimObject::Cube);
    return road_edge;
}

float calculateDistance(float x1, float y1, float x2, float y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

static Entity makeCube(Engine &ctx, const MapVector2 &p1, const MapVector2 &p2, const MapVector2 &p3,
                            const MapVector2 &p4, const EntityType &type) {
    MapVector2 points[] = {p1, p2, p3, p4};

    // Calculate distances between consecutive points
    float lengths[4];
    for (int i = 0; i < 4; ++i)
    {
        MapVector2 &p_start = points[i];
        MapVector2 &p_end = points[(i + 1) % 4]; // Wrap around to the first point
        lengths[i] = calculateDistance(p_start.x, p_start.y, p_end.x, p_end.y);
    }

    int maxLength_i = 0;
    int minLength_i = 0;
    for (int i = 1; i < 4; ++i) {
        if (lengths[i] > lengths[maxLength_i])
            maxLength_i = i;
        if (lengths[i] < lengths[minLength_i])
            minLength_i = i;
    }

    MapVector2 &start = points[maxLength_i];
    MapVector2 &end = points[(maxLength_i + 1) % 4];

    // Calculate rotation angle (assuming longer side is used to calculate angle)
    float angle = atan2(end.y - start.y, end.x - start.x);

    auto speed_bump = ctx.makeRenderableEntity<PhysicsEntity>();
    ctx.get<RoadInterfaceEntity>(speed_bump).e = ctx.makeEntity<RoadInterface>();

    auto pos = Vector3{.x = (p1.x + p2.x + p3.x + p4.x)/4 - ctx.data().mean.x, .y = (p1.y + p2.y + p3.y + p4.y)/4 - ctx.data().mean.y, .z = 1 + consts::lidarRoadLineOffset};
    auto rot = Quat::angleAxis(angle, madrona::math::up);
    auto scale = Diag3x3{.d0 = lengths[maxLength_i]/2, .d1 = lengths[minLength_i]/2, .d2 = 0.1};
    setRoadEntitiesProps(ctx, speed_bump, pos, rot, scale, type, ObjectID{(int32_t)SimObject::SpeedBump}, ResponseType::Static);
    registerRigidBodyEntity(ctx, speed_bump, SimObject::SpeedBump);
    return speed_bump;
}

static Entity makeStopSign(Engine &ctx, const MapVector2 &p1) {
    float x1 = p1.x;
    float y1 = p1.y;

    auto stop_sign = ctx.makeRenderableEntity<PhysicsEntity>();
    ctx.get<RoadInterfaceEntity>(stop_sign).e = ctx.makeEntity<RoadInterface>();
    
    auto pos = Vector3{.x = x1 - ctx.data().mean.x, .y = y1 - ctx.data().mean.y, .z = 1};
    auto rot = Quat::angleAxis(0, madrona::math::up);
    auto scale = Diag3x3{.d0 = 0.2, .d1 = 0.2, .d2 = 1};
    setRoadEntitiesProps(ctx, stop_sign, pos, rot, scale, EntityType::StopSign, ObjectID{(int32_t)SimObject::StopSign}, ResponseType::Static);
    registerRigidBodyEntity(ctx, stop_sign, SimObject::StopSign);
    return stop_sign;
}

static inline void createRoadEntities(Engine &ctx, const MapRoad &roadInit, CountT &idx) {
    if (idx >= consts::kMaxRoadEntityCount)
        return;
    switch (roadInit.type)
    {
        case EntityType::RoadEdge:
        case EntityType::RoadLine:
        case EntityType::RoadLane:
        {
            size_t numPoints = roadInit.numPoints;
            for (size_t j = 1; j <= numPoints - 1; j++)
            {
                auto road = ctx.data().roads[idx] = makeRoadEdge(ctx, roadInit.geometry[j - 1], roadInit.geometry[j], roadInit.type);
                ctx.data().road_ifaces[idx++] = ctx.get<RoadInterfaceEntity>(road).e;
                if (idx >= consts::kMaxRoadEntityCount) return;
            }
            break;
        }
        case EntityType::CrossWalk:
        case EntityType::SpeedBump:
        {
            assert(roadInit.numPoints >= 4);
            // TODO: Speed Bump are not guranteed to have 4 points. Need to handle this case.
            auto road = ctx.data().roads[idx] = makeCube(ctx, roadInit.geometry[0], roadInit.geometry[1], roadInit.geometry[2], roadInit.geometry[3], roadInit.type);
            ctx.data().road_ifaces[idx++] = ctx.get<RoadInterfaceEntity>(road).e;
            break;
        }
        case EntityType::StopSign:
        {
            assert(roadInit.numPoints >= 1);
            // TODO: Stop Sign are not guranteed to have 1 point. Need to handle this case.
            auto road = ctx.data().roads[idx] = makeStopSign(ctx, roadInit.geometry[0]);
            ctx.data().road_ifaces[idx++] = ctx.get<RoadInterfaceEntity>(road).e;
            break;
        }
        default:
            return;
    }
}

static void createFloorPlane(Engine &ctx)
{
    ctx.data().floorPlane = ctx.makeRenderableEntity<PhysicsEntity>();
    setRoadEntitiesProps(ctx, ctx.data().floorPlane, Vector3{.x = 0, .y = 0, .z = 0},
                         Quat::angleAxis(0, madrona::math::up),
                         Diag3x3{.d0 = 100, .d1 = 100, .d2 = 0.1},
                         EntityType::None, ObjectID{(int32_t)SimObject::Plane}, ResponseType::Static);
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);
}

void createPaddingEntities(Engine &ctx) {
    for (CountT agentIdx = ctx.data().numAgents;
         agentIdx < consts::kMaxAgentCount; ++agentIdx) {
        Entity &agent_iface = ctx.data().agent_ifaces[agentIdx] = ctx.makeEntity<AgentInterface>();
        ctx.get<AgentID>(agent_iface) = AgentID{.id = -1};
        resetAgentInterface(ctx, agent_iface, EntityType::None, ResponseType::Static, 0, 1);
        ctx.get<ControlledState>(agent_iface) = ControlledState{.controlled = 0};
        auto &agent_map_obs = ctx.get<AgentMapObservations>(agent_iface);
        for (CountT i = 0; i < consts::kMaxAgentMapObservationsCount; i++) {
            agent_map_obs.obs[i] = MapObservation::zero();
        }
        auto &self_obs = ctx.get<SelfObservation>(agent_iface);
        self_obs = SelfObservation::zero();

        auto &partner_obs = ctx.get<PartnerObservations>(agent_iface);
        for (CountT i = 0; i < consts::kMaxAgentCount-1; i++) {
            partner_obs.obs[i] = PartnerObservation::zero();
        }

    }

    for (CountT roadIdx = ctx.data().numRoads;
         roadIdx < consts::kMaxRoadEntityCount; ++roadIdx) {
        Entity &e = ctx.data().road_ifaces[roadIdx] = ctx.makeEntity<RoadInterface>();
        ctx.get<MapObservation>(e) = MapObservation::zero();
    }
}

void createCameraEntity(Engine &ctx)
{
    auto camera = ctx.makeRenderableEntity<CameraAgent>();
    ctx.get<Position>(camera) = Vector3{.x = 0, .y = 0, .z = 20};
    ctx.get<Rotation>(camera) = (math::Quat::angleAxis(0, math::up) *
            math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();

    render::RenderingSystem::attachEntityToView(ctx,
        camera,
        150.f, 0.001f,
        1.5f * math::up);

    ctx.data().camera_agent = camera;
}

static inline bool shouldAgentBeCreated(Engine &ctx, const MapObject &agentInit)
{
    if (ctx.data().params.IgnoreNonVehicles &&
        (agentInit.type == EntityType::Pedestrian || agentInit.type == EntityType::Cyclist))
    {
        return false;
    }
    if (ctx.data().params.initOnlyValidAgentsAtFirstStep && !agentInit.valid[0])
    {
        return false;
    }

    return true;
}

void createPersistentEntities(Engine &ctx) {
    // createFloorPlane(ctx);

    const auto& map = ctx.singleton<Map>();

    if (ctx.data().enableRender)
    {
        createCameraEntity(ctx);
    }

    ctx.data().mean = {0, 0};
    ctx.data().mean.x = map.mean.x;
    ctx.data().mean.y = map.mean.y;
    ctx.data().numControlledAgents = 0;
    ctx.singleton<ResetMap>().reset = 0;

    CountT agentIdx = 0;
    for (CountT agentCtr = 0; agentCtr < map.numObjects && agentIdx < consts::kMaxAgentCount; ++agentCtr) {
        const auto &agentInit = map.objects[agentCtr];

        if (not shouldAgentBeCreated(ctx, agentInit))
        {
            continue;
        }

        auto agent = createAgent(ctx, agentInit);
        ctx.data().agent_ifaces[agentIdx] = ctx.get<AgentInterfaceEntity>(agent).e;
        ctx.get<AgentID>(ctx.data().agent_ifaces[agentIdx]) = AgentID{.id = static_cast<int32_t>(agentIdx)};
        ctx.data().agents[agentIdx++] = agent;
    }
    ctx.data().numAgents = agentIdx;

    CountT roadIdx = 0;
    for(CountT roadCtr = 0; roadCtr < map.numRoads && roadIdx < consts::kMaxRoadEntityCount; roadCtr++)
    {
        const auto &roadInit = map.roads[roadCtr];
        createRoadEntities(ctx, roadInit, roadIdx);
    }
    ctx.data().numRoads = roadIdx;

    auto &shape = ctx.singleton<Shape>();
    shape.agentEntityCount = ctx.data().numAgents;
    shape.roadEntityCount = ctx.data().numRoads;

    createPaddingEntities(ctx);

    for (CountT i = 0; i < ctx.data().numAgents; i++) {
        Entity cur_agent = ctx.data().agents[i];
        OtherAgents &other_agents = ctx.get<OtherAgents>(cur_agent);
        CountT out_idx = 0;
        for (CountT j = 0; j < ctx.data().numAgents; j++)
        {
            if (i == j)
            {
                continue;
            }

            Entity other_agent = ctx.data().agents[j];
            other_agents.e[out_idx++] = other_agent;
        }
    }
}

static void resetPersistentEntities(Engine &ctx)
{
    for (CountT idx = 0; idx < ctx.data().numAgents; ++idx)
    {
        Entity agent = ctx.data().agents[idx];
        resetAgent(ctx, agent);
        registerRigidBodyEntity(ctx, agent, SimObject::Agent);
    }

    for (CountT idx = 0; idx < ctx.data().numRoads; idx++)
    {
      Entity road = ctx.data().roads[idx];
      if(road == Entity::none()) break;
      SimObject simObjType = static_cast<SimObject>(ctx.get<ObjectID>(road).idx);
      registerRigidBodyEntity(ctx, road, simObjType);
    }
}

void destroyWorld(Engine &ctx)
{
    for (CountT idx = 0; idx < ctx.data().numAgents; ++idx)
    {
        Entity agent = ctx.data().agents[idx];
        ctx.destroyRenderableEntity(agent);
    }
    for (CountT idx = 0; idx < ctx.data().numRoads; idx++)
    {
        Entity road = ctx.data().roads[idx];
        ctx.destroyRenderableEntity(road);
    }
    if (ctx.data().enableRender)
    {
        ctx.destroyRenderableEntity(ctx.data().camera_agent);
    }
    for (CountT idx = 0; idx < consts::kMaxAgentCount; ++idx)
    {
        Entity agent_iface = ctx.data().agent_ifaces[idx];
        ctx.destroyEntity(agent_iface);
    }
    for (CountT idx = 0; idx < consts::kMaxRoadEntityCount; ++idx)
    {
        Entity road_iface = ctx.data().road_ifaces[idx];
        ctx.destroyEntity(road_iface);
    }
    ctx.data().numAgents = 0;
    ctx.data().numRoads = 0;
    ctx.data().numControlledAgents = 0;
    ctx.data().mean = {0, 0};
}


void resetWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
}

}
