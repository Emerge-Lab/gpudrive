#include "level_gen.hpp"

namespace GPUHideSeek {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace consts {
inline constexpr float doorWidth = consts::worldWidth / 3.f;
}

static inline void setupRigidBodyEntity(
    Engine &ctx,
    Entity e,
    Vector3 pos,
    Quat rot,
    SimObject sim_obj,
    ResponseType response_type = ResponseType::Dynamic,
    Diag3x3 scale = {1, 1, 1})
{
    ObjectID obj_id { (int32_t)sim_obj };

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
}

static void registerRigidBodyEntity(
    Engine &ctx,
    Entity e,
    SimObject sim_obj)
{
    ObjectID obj_id { (int32_t)sim_obj };
    ctx.get<broadphase::LeafID>(e) =
        RigidBodyPhysicsSystem::registerEntity(ctx, e, obj_id);
}

#if 0
static Entity makeButtonEntity(Engine &ctx, Vector2 pos, Vector2 scale)
{
    Entity e = ctx.makeEntity<ButtonEntity>();
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
#endif

// Creates floor, outer walls, and agent entities.
// All these entities persist across all episodes.
void createPersistentEntities(Engine &ctx)
{
    // Create the floor entity, just a simple static plane.
    ctx.data().floorPlane = ctx.makeEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().floorPlane,
        Vector3 { 0, 0, 0 },
        Quat { 1, 0, 0, 0 },
        SimObject::Plane,
        ResponseType::Static);

    // Create the outer wall entities
    // Left

    ctx.data().borders[0] = ctx.makeEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[0],
        Vector3 {
            0,
            -consts::wallWidth / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::worldWidth + consts::wallWidth * 2,
            consts::wallWidth,
            2.f,
        });

    // Top
    ctx.data().borders[1] = ctx.makeEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[1],
        Vector3 {
            consts::worldWidth / 2.f + consts::wallWidth / 2.f,
            consts::worldLength / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            consts::worldLength,
            2.f,
        });

    // Bottom
    ctx.data().borders[2] = ctx.makeEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[2],
        Vector3 {
            -consts::worldWidth / 2.f - consts::wallWidth / 2.f,
            consts::worldLength / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            consts::worldLength,
            2.f,
        });

    // Create agent entities. Note that this leaves a lot of components
    // uninitialized, these will be set during world generation, which is
    // called for every episode.
    for (CountT i = 0; i < consts::numAgents; ++i) {
        Entity agent = ctx.data().agents[i] = ctx.makeEntity<Agent>();

        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.get<ObjectID>(agent) = ObjectID { (int32_t)SimObject::Agent };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
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

static inline float randInRangeCentered(Engine &ctx, float range)
{
    return ctx.data().rng.rand() * range - range / 2.f;
}

static inline float randBetween(Engine &ctx, float min, float max)
{
    return ctx.data().rng.rand() * (max - min) + min;
}

static void resetPersistentEntities(Engine &ctx)
{
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

     for (CountT i = 0; i < 3; i++) {
         Entity wall_entity = ctx.data().borders[i];
         registerRigidBodyEntity(ctx, wall_entity, SimObject::Wall);
     }

     for (CountT i = 0; i < consts::numAgents; i++) {
         Entity agent_entity = ctx.data().agents[i];
         registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);

         ctx.get<viz::VizCamera>(agent_entity) =
             viz::VizRenderingSystem::setupView(ctx, 90.f, 0.001f,
                 1.5f * math::up, (int32_t)i);

         // Place the agents near the starting wall
         Vector3 pos {
             randInRangeCentered(ctx, 
                 consts::worldWidth / 2.f - 2.5f * consts::agentRadius),
             randBetween(ctx, 0.f, consts::distancePerProgress / 2.f) +
                 1.1f * consts::agentRadius,
             0.f,
         };

         if (i % 2 == 0) {
             pos.x += consts::worldWidth / 4.f;
         } else {
             pos.x -= consts::worldWidth / 4.f;
         }

         ctx.get<Position>(agent_entity) = pos;
         ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
             randInRangeCentered(ctx, math::pi / 4.f),
             math::up);

         ctx.get<Progress>(agent_entity).numProgressIncrements = 0;

         ctx.get<Velocity>(agent_entity) = {
             Vector3::zero(),
             Vector3::zero(),
         };
         ctx.get<ExternalForce>(agent_entity) = Vector3::zero();
         ctx.get<ExternalTorque>(agent_entity) = Vector3::zero();
         ctx.get<Action>(agent_entity) = Action {
             .x = consts::numMoveBuckets / 2,
             .y = consts::numMoveBuckets / 2,
             .r = consts::numMoveBuckets / 2,
         };

         ctx.get<Done>(agent_entity).v = 0;
     }
}

// Builds the two walls & door that block the end of the challenge room
static Entity makeEndWall(Engine &ctx,
                          Room &room,
                          CountT room_idx)
{
    float y_pos = consts::roomLength * (room_idx + 1) -
        consts::wallWidth / 2.f;

    // Quarter door of buffer on both sides, place door and then build walls
    // up to the door gap on both sides
    float door_center = randBetween(ctx, 0.75f * consts::doorWidth, 
        consts::worldWidth - 0.75f * consts::doorWidth);
    float left_len = door_center - 0.5f * consts::doorWidth;
    Entity left_wall = ctx.makeEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        left_wall,
        Vector3 {
            (-consts::worldWidth + left_len) / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        ResponseType::Static,
        Diag3x3 {
            left_len,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, left_wall, SimObject::Wall);

    float right_len =
        consts::worldWidth - door_center - 0.5f * consts::doorWidth;
    Entity right_wall = ctx.makeEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        right_wall,
        Vector3 {
            (consts::worldWidth - right_len) / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        ResponseType::Static,
        Diag3x3 {
            right_len,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, right_wall, SimObject::Wall);

    Entity door = ctx.makeEntity<DoorEntity>();
    setupRigidBodyEntity(
        ctx,
        door,
        Vector3 {
            door_center - consts::worldWidth / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Door,
        ResponseType::Static,
        Diag3x3 {
            consts::doorWidth * 0.8f,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, door, SimObject::Door);
    ctx.get<OpenState>(door).isOpen = false;

    room.separators[0] = left_wall;
    room.separators[1] = right_wall;
    room.door = door;

    return Entity::none();
}

static void makeSingleButtonRoom(Engine &ctx,
                                      Room &room,
                                      float y_min,
                                      float y_max)
{
    float button_x = randInRangeCentered(ctx,
        consts::worldWidth / 2.f - consts::buttonWidth);
    float button_y = randBetween(ctx, y_min + consts::roomLength / 4.f,
        y_max - consts::wallWidth - consts::buttonWidth / 2.f);

    Entity button = ctx.makeEntity<ButtonEntity>();
    ctx.get<Position>(button) = Vector3 {
        button_x,
        button_y,
        0.f,
    };
    ctx.get<Rotation>(button) = Quat { 1, 0, 0, 0 };
    ctx.get<Scale>(button) = Diag3x3 {
        consts::buttonWidth,
        consts::buttonWidth,
        0.2f,
    };
    ctx.get<ObjectID>(button) = ObjectID { (int32_t)SimObject::Button };
    ctx.get<ButtonProperties>(button).isPersistent = true;

    room.entities[0].type = DynEntityType::Button;
    room.entities[0].e = button;
    for (CountT i = 1; i < consts::maxEntitiesPerRoom; i++) {
        room.entities[i].type = DynEntityType::None;
    }

    ctx.get<LinkedDoor>(button).e =
        room.door;
}

static void makeRandomRoom(Engine &ctx,
                                Room &room,
                                CountT room_idx)
{
    float room_y_min = room_idx * consts::roomLength;
    float room_y_max = (room_idx + 1) * consts::roomLength;

    makeSingleButtonRoom(
        ctx, room, room_y_min, room_y_max);
}

static void generateLevel(Engine &ctx)
{
    LevelState &level = ctx.singleton<LevelState>();

    {
        Room &room = level.rooms[0];
        makeEndWall(ctx, room, 0);
        makeSingleButtonRoom(ctx, room, 0, consts::roomLength);
    }
    
    for (CountT i = 1; i < consts::numRooms; i++) {
        Room &room = level.rooms[i];
        makeEndWall(ctx, room, i);
        makeRandomRoom(ctx, room, i);
    }
}

// Randomly generate a new world for a training episode
void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
    generateLevel(ctx);
}

}
