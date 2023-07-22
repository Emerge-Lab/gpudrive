#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/physics.hpp>
#include <madrona/viz/system.hpp>

#include "consts.hpp"

namespace GPUHideSeek {

// Include several madrona types into the simulator namespace for convenience
using madrona::Entity;
using madrona::CountT;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::phys::Velocity;
using madrona::phys::ResponseType;
using madrona::phys::ExternalForce;
using madrona::phys::ExternalTorque;

// WorldReset is a per-world singleton component that causes the current
// episode to be terminated and the world regenerated
// (Singleton components like WorldReset can be accessed via Context::singleton
// (eg ctx.singleton<WorldReset>().reset = 1)
struct WorldReset {
    int32_t reset;
};

// Discrete action component. Ranges are defined by consts::numMoveBuckets (5),
// repeated here for clarity
struct Action {
    int32_t x; // [-2, 2]
    int32_t y; // [-2, 2]
    int32_t r; // [-2, 2]
    int32_t g; // 0 = do nothing, 1 = grab / release
};

// Per-agent reward
// Exported as an [N * A, 1] float tensor to training code
struct Reward {
    float v;
};

// Per-agent component that indicates that the agent's episode is finished
// This is exported per-agent for simplicity in the training code
struct Done {
    // Currently bool components are not supported due to
    // padding issues, so Done is an int32_t
    int32_t v;
};

// Agent position rescaled to the bounds of the play area to assist training
struct PositionObservation {
    float roomX;
    float roomY;
    float globalX;
    float globalY;
    float globalZ;
    float theta;
};

// Entity ID of the other agents as well as a place to cache their rewards
// so each agent can get half of their partner's reward as well
struct OtherAgents {
    madrona::Entity e[consts::numAgents - 1];
    float rewards[consts::numAgents - 1];
};

// The state of the world is passed to each agent in terms of egocentric
// polar coordinates. theta is degrees off agent forward.
struct PolarObservation {
    float r;
    float theta;
};

// Egocentric observations of other agents
struct ToOtherAgents {
    PolarObservation obs[consts::numAgents - 1];
};

// Per-agent egocentric observations for the interactable entities
// in the current room.
struct EntityObservation {
    float encodedType;
    PolarObservation polar;
};

struct ToRoomEntities {
    EntityObservation obs[consts::maxEntitiesPerRoom];
};

// ToRoomEntities is exported as a N, maxEntitiesPerRoom, 3 tensor to pytorch
static_assert(sizeof(ToRoomEntities) == sizeof(float) *
    consts::maxEntitiesPerRoom * 3);

// Linear depth values in a circle around the agent
struct Lidar {
    float depth[consts::numLidarSamples];
};

// Tracks progress the agent has made through the challenge, used to add
// reward when more progress has been made
struct Progress {
    int32_t numProgressIncrements;
};

// Tracks if an agent is currently grabbing another entity
struct GrabState {
    Entity constraintEntity;
};


// A per-door component that tracks whether or not the door should be open.
struct OpenState {
    bool isOpen;
};

// Linked buttons that control the door opening and whether or not the door
// should remain open after the buttons are pressed once.
struct DoorProperties {
    Entity buttons[consts::maxEntitiesPerRoom];
    int32_t numButtons;
    bool isPersistent;
};

// Similar to OpenState, true during frames where a button is pressed
struct ButtonState {
    bool isPressed;
};

enum class RoomEntityType : uint32_t {
    None,
    Button,
    Cube,
    NumTypes,
};

struct RoomEntityState {
    Entity e;
    RoomEntityType type;
};

struct Room {
    // These are entities the agent will interact with
    RoomEntityState entities[consts::maxEntitiesPerRoom];

    // The walls that separate this room from the next
    Entity walls[2];

    // The door the agents need to figure out how to lower
    Entity door;
};

struct LevelState {
    Room rooms[consts::numRooms];
};

/* ECS Archetypes for the game */

// There are 2 Agents in the environment trying to get to the destination
struct Agent : public madrona::Archetype<
    // Basic components required for physics. Note that the current physics
    // implementation requires archetypes to have these components first
    // in this exact order.
    Position,
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,

    // Internal logic state.
    GrabState,
    Progress,

    // Input
    Action,

    // Observations
    PositionObservation,
    OtherAgents,
    ToOtherAgents,
    ToRoomEntities,
    Lidar,

    // Reward, episode termination
    Reward,
    Done,

    // Visualization: In addition to the fly camera, src/viewer.cpp can
    // view the scene from the perspective of entities with this component
    madrona::viz::VizCamera
> {};

// Archetype for the doors blocking the end of each challenge room
struct DoorEntity : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,
    OpenState,
    DoorProperties
> {};

// Archetype for the button objects that open the doors
// Buttons don't have collision but are rendered
struct ButtonEntity : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    ButtonState
> {};

// Generic archetype for entities that need physics but no other special state
struct PhysicsEntity : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID
> {};

}
