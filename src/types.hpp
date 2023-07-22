#pragma once

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

struct WorldReset {
    int32_t reset;
};

struct OpenState {
    bool isOpen;
};

struct LinkedDoor {
    Entity e;
};

struct ButtonProperties {
    bool isPersistent;
};

struct Progress {
    int32_t numProgressIncrements;
};

// Discrete action component
struct Action {
    int32_t x;
    int32_t y;
    int32_t r;
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
    float x;
    float y;
    float z;
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

struct ToOtherAgents {
    PolarObservation obs[consts::numAgents - 1];
};

struct EntityObservation {
    float encodedType;
    PolarObservation polar;
};

struct ToDynamicEntities {
    EntityObservation obs[consts::numChallenges][
        consts::maxEntitiesPerChallenge];
};

// ToDynamicEntities is exported as a
// N, numChallenges, maxEntitiesPerChallenge, 3 tensor to pytorch
static_assert(sizeof(ToDynamicEntities) == 3 * sizeof(float) *
              consts::numChallenges * consts::maxEntitiesPerChallenge);

struct Lidar {
    float depth[30];
};

enum class DynEntityType : uint32_t {
    None,
    Button,
    Block,
    NumTypes,
};

struct DynEntityState {
    Entity e;
    DynEntityType type;
};

struct ChallengeState {
    // These are entities the agent will interact with
    DynEntityState entities[consts::maxEntitiesPerChallenge];

    // The walls that separate this challenge from the next
    Entity separators[2];

    // The door the agents need to figure out how to lower
    Entity door;
};

struct LevelState {
    ChallengeState challenges[consts::numChallenges];
};

/* ECS Archetypes for the game */

// There are 2 Agents in the environment trying to get to the destination
struct Agent : public madrona::Archetype<
    // Basic things
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

    // Internal state
    Progress,

    // Input
    Action,

    // Observations
    PositionObservation,
    OtherAgents,
    ToOtherAgents,
    ToDynamicEntities,
    Lidar,

    // Data for training code
    Reward,
    Done,

    // Render
    madrona::viz::VizCamera
> {};

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
    OpenState
> {};

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

// Button objects don't have collision
struct ButtonEntity : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    ButtonProperties,
    LinkedDoor
> {};


}
