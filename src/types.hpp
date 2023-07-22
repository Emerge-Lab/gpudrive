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
    Progress,

    // Input
    Action,

    // Observations
    PositionObservation,
    OtherAgents,
    ToOtherAgents,
    ToDynamicEntities,
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
    OpenState
> {};

// Archetype for the button objects that open the doors
// Buttons don't have collision but are rendered
struct ButtonEntity : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    ButtonProperties,
    LinkedDoor
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
