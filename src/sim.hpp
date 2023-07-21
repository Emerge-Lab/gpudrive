#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>
#include <madrona/physics.hpp>
#include <madrona/viz/system.hpp>

#include "init.hpp"
#include "rng.hpp"

namespace GPUHideSeek {

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

namespace consts {

// Each environment is composed of numChallenges rooms that the agents
// must solve in order to maximize their reward.
inline constexpr CountT numChallenges = 4;

inline constexpr CountT numAgents = 2;

inline constexpr float worldLength = 60.f;
inline constexpr float worldWidth = 20.f;
inline constexpr float wallWidth = 1.f;
inline constexpr float buttonWidth = 0.8f;
inline constexpr float agentRadius = 1.f;
inline constexpr float challengeLength =
    worldLength / numChallenges;

// Units of distance along the environment needed for further reward
constexpr float distancePerProgress = 4.f;

// How many discrete options for each movement action
inline constexpr CountT numMoveBuckets = 5;
inline constexpr CountT numLidarSamples = 30;
inline constexpr CountT maxEntitiesPerChallenge = 5;

}

// This enum is used by the Sim and Manager classes to track the export slots
// for each component exported to the training code.
enum class ExportID : uint32_t {
    Reset,
    Action,
    Reward,
    Done,
    PositionObservation,
    ToOtherAgents,
    ToDynamicEntities,
    Lidar,
    NumExports,
};

// Stores values for the ObjectID component that links entities to
// render / physics assets.
enum class SimObject : uint32_t {
    Cube,
    Wall,
    Agent,
    Plane,
    NumObjects,
};

enum class DynamicEntityType : uint32_t {
    None,
    Button,
    Block,
    NumTypes,
};

class Engine;

struct WorldReset {
    int32_t reset;
};

struct OpenState {
    int32_t isOpen;
};

struct ButtonState {
    bool isPressed;
};

struct Progress {
    int32_t numProgressIncrements;
};

struct Action {
    int32_t x;
    int32_t y;
    int32_t r;
};

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

// Entity ID of the other agents
struct OtherAgents {
    madrona::Entity e[consts::numAgents - 1];
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

struct WallObject : public madrona::Archetype<
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

struct PhysicsObject : public madrona::Archetype<
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
struct ButtonObject : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    ButtonState
> {};

// The Sim class encapsulates the per-world state of the simulation.
// Sim is always available by calling ctx.data() given a reference
// to the Engine / Context object that is passed to each ECS system.
//
// Per-World state that is frequently accessed but only used by a few
// ECS systems should be put in a singleton component rather than
// in this class in order to ensure efficient access patterns.
struct Sim : public madrona::WorldBase {
    struct Config {
        bool enableViewer;
        bool autoReset;
    };

    // Sim::registerTypes is called during initialization
    // to register all components & archetypes with the ECS.
    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    // Sim::setupTasks is called during initialization to build
    // the system task graph that will be invoked by the 
    // Manager class (src/mgr.hpp) for each step.
    static void setupTasks(madrona::TaskGraph::Builder &builder,
                           const Config &cfg);

    // The constructor is called for each world during initialization.
    // Config is global across all worlds, while WorldInit (src/init.hpp)
    // can contain per-world initialization data, created in (src/mgr.cpp)
    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &init);

    // EpisodeManager globally tracks episode IDs with an atomic across the
    // simulation.
    EpisodeManager *episodeMgr;

    // Simple random number generator seeded with episode ID.
    RNG rng;

    // Floor plane entity, constant across all episodes.
    Entity floorPlane;

    // Border wall entities: 3 walls to the left, up and down that define
    // play area. These are constant across all episodes.
    Entity borders[3];

    // Agent entity references. This entities live across all episodes
    // and are just reset to the start of the level on reset.
    Entity agents[consts::numAgents];

    // Dynamic entities that need to be cleaned up when generating a new
    // environment.
    Entity dynamicEntities[
        consts::maxEntitiesPerChallenge * consts::numChallenges];
    int32_t numDynamicEntities;

    // Current step within this episode
    int32_t curEpisodeStep;

    // Episode ID number
    int32_t curEpisodeIdx;

    // Should the environment automatically reset (generate a new episode)
    // at the end of each episode?
    bool autoReset;

    // Are we visualizing the simulation in the viewer?
    bool enableVizRender;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
