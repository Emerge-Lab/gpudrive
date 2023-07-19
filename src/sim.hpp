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

inline constexpr CountT numAgents = 2;

// NOTE: There will be as many buttons as there are rooms.
inline constexpr CountT maxRooms = 7;

inline constexpr CountT maxDoorsPerRoom = 6;

inline constexpr CountT numLidarSamples = 30;

inline constexpr float worldBounds = 18.f;

inline constexpr CountT numMoveBuckets = 5;

}

enum class ExportID : uint32_t {
    Reset,
    Action,
    Reward,
    Done,
    PositionObservation,
    ToOtherAgents,
    ToButtons,
    ToGoal,
    Lidar,
    Seed,
    NumExports,
};

class Engine;

struct WorldReset {
    int32_t reset;
};

struct OpenState {
    CountT isOpen;
};

struct Action {
    int32_t x;
    int32_t y;
    int32_t r;
};

struct Reward {
    float v;
};

struct Done {
    int32_t v;
};

struct GlobalDebugPositions {
    madrona::math::Vector2 agentPositions[consts::numAgents];
};

struct PositionObservation {
    float x;
    float y;
};

// Entity ID of the other agents
struct OtherAgents {
    madrona::Entity e[consts::numAgents - 1];
};

struct PolarCoord {
    float r;
    float theta;
};

// Relative position of the other agents.
struct ToOtherAgents {
    PolarCoord obs[consts::numAgents - 1];
};

struct ToButtons {
    PolarCoord obs[consts::maxRooms];
};

struct ToGoal {
    PolarCoord obs;
};

struct Lidar {
    float depth[30];
};

struct Seed {
    int32_t seed;
};

static_assert(sizeof(Action) == 3 * sizeof(int32_t));

struct ButtonObject : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID
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

    // Inputs
    Action,

    // Observations
    PositionObservation,
    OtherAgents,
    ToOtherAgents,
    ToButtons,
    ToGoal,
    Lidar,
    Reward,
    Done,

    // RNG
    Seed,

    // Render
    madrona::viz::VizCamera
> {};

struct RewardTracker {
    static inline constexpr int32_t gridWidth = 11;
    static inline constexpr int32_t gridMaxX = gridWidth / 2;
    static inline constexpr int32_t gridHeight = 11;
    static inline constexpr int32_t gridMaxY = gridHeight / 2;

    uint32_t visited[gridHeight][gridWidth];
    uint32_t numNewCellsVisited;
    uint32_t numNewButtonsVisited;
    uint32_t outOfBounds;
};

struct EnvRoom {
    Entity doors[consts::maxDoorsPerRoom];
};

struct ButtonInfo {
    madrona::math::Vector2 pos;
    uint32_t visited;
};

struct Room {
    static constexpr CountT kTmpPadSpace = 4;

    madrona::math::Vector2 offset;
    madrona::math::Vector2 extent;

    // If this is negative, we are at the leaf
    CountT splitPlus;
    CountT splitNeg;
    CountT parent;

    CountT leafIdx;

    CountT doorCount;
    CountT tmpOffset;
    CountT doors[consts::maxDoorsPerRoom+kTmpPadSpace];

    // Eligible for door
    bool isEligible;

    bool splitHorizontal;
    float splitFactor;

    ButtonInfo button;
    Entity buttonEntity;

    inline void addDoor(CountT at, CountT doorIdx)
    {
        if (doors[at] == -1)
            doors[at] = doorIdx;
        else
            doors[consts::maxDoorsPerRoom + (tmpOffset++)] = doorIdx;

        doorCount++;
    }
};

struct Sim : public madrona::WorldBase {
    struct Config {
        bool enableBatchRender;
        bool enableViewer;
        bool autoReset;
    };

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraph::Builder &builder,
                           const Config &cfg);

    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &init);

    EpisodeManager *episodeMgr;
    RNG rng;

    // Rooms
    CountT roomCount;
    Room *rooms;
    CountT srcRoom;
    CountT dstRoom;

    CountT leafCount;
    CountT *leafs;

    // Walls (which can retract into the ground)
    CountT numWalls;
    Entity *walls;

    Entity floorPlane;

    // Points into the wall entity array
    Entity *doors;
    CountT numDoors;

    // Agents which will try to get to the destination
    Entity agents[consts::numAgents];

    int32_t curEpisodeStep;
    int32_t curEpisodeIdx;
    bool enableBatchRender;
    bool enableVizRender;
    bool autoReset;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
