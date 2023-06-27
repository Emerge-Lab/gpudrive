#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>
#include <madrona/physics.hpp>
#include <madrona/viz/system.hpp>

#include "init.hpp"
#include "rng.hpp"

namespace madrona::math {

// x maps to r; y maps to theta
using PolarVector2 = Vector2;
    
}

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

inline constexpr uint32_t numAgents = 2;

// NOTE: There will be as many buttons as there are rooms.
inline constexpr uint32_t maxRooms = 7;

inline constexpr uint32_t maxDoorsPerRoom = 6;

}

enum class ExportIDs : uint32_t {
    Reset = 0,
    Action = 2,
};

class Engine;

struct WorldReset {
    int32_t resetLevel;
    int32_t numHiders;
    int32_t numSeekers;
};

struct OpenState {
    CountT isOpen;
};

struct Action {
    int32_t x;
    int32_t y;
    int32_t r;
};

struct AgentActiveMask {
    float mask;
};

struct GlobalDebugPositions {
    madrona::math::Vector2 agentPositions[consts::numAgents];
};

// Relative position of the other agent.
struct RelativeAgentObservations {
    madrona::math::PolarVector2 obs[consts::numAgents - 1];
};

struct RelativeButtonObservations {
    madrona::math::PolarVector2 obs[consts::maxRooms];
};

struct RelativeDestinationObservations {
    madrona::math::PolarVector2 obs;
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

struct DynamicObject : public madrona::Archetype<
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
    RelativeAgentObservations,
    RelativeButtonObservations,
    RelativeDestinationObservations,
    Lidar,

    // RNG
    Seed,

    // Render
    madrona::viz::VizCamera
> {};

struct ButtonState {
    CountT isPressed;
    CountT boundDoor;
};

struct Button : public madrona::Archetype<
    Position,
    Scale,
    ObjectID,
    ButtonState
> {};

struct Config {
    bool enableBatchRender;
    bool enableViewer;
    bool autoReset;
};

struct EnvRoom {
    Entity doors[consts::maxDoorsPerRoom];
};

struct ButtonInfo {
    madrona::math::Vector2 start;
    madrona::math::Vector2 end;
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

    void addDoor(CountT at, CountT doorIdx)
    {
        if (doors[at] == -1)
            doors[at] = doorIdx;
        else
            doors[consts::maxDoorsPerRoom + (tmpOffset++)] = doorIdx;

        doorCount++;
    }
};

struct Sim : public madrona::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraph::Builder &builder,
                           const Config &cfg);

    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &init);

    EpisodeManager *episodeMgr;
    float *rewardBuffer;
    uint8_t *doneBuffer;
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

    CountT curEpisodeStep;
    CountT minEpisodeEntities;
    CountT maxEpisodeEntities;

    uint32_t curEpisodeSeed;
    bool enableBatchRender;
    bool enableVizRender;
    bool autoReset;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};


}
