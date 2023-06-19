#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/mw.hpp>

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

static constexpr uint32_t numAgents = 2;

// NOTE: There will be as many buttons as there are rooms.
static constexpr uint32_t maxRooms = 10;

static constexpr uint32_t maxDoorsPerRoom = 6;

}

class Engine;

struct WorldReset {
    int32_t resetLevel;
    int32_t numHiders;
    int32_t numSeekers;
};

struct GrabData {
    Entity constraintEntity;
};
enum class AgentType : uint32_t {

    Agent = 0,
    Camera = 1,
};

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

struct OpenState {
    bool isOpen;
};

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

// There are 2 Agents in the environment trying to get to the destination
struct Agent : public madrona::Archetype<
    // Basic things
    Position,
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    Action,
    AgentType,

    // Physics
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    ExternalForce,
    madrona::phys::solver::PreSolveVelocity,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,

    // Observations
    RelativeAgentObservations,
    RelativeButtonObservations,
    RelativeDestinationObservations,
    Lidar,

    // RNG
    Seed,

    // Render
    madrona::render::ViewSettings
> {};

struct ButtonState {
    bool isPressed;
    int boundDoor;
};

struct Button : public madrona::Archetype<
    Position,
    Scale,
    ObjectID,
    ButtonState,
    madrona::render::ViewSettings
> {};

struct Door : public madrona::Archetype<
    Position,
    Scale,
    Rotation,
    ObjectID,

    // Physics
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,

    // Render
    madrona::render::ViewSettings
> {};

// Camera
struct CameraAgent : public madrona::Archetype<
    Position,
    Rotation,
    madrona::render::ViewSettings
> {};

struct Config {
    bool enableRender;
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
    madrona::math::Vector2 offset;
    madrona::math::Vector2 extent;

    // If this is negative, we are at the leaf
    int8_t splitPlus;
    int8_t splitNeg;
    int8_t parent;

    int8_t leafIdx;

    uint32_t doorCount;
    int8_t doors[consts::maxDoorsPerRoom];

    // Eligible for door
    bool isEligible;

    bool splitHorizontal;
    float splitFactor;

    ButtonInfo button;
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
    uint32_t roomCount;
    Room *rooms;
    uint32_t srcRoom;
    uint32_t dstRoom;

    uint32_t leafCount;
    uint32_t *leafs;

    // Walls (which can retract into the ground)
    uint32_t numWalls;
    Entity *walls;

    Entity floorPlane;

    // Points into the wall entity array
    Entity *doors;
    uint32_t numDoors;

    // Agents which will try to get to the destination
    Entity agents[consts::numAgents];

    CountT curEpisodeStep;
    CountT minEpisodeEntities;
    CountT maxEpisodeEntities;

    uint32_t curEpisodeSeed;
    bool enableRender;
    bool autoReset;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};


}
