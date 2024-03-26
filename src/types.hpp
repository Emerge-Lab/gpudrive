#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/physics.hpp>
#include <madrona/viz/system.hpp>

#include "consts.hpp"

namespace gpudrive {

// Include several madrona types into the simulator namespace for convenience
using madrona::Entity;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::phys::Velocity;
using madrona::phys::ResponseType;
using madrona::phys::ExternalForce;
using madrona::phys::ExternalTorque;

// This enum is used to track the type of each entity
// The order of the enum is important and should not be changed
// The order is {Road types that can be reduced, Road types that cannot be reduced, agent types, other types}
enum class EntityType : uint32_t {
    None,
    RoadEdge,
    RoadLine,
    RoadLane,
    CrossWalk,
    SpeedBump,
    StopSign,
    Vehicle,
    Pedestrian,
    Cyclist,
    Padding,
    NumTypes,
};

// Constants computed from train files.
constexpr size_t MAX_OBJECTS = 515;
constexpr size_t MAX_ROADS = 956;
constexpr size_t MAX_POSITIONS = 91;
constexpr size_t MAX_GEOMETRY = 1746;

// Cannot use Madrona::math::Vector2 because it is not a POD type.
// Getting all zeros if using any madrona types.
struct MapVector2
{
    float x;
    float y;
};

struct MapObject
{
    MapVector2 position[MAX_POSITIONS];
    float width;
    float length;
    float heading[MAX_POSITIONS];
    MapVector2 velocity[MAX_POSITIONS];
    bool valid[MAX_POSITIONS];
    MapVector2 goalPosition;
    EntityType type;

    uint32_t numPositions;
    uint32_t numHeadings;
    uint32_t numVelocities;
    uint32_t numValid;
    MapVector2 mean;
};

struct MapRoad
{
    // std::array<MapPosition, MAX_POSITIONS> geometry;
    MapVector2 geometry[MAX_GEOMETRY];
    EntityType type;
    uint32_t numPoints;
    MapVector2 mean;
};

struct Map
{
    MapObject objects[MAX_OBJECTS];
    MapRoad roads[MAX_ROADS];

    uint32_t numObjects;
    uint32_t numRoads;
    uint32_t numRoadSegments;
    MapVector2 mean;

    // Constructor
    Map() = default;
};

struct BicycleModel {
    madrona::math::Vector2 position;
    float heading;
    float speed;
};

struct VehicleSize {
  float length;
  float width;
};

struct Goal{
    madrona::math::Vector2 position;
};
// WorldReset is a per-world singleton component that causes the current
// episode to be terminated and the world regenerated
// (Singleton components like WorldReset can be accessed via Context::singleton
// (eg ctx.singleton<WorldReset>().reset = 1)
struct WorldReset {
    int32_t reset;
};

// TODO(samk): need to wrap elements in std::optional to match Nocturne?
struct Action {
    float acceleration;
    float steering;
    float headAngle;
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

// Observation state for the current agent.
// Positions are rescaled to the bounds of the play area to assist training.
struct SelfObservation {
    float speed;
    VehicleSize vehicle_size;
    Goal goal;
    float collisionState;
};

struct MapObservation {
    madrona::math::Vector2 position;
    float heading;
    float type;
};

struct PartnerObservation {
    float speed;
    madrona::math::Vector2 position;
    float heading;
    VehicleSize vehicle_size;
    float type;
};

// Egocentric observations of other agents
struct PartnerObservations {
    PartnerObservation obs[consts::kMaxAgentCount - 1];
};

// PartnerObservations is exported as a
// [N, A, consts::numAgents - 1, 3] // tensor to pytorch
static_assert(sizeof(PartnerObservations) == sizeof(float) *
    (consts::kMaxAgentCount - 1) * 7);

struct AgentMapObservations {
    MapObservation obs[consts::kMaxRoadEntityCount];
};



struct LidarSample {
    float depth;
    float encodedType;
};

// Linear depth values and entity type in a circle around the agent
struct Lidar {
    LidarSample samples[consts::numLidarSamples];
};

// Number of steps remaining in the episode. Allows non-recurrent policies
// to track the progression of time.
struct StepsRemaining {
    uint32_t t;
};

// Can be refactored for rewards
struct Progress {
    float maxY;
};

// Per-agent component storing Entity IDs of the other agents. Used to
// build the egocentric observations of their state.
struct OtherAgents {
    madrona::Entity e[consts::kMaxAgentCount - 1];
};

struct Trajectory {
    madrona::math::Vector2 positions[consts::kTrajectoryLength];
    madrona::math::Vector2 velocities[consts::kTrajectoryLength];
    float headings[consts::kTrajectoryLength];
    int32_t valids[consts::kTrajectoryLength];
};

struct Shape {
    int32_t agentEntityCount;
    int32_t roadEntityCount;
};

enum class ControlMode {
   EXPERT,
   BICYCLE
};

struct ControlledState {
   ControlMode controlledState; // 0: controlled by expert, 1: controlled by action inputs. Default: 1
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
    madrona::phys::CollisionEvent,
  
    // Internal logic state.
    Progress,
    OtherAgents,
    EntityType,

    // gpudrive
    BicycleModel,
    VehicleSize,
    Goal,
    Trajectory,
    ControlledState,
    // Input
    Action,

    // Observations
    SelfObservation,
    PartnerObservations,
    AgentMapObservations,
    Lidar,
    StepsRemaining,

    // Reward, episode termination
    Reward,
    Done,

    // Visualization: In addition to the fly camera, src/viewer.cpp can
    // view the scene from the perspective of entities with this component
    madrona::viz::VizCamera
> {};

// Generic archetype for entities that need physics but don't have custom
// logic associated with them.
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
    madrona::phys::broadphase::LeafID,
    MapObservation,
    EntityType
> {};

}
