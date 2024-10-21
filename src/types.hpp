#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>

#include "consts.hpp"

namespace gpudrive
{
    // Include several madrona types into the simulator namespace for convenience
    using madrona::Entity;
    using madrona::base::ObjectID;
    using madrona::base::Position;
    using madrona::base::Rotation;
    using madrona::base::Scale;
    using madrona::phys::ResponseType;
    using madrona::phys::Velocity;

    // This enum is used to track the type of each entity
    // The order of the enum is important and should not be changed
    // The order is {Road types that can be reduced, Road types that cannot be reduced, agent types, other types}
    enum class EntityType : uint32_t
    {
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

struct AgentID {
    int32_t id;
};

    struct VehicleSize
    {
        float length;
        float width;
    };

    struct Goal
    {
        madrona::math::Vector2 position;
    };

    // WorldReset is a per-world singleton component that causes the current
    // episode to be terminated and the world regenerated
    // (Singleton components like WorldReset can be accessed via Context::singleton
    // (eg ctx.singleton<WorldReset>().reset = 1)
    struct WorldReset
    {
        int32_t reset;
    };

struct ResetMap {
    int32_t reset;
};   
     struct ClassicAction
    {
        float acceleration;
        float steering;
        float headAngle;
    };

    struct DeltaAction
    {
        float dx;
        float dy;
        float dyaw;
    };

    struct StateAction
    {
        Position position; // 3 floats
        float yaw;         // 1 float
        Velocity velocity; // 6 floats
    };

    union Action
    {
        ClassicAction classic;
        DeltaAction delta;
        StateAction state;
    };

    const size_t ActionExportSize = 3 + 1 + 6;

    static_assert(sizeof(Action) == sizeof(float) * ActionExportSize);

    // Per-agent reward
    // Exported as an [N * A, 1] float tensor to training code
    struct Reward
    {
        float v;
    };

    // Per-agent component that indicates that the agent's episode is finished
    // This is exported per-agent for simplicity in the training code
    struct Done
    {
        // Currently bool components are not supported due to
        // padding issues, so Done is an int32_t
        int32_t v;
    };

    struct Info
    {
        int collidedWithRoad;
        int collidedWithVehicle;
        int collidedWithNonVehicle;
        int reachedGoal;
        int type;

        static inline Info zero()
        {
            return Info{
                .collidedWithRoad = 0,
                .collidedWithVehicle = 0,
                .collidedWithNonVehicle = 0,
                .reachedGoal = 0,
                .type = static_cast<int>(EntityType::Padding)};
        }
    };

    const size_t InfoExportSize = 5;

    static_assert(sizeof(Info) == sizeof(int) * InfoExportSize);

    // Observation state for the current agent.
    // Positions are rescaled to the bounds of the play area to assist training.
    struct SelfObservation
    {
        float speed;
        VehicleSize vehicle_size;
        Goal goal;
        float collisionState;
    float id;
        static inline SelfObservation zero()
        {
            return SelfObservation{
                .speed = 0,
                .vehicle_size = {0, 0},
                .goal = {.position = {0, 0}},
                .collisionState = 0,
            .id = -1};
        }
    };

    const size_t SelfObservationExportSize = 7;

    static_assert(sizeof(SelfObservation) == sizeof(float) * SelfObservationExportSize);

    struct MapObservation
    {
        madrona::math::Vector2 position;
        Scale scale;
        float heading;
        float type;

        static inline MapObservation zero()
        {
            return MapObservation{
                .position = {0, 0},
                .scale = madrona::math::Diag3x3{0, 0, 0},
                .heading = 0,
                .type = static_cast<float>(EntityType::None)};
        }
    };

    const size_t MapObservationExportSize = 7;

    static_assert(sizeof(MapObservation) == sizeof(float) * MapObservationExportSize);

    struct PartnerObservation
    {
        float speed;
        madrona::math::Vector2 position;
        float heading;
        VehicleSize vehicle_size;
        float type;
    float id;

    static inline PartnerObservation zero() {
        return PartnerObservation{
            .speed = 0,
            .position = {0, 0},
            .heading = 0,
            .vehicle_size = {0, 0},
            .type = static_cast<float>(EntityType::None),
            .id = -1};
    }
};

    // Egocentric observations of other agents
    struct PartnerObservations
    {
        PartnerObservation obs[consts::kMaxAgentCount - 1];
    };

    const size_t PartnerObservationExportSize = 8;

    static_assert(sizeof(PartnerObservations) == sizeof(float) *
                                                     (consts::kMaxAgentCount - 1) * PartnerObservationExportSize);

    struct AgentMapObservations
    {
        MapObservation obs[consts::kMaxAgentMapObservationsCount];
    };

    const size_t AgentMapObservationExportSize = 7;

    static_assert(sizeof(AgentMapObservations) ==
                  sizeof(float) * consts::kMaxAgentMapObservationsCount *
                      AgentMapObservationExportSize);

    struct LidarSample
    {
        float depth;
        float encodedType;
        madrona::math::Vector2 position;
    };

    // Linear depth values and entity type in a circle around the agent
    struct Lidar
    {
        LidarSample samplesCars[consts::numLidarSamples];
        LidarSample samplesRoadEdges[consts::numLidarSamples];
        LidarSample samplesRoadLines[consts::numLidarSamples];
    };

    const size_t LidarExportSize = 3 * consts::numLidarSamples * 4;

    static_assert(sizeof(Lidar) == sizeof(float) * LidarExportSize);
    // Number of steps remaining in the episode. Allows non-recurrent policies
    // to track the progression of time.
    struct StepsRemaining
    {
        uint32_t t;
    };

    // Can be refactored for rewards
    struct Progress
    {
        float maxY;
    };

    // Per-agent component storing Entity IDs of the other agents. Used to
    // build the egocentric observations of their state.
    struct OtherAgents
    {
        madrona::Entity e[consts::kMaxAgentCount - 1];
    };

    struct Trajectory
    {
        madrona::math::Vector2 positions[consts::kTrajectoryLength];
        madrona::math::Vector2 velocities[consts::kTrajectoryLength];
        float headings[consts::kTrajectoryLength];
        float valids[consts::kTrajectoryLength];
        Action inverseActions[consts::kTrajectoryLength];
    };

    const size_t TrajectoryExportSize = 2 * 2 * consts::kTrajectoryLength + 2 * consts::kTrajectoryLength + ActionExportSize * consts::kTrajectoryLength;

    static_assert(sizeof(Trajectory) == sizeof(float) * TrajectoryExportSize);

    struct Shape
    {
        int32_t agentEntityCount;
        int32_t roadEntityCount;
    };

    struct ControlledState
    {
        int32_t controlled; // default: 1
    };

    struct CollisionDetectionEvent
    {
        madrona::AtomicI32 hasCollided{false};
    };

    struct AbsoluteRotation
    {
        Rotation rotationAsQuat;
        float rotationFromAxis;
    };

    struct AbsoluteSelfObservation
    {
        Position position;
        AbsoluteRotation rotation;
        Goal goal;
        VehicleSize vehicle_size;
    };

    const size_t AbsoluteSelfObservationExportSize = 12; // 3 + 4 + 1 + 2 + 2

    static_assert(sizeof(AbsoluteSelfObservation) == sizeof(float) * AbsoluteSelfObservationExportSize);

    struct AgentInterface : public madrona::Archetype<
                                Action,
                                Reward,
                                Done,
                                Info,
                                // Observations
                                SelfObservation,
                                AbsoluteSelfObservation,
                                PartnerObservations,
                                AgentMapObservations,
                                Lidar,
                                StepsRemaining,
                                ResponseType,
                                Trajectory,
    AgentID,

                                ControlledState // Drive Logic

                                >
    {
    };

    struct AgentInterfaceEntity
    {
        madrona::Entity e;
    };

    // Needed so that the taskgraph doesnt run on InterfaceEntity from roads
    struct RoadInterfaceEntity
    {
        madrona::Entity e;
    };

    /* ECS Archetypes for the game */

    struct CameraAgent : public madrona::Archetype<
                             Position,
                             Rotation,
                             madrona::render::RenderCamera,
                             madrona::render::Renderable>
    {
    };

    // There are 2 Agents in the environment trying to get to the destination
    struct Agent : public madrona::Archetype<
                       // Basic components required for physics. Note that the current physics
                       // implementation requires archetypes to have these components first
                       // in this exact order.
                       Position,
                       Rotation,
                       Scale,
                       ObjectID,
                       ResponseType,
                       madrona::phys::broadphase::LeafID,
                       Velocity,
                       CollisionDetectionEvent,

                       // Internal logic state.
                       Progress,
                       OtherAgents,
                       EntityType,

                       VehicleSize,
                       Goal,
                       // Interface
                       AgentInterfaceEntity,
                       // Visualization: In addition to the fly camera, src/viewer.cpp can
                       // view the scene from the perspective of entities with this component
                       madrona::render::RenderCamera,
                       // All entities with the Renderable component will be drawn by the
                       // viewer and batch renderer
                       madrona::render::Renderable

                       >
    {
    };

    struct RoadInterface : public madrona::Archetype<
                               MapObservation>
    {
    };

    // Generic archetype for entities that need physics but don't have custom
    // logic associated with them.
    struct PhysicsEntity : public madrona::Archetype<
                               Position,
                               Rotation,
                               Scale,
                               ObjectID,
                               ResponseType,
                               madrona::phys::broadphase::LeafID,
                               Velocity,
                               RoadInterfaceEntity,
                               EntityType,
                               madrona::render::Renderable>
    {
    };

}
