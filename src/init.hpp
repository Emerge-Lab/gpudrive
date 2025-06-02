#pragma once

#include <madrona/physics.hpp>
#include "types.hpp"

namespace madrona_gpudrive
{
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
        VehicleSize vehicle_size;
        float heading[MAX_POSITIONS];
        MapVector2 velocity[MAX_POSITIONS];
        bool valid[MAX_POSITIONS];
        MapVector2 goalPosition;
        EntityType type;
        MetaData metadata;

        uint32_t numPositions;
        uint32_t numHeadings;
        uint32_t numVelocities;
        uint32_t numValid;
        uint32_t id;
        MapVector2 mean;
        bool markAsExpert{false};
        float vbd_trajectories[consts::kTrajectoryLength][6];  // x, y, yaw, vx, vy, valid
    };

    struct MapRoad
    {
        // std::array<MapPosition, MAX_POSITIONS> geometry;
        MapVector2 geometry[MAX_GEOMETRY];
        uint32_t id;
        MapType mapType;
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

        char mapName[32];

        char scenarioId[32];

        // Constructor
        Map() = default;
    };

    struct EpisodeManager
    {
        madrona::AtomicU32 curEpisode;
    };

    enum class RewardType : uint32_t
    {
        DistanceBased,  // negative distance to goal
        OnGoalAchieved, // 1 if on goal, 0 otherwise
        Dense           // negative distance to expert trajectory
    };

    struct RewardParams
    {
        RewardType rewardType;
        float distanceToGoalThreshold;
        float distanceToExpertThreshold;
    };

    enum class CollisionBehaviour : uint32_t
    {
        AgentStop,
        AgentRemoved,
        Ignore
    };

    enum class GoalBehaviour {
        Remove,  // Teleport to padding position
        Stop,    // Agent remains in place but is marked as done
        Ignore,  // Agent continues to exist in the scene but is marked as done
    };

    enum class DynamicsModel : uint32_t
    {
        Classic,
        InvertibleBicycle,
        DeltaLocal,
        State
    };

    enum class FindRoadObservationsWith
    {
        KNearestEntitiesWithRadiusFiltering,
        AllEntitiesWithRadiusFiltering
    };

    struct Parameters
    {
        float polylineReductionThreshold;
        float observationRadius;
        float viewConeHalfAngle;
        bool viewOccludeObjects;
        RewardParams rewardParams;
        CollisionBehaviour collisionBehaviour = CollisionBehaviour::AgentStop; // Default: AgentStop
        GoalBehaviour goalBehaviour = GoalBehaviour::Remove;  // Default to current behavior
        uint32_t maxNumControlledAgents = 10000;                               // Arbitrary high number to by default control all vehicles
        bool IgnoreNonVehicles = false;                                        // Default: false
        FindRoadObservationsWith roadObservationAlgorithm{
            FindRoadObservationsWith::KNearestEntitiesWithRadiusFiltering};
        bool initOnlyValidAgentsAtFirstStep = true; // Default: true
        bool isStaticAgentControlled = false;       // Default: false
        bool enableLidar = false;
        bool disableClassicalObs = false;
        DynamicsModel dynamicsModel = DynamicsModel::Classic;
        bool readFromTracksToPredict = false;       // Default: false - for wosac initialization mode
        uint32_t initSteps = 0;
        bool controlExperts = false; // Default: false - for wosac initialization mode
    };

    struct WorldInit
    {
        EpisodeManager *episodeMgr;
        madrona::phys::ObjectManager *rigidBodyObjMgr;
        Map *map;
        const Parameters *params;
    };

}
