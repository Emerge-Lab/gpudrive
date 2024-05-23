#pragma once

#include <madrona/physics.hpp>
#include "types.hpp"

namespace gpudrive
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

    enum class DatasetInitOptions : uint32_t
    {
        FirstN,
        RandomN,
        PadN, // Pad the worlds by repeating the first world.
        ExactN, // Will fail if N != NumWorlds
    };
  
    enum class CollisionBehaviour: uint32_t
    {
        AgentStop,
        AgentRemoved,
        Ignore
    };

    enum class FindRoadObservationsWith {
      KNearestEntitiesWithRadiusFiltering,
      AllEntitiesWithRadiusFiltering
    };

    struct Parameters
    {
        float polylineReductionThreshold;
        float observationRadius;
        RewardParams rewardParams;
        DatasetInitOptions datasetInitOptions;
        CollisionBehaviour collisionBehaviour = CollisionBehaviour::AgentStop; // Default: AgentStop
        uint32_t maxNumControlledVehicles = 10000; // Arbitrary high number to by default control all vehicles 
        bool IgnoreNonVehicles = false; // Default: false
        FindRoadObservationsWith roadObservationAlgorithm{
            FindRoadObservationsWith::KNearestEntitiesWithRadiusFiltering};
        bool initOnlyValidAgentsAtFirstStep = true; // Default: true
    };

    struct WorldInit
    {
        EpisodeManager *episodeMgr;
        madrona::phys::ObjectManager *rigidBodyObjMgr;
        Map *map;
        const Parameters *params;
    };

}
