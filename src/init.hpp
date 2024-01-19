#pragma once

#include <madrona/physics.hpp>
#include <madrona/exec_mode.hpp>
#include <madrona/math.hpp>
#include <madrona/physics.hpp>
#include <madrona/types.hpp>

namespace madrona::viz {
struct VizECSBridge;
}

namespace gpudrive
{
    // Constants computed from train files.
    constexpr size_t MAX_OBJECTS = 515;
    constexpr size_t MAX_ROADS = 956;
    constexpr size_t MAX_POSITIONS = 91;
    constexpr size_t MAX_GEOMETRY = 1746;
    
    enum class MapObjectType : uint32_t
    {
        Vehicle,
        Pedestrian,
        Cyclist,
        Invalid
    };

    enum class MapRoadType : uint32_t
    {
        RoadEdge,
        RoadLine,
        Lane,
        CrossWalk,
        SpeedBump,
        StopSign,
        Invalid
    };

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
        MapObjectType type;

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
        MapRoadType type;
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
    struct WorldInit
    {
        EpisodeManager *episodeMgr;
        madrona::phys::ObjectManager *rigidBodyObjMgr;
        const madrona::viz::VizECSBridge *vizBridge;
        Map *map;
        madrona::ExecMode mode;
    };

}