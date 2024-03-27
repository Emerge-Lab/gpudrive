#pragma once

#include <madrona/physics.hpp>
#include "types.hpp"

namespace gpudrive
{
    struct EpisodeManager
    {
        madrona::AtomicU32 curEpisode;
    };

    enum class RewardType : uint32_t
    {
        DistanceBased, // negative distance to goal
        OnGoalAchieved, // 1 if on goal, 0 otherwise
        Dense // negative distance to expert trajectory
    };

    struct RewardParams
    {
        RewardType rewardType;
        float distanceToGoalThreshold;
        float distanceToExpertThreshold;
    };

    enum class CollisionBehaviour: uint32_t
    {
        AgentStop,
        AgentRemoved,
        Ignore
    };

    enum class DatasetInitOptions : uint32_t
    {
        FirstN,
        RandomN,
        PadN, // Pad the worlds by repeating the first world.
        ExactN, // Will fail if N != NumWorlds
    };

    struct Parameters
    {
        float polylineReductionThreshold;
        float observationRadius;
        RewardParams rewardParams;
        CollisionBehaviour collisionBehaviour = CollisionBehaviour::AgentStop; // Default: AgentStop
        DatasetInitOptions datasetInitOptions;
        uint32_t maxNumControlledVehicles = 10000; // Arbitrary high number to by default control all vehicles 
    };

    struct WorldInit
    {
        EpisodeManager *episodeMgr;
        madrona::phys::ObjectManager *rigidBodyObjMgr;
        const madrona::viz::VizECSBridge *vizBridge;
        gpudrive::Map *map;
        const Parameters *params;
    };

}