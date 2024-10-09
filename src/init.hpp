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

    enum class CollisionBehaviour: uint32_t
    {
        AgentStop,
        AgentRemoved,
        Ignore
    };

    enum class DynamicsModel: uint32_t
    {
        Classic,
        InvertibleBicycle,
        DeltaLocal,
        State
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
        CollisionBehaviour collisionBehaviour = CollisionBehaviour::AgentStop; // Default: AgentStop
        uint32_t maxNumControlledAgents = 10000; // Arbitrary high number to by default control all vehicles 
        bool IgnoreNonVehicles = false; // Default: false
        FindRoadObservationsWith roadObservationAlgorithm{
            FindRoadObservationsWith::KNearestEntitiesWithRadiusFiltering};
        bool initOnlyValidAgentsAtFirstStep = true; // Default: true
        bool isStaticAgentControlled = false; // Default: false
        bool enableLidar = false;
        bool disableClassicalObs = false;
        DynamicsModel dynamicsModel = DynamicsModel::Classic;

    };

    struct WorldInit
    {
        EpisodeManager *episodeMgr;
        madrona::phys::ObjectManager *rigidBodyObjMgr;
        Map *map;
        const Parameters *params;
    };

}
