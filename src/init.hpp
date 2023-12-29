#pragma once

#include <madrona/physics.hpp>

namespace gpudrive {

struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
};

struct Parameters{
    std::string jsonPath; // This is supposed to be replaced with a directory path in future.
    float polylineReductionThreshold;
    float observationRadius;
    madrona::CountT numAgents = 10;
    madrona::CountT numRoadSegments = 500;
    // More params to be added here.
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    madrona::phys::ObjectManager *rigidBodyObjMgr;
    const madrona::viz::VizECSBridge *vizBridge;
    Parameters params;
};

}
