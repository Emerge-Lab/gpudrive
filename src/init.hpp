#pragma once

#include "consts.hpp"
#include <madrona/exec_mode.hpp>
#include <madrona/math.hpp>
#include <madrona/physics.hpp>
#include <madrona/types.hpp>

namespace madrona::viz {
struct VizECSBridge;
}

namespace gpudrive {

struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
};

struct AgentInit {
    float xCoord;
    float yCoord;
    float length;
    float width;
    float heading;
    float speedX;
    float speedY;
    float goalX;
    float goalY;
};

// TODO: reuse ObjectID?
enum class RoadInitType : uint64_t { SpeedBump, StopSign, RoadEdge, Lane };

struct RoadInit {
    RoadInitType type;
    madrona::CountT numPoints;
    madrona::math::Vector2 points[consts::kMaxRoadGeometryLength];
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
    AgentInit *agentInits;
    madrona::CountT agentInitsCount;
    RoadInit *roadInits;
    madrona::CountT roadInitsCount;
    madrona::ExecMode mode;
    madrona::CountT computeEntityUpperBound() const;
    Parameters params;
};

}
