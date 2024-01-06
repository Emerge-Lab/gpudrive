#pragma once

#include <madrona/types.hpp>

namespace gpudrive {

namespace consts {
// Each random world is composed of a fixed number of rooms that the agents
// must solve in order to maximize their reward.
inline constexpr madrona::CountT numRooms = 3;

// TODO: Rename numAgents to maxNumAgents and numRoadSegments to
// maxNumRoadSegments
inline constexpr madrona::CountT numAgents = 17;
inline constexpr madrona::CountT numRoadSegments = 5448;

// Maximum number of interactive objects per challenge room. This is needed
// in order to setup the fixed-size learning tensors appropriately.
inline constexpr madrona::CountT maxEntitiesPerRoom = 6;

// Various world / entity size parameters
inline constexpr float worldLength = 40.f;
inline constexpr float worldWidth = 20.f;
inline constexpr float agentRadius = 1.f;

// Each unit of distance forward (+ y axis) rewards the agents by this amount
inline constexpr float rewardPerDist = 0.05f;
// Each step that the agents don't make additional progress they get a small
// penalty reward
inline constexpr float slackReward = -0.005f;

// Steps per episode
inline constexpr int32_t episodeLen = 200;

// Number of lidar samples, arranged in circle around agent
inline constexpr madrona::CountT numLidarSamples = 30;

// Time (seconds) per step
inline constexpr float deltaT = 0.04f;


// Number of physics substeps
inline constexpr madrona::CountT numPhysicsSubsteps = 0.f;

inline constexpr float zDimensionScale = 1;
inline constexpr float xDimensionScaleRoadSegment = 1;

inline constexpr madrona::CountT kTrajectoryLength = 1;

inline constexpr madrona::CountT kMaxRoadGeometryLength = 1810;
}

}
