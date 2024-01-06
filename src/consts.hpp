#pragma once

#include <madrona/types.hpp>

namespace gpudrive {

namespace consts {
// TODO: Rename numAgents to maxNumAgents and numRoadSegments to
// maxNumRoadSegments
inline constexpr madrona::CountT numAgents = 17;
inline constexpr madrona::CountT numRoadSegments = 5448;

// Various world / entity size parameters
inline constexpr float worldLength = 40.f;

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
