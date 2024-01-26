#pragma once

#include <limits>
#include <madrona/math.hpp>
#include <madrona/types.hpp>

namespace gpudrive {

namespace consts {

inline constexpr madrona::CountT kMaxAgentCount = 3;
inline constexpr madrona::CountT kMaxRoadEntityCount = 5000;

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

inline constexpr madrona::math::Vector3 kPaddingPosition = {
    0, 0, std::numeric_limits<float>::max()};
}

}
