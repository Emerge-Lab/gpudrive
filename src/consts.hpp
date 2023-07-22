#pragma once

#include <madrona/types.hpp>

namespace GPUHideSeek {

namespace consts {
// Each random world is composed of a fixed number of rooms that the agents
// must solve in order to maximize their reward.
inline constexpr madrona::CountT numRooms = 4;

// Generated levels assume 2 agents
inline constexpr madrona::CountT numAgents = 2;

// Maximum number of interactive objects per challenge room. This is needed
// in order to setup the fixed-size learning tensors appropriately.
inline constexpr madrona::CountT maxEntitiesPerRoom = 3;

// Various world / entity size parameters
inline constexpr float worldLength = 60.f;
inline constexpr float worldWidth = 20.f;
inline constexpr float wallWidth = 1.f;
inline constexpr float buttonWidth = 0.8f;
inline constexpr float agentRadius = 1.f;
inline constexpr float roomLength = worldLength / numRooms;

// Units of distance along the environment needed for further reward
constexpr float distancePerProgress = 4.f;

// How many discrete options for each movement action
inline constexpr madrona::CountT numMoveBuckets = 5;

// Number of lidar samples, arranged in circle around agent
inline constexpr madrona::CountT numLidarSamples = 30;


}


}
