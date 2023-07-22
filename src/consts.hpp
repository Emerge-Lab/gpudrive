#pragma once

namespace GPUHideSeek {

namespace consts {
// Each environment is composed of numChallenges rooms that the agents
// must solve in order to maximize their reward.
inline constexpr madrona::CountT numChallenges = 4;

// Generated levels assume 2 agents
inline constexpr madrona::CountT numAgents = 2;

// Maximum number of interactive objects per challenge room
inline constexpr madrona::CountT maxEntitiesPerChallenge = 3;

// Various world / entity size parameters
inline constexpr float worldLength = 60.f;
inline constexpr float worldWidth = 20.f;
inline constexpr float wallWidth = 1.f;
inline constexpr float buttonWidth = 0.8f;
inline constexpr float agentRadius = 1.f;
inline constexpr float challengeLength =
    worldLength / numChallenges;


// Units of distance along the environment needed for further reward
constexpr float distancePerProgress = 4.f;

// How many discrete options for each movement action
inline constexpr madrona::CountT numMoveBuckets = 5;

// Number of lidar samples, arranged in circle around agent
inline constexpr madrona::CountT numLidarSamples = 30;


}


}
