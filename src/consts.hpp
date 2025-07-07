#pragma once

#include <limits>
#include <madrona/math.hpp>
#include <madrona/types.hpp>

namespace madrona_gpudrive {

namespace consts {

inline constexpr madrona::CountT kMaxAgentCount = 64;
inline constexpr madrona::CountT kMaxRoadEntityCount = 10000;
inline constexpr madrona::CountT kMaxAgentMapObservationsCount = 200;

inline constexpr bool useEstimatedYaw = true;

inline constexpr float staticThreshold = 0.2f;

// Various world / entity size parameters
inline constexpr float worldLength = 40.f;

// This factor rescales the length of the vehicles by a tiny amount
// To account for the fact that noise occasionally puts vehicles into initial
// collisions. This is a dataset artifact that we are handling here like this.
inline constexpr float vehicleLengthScale = 0.7f;

// Each unit of distance forward (+ y axis) rewards the agents by this amount
inline constexpr float rewardPerDist = 0.05f;
// Each step that the agents don't make additional progress they get a small
// penalty reward
inline constexpr float slackReward = -0.005f;

// Steps per episode
inline constexpr int32_t episodeLen = 91;

// Number of lidar samples, arranged in circle around agent
inline constexpr madrona::CountT numLidarSamples = 50;

// These offsets define the offset in the z-axis to throw the lidar rays
// Different objects are on different heights in the world so as to show in the lidar rays
// In total we throw 3*numLidarSamples rays, 1 for the car, 1 for the road edge, 1 for the road line
inline constexpr float lidarCarOffset = 0.5f;
inline constexpr float lidarRoadEdgeOffset = 0.1f;
inline constexpr float lidarRoadLineOffset = -0.1f;
inline constexpr float lidarDistance = 200.f;
inline constexpr float lidarAngle = madrona::math::pi / 3; // The angle between the normal and the lidar ray in the extreme. By default we define a 120 degree view cone.

// Bev observation constants
inline constexpr int bev_rasterization_resolution = 200;

// Time (seconds) per step
inline constexpr float deltaT = 0.04f;

// Number of physics substeps
inline constexpr madrona::CountT numPhysicsSubsteps = 0.f;

inline constexpr float zDimensionScale = 1;
inline constexpr float xDimensionScaleRoadSegment = 1;

inline constexpr madrona::CountT kTrajectoryLength = 91; // Nocturne has 90 timesteps per episode. making it 91 as a buffer.

inline constexpr madrona::CountT kMaxRoadGeometryLength = 1810;

inline constexpr madrona::math::Vector3 kPaddingPosition = {
    -11000, -11000, std::numeric_limits<float>::max()};
}

}
