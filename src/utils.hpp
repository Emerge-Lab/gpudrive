#pragma once

#include "types.hpp"
#include <cmath>
#include <madrona/components.hpp>
#include <madrona/mw_gpu_entry.hpp>

namespace madrona_gpudrive {
namespace utils {

template <typename T> inline float NormalizeAngle(T angle) {
const T ret = fmod(angle, madrona::math::pi_m2);
  return ret > madrona::math::pi ? ret - madrona::math::pi_m2 : (ret < -madrona::math::pi ? ret + madrona::math::pi_m2 : ret);
}

template <typename T> inline T AngleAdd(T lhs, T rhs) {
  return NormalizeAngle<T>(lhs + rhs);
}

inline float quatToYaw(madrona::base::Rotation q) {
  // From
  // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_(in_3-2-1_sequence)_conversion
  return atan2(2.0f * (q.w * q.z + q.x * q.y),
               1.0f - 2.0f * (q.y * q.y + q.z * q.z));
}

class ReferenceFrame {
public:
  ReferenceFrame(const madrona::math::Vector2 &position,
                 const madrona::base::Rotation &rotation)
      : referenceRotation(rotation), referencePosition(position) {}

  madrona_gpudrive::MapObservation
  observationOf(const madrona::math::Vector3 &position,
                const madrona::base::Rotation &rotation, const Scale &scale,
                madrona_gpudrive::EntityType type, float id, MapType mapType = MapType::UNKNOWN) const {
    return madrona_gpudrive::MapObservation{.position = relative(position),
                                    .scale = scale,
                                    .heading = relative(rotation),
                                    .type = static_cast<float>(type),
                                    .id = static_cast<float>(id),
                                    .mapType = static_cast<float>(mapType)};
  }

  float distanceTo(const madrona::math::Vector3 &position) const {
    return relative(position).length();
  }

private:
  madrona::math::Vector2
  relative(const madrona::math::Vector3 &absolutePos) const {
    auto relativePosition = absolutePos.xy() - referencePosition;

    return referenceRotation.inv()
        .rotateVec({relativePosition.x, relativePosition.y, 0})
        .xy();
  }

  float relative(const madrona::base::Rotation &absoluteRot) const {
    return madrona_gpudrive::utils::quatToYaw(referenceRotation.inv() * absoluteRot);
  }

  madrona::math::Vector2 referencePosition;
  madrona::base::Rotation referenceRotation;
};
}}
