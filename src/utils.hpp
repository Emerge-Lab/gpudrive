#pragma once

#include "types.hpp"
#include <cmath>
#include <madrona/components.hpp>
#include <madrona/mw_gpu_entry.hpp>

namespace gpudrive {
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

}}
