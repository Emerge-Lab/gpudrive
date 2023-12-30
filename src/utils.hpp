#pragma once

#include <cmath>
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

}}
