#pragma once

#include <cmath>
#include <madrona/mw_gpu_entry.hpp>


namespace gpudrive {
namespace utils {

template <typename T> inline float NormalizeAngle(T angle) {
#ifdef MADRONA_GPU_MODE
  const T ret = fmod(angle, madrona::math::pi_m2);
#else
  const T ret = std::fmod(angle, madrona::math::pi_m2);
#endif

  return ret > madrona::math::pi ? ret - madrona::math::pi_m2 : (ret < -madrona::math::pi ? ret + madrona::math::pi_m2 : ret);
}

template <typename T> inline T AngleAdd(T lhs, T rhs) {
  return NormalizeAngle<T>(lhs + rhs);
}

}}
