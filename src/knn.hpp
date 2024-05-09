#pragma once

#include "binary_heap.hpp"
#include "types.hpp"
#include <algorithm>
#include <madrona/math.hpp>
#include <madrona/types.hpp>

#ifndef MADRONA_GPU_MODE
#include <vector>
#endif

namespace {
bool cmp(const gpudrive::MapObservation &lhs, const gpudrive::MapObservation &rhs) {
  return lhs.position.length2() < rhs.position.length2();
}

void fillZeros(gpudrive::MapObservation *begin,
               gpudrive::MapObservation *beyond) {
  while (begin < beyond) {
    *begin++ =
        gpudrive::MapObservation{.position = {0, 0},
                                 .scale = madrona::math::Diag3x3{0, 0, 0},
                                 .heading = 0.f,
                                 .type = (float)gpudrive::EntityType::None};
  }
}

gpudrive::MapObservation
relativeObservation(const gpudrive::MapObservation &absoluteObservation,
                    const madrona::base::Rotation &referenceRotation,
                    const madrona::math::Vector2 &referencePosition) {
  auto relativePosition =
      madrona::math::Vector2{.x = absoluteObservation.position.x,
                             .y = absoluteObservation.position.y} -
      referencePosition;

  return gpudrive::MapObservation{
      .position = referenceRotation.inv()
                      .rotateVec({relativePosition.x, relativePosition.y, 0})
                      .xy(),
      .scale = absoluteObservation.scale,
      .heading = absoluteObservation.heading,
      .type = absoluteObservation.type};
}

bool isObservationsValid(gpudrive::Engine &ctx,
                         gpudrive::MapObservation *observations,
                         madrona::CountT K,
                         const madrona::base::Rotation &referenceRotation,
                         const madrona::math::Vector2 &referencePosition) {
#ifdef MADRONA_GPU_MODE
  return true;
#else
  const auto roadCount = ctx.data().numRoads;

  std::vector<gpudrive::MapObservation> sortedObservations;
  sortedObservations.reserve(roadCount);

  for (madrona::CountT roadIdx = 0; roadIdx < roadCount; ++roadIdx) {
    const auto &currentObservation =
        ctx.get<gpudrive::MapObservation>(ctx.data().roads[roadIdx]);
    sortedObservations.emplace_back(relativeObservation(
        currentObservation, referenceRotation, referencePosition));
  }

  std::sort(sortedObservations.begin(), sortedObservations.end(), cmp);
  std::sort(observations, observations + K, cmp);

  return std::equal(observations, observations + K, sortedObservations.begin(),
                    sortedObservations.begin() + K,
                    [](const gpudrive::MapObservation &lhs,
                       const gpudrive::MapObservation &rhs) {
                      return lhs.position.x == rhs.position.x &&
                             lhs.position.y == rhs.position.y;
                    });
#endif
}
} // namespace

namespace gpudrive {

template <madrona::CountT K>
void selectKNearestRoadEntities(Engine &ctx, const Rotation &referenceRotation,
                                const madrona::math::Vector2 &referencePosition,
                                gpudrive::MapObservation *heap) {
  const Entity *roads = ctx.data().roads;
  const auto roadCount = ctx.data().numRoads;

  for (madrona::CountT i = 0; i < std::min(roadCount, K); ++i) {
    heap[i] = relativeObservation(ctx.get<gpudrive::MapObservation>(roads[i]),
                                  referenceRotation, referencePosition);
  }

  if (roadCount < K) {
    fillZeros(heap + roadCount, heap + K);
    return;
  }

  make_heap(heap, heap + K, cmp);

  for (madrona::CountT roadIdx = K; roadIdx < roadCount; ++roadIdx) {
    auto currentObservation =
        relativeObservation(ctx.get<gpudrive::MapObservation>(roads[roadIdx]),
                            referenceRotation, referencePosition);

    const auto &kthNearestObservation = heap[0];
    bool isCurrentObservationCloser =
        cmp(currentObservation, kthNearestObservation);

    if (not isCurrentObservationCloser) {
      continue;
    }

    pop_heap(heap, heap + K, cmp);

    heap[K - 1] = currentObservation;
    push_heap(heap, heap + K, cmp);
  }

  assert(
      isObservationsValid(ctx, heap, K, referenceRotation, referencePosition));

  madrona::CountT newBeyond{K};
  {
    madrona::CountT idx{0};
    while (idx < newBeyond) {
      if (heap[idx].position.length() <= ctx.data().params.observationRadius) {
        ++idx;
        continue;
      }

      heap[idx] = heap[--newBeyond];
    }
  }

  fillZeros(heap + newBeyond, heap + K);
}

} // namespace gpudrive
