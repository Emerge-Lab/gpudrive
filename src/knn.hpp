#pragma once

#include "binary_heap.hpp"
#include "types.hpp"
#include <algorithm>
#include <madrona/math.hpp>
#include <madrona/types.hpp>
#include "utils.hpp"

#ifndef MADRONA_GPU_MODE
#include <vector>
#endif

namespace {
bool cmp(const madrona_gpudrive::MapObservation &lhs, const madrona_gpudrive::MapObservation &rhs) {
  return lhs.position.length2() < rhs.position.length2();
}

void fillZeros(madrona_gpudrive::MapObservation *begin,
               madrona_gpudrive::MapObservation *beyond) {
  while (begin < beyond) {
    *begin++ =
        madrona_gpudrive::MapObservation{.position = {0, 0},
                                 .scale = madrona::math::Diag3x3{0, 0, 0},
                                 .heading = 0.f,
                                 .type = (float)madrona_gpudrive::EntityType::None};
  }
}

madrona_gpudrive::MapObservation
relativeObservation(const madrona_gpudrive::MapObservation &absoluteObservation,
                    const madrona::base::Rotation &referenceRotation,
                    const madrona::math::Vector2 &referencePosition) {
  auto relativePosition =
      madrona::math::Vector2{.x = absoluteObservation.position.x,
                             .y = absoluteObservation.position.y} -
      referencePosition;

  return madrona_gpudrive::MapObservation{
      .position = referenceRotation.inv()
                      .rotateVec({relativePosition.x, relativePosition.y, 0})
                      .xy(),
      .scale = absoluteObservation.scale,
      .heading =  madrona_gpudrive::utils::quatToYaw(referenceRotation.inv() * madrona::math::Quat::angleAxis(absoluteObservation.heading,madrona::math::up)),
      .type = absoluteObservation.type};
}


bool isObservationsValid(madrona_gpudrive::Engine &ctx,
                         madrona_gpudrive::MapObservation *observations,
                         madrona::CountT K,
                         const madrona::base::Rotation &referenceRotation,
                         const madrona::math::Vector2 &referencePosition) {
#ifdef MADRONA_GPU_MODE
  return true;
#else
  const auto roadCount = ctx.data().numRoads;

  std::vector<madrona_gpudrive::MapObservation> sortedObservations;
  sortedObservations.reserve(roadCount);

  for (madrona::CountT roadIdx = 0; roadIdx < roadCount; ++roadIdx) {
    auto &road_iface = ctx.get<madrona_gpudrive::RoadInterfaceEntity>(ctx.data().roads[roadIdx]).e;
    const auto &currentObservation =
        ctx.get<madrona_gpudrive::MapObservation>(road_iface);
    sortedObservations.emplace_back(relativeObservation(
        currentObservation, referenceRotation, referencePosition));
  }

  std::sort(sortedObservations.begin(), sortedObservations.end(), cmp);
  std::sort(observations, observations + K, cmp);

  return std::equal(observations, observations + K, sortedObservations.begin(),
                    sortedObservations.begin() + K,
                    [](const madrona_gpudrive::MapObservation &lhs,
                       const madrona_gpudrive::MapObservation &rhs) {
                      return lhs.position.x == rhs.position.x &&
                             lhs.position.y == rhs.position.y;
                    });
#endif
}

madrona::CountT radiusFilter(madrona_gpudrive::MapObservation *heap, madrona::CountT K, float radius) {
  madrona::CountT newBeyond{K};

  madrona::CountT idx{0};
  while (idx < newBeyond) {
    if (heap[idx].position.length() <= radius) {
      ++idx;
      continue;
    }

    heap[idx] = heap[--newBeyond];
  }

  return newBeyond;
}

} // namespace

namespace madrona_gpudrive {

template <madrona::CountT K>
void selectKNearestRoadEntities(Engine &ctx, const Rotation &referenceRotation,
                                const madrona::math::Vector2 &referencePosition,
                                madrona_gpudrive::MapObservation *heap) {
  const Entity *roads = ctx.data().roads;
  const auto roadCount = ctx.data().numRoads;

  utils::ReferenceFrame referenceFrame(referencePosition, referenceRotation);

  for (madrona::CountT i = 0; i < std::min(roadCount, K); ++i) {
    heap[i] =
        referenceFrame.observationOf(ctx.get<madrona::base::Position>(roads[i]),
                                     ctx.get<madrona::base::Rotation>(roads[i]),
                                     ctx.get<madrona::base::Scale>(roads[i]),
                                     ctx.get<madrona_gpudrive::EntityType>(roads[i]),
                                     static_cast<float>(ctx.get<RoadMapId>(roads[i]).id),
                                     ctx.get<MapType>(roads[i]));
  }

  if (roadCount < K) {
    auto newBeyond = radiusFilter(heap, roadCount, ctx.data().params.observationRadius);
    fillZeros(heap + newBeyond, heap + K);
    return;
  }

  make_heap(heap, heap + K, cmp);

  for (madrona::CountT roadIdx = K; roadIdx < roadCount; ++roadIdx) {
    auto currentObservation = referenceFrame.observationOf(
        ctx.get<madrona::base::Position>(roads[roadIdx]),
        ctx.get<madrona::base::Rotation>(roads[roadIdx]),
        ctx.get<madrona::base::Scale>(roads[roadIdx]),
        ctx.get<madrona_gpudrive::EntityType>(roads[roadIdx]),
        static_cast<float>(ctx.get<RoadMapId>(roads[roadIdx]).id),
        ctx.get<MapType>(roads[roadIdx]));

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

  auto newBeyond = radiusFilter(heap, K, ctx.data().params.observationRadius);
  fillZeros(heap + newBeyond, heap + K);
}

} // namespace madrona_gpudrive
