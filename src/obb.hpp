#pragma once

#include "utils.hpp"
#include <madrona/components.hpp>
#include <madrona/math.hpp>

namespace madrona_gpudrive {

// This code is based on
// https://www.flipcode.com/archives/2D_OBB_Intersection.shtml
struct OrientedBoundingBox2D {
  static OrientedBoundingBox2D from(const madrona::base::Position &position,
                                    const madrona::base::Rotation &rotation,
                                    const madrona::base::Scale &scale) {
    float theta{utils::quatToYaw(rotation)};
    madrona::math::Vector2 X{cosf(theta), sinf(theta)};
    madrona::math::Vector2 Y{-sinf(theta), cosf(theta)};

    X *= scale.d0;
    Y *= scale.d1;

    madrona::math::Vector2 center{.x = position.x, .y = position.y};

    OrientedBoundingBox2D obb;
    obb.corners[0] = center - X - Y;
    obb.corners[1] = center + X - Y;
    obb.corners[2] = center + X + Y;
    obb.corners[3] = center - X + Y;

    obb.updateAxes();

    return obb;
  }
  static bool hasCollided(const OrientedBoundingBox2D &obbA,
                          const OrientedBoundingBox2D &obbB) {
    return obbA.overlaps(obbB) && obbB.overlaps(obbA);
  }

  void updateAxes() {
    axes[0] = corners[1] - corners[0];
    axes[1] = corners[3] - corners[0];

    // Make the length of each axes 1/edge length so we know any
    // dot product must be less than 1 to fall within the edge.

    for (int a = 0; a < 2; ++a) {
      axes[a] /= axes[a].length2();
      origin[a] = corners[0].dot(axes[a]);
    }
  }
  bool overlaps(const OrientedBoundingBox2D &other) const {
    for (int a = 0; a < 2; ++a) {
      float t = other.corners[0].dot(axes[a]);

      // Find the extent of box 2 on axis a
      float tMin = t;
      float tMax = t;

      for (int c = 1; c < 4; ++c) {
        t = other.corners[c].dot(axes[a]);

        if (t < tMin) {
          tMin = t;
        } else if (t > tMax) {
          tMax = t;
        }
      }

      // We have to subtract off the origin

      // See if [tMin, tMax] intersects [0, 1]
      if ((tMin > 1 + origin[a]) || (tMax < origin[a])) {
        // There was no intersection along this dimension;
        // the boxes cannot possibly overlap.
        return false;
      }
    }

    // There was no dimension along which there is no intersection.
    // Therefore the boxes overlap.
    return true;
  }

  // Corners of OBB in the global coordinate system
  madrona::math::Vector2 corners[4];

  // Two edges of the box extended away from corner[0]
  madrona::math::Vector2 axes[2];

  // origin[a] = corner[0].dot(axis[a])
  float origin[2];
};

} // namespace madrona_gpudrive
