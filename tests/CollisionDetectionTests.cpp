#include "obb.hpp"
#include <gtest/gtest.h>
#include <madrona/components.hpp>
#include <madrona/math.hpp>

using namespace madrona;
using namespace madrona::base;
using namespace madrona::math;
using madrona_gpudrive::OrientedBoundingBox2D;

TEST(CollisionDetectionAxisAligned, Colliding) {
  Position origin{{.x = 0, .y = 0, .z = 0}};
  Position offset{{.x = 1, .y = 1, .z = 1}};

  Scale unitCubeScale{{.d0 = 1, .d1 = 1, .d2 = 1}};

  Rotation noRotation{Quat::angleAxis(0, madrona::math::up)};

  auto obbA = OrientedBoundingBox2D::from(origin, noRotation, unitCubeScale);
  auto obbB =
      OrientedBoundingBox2D::from(origin + offset, noRotation, unitCubeScale);

  EXPECT_TRUE(OrientedBoundingBox2D::hasCollided(obbA, obbB));
}

TEST(CollisionDetectionAxisAligned, NotColliding) {
  Position origin{{.x = 0, .y = 0, .z = 0}};
  Position offset{{.x = 2, .y = 2, .z = 0}};

  Scale unitCubeScale{{.d0 = 0.5, .d1 = 0.5, .d2 = 1}};

  Rotation noRotation{Quat::angleAxis(0, madrona::math::up)};

  auto obbA = OrientedBoundingBox2D::from(origin, noRotation, unitCubeScale);
  auto obbB =
      OrientedBoundingBox2D::from(origin + offset, noRotation, unitCubeScale);

  EXPECT_FALSE(OrientedBoundingBox2D::hasCollided(obbA, obbB));
}

TEST(CollisionDetectionAxisAligned, PointIntersection) {
  Position origin{{.x = 0, .y = 0, .z = 0}};
  Position offset{{.x = 1, .y = 1, .z = 0}};

  Scale unitCubeScale{{.d0 = 0.5, .d1 = 0.5, .d2 = 0.5}};

  Rotation noRotation{Quat::angleAxis(0, madrona::math::up)};

  auto obbA = OrientedBoundingBox2D::from(origin, noRotation, unitCubeScale);
  auto obbB =
      OrientedBoundingBox2D::from(origin + offset, noRotation, unitCubeScale);

  EXPECT_TRUE(OrientedBoundingBox2D::hasCollided(obbA, obbB));
}

TEST(CollisionDetectionAxisAligned, OneInsideOther) {
  Position origin{{.x = 0, .y = 0, .z = 0}};

  Scale unitCubeScale{{.d0 = 1, .d1 = 1, .d2 = 1}};
  Scale halfUnitCubeScale{{.d0 = 0.5, .d1 = 0.5, .d2 = 0.5}};

  Rotation noRotation{Quat::angleAxis(0, madrona::math::up)};

  auto obbA = OrientedBoundingBox2D::from(origin, noRotation, unitCubeScale);
  auto obbB =
      OrientedBoundingBox2D::from(origin, noRotation, halfUnitCubeScale);

  EXPECT_TRUE(OrientedBoundingBox2D::hasCollided(obbA, obbB));
}

TEST(CollisionDetectionNonAxisAligned, ExhaustiveRotations) {
  Position posA{{.x = 0, .y = 0, .z = 0}};
  Position posB{{.x = 0.5, .y = 0.5, .z = 0}};
  Scale unitCubeScale{{.d0 = 1, .d1 = 1, .d2 = 1}};
  Rotation rotA{Quat::angleAxis(0, madrona::math::up)};

  for (float degrees = 0; degrees < 360; degrees += 15) {
    Rotation rotB{Quat::angleAxis(toRadians(degrees), madrona::math::up)};

    auto obbA = OrientedBoundingBox2D::from(posA, rotA, unitCubeScale);
    auto obbB = OrientedBoundingBox2D::from(posB, rotB, unitCubeScale);

    EXPECT_TRUE(OrientedBoundingBox2D::hasCollided(obbA, obbB));
  }
}
