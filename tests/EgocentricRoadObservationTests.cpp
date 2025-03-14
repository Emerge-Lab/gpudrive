#include "utils.hpp"
#include <gtest/gtest.h>

using namespace madrona;
using namespace madrona::base;
using namespace madrona::math;
using madrona_gpudrive::utils::ReferenceFrame;

TEST(EgocentricRoadObservationTests, Relative) {
  ReferenceFrame rf{Vector2{.x = 3, .y = 0},
                    Quat::angleAxis(toRadians(90), madrona::math::up)};

  auto mapObs =
      rf.observationOf(Vector3{.x = 3, .y = 3, .z = 0},
                       Quat::angleAxis(toRadians(270), madrona::math::up),
                       Scale{{.d0 = 10, .d1 = 0.1, .d2 = 0.1}},
                       static_cast<madrona_gpudrive::EntityType>(0), 0);

  EXPECT_LT(mapObs.position.x - 3, 0.000001);
  EXPECT_LT(mapObs.position.y - 0, 0.000001);
  EXPECT_EQ(mapObs.heading, toRadians(180));
}
