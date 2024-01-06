#include "init.hpp"

namespace gpudrive{

madrona::CountT WorldInit::computeEntityUpperBound() const {
    madrona::CountT max_total_entities{agentInitsCount};
      
    for (madrona::CountT idx = 0; idx < roadInitsCount; ++idx) {
      const auto& roadInit = roadInits[idx];
      if (roadInit.type == RoadInitType::StopSign or roadInit.type == RoadInitType::SpeedBump) {
	++max_total_entities;
      } else {
	// Because level_gen.cpp:createRoadEntities() may skip over points which are too close together, this is effectively an over-estimate. If the logic to skip over road points was executed earlier, this could instead be an exact calculation.
	max_total_entities += roadInit.numPoints;
      }
    }

    return max_total_entities;
  }

}
