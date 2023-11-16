#pragma once

#include "init.hpp"
#include "sim.hpp"

#include <madrona/types.hpp>

namespace gpudrive {

void createPersistentEntities(Engine &ctx, const AgentInit *agentInits,
                              madrona::CountT agentCount, RoadInit *roadInits,
                              madrona::CountT roadInitsCount);

// First, destroys any non-persistent state for the current world and then
// generates a new play area.
void generateWorld(Engine &ctx);

}
