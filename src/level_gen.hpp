#pragma once

#include "init.hpp"
#include "sim.hpp"

#include <madrona/types.hpp>

namespace gpudrive {

void createPersistentEntities(Engine &ctx, Map *map);

// First, destroys any non-persistent state for the current world and then
// generates a new play area.
void generateWorld(Engine &ctx);

}
