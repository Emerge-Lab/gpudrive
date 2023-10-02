#pragma once

#include "sim.hpp"

namespace gpudrive {

void createPersistentEntities(Engine &ctx);

// Randomly generate a new world for a training episode
// First, destroys any non-persistent state for the current world and then
// generates a new play area.
void generateWorld(Engine &ctx);

}
