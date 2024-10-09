#pragma once

#include "sim.hpp"

namespace gpudrive {

void createPersistentEntities(Engine &ctx);

void resetWorld(Engine &ctx);
// First, destroys any non-persistent state for the current world and then
// generates a new play area.
void generateWorld(Engine &ctx);

}
