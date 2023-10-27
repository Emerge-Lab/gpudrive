#pragma once

#include "sim.hpp"

namespace gpudrive {

void createPersistentEntities(Engine &ctx, const std::string &path);

// First, destroys any non-persistent state for the current world and then
// generates a new play area.
void generateWorld(Engine &ctx);

}
