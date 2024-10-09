#pragma once

#include "sim.hpp"

namespace gpudrive {

void createPersistentEntities(Engine &ctx);

void resetWorld(Engine &ctx);

// Destroys all entities in the world
void destroyWorld(Engine &ctx);

}
