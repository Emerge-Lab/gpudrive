#pragma once

#include "sim.hpp"

namespace GPUHideSeek {

// Creates the wall entities / generates room plan. Wall entities
// get destroyed during environment reset
void generateEnvironment(Engine &ctx);

// Just creates the agent entities - these just get created once
// and moved upon environment resets
void createAgents(Engine &ctx);

// Creates the floor
void createFloor(Engine &ctx);

static constexpr float BUTTON_WIDTH = 1.0f/22.0f;

}
