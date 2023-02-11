#pragma once

#include "sim.hpp"

namespace GPUHideSeek {

void generateEnvironment(Engine &ctx,
                         CountT level_id,
                         CountT num_hiders,
                         CountT num_seekers);

}
