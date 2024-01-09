#pragma once

#include "consts.hpp"
#include "types.hpp"

#include <madrona/exec_mode.hpp>
#include <madrona/math.hpp>
#include <madrona/physics.hpp>
#include <madrona/types.hpp>

namespace madrona::viz {
struct VizECSBridge;
}

namespace gpudrive {

struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    madrona::phys::ObjectManager *rigidBodyObjMgr;
    const madrona::viz::VizECSBridge *vizBridge;
    gpudrive::Map* map;
};

}
