#pragma once

#include <madrona/physics.hpp>

namespace GPUHideSeek {

struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    madrona::phys::ObjectManager *rigidBodyObjMgr;
    const madrona::viz::VizECSBridge *vizBridge;
};

}
