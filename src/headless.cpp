#include "mgr.hpp"

#include <cstdio>
#include <string>
#include <filesystem>

using namespace madrona;
using namespace madrona::viz;

int main(int argc, char *argv[])
{
    using namespace GPUHideSeek;

    if (argc < 3) {
        fprintf(stderr, "%s NUM_WORLDS NUM_STEPS\n", argv[0]);
        return -1;
    }
    uint64_t num_worlds = std::stoul(argv[1]);
    uint64_t num_steps = std::stoul(argv[2]);

    Manager mgr({
        .execMode = ExecMode::CPU,
        .gpuID = 0,
        .numWorlds = (uint32_t)num_worlds,
        .renderWidth = 0,
        .renderHeight = 0,
        .autoReset = false,
        .enableBatchRender = false,
    });

    for (CountT i = 0; i < (CountT)num_worlds; i++) {
        mgr.triggerReset(i);
    }

    for (CountT i = 0; i < (CountT)num_steps; i++) {
        mgr.step();
    }
}
