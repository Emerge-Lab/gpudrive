#include <madrona/viz/viewer.hpp>

#include "mgr.hpp"

#include <filesystem>

using namespace madrona;
using namespace madrona::viz;

int main(int argc, char *argv[])
{
    using namespace GPUHideSeek;

    (void)argc;
    (void)argv;

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk({
        (std::filesystem::path(DATA_DIR) / "sphere.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "plane.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "cylinder_render.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "ramp_render.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "elongated_render.obj").c_str(),
    }, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    uint32_t num_worlds = 2;

    Viewer viewer({
        .gpuID = 0,
        .renderWidth = 2730,
        .renderHeight = 1536,
        .numWorlds = num_worlds,
        .maxViewsPerWorld = 2,
        .maxInstancesPerWorld = 1000,
        .defaultSimTickRate = 10,
        .execMode = ExecMode::CPU,
    });

    viewer.loadObjects(render_assets->objects);

    Manager mgr({
        .execMode = ExecMode::CPU,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .renderWidth = 0,
        .renderHeight = 0,
        .autoReset = false,
        .enableBatchRender = false,
        .debugCompile = false,
    }, viewer.rendererBridge());

    for (CountT i = 0; i < (CountT)num_worlds; i++) {
        mgr.triggerReset(i);
    }

    mgr.step();

    viewer.loop([&mgr](CountT world_idx, CountT agent_idx,
                       const Viewer::UserInput &input) {
        using Key = Viewer::KeyboardKey;

        int32_t x = 0;
        int32_t y = 0;
        int32_t r = 0;

        if (input.keyPressed(Key::R)) {
            mgr.triggerReset(world_idx);
        }

        if (input.keyPressed(Key::W)) {
            y += 5;
        }
        if (input.keyPressed(Key::S)) {
            y -= 5;
        }

        if (input.keyPressed(Key::D)) {
            x += 5;
        }
        if (input.keyPressed(Key::A)) {
            x -= 5;
        }

        if (input.keyPressed(Key::Q)) {
            r += 5;
        }
        if (input.keyPressed(Key::E)) {
            r -= 5;
        }

        mgr.setAction(world_idx, agent_idx, x, y, r);
    }, [&mgr]() {
        mgr.step();
    });
}
