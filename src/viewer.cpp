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

    Viewer viewer({
        .gpuID = 0,
        .renderWidth = 2730,
        .renderHeight = 1536,
        .numWorlds = 1,
        .maxViewsPerWorld = 6,
        .maxInstancesPerWorld = 1000,
        .execMode = ExecMode::CPU,
    });

    viewer.loadObjects(render_assets->objects);

    Manager mgr({
        .execMode = ExecMode::CPU,
        .gpuID = 0,
        .numWorlds = 1,
        .renderWidth = 0,
        .renderHeight = 0,
        .autoReset = false,
        .enableBatchRender = false,
        .debugCompile = false,
    }, viewer.rendererBridge());

    viewer.loop([&mgr]() {
        mgr.step();
    });
}
