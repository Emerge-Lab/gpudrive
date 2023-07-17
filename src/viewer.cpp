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
        (std::filesystem::path(DATA_DIR) / "sphere.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "plane.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "cylinder_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "ramp_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "elongated_render.obj").string().c_str(),
    }, Span<char>(import_err.data(), import_err.size()));

    std::array<imp::SourceMaterial, 3> materials = {{
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 1.f, 0.f },
        { math::Vector4{1.0f, 0.1f, 0.1f, 0.0f}, -1, 1.f, 0.f },
        { math::Vector4{0.1f, 0.1f, 1.0f, 0.0f}, -1, 1.f, 0.f }
    }};

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

    const_cast<uint32_t&>(render_assets->objects[0].meshes[0].materialIDX) = 0;
    const_cast<uint32_t&>(render_assets->objects[1].meshes[0].materialIDX) = 0;
    const_cast<uint32_t&>(render_assets->objects[2].meshes[0].materialIDX) = 1;
    const_cast<uint32_t&>(render_assets->objects[3].meshes[0].materialIDX) = 0;
    const_cast<uint32_t&>(render_assets->objects[4].meshes[0].materialIDX) = 2;
    const_cast<uint32_t&>(render_assets->objects[5].meshes[0].materialIDX) = 0;
    const_cast<uint32_t&>(render_assets->objects[6].meshes[0].materialIDX) = 0;

    viewer.loadObjects(render_assets->objects, materials, {});

    viewer.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -1.5f}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });

    Manager mgr({
        .execMode = ExecMode::CPU,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .renderWidth = 0,
        .renderHeight = 0,
        .autoReset = false,
        .enableBatchRender = false,
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
