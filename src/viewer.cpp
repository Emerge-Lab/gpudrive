#include <madrona/viz/viewer.hpp>

#include "mgr.hpp"

#include <filesystem>
#include <fstream>

using namespace madrona;
using namespace madrona::viz;

HeapArray<int32_t> readReplayLog(const char *path)
{
    std::ifstream replay_log(path, std::ios::binary);
    replay_log.seekg(0, std::ios::end);
    int64_t size = replay_log.tellg();
    replay_log.seekg(0, std::ios::beg);

    HeapArray<int32_t> log(size / sizeof(int32_t));

    replay_log.read((char *)log.data(), (size / sizeof(int32_t)) * sizeof(int32_t));

    return log;
}

int main(int argc, char *argv[])
{
    using namespace GPUHideSeek;

    constexpr int64_t num_views = 2;

    uint32_t num_worlds = 1;
    if (argc >= 2) {
        num_worlds = (uint32_t)atoi(argv[1]);
    }

    ExecMode exec_mode = ExecMode::CPU;
    if (argc >= 3) {
        if (!strcmp("--cpu", argv[2])) {
            exec_mode = ExecMode::CPU;
        } else if (!strcmp("--cuda", argv[2])) {
            exec_mode = ExecMode::CUDA;
        }
    }

    const char *replay_log_path = nullptr;
    if (argc >= 4) {
        replay_log_path = argv[3];
    }

    auto replay_log = Optional<HeapArray<int32_t>>::none();
    uint32_t cur_replay_step = 0;
    uint32_t num_replay_steps = 0;
    if (replay_log_path != nullptr) {
        replay_log = readReplayLog(replay_log_path);
        num_replay_steps = replay_log->size() / (num_worlds * num_views * 3);
    }

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

    Viewer viewer({
        .gpuID = 0,
        .renderWidth = 2730,
        .renderHeight = 1536,
        .numWorlds = num_worlds,
        .maxViewsPerWorld = num_views,
        .maxInstancesPerWorld = 1000,
        .defaultSimTickRate = 10,
        .execMode = exec_mode,
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
        .execMode = exec_mode,
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

    auto replayStep = [&]() {
        if (cur_replay_step == num_replay_steps) {
            cur_replay_step = 0;
            for (uint32_t i = 0; i < num_worlds; i++) {
                mgr.triggerReset(i);
                mgr.step();
            }
        }

        printf("Step: %u\n", cur_replay_step);

        for (uint32_t i = 0; i < num_worlds; i++) {
            for (uint32_t j = 0; j < num_views; j++) {
                uint32_t base_idx = 0;
                base_idx = 3 * (cur_replay_step * num_views * num_worlds +
                    i * num_views + j);

                int32_t x = (*replay_log)[base_idx];
                int32_t y = (*replay_log)[base_idx + 1];
                int32_t r = (*replay_log)[base_idx + 2];

                mgr.setAction(i, j, x, y, r);
            }
        }

        cur_replay_step++;
    };

    viewer.loop([&mgr](CountT world_idx, CountT agent_idx,
                       const Viewer::UserInput &input) {
        using Key = Viewer::KeyboardKey;

        int32_t x = 2;
        int32_t y = 2;
        int32_t r = 2;

        if (input.keyPressed(Key::R)) {
            mgr.triggerReset(world_idx);
        }

        if (input.keyPressed(Key::W)) {
            y += 2;
        }
        if (input.keyPressed(Key::S)) {
            y -= 2;
        }

        if (input.keyPressed(Key::D)) {
            x += 2;
        }
        if (input.keyPressed(Key::A)) {
            x -= 2;
        }

        if (input.keyPressed(Key::Q)) {
            r += 2;
        }
        if (input.keyPressed(Key::E)) {
            r -= 2;
        }

        mgr.setAction(world_idx, agent_idx, x, y, r);
    }, [&mgr, &replay_log, &replayStep]() {
        if (replay_log.has_value()) {
            replayStep();
        }

        mgr.step();
    });
}
