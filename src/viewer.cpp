#include <madrona/viz/viewer.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>
#include <optional>

using namespace madrona;
using namespace madrona::viz;

static inline float srgbToLinear(float srgb)
{
    if (srgb <= 0.04045f) {
        return srgb / 12.92f;
    }

    return powf((srgb + 0.055f) / 1.055f, 2.4f);
}

static inline math::Vector4 rgb8ToFloat(uint8_t r, uint8_t g, uint8_t b)
{
    return {
        srgbToLinear((float)r / 255.f),
        srgbToLinear((float)g / 255.f),
        srgbToLinear((float)b / 255.f),
        1.f,
    };
}

static HeapArray<float> readReplayLog(const char *path) {
    std::ifstream replay_log(path, std::ios::binary);
    replay_log.seekg(0, std::ios::end);
    int64_t size = replay_log.tellg();
    replay_log.seekg(0, std::ios::beg);

    HeapArray<float> log(size / sizeof(float));

    replay_log.read((char *)log.data(), (size / sizeof(float)) * sizeof(float));

    return log;
}

int main(int argc, char *argv[])
{
    using namespace gpudrive;

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

    std::optional<HeapArray<float>> replay_log;
    uint32_t cur_replay_step = 0;
    uint32_t num_replay_steps = 0;
    if (replay_log_path != nullptr) {
        replay_log = readReplayLog(replay_log_path);
        num_replay_steps = replay_log->size() / (num_worlds * num_views * 4);
    }

    std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;
    render_asset_paths[(size_t)SimObject::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_render.obj").string();
    render_asset_paths[(size_t)SimObject::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();
    render_asset_paths[(size_t)SimObject::StopSign] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::SpeedBump] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();

    std::array<const char *, (size_t)SimObject::NumObjects> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    auto materials = std::to_array<imp::SourceMaterial>({
        { rgb8ToFloat(191, 108, 10), -1, 0.8f, 1.0f },
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,},
        { math::Vector4{1.f, 1.f, 1.f, 0.0f}, 1, 0.5f, 1.0f,},
        { rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },
        { math::Vector4{0.5f, 0.3f, 0.3f, 0.0f},  0, 0.8f, 0.2f,},
        { rgb8ToFloat(230, 20, 20),   -1, 0.8f, 1.0f },
        { rgb8ToFloat(230, 230, 20),   -1, 0.8f, 1.0f },
        { rgb8ToFloat(255,0,0), -1, 0.8f, 1.0f},
        { rgb8ToFloat(0,0,0), -1, 0.8f, 0.2f}
    });

    // math::Quat initial_camera_rotation = math::Quat::angleAxis(0, math::up).normalize();
    math::Quat initial_camera_rotation =
            (math::Quat::angleAxis(0, math::up) *
            math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();

    Viewer viewer({
        .gpuID = 0,
        .renderWidth = 2730,
        .renderHeight = 1536,
        .numWorlds = num_worlds,
        .maxViewsPerWorld = num_views,
        .maxInstancesPerWorld = 450,
        .defaultSimTickRate = 20,
        .cameraMoveSpeed = 20.f,
        .cameraPosition = 20.f * math::up,
        .cameraRotation = initial_camera_rotation,
        .execMode = exec_mode,
    });

    // Override materials
    render_assets->objects[(CountT)SimObject::Cube].meshes[0].materialIDX = 0;
    render_assets->objects[(CountT)SimObject::Agent].meshes[0].materialIDX = 2;
    render_assets->objects[(CountT)SimObject::Agent].meshes[1].materialIDX = 3;
    render_assets->objects[(CountT)SimObject::Agent].meshes[2].materialIDX = 3;
    render_assets->objects[(CountT)SimObject::Plane].meshes[0].materialIDX = 4;
    render_assets->objects[(CountT)SimObject::StopSign].meshes[0].materialIDX = 7;
    render_assets->objects[(CountT)SimObject::SpeedBump].meshes[0].materialIDX = 8;
    // render_assets->objects[(CountT)SimObject::Cylinder].meshes[0].materialIDX = 7;

    viewer.loadObjects(render_assets->objects, materials, {
        { (std::filesystem::path(DATA_DIR) /
           "green_grid.png").string().c_str() },
        { (std::filesystem::path(DATA_DIR) /
           "smile.png").string().c_str() },
    });

    viewer.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .autoReset = replay_log.has_value(),
        .jsonPath = "../maps",
        .params = {
            .polylineReductionThreshold = 1.0,
            .observationRadius = 100.0,
        }
    }, viewer.rendererBridge());

    auto replayStep = [&]() {
        if (cur_replay_step == num_replay_steps - 1) {
            return true;
        }

        printf("Step: %u\n", cur_replay_step);

        for (uint32_t i = 0; i < num_worlds; i++) {
            for (uint32_t j = 0; j < num_views; j++) {
                uint32_t base_idx = 0;
                base_idx = 4 * (cur_replay_step * num_views * num_worlds +
                    i * num_views + j);

                auto acceleration = (*replay_log)[base_idx];
                auto steering = (*replay_log)[base_idx + 1];
                auto headAngle = (*replay_log)[base_idx + 2];


                printf("%d, %d: %f %f %f\n", i, j, acceleration, steering,
                       headAngle);
                mgr.setAction(i, j, acceleration, steering, headAngle);
            }
        }

        cur_replay_step++;

        return false;
    };

    auto self_printer = mgr.selfObservationTensor().makePrinter();
    auto partner_printer = mgr.partnerObservationsTensor().makePrinter();
    auto lidar_printer = mgr.lidarTensor().makePrinter();
    auto steps_remaining_printer = mgr.stepsRemainingTensor().makePrinter();
    auto reward_printer = mgr.rewardTensor().makePrinter();
    auto collisionPrinter = mgr.collisionTensor().makePrinter();

    auto printObs = [&]() {
        printf("Self\n");
        self_printer.print();

        printf("Partner\n");
        partner_printer.print();
        
        printf("Lidar\n");
        lidar_printer.print();

        printf("Steps Remaining\n");
        steps_remaining_printer.print();

        printf("Reward\n");
        reward_printer.print();

        printf("Collision\n");
        collisionPrinter.print();

        printf("\n");
    };

    viewer.loop([&mgr](CountT world_idx, CountT agent_idx,
                       const Viewer::UserInput &input) {
        using Key = Viewer::KeyboardKey;

        float steering{0};
        const float steeringDelta{math::pi / 8};

        float acceleration{0};
        const float accelerationDelta{1};

        if (input.keyPressed(Key::R)) {
            mgr.triggerReset(world_idx);
        }

        bool shift_pressed = input.keyPressed(Key::Shift);

        if (input.keyPressed(Key::W)) {
            acceleration += accelerationDelta;
        }
        if (input.keyPressed(Key::S)) {
            acceleration -= accelerationDelta;
        }

        if (input.keyPressed(Key::D)) {
            steering += steeringDelta;
        }
        if (input.keyPressed(Key::A)) {
            steering -= steeringDelta;
        }

        mgr.setAction(world_idx, agent_idx, acceleration, steering, 0);

    }, [&]() {
        if (replay_log.has_value()) {
            bool replay_finished = replayStep();

            if (replay_finished) {
                viewer.stopLoop();
            }
        }

        mgr.step();
        
        printObs();
    }, []() {});
}
