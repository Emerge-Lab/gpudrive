#include <madrona/viz/viewer.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>
#include <optional>

#include <iostream>

using namespace madrona;
using namespace madrona::viz;

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
    using namespace madrona_gpudrive;

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

    bool enable_batch_renderer =
#ifdef MADRONA_MACOS
        false;
#else
        true;
#endif

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("GPUDrive", 1920, 1080);
    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .scenes = {"../data/processed/examples/tfrecord-00001-of-01000_307.json"},
        .params = {
            .polylineReductionThreshold = 1.0,
            .observationRadius = 100.0,
            .maxNumControlledAgents = 0
        },
        .enableBatchRenderer = enable_batch_renderer,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
    });

    madrona::CountT stepCtr = 0;
    // math::Quat initial_camera_rotation = math::Quat::angleAxis(0, math::up).normalize();
    math::Quat initial_camera_rotation =
            (math::Quat::angleAxis(0, math::up) *
            math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();

    Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = 20,
        .cameraMoveSpeed = 20.f,
        .cameraPosition = 100.f * math::up,
        .cameraRotation = initial_camera_rotation,
    });

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

        printf("\n");
    };

    viewer.loop(
    [&mgr](CountT world_idx, const Viewer::UserInput &input) {

        using Key = Viewer::KeyboardKey;
        if (input.keyHit(Key::R)) {
            mgr.reset({(int)world_idx});
        }
        (void)world_idx;
    },
    [&mgr](CountT world_idx, CountT agent_idx,
            const Viewer::UserInput &input) {
        using Key = Viewer::KeyboardKey;

        float steering{0};
        const float steeringDelta{math::pi / 8};

        float acceleration{0};
        const float accelerationDelta{1};

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
        stepCtr++;

        if(stepCtr % consts::episodeLen == 0) {
            mgr.reset({0});
        }

        // printObs();
    }, []() {});
}
