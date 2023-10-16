#include "mgr.hpp"

#include <cstdio>
#include <chrono>
#include <string>
#include <filesystem>
#include <fstream>
#include <random>

using namespace madrona;
using namespace madrona::viz;

[[maybe_unused]] static void
saveWorldActions(const HeapArray<float> &action_store, int32_t total_num_steps,
                 int32_t world_idx) {
  const float *world_base =
      action_store.data() + world_idx * total_num_steps * 2 * 3;

  std::ofstream f("/tmp/actions", std::ios::binary);
  f.write((char *)world_base, sizeof(float) * total_num_steps * 2 * 3);
}

int main(int argc, char *argv[])
{
    using namespace gpudrive;

    if (argc < 4) {
        fprintf(stderr, "%s TYPE NUM_WORLDS NUM_STEPS [--rand-actions]\n", argv[0]);
        return -1;
    }
    std::string type(argv[1]);

    ExecMode exec_mode;
    if (type == "CPU") {
        exec_mode = ExecMode::CPU;
    } else if (type == "CUDA") {
        exec_mode = ExecMode::CUDA;
    } else {
        fprintf(stderr, "Invalid ExecMode\n");
        return -1;
    }

    uint64_t num_worlds = std::stoul(argv[2]);
    uint64_t num_steps = std::stoul(argv[3]);

    HeapArray<float> action_store(num_worlds * 2 * num_steps * 3);

    bool rand_actions = false;
    if (argc >= 5) {
        if (std::string(argv[4]) == "--rand-actions") {
            rand_actions = true;
        }
    }

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = (uint32_t)num_worlds,
        .autoReset = false,
    });

    std::random_device rd;
    std::mt19937 rand_gen(rd());
    std::uniform_real_distribution<float> act_rand(0, 4);

    auto start = std::chrono::system_clock::now();

    for (CountT i = 0; i < (CountT)num_steps; i++) {
        if (rand_actions) {
            for (CountT j = 0; j < (CountT)num_worlds; j++) {
                for (CountT k = 0; k < 2; k++) {
                  auto acceleration = act_rand(rand_gen);
                  auto steering = act_rand(rand_gen);
                  auto headAngle = act_rand(rand_gen);

                  mgr.setAction(j, k, acceleration, steering, headAngle);

                  int64_t base_idx = j * num_steps * 2 * 3 + i * 2 * 3 + k * 3;
                  action_store[base_idx] = acceleration;
                  action_store[base_idx + 1] = steering;
                  action_store[base_idx + 2] = headAngle;
                }
            }
        }
        mgr.step();
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    float fps = (double)num_steps * (double)num_worlds / elapsed.count();
    printf("FPS %f\n", fps);
}
