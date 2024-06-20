#include "mgr.hpp"
#include "consts.hpp"
#include "types.hpp"

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <string>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

using namespace madrona;

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

    Manager mgr({.execMode = exec_mode,
                 .gpuID = 0,
                 .numWorlds = (uint32_t)num_worlds,
                 .jsonPath = "tests/testJsons",
                 .params = {
                     .polylineReductionThreshold = 1.0,
                     .observationRadius = 100.0,
                     .rewardParams = {.rewardType = RewardType::DistanceBased,
                                      .distanceToGoalThreshold = 0.5,
                                      .distanceToExpertThreshold = 0.5},
                     .datasetInitOptions = DatasetInitOptions::ExactN,
                     .maxNumControlledVehicles = 0,
                 }});

    std::random_device rd;
    std::mt19937 rand_gen(rd());
    std::uniform_real_distribution<float> acc_gen(-3.0,2.0);
    std::uniform_real_distribution<float> steer_gen(-0.7,0.7);

    auto worldToShape = mgr.getShapeTensorFromDeviceMemory(num_worlds);

    const auto start = std::chrono::steady_clock::now();
    for (CountT i = 0; i < (CountT)num_steps; i++) {
        if (rand_actions) {
            for (CountT j = 0; j < (CountT)num_worlds; j++) {
	        auto agentCount = worldToShape.at(j).agentEntityCount;
                for (CountT k = 0; k < agentCount; k++) {
                    float acc = acc_gen(rand_gen);
                    float steer = steer_gen(rand_gen);
                    float head = 0;

                    mgr.setAction(j, k, acc, steer, head);

                    int64_t base_idx = j * num_steps * 2 * 3 + i * 2 * 3 + k * 3;
                    action_store[base_idx] = acc;
                    action_store[base_idx + 1] = steer;
                    action_store[base_idx + 2] = head;
                }
            }
        }
        mgr.step();
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    float fps = (double)num_steps * (double)num_worlds / elapsed.count();
    printf("FPS %f\n", fps);

    uint64_t totalAgentCount{0};
    for (CountT j = 0; j < (CountT)num_worlds; j++) {
      auto agentCount = worldToShape.at(j).agentEntityCount;
      totalAgentCount += agentCount;
    }

    float fpsNormalized = fps * totalAgentCount;
    printf("Agent-Normalized FPS %f\n", fpsNormalized);
}
