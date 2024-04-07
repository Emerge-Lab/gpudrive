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

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = (uint32_t)num_worlds,
        .autoReset = false,
        .jsonPath = "/home/aarav/nocturne_data/formatted_json_v2_no_tl_valid",
        .params = {
            .polylineReductionThreshold = 1.0,
            .observationRadius = 100.0,
            .rewardParams = {
                .rewardType = RewardType::Dense,
                .distanceToGoalThreshold = 0.5,
                .distanceToExpertThreshold = 0.5
            }
        }
    });

    std::random_device rd;
    std::mt19937 rand_gen(rd());
    std::uniform_real_distribution<float> acc_gen(-3.0,2.0);
    std::uniform_real_distribution<float> steer_gen(-0.7,0.7);

    auto start = std::chrono::system_clock::now();
    auto action_printer = mgr.actionTensor().makePrinter();
    auto model_printer = mgr.bicycleModelTensor().makePrinter();
    auto self_printer = mgr.selfObservationTensor().makePrinter();
    auto partner_obs_printer = mgr.partnerObservationsTensor().makePrinter();
    auto map_obs_printer = mgr.mapObservationTensor().makePrinter();
    auto shapePrinter = mgr.shapeTensor().makePrinter();
    auto rewardPrinter = mgr.rewardTensor().makePrinter();
    auto donePrinter = mgr.doneTensor().makePrinter();
    auto controlledStatePrinter = mgr.controlledStateTensor().makePrinter();
    auto agent_map_obs_printer = mgr.agentMapObservationsTensor().makePrinter();

    auto printObs = [&]() {
        // printf("Self\n");
        // self_printer.print();

        // printf("Actions\n");
        // action_printer.print();

        // printf("Model \n");
        // model_printer.print();

        // printf("Partner Obs\n");
        // partner_obs_printer.print();

        // printf("Map Obs\n");
        // map_obs_printer.print();
        // printf("\n");

        printf("Shape\n");
        shapePrinter.print();

        printf("Reward\n");
        rewardPrinter.print();

        printf("Done\n");
        donePrinter.print();

        printf("Controlled State\n");
        controlledStatePrinter.print();
        
        // printf("Agent Map Obs\n");
        // agent_map_obs_printer.print();
    };
    // printObs();

    auto worldToShape =
	mgr.getShapeTensorFromDeviceMemory(exec_mode, num_worlds);


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
        printObs();
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;

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
