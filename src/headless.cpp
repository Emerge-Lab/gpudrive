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

int main(int argc, char *argv[])
{
    using namespace gpudrive;

    if (argc < 3) {
        fprintf(stderr, "%s TYPE NUM_STEPS [--rand-actions]\n", argv[0]);
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

    uint64_t num_steps = std::stoul(argv[2]);
    std::vector<std::string> scenes = {"../data/processed/examples/tfrecord-00001-of-01000_307.json",
         "../data/processed/examples/tfrecord-00003-of-01000_109.json",
         "../data/processed/examples/tfrecord-00012-of-01000_389.json"};
    uint64_t num_worlds = scenes.size();

    bool rand_actions = false;
    if (argc >= 4) {
        if (std::string(argv[3]) == "--rand-actions") {
            rand_actions = true;
        }
    }

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .scenes = scenes,
        .params = {
            .polylineReductionThreshold = 1.0,
            .observationRadius = 100.0,
            .rewardParams = {
                .rewardType = RewardType::DistanceBased,
                .distanceToGoalThreshold = 0.5,
                .distanceToExpertThreshold = 0.5
            },
            .maxNumControlledAgents = 0,
        }
    });

    std::random_device rd;
    std::mt19937 rand_gen(rd());
    std::uniform_real_distribution<float> acc_gen(-3.0,2.0);
    std::uniform_real_distribution<float> steer_gen(-0.7,0.7);

    auto action_printer = mgr.actionTensor().makePrinter();
    auto self_printer = mgr.selfObservationTensor().makePrinter();
    auto partner_obs_printer = mgr.partnerObservationsTensor().makePrinter();
    auto map_obs_printer = mgr.mapObservationTensor().makePrinter();
    auto shapePrinter = mgr.shapeTensor().makePrinter();
    auto rewardPrinter = mgr.rewardTensor().makePrinter();
    auto donePrinter = mgr.doneTensor().makePrinter();
    auto controlledStatePrinter = mgr.controlledStateTensor().makePrinter();
    auto agent_map_obs_printer = mgr.agentMapObservationsTensor().makePrinter();
    auto info_printer = mgr.infoTensor().makePrinter();

    auto printObs = [&]() {
        printf("Self\n");
        self_printer.print();

        // printf("Actions\n");
        // action_printer.print();

        // printf("Partner Obs\n");
        // partner_obs_printer.print();

        // printf("Map Obs\n");
        // map_obs_printer.print();

        // printf("Shape\n");
        // shapePrinter.print();

        // printf("Reward\n");
        // rewardPrinter.print();

        // printf("Done\n");
        // donePrinter.print();

        // printf("Controlled State\n");
        // controlledStatePrinter.print();

        // printf("Agent Map Obs\n");
        // agent_map_obs_printer.print();

        // printf("Info\n");
        // info_printer.print();
    };

    auto worldToShape =
	mgr.getShapeTensorFromDeviceMemory();

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
                }
            }
        }
        mgr.step();
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    printObs();

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
