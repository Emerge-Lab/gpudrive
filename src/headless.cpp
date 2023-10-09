#include "mgr.hpp"

#include <cstdio>
#include <chrono>
#include <string>
#include <filesystem>
#include <fstream>
#include <random>
#include <iostream>

using namespace madrona;
using namespace madrona::viz;

[[maybe_unused]] static void saveWorldActions(
    const HeapArray<int32_t> &action_store,
    int32_t total_num_steps,
    int32_t world_idx)
{
    const int32_t *world_base = action_store.data() + world_idx * total_num_steps * 2 * 3;

    std::ofstream f("/tmp/actions", std::ios::binary);
    f.write((char *)world_base,
            sizeof(uint32_t) * total_num_steps * 2 * 3);
}


template <typename T>
bool validateTensor(const py::Tensor& tensor, const std::vector<T>& expected) {
    int64_t num_elems = 1;
    for (int i = 0; i < tensor.numDims(); i++) {
        num_elems *= tensor.dims()[i];
    }

    // Check if the sizes match
    if (num_elems != expected.size()) {
        std::cerr << "Size mismatch between tensor and expected values." << std::endl;
        return false;
    }

    if constexpr (std::is_same<T, int64_t>::value) {
        if (tensor.type() != py::Tensor::ElementType::Int64) {
            std::cerr << "Type mismatch: Expected Int64." << std::endl;
            return false;
        }
    } else if constexpr (std::is_same<T, float>::value) {
        if (tensor.type() != py::Tensor::ElementType::Float32) {
            std::cerr << "Type mismatch: Expected Float32." << std::endl;
            return false;
        }
    } // Add more types if needed

    switch (tensor.type()) {
        case py::Tensor::ElementType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(tensor.devicePtr());
            for (int64_t i = 0; i < num_elems; ++i) {
                if (ptr[i] != static_cast<int64_t>(expected[i])) {
                    return false;
                }
            }
            break;
        }
        case py::Tensor::ElementType::Float32: {
            float* ptr = static_cast<float*>(tensor.devicePtr());
            for (int64_t i = 0; i < num_elems; ++i) {
                if (ptr[i] != static_cast<float>(expected[i])) {
                    return false;
                }
            }
            break;
        }
        default:
            std::cerr << "Unhandled data type!";
            return false;
    }

    return true;
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

    HeapArray<int32_t> action_store(
        num_worlds * 2 * num_steps * 3);

    bool rand_actions = true;
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
    std::uniform_int_distribution<int32_t> act_rand(0, 4);

    auto start = std::chrono::system_clock::now();
    auto action_printer = mgr.actionTensor().makePrinter();
    auto model_printer = mgr.modelTensor().makePrinter();
    auto self_printer = mgr.selfObservationTensor().makePrinter();
    auto partner_printer = mgr.partnerObservationsTensor().makePrinter();
    auto room_ent_printer = mgr.roomEntityObservationsTensor().makePrinter();
    auto door_printer = mgr.doorObservationTensor().makePrinter();
    auto lidar_printer = mgr.lidarTensor().makePrinter();
    auto steps_remaining_printer = mgr.stepsRemainingTensor().makePrinter();
    auto reward_printer = mgr.rewardTensor().makePrinter();
    auto printObs = [&]() {
        printf("Self\n");
        self_printer.print();

        printf("Actions\n");
        action_printer.print();

        printf("Model \n");
        model_printer.print();

        printf("\n");
    };
    rand_actions = false;
    for (CountT i = 0; i < (CountT)num_steps; i++) {
        // if (rand_actions) {
        //     for (CountT j = 0; j < (CountT)num_worlds; j++) {
        //         for (CountT k = 0; k < 2; k++) {
        //             int32_t x = act_rand(rand_gen);
        //             int32_t y = act_rand(rand_gen);
        //             int32_t r = act_rand(rand_gen);

        //             mgr.setAction(j, k, x, y, r);
                    
        //             int64_t base_idx = j * num_steps * 2 * 3 + i * 2 * 3 + k * 3;
        //             action_store[base_idx] = x;
        //             action_store[base_idx + 1] = y;
        //             action_store[base_idx + 2] = r;
        //         }
        //     }
        // }
        mgr.setAction(0,0,1,1,1); // (world_idx, agent_idx, acceleration, steering, head_angle)
        mgr.setAction(0,1,1,1,1);
        printObs();
        mgr.step();
        printObs();
    }

    std::vector<float> expected = {809.614, -4799.32, 2.78503, -21.443, 790.831, -4794.45, 2.92268, -19.855};
    bool valid = validateTensor(mgr.modelTensor(), expected);
    printf("Model tensor validation: %s\n", valid ? "PASSED" : "FAILED");

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    float fps = (double)num_steps * (double)num_worlds / elapsed.count();
    printf("FPS %f\n", fps);
}
