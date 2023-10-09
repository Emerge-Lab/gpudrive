#include "gtest/gtest.h"
#include "consts.hpp"
#include "mgr.hpp"
#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>


using namespace madrona;
using nlohmann::json;


float degreesToRadians(float degrees) {
    return degrees * M_PI / 180.0;
}

// TODO: Add the dynamic files here to be able to test from any json file.

const float EPSILON = 0.00001f; // Define epsilon as a constant

class BicycleKinematicModelTest : public ::testing::Test {
protected:
    gpudrive::Manager mgr = gpudrive::Manager({
        .execMode = ExecMode::CPU,
        .gpuID = 0,
        .numWorlds = 1,
        .autoReset = false,
    });
    
    int64_t num_agents = gpudrive::consts::numAgents;
    int64_t num_steps = 10;
    int64_t num_worlds = 1;
    std::unordered_map<int64_t, float> agent_length_map;
    std::ifstream data = std::ifstream("/home/aarav/gpudrive/nocturne_data/formatted_json_v2_no_tl_valid/tfrecord-00100-of-00150_139.json");
    
    std::vector<float> initialState;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> acc_distribution;
    std::uniform_real_distribution<float> steering_distribution; 
    void SetUp() override {
        json rawJson;
        data >> rawJson;
        for (int i = 0; i < num_agents; i++) {
            auto& obj = rawJson["objects"][i];
            initialState.push_back(obj["position"][0]["x"]);
            initialState.push_back(obj["position"][0]["y"]);
            initialState.push_back(degreesToRadians(obj["heading"][0]));
            initialState.push_back(obj["velocity"][0]["x"]);
            agent_length_map[i] = obj["length"];
        }
        acc_distribution = std::uniform_real_distribution<float>(-3.0, 2.0);
        steering_distribution = std::uniform_real_distribution<float>(-0.7, 0.7); 
        generator = std::default_random_engine(42);
    }
};


std::tuple<float, float, float, float> StepBicycleModel(float x, float y, float theta, float speed_curr, float acceleration, float steering_action, float dt = 0.1, float L = 1) {

    float v = speed_curr + 0.5 * acceleration * dt; //Nocturne uses average speed

    float beta = atan(tan(steering_action) * (L/2) / L);

    float w = v * cos(beta) * tan(steering_action) / L;

    float x_next = x + v * cos(theta + beta) * dt;
    float y_next = y + v * sin(theta + beta) * dt;

    float theta_next = std::fmod(theta + w * dt, M_PI*2); // Clipping necessary to follow the implementation in madrona
    theta_next = theta_next > M_PI ? theta_next -  M_PI*2 : (theta_next < - M_PI ? theta_next +  M_PI*2 : theta_next);
    
    float speed_next = speed_curr + acceleration * dt;
    return std::make_tuple(x_next, y_next, theta_next, speed_next);
}

template <typename T>
std::pair<bool, std::string> validateTensor(const py::Tensor& tensor, const std::vector<T>& expected) {
    int64_t num_elems = 1;
    for (int i = 0; i < tensor.numDims(); i++) {
        num_elems *= tensor.dims()[i];
    }

    if (num_elems != expected.size()) {
        return {false, "Size mismatch between tensor and expected values."};
    }

    if constexpr (std::is_same<T, int64_t>::value) {
        if (tensor.type() != py::Tensor::ElementType::Int64) {
            return {false, "Type mismatch: Expected Int64."};
        }
    } else if constexpr (std::is_same<T, float>::value) {
        if (tensor.type() != py::Tensor::ElementType::Float32) {
             return {false, "Type mismatch: Expected Float32."};
        }
    } 

    switch (tensor.type()) {
        case py::Tensor::ElementType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(tensor.devicePtr());
            for (int64_t i = 0; i < num_elems; ++i) {
                if(std::abs(ptr[i] - static_cast<int64_t>(expected[i])) > EPSILON) {
                    return {false, "Value mismatch."};
                }
            }
            break;
        }
        case py::Tensor::ElementType::Float32: {
            float* ptr = static_cast<float*>(tensor.devicePtr());
            for (int64_t i = 0; i < num_elems; ++i) {
                auto orig = static_cast<float>(ptr[i]);
                auto exp = expected[i];
                if(std::abs(orig - exp) > EPSILON) {
                    return {false, "Value mismatch."};
                }
            }
            break;
        }
        default:
            return {false, "Unhandled data type!"};
    }

    return {true, ""};
}

std::vector<float> flatten_obs(const py::Tensor& obs) {
    int64_t num_elems = 1;
    for (int i = 0; i < obs.numDims(); i++) {
        num_elems *= obs.dims()[i];
    }
    float* ptr = static_cast<float*>(obs.devicePtr());
    std::vector<float> flattened;
    for(int i = 0 ; i < num_elems; i++)
    {
        flattened.push_back(static_cast<float>(ptr[i]));
    }
    return flattened;
}

TEST_F(BicycleKinematicModelTest, TestModelEvolution) {
    std::vector<float> expected;
    //Check first step -
    for(int i = 0; i < num_agents; i++)
    {
        auto [x_next, y_next, theta_next, speed_next] = StepBicycleModel(initialState[4*i], initialState[4*i+1], initialState[4*i+2], initialState[4*i+3], 0, 0, 0.1, agent_length_map[i]);
        expected.push_back(x_next);
        expected.push_back(y_next);
        expected.push_back(theta_next);
        expected.push_back(speed_next);
    }
    auto obs = mgr.bicycleModelTensor(); 
    auto [valid, errorMsg] = validateTensor(obs, initialState);
    ASSERT_TRUE(valid);
    
    for(int i = 0; i < num_steps; i++)
    {
        expected.clear();
        auto prev_state = flatten_obs(obs); // Due to floating point errors, we cannot use the expected values from the previous step so as not to accumulate errors.
        for(int j = 0; j < num_agents; j++)
        {
            float acc =  acc_distribution(generator);
            float steering = steering_distribution(generator);
            mgr.setAction(0,j,acc,steering,0);
            auto [x_next, y_next, theta_next, speed_next] = StepBicycleModel(prev_state[4*j], prev_state[4*j+1], prev_state[4*j+2], prev_state[4*j+3], acc, steering, 0.1, agent_length_map[j]);
            expected.push_back(x_next);
            expected.push_back(y_next);
            expected.push_back(theta_next);
            expected.push_back(speed_next);
        }
        mgr.step();
        obs = mgr.bicycleModelTensor(); 
        std::tie(valid, errorMsg) = validateTensor(obs, expected);
        ASSERT_TRUE(valid);
    }

}


