#include "gtest/gtest.h"
#include "consts.hpp"
#include "mgr.hpp"
#include <nlohmann/json.hpp>
#include "test_utils.hpp"

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

class BicycleKinematicModelTest : public ::testing::Test {
protected:
    gpudrive::Manager mgr = gpudrive::Manager({
        .execMode = ExecMode::CPU,
        .gpuID = 0,
        .numWorlds = 1,
        .autoReset = false,
    });
    
    int64_t num_agents = gpudrive::consts::numAgents;
    int64_t num_roads = gpudrive::consts::numRoadSegments;
    int64_t num_steps = 10;
    int64_t num_worlds = 1;
    int64_t numEntities = 0;

    std::pair<float, float> mean = {0, 0};

    std::unordered_map<int64_t, float> agent_length_map;
    std::ifstream data = std::ifstream("/home/aarav/gpudrive/nocturne_data/formatted_json_v2_no_tl_valid/tfrecord-00004-of-00150_246.json");
    std::vector<float> initialState;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> acc_distribution;
    std::uniform_real_distribution<float> steering_distribution; 
    void SetUp() override {
        json rawJson;
        data >> rawJson;
        for (const auto &obj : rawJson["objects"])
        {
            if (obj["type"] != "vehicle")
            {
                continue;
            }
            numEntities++;
            float newX = obj["position"][0]["x"];
            float newY = obj["position"][0]["y"];

            // Update mean incrementally
            mean.first += (newX - mean.first) / numEntities;
            mean.second += (newY - mean.second) / numEntities;
        }
        for (const auto &obj: rawJson["roads"])
        {
            for (const auto &point: obj["geometry"])
            {   
                numEntities++;
                float newX = point["x"];
                float newY = point["y"];

                // Update mean incrementally
                mean.first += (newX - mean.first) / numEntities;
                mean.second += (newY - mean.second) / numEntities;
            }
        }
        std::cout<<"CTEST Mean x: "<<mean.first<<" Mean y: "<<mean.second<<std::endl;
        int64_t n_agents = 0;
        for (const auto &obj : rawJson["objects"]) {
            if(n_agents == num_agents)
            {
                break;
            }
            if (obj["type"] != "vehicle") {
                continue;
            }
            initialState.push_back(float(obj["position"][0]["x"]) - mean.first);
            initialState.push_back(float(obj["position"][0]["y"]) - mean.second);
            initialState.push_back(degreesToRadians(obj["heading"][0]));
            initialState.push_back(math::Vector2{.x = obj["velocity"][0]["x"], .y = obj["velocity"][0]["y"]}.length());
            agent_length_map[n_agents++] = obj["length"];
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

// TEST_F(BicycleKinematicModelTest, TestModelEvolution) {
//     std::vector<float> expected;
//     //Check first step -
//     for(int i = 0; i < num_agents; i++)
//     {
//         auto [x_next, y_next, theta_next, speed_next] = StepBicycleModel(initialState[4*i], initialState[4*i+1], initialState[4*i+2], initialState[4*i+3], 0, 0, 0.1, agent_length_map[i]);
//         expected.push_back(x_next);
//         expected.push_back(y_next);
//         expected.push_back(theta_next);
//         expected.push_back(speed_next);
//     }
//     auto obs = mgr.bicycleModelTensor();
//     auto [valid, errorMsg] = test_utils::validateTensor(obs, initialState);
//     ASSERT_TRUE(valid);
    
//     for(int i = 0; i < num_steps; i++)
//     {
//         expected.clear();
//         auto prev_state = test_utils::flatten_obs(obs); // Due to floating point errors, we cannot use the expected values from the previous step so as not to accumulate errors.
//         for(int j = 0; j < num_agents; j++)
//         {
//             float acc =  acc_distribution(generator);
//             float steering = steering_distribution(generator);
//             mgr.setAction(0,j,acc,steering,0);
//             auto [x_next, y_next, theta_next, speed_next] = StepBicycleModel(prev_state[4*j], prev_state[4*j+1], prev_state[4*j+2], prev_state[4*j+3], acc, steering, 0.1, agent_length_map[j]);
//             expected.push_back(x_next);
//             expected.push_back(y_next);
//             expected.push_back(theta_next);
//             expected.push_back(speed_next);
//         }
//         mgr.step();
//         obs = mgr.bicycleModelTensor(); 
//         std::tie(valid, errorMsg) = test_utils::validateTensor(obs, expected);
//         ASSERT_TRUE(valid);
//     }

// }


