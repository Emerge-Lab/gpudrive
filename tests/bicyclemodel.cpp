#include "gtest/gtest.h"
#include "consts.hpp"
#include "mgr.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>


using namespace madrona;
using nlohmann::json;

// TODO: Add the dynamic files here to be able to test from any json file.

class BicycleKinematicModelTest : public ::testing::Test {
protected:    
    madrona_gpudrive::Manager mgr = madrona_gpudrive::Manager({
        .execMode = ExecMode::CPU,
        .gpuID = 0,
        .scenes = {"testJsons/test.json"},
        .params = {
            .polylineReductionThreshold = 0.0,
            .observationRadius = 100.0,
            .collisionBehaviour = madrona_gpudrive::CollisionBehaviour::Ignore,
            .initOnlyValidAgentsAtFirstStep = false,
            .dynamicsModel = madrona_gpudrive::DynamicsModel::Classic
        }
    });
    
    uint32_t num_agents = madrona_gpudrive::consts::kMaxAgentCount;
    int64_t num_roads = madrona_gpudrive::consts::kMaxRoadEntityCount;
    int64_t num_steps = 10;
    int64_t num_worlds = 1;
    int64_t numEntities = 0;

    std::pair<float, float> mean = {0, 0};

    std::unordered_map<int64_t, float> agent_length_map;
    std::unordered_map<int64_t, float> agent_width_map;
    std::ifstream data = std::ifstream("testJsons/test.json");
    std::vector<float> initialState;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> acc_distribution;
    std::uniform_real_distribution<float> steering_distribution; 
    madrona::py::Tensor::Printer absolute_obs_printer =  mgr.absoluteSelfObservationTensor().makePrinter();
    void SetUp() override {
        json rawJson;
        data >> rawJson;
        mean = test_utils::calcMean(rawJson);
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
            agent_length_map[n_agents] = (float)obj["length"];
            agent_width_map[n_agents] = (float)obj["width"];
            initialState.push_back(float(obj["position"][0]["x"]) - mean.first);
            initialState.push_back(float(obj["position"][0]["y"]) - mean.second);
            float_t theta = obj["heading"][0];
            theta = theta > M_PI ? theta -  M_PI*2 : (theta < - M_PI ? theta +  M_PI*2 : theta);
            initialState.push_back(theta);
            initialState.push_back(math::Vector2{.x = obj["velocity"][0]["x"], .y = obj["velocity"][0]["y"]}.length());
            n_agents++;
        }
        acc_distribution = std::uniform_real_distribution<float>(-3.0, 2.0);
        steering_distribution = std::uniform_real_distribution<float>(-0.7, 0.7); 
        generator = std::default_random_engine(42);

        auto shape_tensor = mgr.shapeTensor();
        int32_t *ptr = static_cast<int32_t *>(shape_tensor.devicePtr());
        num_agents = 1;
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
    
    float speed_next = abs(speed_curr + acceleration * dt);
    return std::make_tuple(x_next, y_next, theta_next, speed_next);
}

std::pair<bool, std::string> validateBicycleModel(const py::Tensor &abs_obs, const py::Tensor &self_obs, const std::vector<float> &expected, const uint32_t num_agents)
{
    int64_t num_elems = 1;
    for (int i = 0; i < abs_obs.numDims(); i++)
    {
        num_elems *= abs_obs.dims()[i];
    }

    if (num_agents * madrona_gpudrive::AbsoluteSelfObservationExportSize > num_elems)
    {
        return {false, "Expected number of elements is less than the number of agents."};
    }

    num_elems = 1;
    for (int i = 0; i < self_obs.numDims(); i++)
    {
        num_elems *= self_obs.dims()[i];
    }

    if (num_agents * madrona_gpudrive::SelfObservationExportSize > num_elems)
    {
        return {false, "Expected number of elements is less than the number of agents."};
    }

    float *ptr = static_cast<float *>(abs_obs.devicePtr());

    for (int64_t i = 0, agent_idx = 0; i < num_agents * madrona_gpudrive::AbsoluteSelfObservationExportSize;)
    {
        auto x = static_cast<float>(ptr[i]);
        auto y = static_cast<float>(ptr[i + 1]);
        auto rot = static_cast<float>(ptr[i + 7]);
        auto x_exp = expected[agent_idx];
        auto y_exp = expected[agent_idx + 1];
        auto rot_exp = expected[agent_idx + 2];

        i += madrona_gpudrive::AbsoluteSelfObservationExportSize;
        agent_idx += 4;

        if (std::abs(x - x_exp) > test_utils::EPSILON || std::abs(y - y_exp) > test_utils::EPSILON || std::abs(rot - rot_exp) > test_utils::EPSILON)
        {
            return {false, "Value mismatch."};
        }
    }
    
    ptr = static_cast<float *>(self_obs.devicePtr());
    for (int64_t i = 0, agent_idx = 0; i < num_agents * madrona_gpudrive::SelfObservationExportSize;)
    {
        auto speed = static_cast<float>(ptr[i]);
        auto speed_exp = expected[agent_idx+3];

        if(std::abs(speed - speed_exp) > test_utils::EPSILON)
        {
            return {false, "Value mismatch."};
        }

        agent_idx += 4;
        i += madrona_gpudrive::SelfObservationExportSize;
    }

    return {true, ""};
}

std::vector<float> parseBicycleModel(const py::Tensor &abs_obs, const py::Tensor &self_obs, const uint32_t num_agents)
{
    std::vector<float> obs;
    obs.resize(num_agents * 4);
    float *ptr = static_cast<float *>(abs_obs.devicePtr());
    for (int i = 0, agent_idx = 0; i < num_agents * madrona_gpudrive::AbsoluteSelfObservationExportSize;)
    {
        obs[agent_idx] = static_cast<float>(ptr[i]);
        obs[agent_idx+1] = static_cast<float>(ptr[i+1]);
        obs[agent_idx+2] = static_cast<float>(ptr[i+7]);
        agent_idx += 4;
        i+=madrona_gpudrive::AbsoluteSelfObservationExportSize;
    }
    ptr = static_cast<float *>(self_obs.devicePtr());
    for (int i = 0, agent_idx = 0; i < num_agents * madrona_gpudrive::SelfObservationExportSize;)
    {
        obs[agent_idx+3] = static_cast<float>(ptr[i]);
        agent_idx += 4;
        i+=madrona_gpudrive::SelfObservationExportSize;
    }
    return obs;
}

TEST_F(BicycleKinematicModelTest, TestModelEvolution) {

    auto printObs = [&]() {
        printf("Absolute: \n");
        absolute_obs_printer.print();
        printf("\n");
    };

    auto printVector = [](const std::vector<float>& v) {
        std::cout << "Vector: \n";
        for(auto i : v) {
            std::cout << i << " ";
        }
        std::cout << "\n";
    };
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
    auto abs_obs = mgr.absoluteSelfObservationTensor();
    auto self_obs = mgr.selfObservationTensor();
    auto [valid, errorMsg] = validateBicycleModel(abs_obs, self_obs, initialState, num_agents);
    ASSERT_TRUE(valid);
    printObs();
    
    for(int i = 0; i < num_steps; i++)
    {
        expected.clear();
        printObs();
        auto prev_state = parseBicycleModel(abs_obs, self_obs, num_agents); // Due to floating point errors, we cannot use the expected values from the previous step so as not to accumulate errors.
        printVector(prev_state);
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
        abs_obs = mgr.absoluteSelfObservationTensor(); 
        self_obs = mgr.selfObservationTensor();
        std::tie(valid, errorMsg) = validateBicycleModel(abs_obs, self_obs, expected, num_agents);
        ASSERT_TRUE(valid);
    }

}
