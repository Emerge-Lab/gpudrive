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


float degreesToRadians(float degrees) {
    return degrees * M_PI / 180.0;
}

class ObservationsTest : public ::testing::Test {
protected:    
    int64_t num_agents = 3;
    int64_t num_roads = 4500;
    
    gpudrive::Manager mgr = gpudrive::Manager({
        .execMode = ExecMode::CPU,
        .gpuID = 0,
        .numWorlds = 1,
        .autoReset = false,
        .jsonPath = "test.json",
        .params = {
            .polylineReductionThreshold = 0.0,
            .observationRadius = 100.0
        }
    });
    int64_t num_steps = 10;
    int64_t num_worlds = 1;
    int64_t numEntities = 0;

    std::pair<float, float> mean = {0, 0};

    std::unordered_map<int64_t, float> agent_length_map;
    std::ifstream data = std::ifstream("test.json");
    std::vector<std::vector<std::pair<float, float>>> roadGeoms;
    std::vector<float> roadTypes;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> acc_distribution;
    std::uniform_real_distribution<float> steering_distribution; 
    void SetUp() override {
        json rawJson;
        data >> rawJson;
        mean = test_utils::calcMean(rawJson);

        std::cout<<"CTEST Mean x: "<<mean.first<<" Mean y: "<<mean.second<<std::endl;

        int64_t n_roads = 0;
        roadGeoms.reserve(num_roads);
        for (const auto &obj : rawJson["roads"]) {
            std::vector<std::pair<float, float>> roadGeom;
            for (const auto &point: obj["geometry"])
            {
                roadGeom.push_back({point["x"], point["y"]});
            }
            roadGeoms.push_back(roadGeom);

            if(obj["type"] == "road_edge")
            {
                roadTypes.push_back(0);
            }
            else if(obj["type"] == "road_line")
            {
                roadTypes.push_back(1);
            }
            else if(obj["type"] == "lane")
            {
                roadTypes.push_back(2);
            }
            else if(obj["type"] == "crosswalk")
            {
                roadTypes.push_back(3);
            }
            else if(obj["type"] == "speed_bump")
            {
                roadTypes.push_back(4);
            }
            else if(obj["type"] == "stop_sign")
            {
                roadTypes.push_back(5);
            }
            else if(obj["type"] == "invalid")
            {
                roadTypes.push_back(6);
            }
        }
    }
};

TEST_F(ObservationsTest, TestObservations) {
    auto obs = mgr.mapObservationTensor();
    auto flat_obs = test_utils::flatten_obs(obs);

    int64_t idx = 4; // Skip the first 4 points which are garbage for some reason.
    for(int64_t i = 0; i < roadGeoms.size(); i++)
    {
        std::vector<std::pair<float, float>> roadGeom = roadGeoms[i];
        float roadType = roadTypes[i];
        for(int64_t j = 0; j < roadGeom.size() - 1; j++)
        {
            if(roadType > 2)
            {
                float x = (roadGeom[j].first + roadGeom[j+1].first + roadGeom[j+2].first + roadGeom[j+3].first)/4 - mean.first;
                float y = (roadGeom[j].second + roadGeom[j+1].second + roadGeom[j+2].second + roadGeom[j+3].second)/4 - mean.second;

                ASSERT_NEAR(flat_obs[idx], x, test_utils::EPSILON);
                ASSERT_NEAR(flat_obs[idx+1], y, test_utils::EPSILON);

                idx += 4;
                break;
            }

            float x1 = roadGeom[j].first - mean.first;
            float y1 = roadGeom[j].second - mean.second;
            float x2 = roadGeom[j+1].first - mean.first;
            float y2 = roadGeom[j+1].second - mean.second;
            float dx = (x2 + x1)/2;
            float dy = (y2 + y1)/2;

            ASSERT_NEAR(flat_obs[idx], dx, test_utils::EPSILON);
            ASSERT_NEAR(flat_obs[idx+1], dy, test_utils::EPSILON);
            ASSERT_FLOAT_EQ(flat_obs[idx+3], roadType) << "i = " << i << " j = " << j << " idx = " << idx;
            idx += 4;
        }
        if(idx >= flat_obs.size())
        {
            break;
        }
    }
}
