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
    uint32_t num_worlds = 1;
    
    gpudrive::Manager mgr = gpudrive::Manager({
        .execMode = ExecMode::CPU,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .autoReset = false,
        .jsonPath = "testJsons",
        .params = {
            .polylineReductionThreshold = 0.0,
            .observationRadius = 100.0
        }
    });


    std::pair<float, float> mean = {0, 0};

    std::ifstream data = std::ifstream("testJsons/test.json");
    std::vector<std::vector<std::pair<float, float>>> roadGeoms;
    std::vector<float> roadTypes;

    void SetUp() override {
        json rawJson;
        data >> rawJson;
        mean = test_utils::calcMean(rawJson);

        std::cout<<"CTEST Mean x: "<<mean.first<<" Mean y: "<<mean.second<<std::endl;

        int64_t n_roads = 0;
        for (const auto &obj : rawJson["roads"]) {
            std::vector<std::pair<float, float>> roadGeom;
            for (const auto &point: obj["geometry"])
            {
                roadGeom.push_back({point["x"], point["y"]});
            }
            roadGeoms.push_back(roadGeom);

            if(obj["type"] == "road_edge")
            {
                roadTypes.push_back((float)gpudrive::EntityType::RoadEdge);
            }
            else if(obj["type"] == "road_line")
            {
                roadTypes.push_back((float)gpudrive::EntityType::RoadLine);
            }
            else if(obj["type"] == "lane")
            {
                roadTypes.push_back((float)gpudrive::EntityType::RoadLane);
            }
            else if(obj["type"] == "crosswalk")
            {
                roadTypes.push_back((float)gpudrive::EntityType::CrossWalk);
            }
            else if(obj["type"] == "speed_bump")
            {
                roadTypes.push_back((float)gpudrive::EntityType::SpeedBump);
            }
            else if(obj["type"] == "stop_sign")
            {
                roadTypes.push_back((float)gpudrive::EntityType::StopSign);
            }
            else if(obj["type"] == "invalid")
            {
                roadTypes.push_back(0);
            }
        }
    }
};

TEST_F(ObservationsTest, TestObservations) {
    auto obs = mgr.mapObservationTensor();
    auto flat_obs = test_utils::flatten_obs(obs);

    int64_t idx = 0;
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
