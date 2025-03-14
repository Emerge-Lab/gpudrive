#include "test_utils.hpp"

namespace test_utils
{
    std::vector<float> flatten_obs(const py::Tensor &obs)
    {
        int64_t num_elems = 1;
        for (int i = 0; i < obs.numDims(); i++)
        {
            num_elems *= obs.dims()[i];
        }
        float *ptr = static_cast<float *>(obs.devicePtr());
        std::vector<float> flattened;
        for (int i = 0; i < num_elems; i++)
        {
            flattened.push_back(static_cast<float>(ptr[i]));
        }
        return flattened;
    }

    std::pair<float, float> calcMean(const nlohmann::json &rawJson)
    {
        std::pair<float, float> mean = {0, 0};
        int64_t numEntities = 0;
        for (const auto &obj : rawJson["objects"])
        {
            int i = 0;
            for (const auto &pos : obj["position"])
            {
                if(obj["valid"][i++] == false)
                    continue;
                numEntities++;
                float newX = pos["x"];
                float newY = pos["y"];
                // Update mean incrementally
                mean.first += (newX - mean.first) / numEntities;
                mean.second += (newY - mean.second) / numEntities;
            }
        }
        for (const auto &obj : rawJson["roads"])
        {
            for (const auto &point : obj["geometry"])
            {
                numEntities++;
                float newX = point["x"];
                float newY = point["y"];

                // Update mean incrementally
                mean.first += (newX - mean.first) / numEntities;
                mean.second += (newY - mean.second) / numEntities;
            }
        }
        return mean;
    }
}