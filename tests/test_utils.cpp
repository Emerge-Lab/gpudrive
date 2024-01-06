#include "test_utils.hpp"

namespace test_utils
{
    float degreesToRadians(float degrees)
    {
        return degrees * M_PI / 180.0;
    }

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
}