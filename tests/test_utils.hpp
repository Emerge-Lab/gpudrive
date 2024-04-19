#include "mgr.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace madrona;

namespace test_utils
{
    const float EPSILON = 0.001f; // Decreased epsilon to account for floating point errors. TODO: increase floating point precision and ideally use 1e-6 as epsilon.

    template <typename T>
    std::pair<bool, std::string> validateTensor(const py::Tensor &tensor, const std::vector<T> &expected, const uint32_t num_agents)
    {
        int64_t num_elems = 1;
        for (int i = 0; i < tensor.numDims(); i++)
        {
            num_elems *= tensor.dims()[i];
        }

        if (num_agents * gpudrive::BicycleModelExportSize > num_elems)
        {
            return {false, "Expected number of elements is less than the number of agents."};
        }

        if constexpr (std::is_same<T, int64_t>::value)
        {
            if (tensor.type() != py::TensorElementType::Int64)
            {
                return {false, "Type mismatch: Expected Int64."};
            }
        }
        else if constexpr (std::is_same<T, float>::value)
        {
            if (tensor.type() != py::TensorElementType::Float32)
            {
                return {false, "Type mismatch: Expected Float32."};
            }
        }

        switch (tensor.type())
        {
        case py::TensorElementType::Int64:
        {
            int64_t *ptr = static_cast<int64_t *>(tensor.devicePtr());
            for (int64_t i = 0; i < num_agents * gpudrive::BicycleModelExportSize; ++i)
            {
                if (std::abs(ptr[i] - static_cast<int64_t>(expected[i])) > EPSILON)
                {
                    return {false, "Value mismatch."};
                }
            }
            break;
        }
        case py::TensorElementType::Float32:
        {
            float *ptr = static_cast<float *>(tensor.devicePtr());
            for (int64_t i = 0; i < num_agents * gpudrive::BicycleModelExportSize; ++i)
            {
                auto orig = static_cast<float>(ptr[i]);
                auto exp = expected[i];
                if (std::abs(orig - exp) > EPSILON)
                {
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

    std::vector<float> flatten_obs(const py::Tensor &obs);

    float degreesToRadians(float degrees);

    std::pair<float, float> calcMean(const nlohmann::json &rawJson);
} // namespace utils
