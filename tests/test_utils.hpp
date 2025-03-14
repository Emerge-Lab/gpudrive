#include "mgr.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace madrona;

namespace test_utils
{
    const float EPSILON = 0.001f; // Decreased epsilon to account for floating point errors. TODO: increase floating point precision and ideally use 1e-6 as epsilon.
    
    std::vector<float> flatten_obs(const py::Tensor &obs);

    std::pair<float, float> calcMean(const nlohmann::json &rawJson);
} // namespace utils
