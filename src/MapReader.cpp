#include "MapReader.hpp"
#include "consts.hpp"
#include "init.hpp"

#include <algorithm>
#include <cassert>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

namespace {
template <typename T>
T *copyToArrayOnHostOrDevice(const std::vector<T> &in,
                             madrona::ExecMode hostOrDevice) {
  T *arr{nullptr};
  madrona::CountT len = sizeof(T) * in.size();

  if (hostOrDevice == madrona::ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
    arr = (T *)madrona::cu::allocGPU(len);
    cudaMemcpy(arr, in.data(), sizeof(T) * in.size(), cudaMemcpyHostToDevice);
#else
    FATAL("Madrona was not compiled with CUDA support");
#endif
  } else {
    assert(hostOrDevice == madrona::ExecMode::CPU);

    // This is a copy from CPU to CPU and can be avoided by extracting and
    // releasing in's backing array. For the sake of symmetry with the CUDA
    // scenario, we nevertheless opt to copy the data.
    arr = (T *)malloc(len);
    std::copy(in.cbegin(), in.cend(), arr);
  }

  return arr;
}
} // namespace

namespace gpudrive {

MapReader::MapReader(const std::string &pathToFile) : in_(pathToFile) {
  assert(in_.is_open());
}

void MapReader::doParse() {
  nlohmann::json rawJson;
  in_ >> rawJson;

  for (const auto &obj : rawJson["objects"]) {
    if (obj["type"] != "vehicle") {
      continue;
    }
    auto &agentInit = agentInits_.emplace_back();

    agentInit.xCoord = obj["position"][0]["x"];
    agentInit.yCoord = obj["position"][0]["y"];
    agentInit.length = obj["length"];
    agentInit.width = obj["width"];
    agentInit.heading = obj["heading"][0];
    agentInit.speedX = obj["velocity"][0]["x"];
    agentInit.speedY = obj["velocity"][0]["y"];
    agentInit.goalX = obj["goalPosition"]["x"];
    agentInit.goalY = obj["goalPosition"]["y"];
  }
  assert(agentInits_.size() <= consts::numAgents);

  for (const auto &obj : rawJson["roads"]) {
    auto &rawPoints = obj["geometry"];
    auto &roadInit = roadInits_.emplace_back();

    auto rawType = obj["type"];
    if (rawType == "road_edge" || rawType == "lane") {
      roadInit.type =
          rawType == "road_edge" ? RoadInitType::RoadEdge : RoadInitType::Lane;

      madrona::CountT idx;
      for (idx = 0; idx < static_cast<madrona::CountT>(rawPoints.size());
           ++idx) {
        assert(idx < consts::kMaxRoadGeometryLength);
        roadInit.points[idx] =
            madrona::math::Vector2{rawPoints[idx]["x"], rawPoints[idx]["y"]};
      }
      roadInit.numPoints = idx;

    } else if (rawType == "speed_bump") {
      roadInit.type = RoadInitType::SpeedBump;

      assert(rawPoints.size() == 4);
      roadInit.points[0] =
          madrona::math::Vector2{rawPoints[0]["x"], rawPoints[0]["y"]};
      roadInit.points[1] =
          madrona::math::Vector2{rawPoints[1]["x"], rawPoints[1]["y"]};
      roadInit.points[2] =
          madrona::math::Vector2{rawPoints[2]["x"], rawPoints[2]["y"]};
      roadInit.points[3] =
          madrona::math::Vector2{rawPoints[3]["x"], rawPoints[3]["y"]};

      roadInit.numPoints = 4;
    } else if (rawType == "stop_sign") {
      roadInit.type = RoadInitType::StopSign;

      assert(rawPoints.size() == 1);
      roadInit.points[0] =
          madrona::math::Vector2{rawPoints[0]["x"], rawPoints[0]["y"]};

      roadInit.numPoints = 1;
    } else {
      assert(false);
    }
  }
  assert(roadInits_.size() <= consts::numRoadSegments);
}

std::tuple<AgentInit *, madrona::CountT, RoadInit *, madrona::CountT>
MapReader::parseAndWriteOut(const std::string &path,
                            madrona::ExecMode executionMode) {
  MapReader reader(path);
  reader.doParse();

  auto agentInitsBuffer =
      copyToArrayOnHostOrDevice(reader.agentInits_, executionMode);
  auto roadInitsBuffer =
      copyToArrayOnHostOrDevice(reader.roadInits_, executionMode);

  return std::make_tuple(
      agentInitsBuffer, static_cast<madrona::CountT>(reader.agentInits_.size()),
      roadInitsBuffer, static_cast<madrona::CountT>(reader.roadInits_.size()));
}

} // namespace gpudrive
