#pragma once

#include <fstream>
#include <madrona/exec_mode.hpp>
#include <tuple>

#include "init.hpp"

namespace gpudrive {

struct AgentInit;

class MapReader {
public:
  static std::tuple<gpudrive::Map*, std::pair<uint32_t, uint32_t>> parseAndWriteOut(const std::string &path, madrona::ExecMode executionMode, float polylineReductionThreshold);

private:
  MapReader(const std::string &pathToFile);
  ~MapReader();
  void doParse(float polylineReductionThreshold);

  std::ifstream in_;
  gpudrive::Map *map_;
};

} // namespace gpudrive
