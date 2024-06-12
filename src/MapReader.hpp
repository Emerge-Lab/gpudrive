#pragma once

#include <fstream>
#include <madrona/exec_mode.hpp>
#include <madrona/types.hpp>
#include <vector>

#include "init.hpp"

namespace gpudrive {

struct AgentInit;

class MapReader {
public:
  static gpudrive::Map *
  parseAndWriteOut(const std::string &path, madrona::ExecMode executionMode,
                   float polylineReductionThreshold,
                   const std::vector<madrona::CountT> &vehiclesToSkip);

private:
  MapReader(const std::string &pathToFile);
  ~MapReader();
  void doParse(float polylineReductionThreshold,
               const std::vector<madrona::CountT> &vehiclesToSkip);

  std::ifstream in_;
  gpudrive::Map *map_;
};

} // namespace gpudrive
