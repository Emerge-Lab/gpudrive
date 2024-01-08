#pragma once

#include "init.hpp"
#include <fstream>
#include <madrona/exec_mode.hpp>
#include <madrona/types.hpp>
#include <string>
#include <utility>
#include <vector>

namespace gpudrive {

struct AgentInit;

class MapReader {
public:
  static std::tuple<AgentInit *, madrona::CountT, RoadInit *, madrona::CountT>
  parseAndWriteOut(const std::string &path, madrona::ExecMode executionMode);

private:
  MapReader(const std::string &pathToFile);
  void doParse();

  std::ifstream in_;
  std::vector<AgentInit> agentInits_;
  std::vector<RoadInit> roadInits_;
};

} // namespace gpudrive
