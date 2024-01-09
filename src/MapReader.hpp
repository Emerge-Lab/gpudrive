#pragma once

#include "init.hpp"
#include <fstream>
#include <madrona/exec_mode.hpp>
#include <madrona/types.hpp>
#include <string>
#include <utility>
#include <vector>
#include "types.hpp"

namespace gpudrive {

struct AgentInit;

class MapReader {
public:
  static gpudrive::Map* parseAndWriteOut(const std::string &path, madrona::ExecMode executionMode);

private:
  MapReader(const std::string &pathToFile);
  ~MapReader();
  void doParse();

  std::ifstream in_;
  gpudrive::Map *map_;
};

} // namespace gpudrive
