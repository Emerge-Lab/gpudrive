#pragma once

#include <fstream>
#include <madrona/exec_mode.hpp>
#include <simdjson.h>
#include "init.hpp"

namespace madrona_gpudrive {

struct AgentInit;

class MapReader {
public:
  static madrona_gpudrive::Map* parseAndWriteOut(const std::string &path, madrona::ExecMode executionMode, float polylineReductionThreshold);

private:
  static simdjson::dom::parser& getParser();
  MapReader(const std::string &pathToFile);
  ~MapReader();
  void doParse(float polylineReductionThreshold);

  std::string in_;
  madrona_gpudrive::Map *map_;
};

} // namespace madrona_gpudrive
