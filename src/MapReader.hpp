#pragma once

#include <fstream>
#include <madrona/exec_mode.hpp>

#include "init.hpp"

namespace madrona_gpudrive {

struct AgentInit;

/**
 * @brief The MapReader class is responsible for parsing map data from a file
 * and preparing it for use in the simulation.
 *
 * This class provides a static method to parse a map file, process its
 * contents, and write the resulting data to a memory buffer that can be
 * used by the simulation.
 */
class MapReader {
public:
  /**
   * @brief Parses a map file and writes the data to a memory buffer.
   *
   * This function reads a map file, processes the polylines to reduce their
   * complexity, and then writes the processed data to a memory buffer that
   * can be used by the simulation. The memory layout of the buffer is
   * optimized for the specified execution mode (CPU or CUDA).
   *
   * @param path The path to the map file.
   * @param executionMode The execution mode (CPU or CUDA).
   * @param polylineReductionThreshold The threshold for polyline reduction.
   * @return A pointer to the Map object containing the processed map data.
   */
  static madrona_gpudrive::Map* parseAndWriteOut(const std::string &path, madrona::ExecMode executionMode, float polylineReductionThreshold);

private:
  MapReader(const std::string &pathToFile);
  ~MapReader();
  void doParse(float polylineReductionThreshold);

  std::ifstream in_;
  madrona_gpudrive::Map *map_;
};

} // namespace madrona_gpudrive
