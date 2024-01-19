#include "MapReader.hpp"
#include "json_serialization.hpp"

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

namespace {
gpudrive::Map *copyToArrayOnHostOrDevice(const gpudrive::Map *in,
                             madrona::ExecMode hostOrDevice) {
  gpudrive::Map *map = nullptr;

  if (hostOrDevice == madrona::ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
    map = static_cast<gpudrive::Map*>(madrona::cu::allocGPU(sizeof(gpudrive::Map)));
    cudaMemcpy(map, in, sizeof(gpudrive::Map), cudaMemcpyHostToDevice);
    auto error = cudaGetLastError();
    assert (error == cudaSuccess);
    
#else
    FATAL("Madrona was not compiled with CUDA support");
#endif
  } else {
    assert(hostOrDevice == madrona::ExecMode::CPU);

    // This is a copy from CPU to CPU and can be avoided by extracting and
    // releasing in's backing array. For the sake of symmetry with the CUDA
    // scenario, we nevertheless opt to copy the data.
    map = new gpudrive::Map();
    std::memcpy(map, in, sizeof(gpudrive::Map));
  }

  return map;
}
} // namespace

namespace gpudrive {

MapReader::MapReader(const std::string &pathToFile) : in_(pathToFile) {
  assert(in_.is_open());
  map_ = new gpudrive::Map();
}

MapReader::~MapReader() {
    delete map_;
}

void MapReader::doParse() {
  nlohmann::json rawJson;
  in_ >> rawJson;

  from_json(rawJson, *map_);
}

gpudrive::Map* MapReader::parseAndWriteOut(const std::string &path,
                            madrona::ExecMode executionMode) {
  MapReader reader(path);
  reader.doParse();

  return copyToArrayOnHostOrDevice(reader.map_, executionMode);

} 
} // namespace gpudrive