#pragma once
#ifdef madrona_3d_example_mgr_EXPORTS
#define MGR_EXPORT MADRONA_EXPORT
#else
#define MGR_EXPORT MADRONA_IMPORT
#endif

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/render/mw.hpp>
#include <madrona/viz/system.hpp>

namespace madEscape {

// The Manager class encapsulates the linkage between the outside training
// code and the internal simulation state (src/sim.hpp / src/sim.cpp)
//
// Manager is responsible for initializing the simulator, loading physics
// and rendering assets off disk, and mapping ECS components to tensors
// for learning
class Manager {
public:
    struct Config {
        madrona::ExecMode execMode; // CPU or CUDA
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        bool autoReset; // Immediately generate new world on episode end
    };

    MGR_EXPORT Manager(
        const Config &cfg,
        const madrona::viz::VizECSBridge *viz_bridge = nullptr);
    MGR_EXPORT ~Manager();

    MGR_EXPORT void step();

    // These functions export Tensor objects that link the ECS
    // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
    MGR_EXPORT madrona::py::Tensor resetTensor() const;
    MGR_EXPORT madrona::py::Tensor actionTensor() const;
    MGR_EXPORT madrona::py::Tensor rewardTensor() const;
    MGR_EXPORT madrona::py::Tensor doneTensor() const;
    MGR_EXPORT madrona::py::Tensor selfObservationTensor() const;
    MGR_EXPORT madrona::py::Tensor partnerObservationsTensor() const;
    MGR_EXPORT madrona::py::Tensor roomEntityObservationsTensor() const;
    MGR_EXPORT madrona::py::Tensor doorObservationTensor() const;
    MGR_EXPORT madrona::py::Tensor lidarTensor() const;
    MGR_EXPORT madrona::py::Tensor stepsRemainingTensor() const;

    // These functions are used by the viewer to control the simulation
    // with keyboard inputs in place of DNN policy actions
    MGR_EXPORT void triggerReset(int32_t world_idx);
    MGR_EXPORT void setAction(int32_t world_idx,
                              int32_t agent_idx,
                              int32_t move_amount,
                              int32_t move_angle,
                              int32_t rotate,
                              int32_t grab);

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;
};

}
