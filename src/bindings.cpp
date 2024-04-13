#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace gpudrive
{

    // This file creates the python bindings used by the learning code.
    // Refer to the nanobind documentation for more details on these functions.
    NB_MODULE(gpudrive, m)
    {
        // Each simulator has a madrona submodule that includes base types
        // like madrona::py::Tensor and madrona::py::PyExecMode.
        madrona::py::setupMadronaSubmodule(m);

        // Define RewardType enum
        nb::enum_<RewardType>(m, "RewardType")
            .value("DistanceBased", RewardType::DistanceBased)
            .value("OnGoalAchieved", RewardType::OnGoalAchieved)
            .value("Dense", RewardType::Dense);

        // Define RewardParams class
        nb::class_<RewardParams>(m, "RewardParams")
            .def(nb::init<>()) // Default constructor
            .def_rw("rewardType", &RewardParams::rewardType)
            .def_rw("distanceToGoalThreshold", &RewardParams::distanceToGoalThreshold)
            .def_rw("distanceToExpertThreshold", &RewardParams::distanceToExpertThreshold);

        nb::enum_<DatasetInitOptions>(m, "DatasetInitOptions")
            .value("FirstN", DatasetInitOptions::FirstN)
            .value("RandomN", DatasetInitOptions::RandomN)
            .value("PadN", DatasetInitOptions::PadN)
            .value("ExactN", DatasetInitOptions::ExactN);

        // Define Parameters class
        nb::class_<Parameters>(m, "Parameters")
            .def(nb::init<>()) // Default constructor
            .def_rw("polylineReductionThreshold", &Parameters::polylineReductionThreshold)
            .def_rw("observationRadius", &Parameters::observationRadius)
            .def_rw("datasetInitOptions", &Parameters::datasetInitOptions)
            .def_rw("rewardParams", &Parameters::rewardParams)
            .def_rw("collisionBehaviour", &Parameters::collisionBehaviour)
            .def_rw("maxNumControlledVehicles", &Parameters::maxNumControlledVehicles);

        // Define CollisionBehaviour enum
        nb::enum_<CollisionBehaviour>(m, "CollisionBehaviour")
            .value("AgentStop", CollisionBehaviour::AgentStop)
            .value("AgentRemoved", CollisionBehaviour::AgentRemoved)
            .value("Ignore", CollisionBehaviour::Ignore);


        // Bindings for Manager class
        nb::class_<Manager>(m, "SimManager")
            .def(
                "__init__", [](Manager *self, madrona::py::PyExecMode exec_mode, int64_t gpu_id, int64_t num_worlds, bool auto_reset, std::string jsonPath, Parameters params, bool enable_batch_renderer = false)
                { new (self) Manager(Manager::Config{
                      .execMode = exec_mode,
                      .gpuID = (int)gpu_id,
                      .numWorlds = (uint32_t)num_worlds,
                      .autoReset = auto_reset,
                      .jsonPath = jsonPath,
                      .params = params,
                      .enableBatchRenderer = enable_batch_renderer}); },
                nb::arg("exec_mode"),
                nb::arg("gpu_id"),
                nb::arg("num_worlds"),
                nb::arg("auto_reset"),
                nb::arg("json_path"),
                nb::arg("params"),
                nb::arg("enable_batch_renderer"))
            .def("step", &Manager::step)
            .def("reset", &Manager::triggerReset)
            .def("reset_tensor", &Manager::resetTensor)
            .def("action_tensor", &Manager::actionTensor)
            .def("reward_tensor", &Manager::rewardTensor)
            .def("done_tensor", &Manager::doneTensor)
            .def("bicycle_model_tensor", &Manager::bicycleModelTensor)
            .def("self_observation_tensor", &Manager::selfObservationTensor)
            .def("map_observation_tensor", &Manager::mapObservationTensor)
            .def("partner_observations_tensor", &Manager::partnerObservationsTensor)
            .def("lidar_tensor", &Manager::lidarTensor)
            .def("steps_remaining_tensor", &Manager::stepsRemainingTensor)
            .def("shape_tensor", &Manager::shapeTensor)
            .def("controlled_state_tensor", &Manager::controlledStateTensor)
            .def("agent_roadmap_tensor", &Manager::agentMapObservationsTensor)
            .def("absolute_self_observation_tensor",
                 &Manager::absoluteSelfObservationTensor)
            .def("agent_roadmap_tensor", &Manager::agentMapObservationsTensor)
            .def("rgb_tensor", &Manager::rgbTensor)
            .def("depth_tensor", &Manager::depthTensor);
    }

}