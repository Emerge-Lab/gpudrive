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

        nb::enum_<FindRoadObservationsWith>(m, "FindRoadObservationsWith")
            .value("KNearestEntitiesWithRadiusFiltering", FindRoadObservationsWith::KNearestEntitiesWithRadiusFiltering)
            .value("AllEntitiesWithRadiusFiltering", FindRoadObservationsWith::AllEntitiesWithRadiusFiltering);

        // Define Parameters class
        nb::class_<Parameters>(m, "Parameters")
            .def(nb::init<>()) // Default constructor
            .def_rw("polylineReductionThreshold", &Parameters::polylineReductionThreshold)
            .def_rw("observationRadius", &Parameters::observationRadius)
            .def_rw("datasetInitOptions", &Parameters::datasetInitOptions)
            .def_rw("rewardParams", &Parameters::rewardParams)
            .def_rw("collisionBehaviour", &Parameters::collisionBehaviour)
            .def_rw("maxNumControlledVehicles", &Parameters::maxNumControlledVehicles)
            .def_rw("IgnoreNonVehicles", &Parameters::IgnoreNonVehicles)
            .def_rw("roadObservationAlgorithm", &Parameters::roadObservationAlgorithm)
            .def_rw("initOnlyValidAgentsAtFirstStep ", &Parameters::initOnlyValidAgentsAtFirstStep)
            .def_rw("enableLidar", &Parameters::enableLidar)
            .def_rw("disableClassicalObs", &Parameters::disableClassicalObs);

        // Define CollisionBehaviour enum
        nb::enum_<CollisionBehaviour>(m, "CollisionBehaviour")
            .value("AgentStop", CollisionBehaviour::AgentStop)
            .value("AgentRemoved", CollisionBehaviour::AgentRemoved)
            .value("Ignore", CollisionBehaviour::Ignore);

        nb::enum_<EntityType>(m, "EntityType")
            .value("_None", EntityType::None)
            .value("RoadEdge", EntityType::RoadEdge)
            .value("RoadLine", EntityType::RoadLine)
            .value("RoadLane", EntityType::RoadLane)
            .value("CrossWalk", EntityType::CrossWalk)
            .value("SpeedBump", EntityType::SpeedBump)
            .value("StopSign", EntityType::StopSign)
            .value("Vehicle", EntityType::Vehicle) 
            .value("Pedestrian", EntityType::Pedestrian)
            .value("Cyclist", EntityType::Cyclist)
            .value("Padding", EntityType::Padding) 
            .value("NumTypes", EntityType::NumTypes);

        // Bindings for Manager class
        nb::class_<Manager>(m, "SimManager")
            .def(
                "__init__", [](Manager *self, madrona::py::PyExecMode exec_mode, int64_t gpu_id, int64_t num_worlds, std::string jsonPath, Parameters params, bool enable_batch_renderer, uint32_t batch_render_view_width, uint32_t batch_render_view_height)
                { new (self) Manager(Manager::Config{
                      .execMode = exec_mode,
                      .gpuID = (int)gpu_id,
                      .numWorlds = (uint32_t)num_worlds,
                      .jsonPath = jsonPath,
                      .params = params,
                      .enableBatchRenderer = enable_batch_renderer,
                      .batchRenderViewWidth = batch_render_view_width,
                      .batchRenderViewHeight = batch_render_view_height});},
                nb::arg("exec_mode"),
                nb::arg("gpu_id"),
                nb::arg("num_worlds"),
                nb::arg("json_path"),
                nb::arg("params"),
                nb::arg("enable_batch_renderer") = false,
                nb::arg("batch_render_view_width") = 64,
                nb::arg("batch_render_view_height") = 64)
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
            .def("valid_state_tensor", &Manager::validStateTensor)
            .def("info_tensor", &Manager::infoTensor)
            .def("rgb_tensor", &Manager::rgbTensor)
            .def("depth_tensor", &Manager::depthTensor)
            .def("response_type_tensor", &Manager::responseTypeTensor);
    }

}
