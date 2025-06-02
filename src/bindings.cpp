#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace madrona_gpudrive
{

    // This file creates the python bindings used by the learning code.
    // Refer to the nanobind documentation for more details on these functions.
    NB_MODULE(madrona_gpudrive, m)
    {
        // Each simulator has a madrona submodule that includes base types
        // like madrona::py::Tensor and madrona::py::PyExecMode.
        madrona::py::setupMadronaSubmodule(m);

        // Add bindings for constants defined in src/consts.hpp
        m.attr("kMaxAgentCount") = consts::kMaxAgentCount;
        m.attr("kMaxRoadEntityCount") = consts::kMaxRoadEntityCount;
        m.attr("kMaxAgentMapObservationsCount") = consts::kMaxAgentMapObservationsCount;
        m.attr("kTrajectoryLength") = consts::kTrajectoryLength;
        m.attr("episodeLen") = consts::episodeLen;
        m.attr("numLidarSamples") = consts::numLidarSamples;
        m.attr("vehicleScale") = consts::vehicleLengthScale;

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

        nb::enum_<FindRoadObservationsWith>(m, "FindRoadObservationsWith")
            .value("KNearestEntitiesWithRadiusFiltering", FindRoadObservationsWith::KNearestEntitiesWithRadiusFiltering)
            .value("AllEntitiesWithRadiusFiltering", FindRoadObservationsWith::AllEntitiesWithRadiusFiltering);

        // Define Parameters class
        nb::class_<Parameters>(m, "Parameters")
            .def(nb::init<>()) // Default constructor
            .def_rw("polylineReductionThreshold", &Parameters::polylineReductionThreshold)
            .def_rw("observationRadius", &Parameters::observationRadius)
            .def_rw("viewConeHalfAngle", &Parameters::viewConeHalfAngle)
            .def_rw("viewOccludeObjects", &Parameters::viewOccludeObjects)
            .def_rw("rewardParams", &Parameters::rewardParams)
            .def_rw("collisionBehaviour", &Parameters::collisionBehaviour)
            .def_rw("goalBehaviour", &Parameters::goalBehaviour)
            .def_rw("maxNumControlledAgents", &Parameters::maxNumControlledAgents)
            .def_rw("IgnoreNonVehicles", &Parameters::IgnoreNonVehicles)
            .def_rw("roadObservationAlgorithm", &Parameters::roadObservationAlgorithm)
            .def_rw("initOnlyValidAgentsAtFirstStep", &Parameters::initOnlyValidAgentsAtFirstStep)
            .def_rw("dynamicsModel", &Parameters::dynamicsModel)
            .def_rw("enableLidar", &Parameters::enableLidar)
            .def_rw("disableClassicalObs", &Parameters::disableClassicalObs)
            .def_rw("isStaticAgentControlled", &Parameters::isStaticAgentControlled)
            .def_rw("readFromTracksToPredict", &Parameters::readFromTracksToPredict)
            .def_rw("initSteps", &Parameters::initSteps)
            .def_rw("controlExperts", &Parameters::controlExperts);

        // Define CollisionBehaviour enum
        nb::enum_<CollisionBehaviour>(m, "CollisionBehaviour")
        .value("AgentStop", CollisionBehaviour::AgentStop)
        .value("AgentRemoved", CollisionBehaviour::AgentRemoved)
        .value("Ignore", CollisionBehaviour::Ignore);

        // Define GoalBehaviour enum
        nb::enum_<GoalBehaviour>(m, "GoalBehaviour")
        .value("Remove", GoalBehaviour::Remove)
        .value("Stop", GoalBehaviour::Stop)
        .value("Ignore", GoalBehaviour::Ignore);

        nb::enum_<DynamicsModel>(m, "DynamicsModel")
            .value("Classic", DynamicsModel::Classic)
            .value("InvertibleBicycle", DynamicsModel::InvertibleBicycle)
            .value("DeltaLocal", DynamicsModel::DeltaLocal)
            .value("State", DynamicsModel::State);

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
		 "__init__", [](Manager *self, madrona::py::PyExecMode exec_mode, int64_t gpu_id, std::vector<std::string> scenes, Parameters params, bool enable_batch_renderer, uint32_t batch_render_view_width, uint32_t batch_render_view_height)
                { new (self) Manager(Manager::Config{
                      .execMode = exec_mode,
                      .gpuID = (int)gpu_id,
                      .scenes = scenes,
                      .params = params,
                      .enableBatchRenderer = enable_batch_renderer,
                      .batchRenderViewWidth = batch_render_view_width,
                      .batchRenderViewHeight = batch_render_view_height});},
                nb::arg("exec_mode"),
                nb::arg("gpu_id"),
                nb::arg("scenes"),
                nb::arg("params"),
                nb::arg("enable_batch_renderer") = false,
                nb::arg("batch_render_view_width") = 64,
                nb::arg("batch_render_view_height") = 64)
            .def("step", &Manager::step)
            .def("reset", &Manager::reset)
            .def("action_tensor", &Manager::actionTensor)
            .def("reward_tensor", &Manager::rewardTensor)
            .def("done_tensor", &Manager::doneTensor)
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
            .def("bev_observation_tensor", &Manager::bevObservationTensor)
            .def("valid_state_tensor", &Manager::validStateTensor)
            .def("info_tensor", &Manager::infoTensor)
            .def("rgb_tensor", &Manager::rgbTensor)
            .def("depth_tensor", &Manager::depthTensor)
            .def("response_type_tensor", &Manager::responseTypeTensor)
            .def("expert_trajectory_tensor", &Manager::expertTrajectoryTensor)
            .def("set_maps", &Manager::setMaps)
            .def("world_means_tensor", &Manager::worldMeansTensor)
            .def("metadata_tensor", &Manager::metadataTensor)
            .def("vbd_trajectory_tensor", &Manager::vbdTrajectoryTensor)
            .def("map_name_tensor", &Manager::mapNameTensor)
            .def("deleteAgents", [](Manager &self, nb::dict py_agents_to_delete) {
                std::unordered_map<int32_t, std::vector<int32_t>> agents_to_delete;

                // Convert Python dict to C++ unordered_map
                for (auto item : py_agents_to_delete) {
                    int32_t key = nb::cast<int32_t>(item.first);
                    std::vector<int32_t> value = nb::cast<std::vector<int32_t>>(item.second);
                    agents_to_delete[key] = value;
                }

                self.deleteAgents(agents_to_delete);
            })
            .def("deleted_agents_tensor", &Manager::deletedAgentsTensor)
            .def("map_name_tensor", &Manager::mapNameTensor)
            .def("scenario_id_tensor", &Manager::scenarioIdTensor);
    }

}
