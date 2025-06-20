#pragma once

#include "init.hpp"
#include "types.hpp"
#include "consts.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include <unordered_set>

namespace madrona_gpudrive
{
    void from_json(const nlohmann::json &j, MapVector2 &p)
    {
        p.x = j.at("x").get<float>();
        p.y = j.at("y").get<float>();
    }

    // Extract Z coordinate from position JSON
    float extract_z_from_position(const nlohmann::json &j)
    {
        if (j.contains("z")) {
            return j.at("z").get<float>();
        }
        return 0.0f; // Default z value if not provided
    }

    void from_json(const nlohmann::json &j, TrafficLightState &tl_state)
    {
        // Set number of states to the size of the state array
        size_t numStates = std::max(j.at("state").size(), static_cast<size_t>(consts::kTrajectoryLength));
        tl_state.numStates = numStates;
        static const std::unordered_map<std::string, TLState> state_map = {
                    {"unknown", TLState::Unknown},
                    {"stop", TLState::Stop},
                    {"caution", TLState::Caution},
                    {"go", TLState::Go}
        };

        // Process each timestep
        for (size_t t = 0; t < numStates; t++) {
            // Get the state string and convert to enum
            if (t < j.at("state").size()) {
                std::string state_str = j.at("state")[t];
                auto it = state_map.find(state_str);
                TLState enum_state = (it != state_map.end()) ? it->second : TLState::Unknown;  // ADD THIS LINE
                tl_state.state[t] = static_cast<float>(enum_state);  // Cast enum to float
            } else {
                tl_state.state[t] = static_cast<float>(TLState::Unknown);  // Cast enum to float
            }

            // Get the x,y,z positions in a more interpretable fashion
            if(t < j.at("x").size())
            {
                tl_state.x[t] = j.at("x")[t].get<float>();
            }
            else
            {
                tl_state.x[t] = -1000.0f;
            }
            if(t < j.at("y").size())
            {
                tl_state.y[t] = j.at("y")[t].get<float>();
            }
            else
            {
                tl_state.y[t] = -1000.0f;
            }
            if(t < j.at("z").size())
            {
                tl_state.z[t] = j.at("z")[t].get<float>();
            }
            else
            {
                tl_state.z[t] = -1000.0f;
            }

            // Get time index and lane id
            if (t < j.at("time_index").size()) {
                tl_state.timeIndex[t] = j.at("time_index")[t];
            } else {
                tl_state.timeIndex[t] = -1;
            }
        }

        tl_state.laneId = static_cast<int32_t>(j.at("lane_id")[0]);

        // Fill any remaining timesteps with default values
        for (size_t t = numStates; t < consts::kTrajectoryLength; t++) {
            tl_state.state[t] = static_cast<float>(TLState::Unknown);
            tl_state.timeIndex[t] = -1;
        }
    }

    void from_json(const nlohmann::json &j, MapObject &obj)
    {
        obj.mean = {0,0};
        uint32_t i = 0;

        float zPositions[10];

        // Check if position array exists and process it
        if (j.contains("position") && j.at("position").is_array()) {
            for (const auto &pos : j.at("position"))
            {
                if (i < MAX_POSITIONS)
                {
                    from_json(pos, obj.position[i]);
                    obj.mean.x += (obj.position[i].x - obj.mean.x)/(i+1);
                    obj.mean.y += (obj.position[i].y - obj.mean.y)/(i+1);

                    // Store z position separately
                    if (i < 10)
                    {
                        float z = extract_z_from_position(pos);
                        zPositions[i] = z;
                    }

                    ++i;
                }
                else
                {
                    break; // Avoid overflow
                }
            }
        }
        obj.numPositions = i;
        
        // Check if required fields exist before accessing
        if (j.contains("width")) {
            j.at("width").get_to(obj.vehicle_size.width);
        } else {
            obj.vehicle_size.width = 0.0f;
        }
        
        if (j.contains("length")) {
            j.at("length").get_to(obj.vehicle_size.length);
        } else {
            obj.vehicle_size.length = 0.0f;
        }
        
        if (j.contains("height")) {
            j.at("height").get_to(obj.vehicle_size.height);
        } else {
            obj.vehicle_size.height = 0.0f;
        }
        
        if (j.contains("id")) {
            j.at("id").get_to(obj.id);
        } else {
            obj.id = 0;
        }

        i = 0;
        if (j.contains("heading") && j.at("heading").is_array()) {
            for (const auto &h : j.at("heading"))
            {
                if (i < MAX_POSITIONS)
                {
                    h.get_to(obj.heading[i]);
                    ++i;
                }
                else
                {
                    break; // Avoid overflow
                }
            }
        }
        obj.numHeadings = i;

        i = 0;
        if (j.contains("velocity") && j.at("velocity").is_array()) {
            for (const auto &v : j.at("velocity"))
            {
                if (i < MAX_POSITIONS)
                {
                    from_json(v, obj.velocity[i]);
                    ++i;
                }
                else
                {
                    break; // Avoid overflow
                }
            }
        }
        obj.numVelocities = i;

        i = 0;
        if (j.contains("valid") && j.at("valid").is_array()) {
            for (const auto &v : j.at("valid"))
            {
                if (i < MAX_POSITIONS)
                {
                    v.get_to(obj.valid[i]);
                    ++i;
                }
                else
                {
                    break; // Avoid overflow
                }
            }
        }
        obj.numValid = i;

        if (j.contains("goalPosition")) {
            from_json(j.at("goalPosition"), obj.goalPosition);
        } else {
            obj.goalPosition = {0.0f, 0.0f};
        }
        
        if (j.contains("type")) {
            std::string type = j.at("type");
            if(type == "vehicle")
                obj.type = EntityType::Vehicle;
            else if(type == "pedestrian")
                obj.type = EntityType::Pedestrian;
            else if(type == "cyclist")
                obj.type = EntityType::Cyclist;
            else
                obj.type = EntityType::None;
        } else {
            obj.type = EntityType::None;
        }

        std::string markAsExpertKey = "mark_as_expert";
        if (j.contains(markAsExpertKey)) {
            from_json(j.at("mark_as_expert"), obj.markAsExpert);
        }

        // Initialize metadata fields to 0
        MetaData::zero(obj.metadata);

        // Calculate average z position from the first 10 positions
        // Store in the avgZ field of map object
        float avgZ = 0.0f;
        float avg_count = 0.0f;
        for (int zi = 0; zi < 10 && zi < obj.numValid; zi++) {
            // Only consider valid z positions
            if (zi < obj.numValid && obj.valid[zi]) {
                avgZ += zPositions[zi];
                avg_count += 1;
            }
        }
        if (avg_count == 0) {
            avgZ = 0.0f; // Avoid division by zero
        }
        else
        {
            avgZ /= avg_count; // Calculate average
        }
        obj.metadata.avgZ = avgZ;

        // Initialize VBD trajectories to zeros
        for (int i = 0; i < consts::kTrajectoryLength; i++) {
            for (int j = 0; j < 6; j++) {
                obj.vbd_trajectories[i][j] = 0.0f;
            }
        }

        // If VBD trajectories exist in the JSON, read them
        if (j.contains("vbd_trajectory") && j.at("vbd_trajectory").is_array()) {
            int vbd_idx = 0;
            for (const auto &vbd_traj : j.at("vbd_trajectory")) {
                if (vbd_idx < consts::kTrajectoryLength) {
                    if (!vbd_traj.is_null() && vbd_traj.is_array()) {
                        size_t traj_size = vbd_traj.size();
                        if (traj_size > 0) obj.vbd_trajectories[vbd_idx][0] = vbd_traj.at(0).get<float>();
                        if (traj_size > 1) obj.vbd_trajectories[vbd_idx][1] = vbd_traj.at(1).get<float>();
                        if (traj_size > 2) obj.vbd_trajectories[vbd_idx][2] = vbd_traj.at(2).get<float>();
                        if (traj_size > 3) obj.vbd_trajectories[vbd_idx][3] = vbd_traj.at(3).get<float>();
                        if (traj_size > 4) obj.vbd_trajectories[vbd_idx][4] = vbd_traj.at(4).get<float>();
                        if (traj_size > 5) obj.vbd_trajectories[vbd_idx][5] = vbd_traj.at(5).get<float>();
                    }
                }
                vbd_idx++;
            }
        }
    }

    void from_json(const nlohmann::json &j, MapRoad &road, float polylineReductionThreshold = 0.0)
    {
        road.mean = {0,0};
        std::string type = j.at("type");
         if(type == "road_edge")
            road.type = EntityType::RoadEdge;
        else if(type == "road_line")
            road.type = EntityType::RoadLine;
        else if(type == "lane")
            road.type = EntityType::RoadLane;
        else if(type == "crosswalk")
            road.type = EntityType::CrossWalk;
        else if(type == "speed_bump")
            road.type = EntityType::SpeedBump;
        else if(type == "stop_sign")
            road.type = EntityType::StopSign;
        else
            road.type = EntityType::None;

        // Check if geometry exists and is not empty
        if (!j.contains("geometry") || j.at("geometry").empty()) {
            road.numPoints = 0;
            return;
        }

        std::vector<MapVector2> geometry_points_;
        for(const auto &point: j.at("geometry"))
        {
            MapVector2 p;
            from_json(point, p);
            geometry_points_.push_back(p);
        }

        const int64_t geometry_size = static_cast<int64_t>(geometry_points_.size());
        if (geometry_size == 0) {
            road.numPoints = 0;
            return;
        }

        const int64_t num_segments = geometry_size - 1;
        const int64_t sample_every_n_ = 1;
        const int64_t num_sampled_points = (num_segments + sample_every_n_ - 1) / sample_every_n_ + 1;
        
        if (num_segments >= 10 && (road.type == EntityType::RoadLane || road.type == EntityType::RoadEdge || road.type == EntityType::RoadLine))
        {
            std::vector<bool> skip(num_sampled_points, false); // This list tracks the points that are skipped
            int64_t k = 0;
            bool skipChanged = true; // This is used to check if the skip list has changed in the last iteration
            while (skipChanged)      // This loop runs O(N^2) in worst case, but it is very fast in practice probably O(NlogN)
            {
                skipChanged = false; // Reset the skipChanged flag
                k = 0;
                while (k < num_sampled_points - 1)
                {
                    int64_t k_1 = k + 1; // k_1 is the next point that is not skipped
                    while (k_1 < num_sampled_points - 1 && skip[k_1])
                    {
                        k_1++; // Keep incrementing k_1 until we find a point that is not skipped
                    }
                    if (k_1 >= num_sampled_points - 1)
                        break;
                    int64_t k_2 = k_1 + 1;
                    while (k_2 < num_sampled_points && skip[k_2])
                    {
                        k_2++; // Keep incrementing k_2 until we find a point that is not skipped
                    }
                    if (k_2 >= num_sampled_points)
                        break;
                    
                    // Add bounds checking for geometry_points_ access
                    int64_t idx1 = k * sample_every_n_;
                    int64_t idx2 = k_1 * sample_every_n_;
                    int64_t idx3 = k_2 * sample_every_n_;
                    
                    if (idx1 >= geometry_size || idx2 >= geometry_size || idx3 >= geometry_size) {
                        break;
                    }
                    
                    auto point1 = geometry_points_[idx1];
                    auto point2 = geometry_points_[idx2];
                    auto point3 = geometry_points_[idx3];
                    float_t area = 0.5 * std::abs((point1.x - point3.x) * (point2.y - point1.y) - (point1.x - point2.x) * (point3.y - point1.y));
                    if (area < polylineReductionThreshold)
                    {                       // If the area is less than the threshold, then we skip the middle point
                        skip[k_1] = true;   // Mark the middle point as skipped
                        k = k_2;            // Skip the middle point and start from the next point
                        skipChanged = true; // Set the skipChanged flag to true
                    }
                    else
                    {
                        k = k_1; // If the area is greater than the threshold, then we don't skip the middle point and start from the next point
                    }
                }
            }

            // Create the road lines
            k = 0;
            if (num_sampled_points > 0) {
                skip[0] = false;
            }
            if (num_sampled_points > 1) {
                skip[num_sampled_points - 1] = false;
            }
            std::vector<MapVector2> new_geometry_points; // This list stores the points that are not skipped
            while (k < num_sampled_points)
            {
                int64_t idx = k * sample_every_n_;
                if (idx < geometry_size && !skip[k])
                {
                    new_geometry_points.push_back(geometry_points_[idx]); // Add the point to the list if it is not skipped
                }
                k++;
            }
            for (size_t i = 0; i < new_geometry_points.size(); i++)
            {
                if(i >= MAX_GEOMETRY)
                    break;
                road.geometry[i] = new_geometry_points[i]; // Create the road lines
            }
            road.numPoints = std::min(new_geometry_points.size(), static_cast<size_t>(MAX_GEOMETRY));
        }
        else
        {
            for (int64_t i = 0; i < num_sampled_points ; ++i)
            {
                if(i >= MAX_GEOMETRY)
                    break;
                int64_t idx = i * sample_every_n_;
                if (idx >= geometry_size) {
                    break;
                }
                road.geometry[i] = geometry_points_[idx];
            }
            road.numPoints = std::min(static_cast<size_t>(num_sampled_points), static_cast<size_t>(MAX_GEOMETRY));
        }

        if (j.contains("id")) {
            road.id = j.at("id").get<uint32_t>();
        }

        if (j.contains("map_element_id"))
        {
            auto mapElementId = j.at("map_element_id").get<int32_t>();

            if(mapElementId == 4 or mapElementId >= static_cast<int32_t>(MapType::NUM_TYPES) or mapElementId < -1)
            {
                road.mapType = MapType::UNKNOWN;
            }
            else
            {
                road.mapType = static_cast<MapType>(mapElementId);
            }
        }
        else
        {
            road.mapType = MapType::UNKNOWN;
        }

        for (int i = 0; i < road.numPoints; i++)
        {
            road.mean.x += (road.geometry[i].x - road.mean.x)/(i+1);
            road.mean.y += (road.geometry[i].y - road.mean.y)/(i+1);
        }

    }

    std::pair<float, float> calc_mean(const nlohmann::json &j)
    {
        std::pair<float, float> mean = {0, 0};
        int64_t numEntities = 0;
        for (const auto &obj : j["objects"])
        {
            int i = 0;
            for (const auto &pos : obj["position"])
            {
                if(obj["valid"][i++] == false)
                    continue;
                numEntities++;
                float newX = pos["x"];
                float newY = pos["y"];
                // Update mean incrementally
                mean.first += (newX - mean.first) / numEntities;
                mean.second += (newY - mean.second) / numEntities;
            }
        }
        for (const auto &obj : j["roads"])
        {
            for (const auto &point : obj["geometry"])
            {
                numEntities++;
                float newX = point["x"];
                float newY = point["y"];

                // Update mean incrementally
                mean.first += (newX - mean.first) / numEntities;
                mean.second += (newY - mean.second) / numEntities;
            }
        }
        return mean;
    }

    void from_json(const nlohmann::json &j, Map &map, float polylineReductionThreshold)
    {
        std::string name = j.at("name").get<std::string>();
        std::strncpy(map.mapName, name.c_str(), sizeof(map.mapName));

        std::string scenario_id = j.at("scenario_id").get<std::string>();
        std::strncpy(map.scenarioId, scenario_id.c_str(), sizeof(map.scenarioId));

        auto mean = calc_mean(j);
        map.mean = {mean.first, mean.second};
        map.numObjects = std::min(j.at("objects").size(), static_cast<size_t>(MAX_OBJECTS));

        const auto& metadata = j.at("metadata");
        int sdc_index = metadata.at("sdc_track_index").get<int>();

        // Create id to object index mapping
        std::unordered_map<int, size_t> idToObjIdx;
        size_t idx = 0;

        // First, identify which objects are tracks_to_predict and objects_of_interest
        std::unordered_set<int> tracks_to_predict_indices;
        std::unordered_set<int> objects_of_interest_ids;

        // Collect tracks_to_predict indices
        for (const auto& track : metadata.at("tracks_to_predict")) {
            int track_index = track.at("track_index").get<int>();
            if (track_index >= 0 && track_index < j.at("objects").size()) {
                tracks_to_predict_indices.insert(track_index);
            } else {
                std::cerr << "Warning: Invalid track_index " << track_index << " in scene " << j.at("name").get<std::string>() << std::endl;
            }
        }

        // Collect objects_of_interest IDs
        for (const auto& obj_id : metadata.at("objects_of_interest")) {
            objects_of_interest_ids.insert(obj_id.get<int>());
        }

        // Initialize SDC first if valid
        if (sdc_index >= 0 && sdc_index < j.at("objects").size()) {
            j.at("objects")[sdc_index].get_to(map.objects[0]);
            map.objects[0].metadata.isSdc = 1.0f;

            // Set additional metadata if needed
            int sdc_id = map.objects[0].id;
            if (tracks_to_predict_indices.find(sdc_index) != tracks_to_predict_indices.end()) {
                map.objects[0].metadata.isTrackToPredict = 1.0f;
                // Find and set difficulty
                for (const auto& track : metadata.at("tracks_to_predict")) {
                    if (track.at("track_index").get<int>() == sdc_index) {
                        map.objects[0].metadata.difficulty = static_cast<float>(track.at("difficulty").get<int>());
                        break;
                    }
                }
            }
            if (objects_of_interest_ids.find(sdc_id) != objects_of_interest_ids.end()) {
                map.objects[0].metadata.isObjectOfInterest = 1.0f;
            }

            idToObjIdx[sdc_id] = 0;
            idx = 1;

            // Remove SDC from sets to avoid double processing
            tracks_to_predict_indices.erase(sdc_index);
            objects_of_interest_ids.erase(sdc_id);
        }

        // Initialize tracks_to_predict objects (excluding SDC)
        for (size_t i = 0; i < j.at("objects").size() && idx < map.numObjects; i++) {
            if (i == sdc_index) continue; // Skip SDC as it's already initialized

            if (tracks_to_predict_indices.find(i) != tracks_to_predict_indices.end()) {
                j.at("objects")[i].get_to(map.objects[idx]);
                map.objects[idx].metadata.isTrackToPredict = 1.0f;

                // Find and set difficulty
                for (const auto& track : metadata.at("tracks_to_predict")) {
                    if (track.at("track_index").get<int>() == static_cast<int>(i)) {
                        map.objects[idx].metadata.difficulty = static_cast<float>(track.at("difficulty").get<int>());
                        break;
                    }
                }

                // Check if also object of interest
                if (objects_of_interest_ids.find(map.objects[idx].id) != objects_of_interest_ids.end()) {
                    map.objects[idx].metadata.isObjectOfInterest = 1.0f;
                    objects_of_interest_ids.erase(map.objects[idx].id);
                }

                idToObjIdx[map.objects[idx].id] = idx;
                idx++;
            }
        }

        // Initialize objects_of_interest (excluding those already processed)
        for (size_t i = 0; i < j.at("objects").size() && idx < map.numObjects; i++) {
            if (i == sdc_index) continue;

            int obj_id = j.at("objects")[i].at("id").get<int>();
            if (objects_of_interest_ids.find(obj_id) != objects_of_interest_ids.end()) {
                j.at("objects")[i].get_to(map.objects[idx]);
                map.objects[idx].metadata.isObjectOfInterest = 1.0f;

                idToObjIdx[map.objects[idx].id] = idx;
                idx++;
            }
        }

        // Initialize all remaining objects
        for (size_t i = 0; i < j.at("objects").size() && idx < map.numObjects; i++) {
            if (i == sdc_index) continue;

            int obj_id = j.at("objects")[i].at("id").get<int>();
            if (idToObjIdx.find(obj_id) == idToObjIdx.end()) { // Check if not already processed
                j.at("objects")[i].get_to(map.objects[idx]);
                idToObjIdx[map.objects[idx].id] = idx;
                idx++;
            }
        }

        // Process roads
        map.numRoads = std::min(j.at("roads").size(), static_cast<size_t>(MAX_ROADS));
        size_t countRoadPoints = 0;
        idx = 0;
        for (const auto &road : j.at("roads")) {
            if (idx >= map.numRoads)
                break;
            from_json(road, map.roads[idx], polylineReductionThreshold);
            size_t roadPoints = map.roads[idx].numPoints;
            countRoadPoints += (map.roads[idx].type <= EntityType::RoadLane) ? (roadPoints - 1) : 1;
            ++idx;
        }
        map.numRoadSegments = countRoadPoints;

        // Process traffic light states if present
        if (j.contains("tl_states")) {
            const auto& tl_states = j.at("tl_states");
            map.numTrafficLights = std::max(tl_states.size(), static_cast<size_t>(consts::kMaxTrafficLightCount));
            map.hasTrafficLights = (map.numTrafficLights > 0);

            size_t idx = 0;
            for (auto &kv : tl_states.items())
            {
                if (idx >= map.numTrafficLights)
                    break;
                kv.value().get_to(map.trafficLightStates[idx++]);
            }
        } else {
            map.numTrafficLights = 0;
            map.hasTrafficLights = false;
        }
    }
}
