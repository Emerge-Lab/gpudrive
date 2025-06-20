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

    void from_json(const nlohmann::json &j, MapObject &obj)
    {
        obj.mean = {0,0};
        uint32_t i = 0;
        int numPositions = j.at("position").size();
        for (const auto &pos : j.at("position"))
        { 
            if (i < MAX_POSITIONS && i < numPositions)
            {
                from_json(pos, obj.position[i]);
                obj.mean.x += (obj.position[i].x - obj.mean.x)/(i+1);
                obj.mean.y += (obj.position[i].y - obj.mean.y)/(i+1);
                ++i;
            }
            else
            {
                break; // Avoid overflow
            }
        }
        obj.numPositions = i;
        j.at("width").get_to(obj.vehicle_size.width);
        j.at("length").get_to(obj.vehicle_size.length);
        j.at("height").get_to(obj.vehicle_size.height);
        j.at("id").get_to(obj.id);

        i = 0;
        for (const auto &h : j.at("heading"))
        {
            if (i < MAX_POSITIONS && i < numPositions)
            {
                h.get_to(obj.heading[i]);
                ++i;
            }
            else
            {
                break; // Avoid overflow
            }
        }
        obj.numHeadings = i;

        i = 0;
        for (const auto &v : j.at("velocity"))
        {
            if (i < MAX_POSITIONS && i < numPositions)
            {
                from_json(v, obj.velocity[i]);
                ++i;
            }
            else
            {
                break; // Avoid overflow
            }
        }
        obj.numVelocities = i;

        i = 0;
        for (const auto &v : j.at("valid"))
        {
            if (i < MAX_POSITIONS && i < numPositions)
            {
                v.get_to(obj.valid[i]);
                ++i;
            }
            else
            {
                break; // Avoid overflow
            }
        }
        obj.numValid = i;


        from_json(j.at("goalPosition"), obj.goalPosition);
        std::string type = j.at("type");
        if(type == "vehicle")
            obj.type = EntityType::Vehicle;
        else if(type == "pedestrian")
            obj.type = EntityType::Pedestrian;
        else if(type == "cyclist")
            obj.type = EntityType::Cyclist;
        else
            obj.type = EntityType::None;

        std::string markAsExpertKey = "mark_as_expert";
        if (j.contains(markAsExpertKey)) {
            from_json(j.at("mark_as_expert"), obj.markAsExpert);
        }

         // Initialize metadata fields to 0
        obj.metadata.isSdc = 0;
        obj.metadata.isObjectOfInterest = 0;
        obj.metadata.isTrackToPredict = 0;
        obj.metadata.difficulty = 0;
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

        
        std::vector<MapVector2> geometry_points_;
        for(const auto &point: j.at("geometry"))
        {
            MapVector2 p;
            from_json(point, p);
            geometry_points_.push_back(p);
        }

        const int64_t num_segments = j["geometry"].size() - 1;
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
                    auto point1 = geometry_points_[k * sample_every_n_];
                    auto point2 = geometry_points_[k_1 * sample_every_n_];
                    auto point3 = geometry_points_[k_2 * sample_every_n_];
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
            skip[0] = false;
            skip[num_sampled_points - 1] = false;
            std::vector<MapVector2> new_geometry_points; // This list stores the points that are not skipped
            while (k < num_sampled_points)
            {
                if (!skip[k])
                {
                    new_geometry_points.push_back(geometry_points_[k * sample_every_n_]); // Add the point to the list if it is not skipped
                }
                k++;
            }
            for (size_t i = 0; i < new_geometry_points.size(); i++)
            {
                if(i==MAX_GEOMETRY)
                    break;
                road.geometry[i] = new_geometry_points[i]; // Create the road lines
            }
            road.numPoints = new_geometry_points.size();
        }
        else
        {
            for (int64_t i = 0; i < num_sampled_points ; ++i)
            {
                if(i==MAX_GEOMETRY)
                    break;
                road.geometry[i] = geometry_points_[i * sample_every_n_]; 
            }
            road.numPoints = num_sampled_points;
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
            map.objects[0].metadata.isSdc = 1;
            
            // Set additional metadata if needed
            int sdc_id = map.objects[0].id;
            if (tracks_to_predict_indices.find(sdc_index) != tracks_to_predict_indices.end()) {
                map.objects[0].metadata.isTrackToPredict = 1;
                // Find and set difficulty
                for (const auto& track : metadata.at("tracks_to_predict")) {
                    if (track.at("track_index").get<int>() == sdc_index) {
                        map.objects[0].metadata.difficulty = track.at("difficulty").get<int>();
                        break;
                    }
                }
            }
            if (objects_of_interest_ids.find(sdc_id) != objects_of_interest_ids.end()) {
                map.objects[0].metadata.isObjectOfInterest = 1;
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
                map.objects[idx].metadata.isTrackToPredict = 1;
                
                // Find and set difficulty
                for (const auto& track : metadata.at("tracks_to_predict")) {
                    if (track.at("track_index").get<int>() == static_cast<int>(i)) {
                        map.objects[idx].metadata.difficulty = track.at("difficulty").get<int>();
                        break;
                    }
                }
                
                // Check if also object of interest
                if (objects_of_interest_ids.find(map.objects[idx].id) != objects_of_interest_ids.end()) {
                    map.objects[idx].metadata.isObjectOfInterest = 1;
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
                map.objects[idx].metadata.isObjectOfInterest = 1;
                
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
    }
}
