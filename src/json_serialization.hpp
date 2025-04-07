#pragma once

#include "init.hpp"
#include "types.hpp"
#include "consts.hpp"
#include <iostream>
#include <simdjson.h>
#include <madrona/json.hpp>

namespace madrona_gpudrive
{

    #define CHECK_JSON_ERROR(err) \
    if (err) { \
        std::cerr << "JSON error: " << simdjson::error_message(err) << std::endl; \
        abort(); \
    }

    // Helper function to safely get values with error checking
    template<typename T>
    T getValueOrDefault(simdjson::ondemand::object &obj, 
                        std::string_view key, 
                        T defaultValue) {
        T result;
        auto error = obj[key].get(result);
        if (error == simdjson::NO_SUCH_FIELD) {
            return defaultValue;
        }
        CHECK_JSON_ERROR(error);
        return result;
    }

    void from_json(const simdjson::dom::element &j, MapVector2 &p)
    {
        double x_val, y_val;
        CHECK_JSON_ERROR(j["x"].get(x_val));
        CHECK_JSON_ERROR(j["y"].get(y_val));

        p.x = static_cast<float>(x_val);
        p.y = static_cast<float>(y_val);
    }

     void from_json(simdjson::dom::element &j, MapObject &obj)
    {
        obj.mean = {0,0};
        uint32_t i = 0;

        // Parse positions array
        simdjson::dom::array positions;
        auto positions_error = j["position"].get_array().get(positions);
        CHECK_JSON_ERROR(positions_error);

        for (auto pos : positions) {
            if (i < MAX_POSITIONS) {
                from_json(pos, obj.position[i]);
                obj.mean.x += (obj.position[i].x - obj.mean.x)/(i+1);
                obj.mean.y += (obj.position[i].y - obj.mean.y)/(i+1);
                ++i;
            }
            else{
                break; // Avoid overflow
            }
        }
        obj.numPositions = i;
        // Get scalar values
        double width, length, height;
        int64_t id;

        CHECK_JSON_ERROR(j["width"].get(width));
        CHECK_JSON_ERROR(j["length"].get(length));
        CHECK_JSON_ERROR(j["height"].get(height));
        CHECK_JSON_ERROR(j["id"].get(id));

        obj.vehicle_size.width = static_cast<float>(width);
        obj.vehicle_size.length = static_cast<float>(length);
        obj.vehicle_size.height = static_cast<float>(height);
        obj.id = static_cast<uint32_t>(id);

        i = 0;
        simdjson::dom::array headings;
        auto headings_error = j["heading"].get_array().get(headings);
        CHECK_JSON_ERROR(headings_error);

        for (auto h : headings) {
            if (i < MAX_POSITIONS) {
                double heading_val;
                CHECK_JSON_ERROR(h.get(heading_val));
                obj.heading[i] = static_cast<float>(heading_val);
                ++i;
            }
            else
            {
                break; // Avoid overflow
            }
        }
        obj.numHeadings = i;

        i = 0;
        simdjson::dom::array velocities;
        auto velocities_error = j["velocity"].get_array().get(velocities);
        CHECK_JSON_ERROR(velocities_error);

        for (auto v : velocities) {
            if (i < MAX_POSITIONS) {
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
        simdjson::dom::array valid;
        auto valid_error = j["valid"].get_array().get(valid);
        CHECK_JSON_ERROR(valid_error);

        for (const auto &v : valid)
        {
            if (i < MAX_POSITIONS)
            {
                bool valid_val;
                CHECK_JSON_ERROR(v.get(valid_val));
                obj.valid[i] = valid_val;
                ++i;
            }
            else
            {
                break; // Avoid overflow
            }
        }
        obj.numValid = i;


        simdjson::dom::element goalPosition;
        auto goalPosition_error = j["goalPosition"].get(goalPosition);
        CHECK_JSON_ERROR(goalPosition_error);
        from_json(goalPosition, obj.goalPosition);

        std::string_view type_view;
        CHECK_JSON_ERROR(j["type"].get(type_view));
        if(type_view == "vehicle")
            obj.type = EntityType::Vehicle;
        else if(type_view == "pedestrian")
            obj.type = EntityType::Pedestrian;
        else if(type_view == "cyclist")
            obj.type = EntityType::Cyclist;
        else
            obj.type = EntityType::None;

        bool mark_as_expert;
        auto error = j["mark_as_expert"].get(mark_as_expert);
        if (error == simdjson::error_code::NO_SUCH_FIELD)
        {
            // Field "mark_as_expert" does not exist; do nothing.
        }
        else
        {
            CHECK_JSON_ERROR(error); // Make sure no other error occurred.
            obj.markAsExpert = mark_as_expert;
        }

         // Initialize metadata fields to 0
        obj.metadata.isSdc = 0;
        obj.metadata.isObjectOfInterest = 0;
        obj.metadata.isTrackToPredict = 0;
        obj.metadata.difficulty = 0;
    }

    void from_json(const simdjson::dom::element &j, MapRoad &road, float polylineReductionThreshold = 0.0)
    {
        road.mean = {0,0};

        std::string_view type_view;
        CHECK_JSON_ERROR(j["type"].get(type_view));
        if(type_view == "road_edge")
            road.type = EntityType::RoadEdge;
        else if(type_view == "road_line")
            road.type = EntityType::RoadLine;
        else if(type_view == "lane")
            road.type = EntityType::RoadLane;
        else if(type_view == "crosswalk")
            road.type = EntityType::CrossWalk;
        else if(type_view == "speed_bump")
            road.type = EntityType::SpeedBump;
        else if(type_view == "stop_sign")
            road.type = EntityType::StopSign;
        else
            road.type = EntityType::None;

        simdjson::dom::array geometry;
        auto geometry_error = j["geometry"].get_array().get(geometry);
        CHECK_JSON_ERROR(geometry_error);

        std::vector<MapVector2> geometry_points_;
        for(const auto &point: geometry)
        {
            MapVector2 p;
            from_json(point, p);
            geometry_points_.push_back(p);
        }

        const int64_t num_segments = geometry.size() - 1;
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

        
        int64_t road_id;
        auto error = j["id"].get(road_id);
        if (error != simdjson::error_code::NO_SUCH_FIELD) {
            CHECK_JSON_ERROR(error);
            road.id = static_cast<uint32_t>(road_id);
        }
        
        int64_t mapElementId;
        auto mapElementId_error = j["map_element_id"].get(mapElementId);

        if(mapElementId_error == simdjson::error_code::NO_SUCH_FIELD)
        {
            road.mapType = MapType::UNKNOWN;
        }
        else{
            CHECK_JSON_ERROR(mapElementId_error); // Make sure no other error occurred.

            if(mapElementId == 4 or mapElementId >= static_cast<int32_t>(MapType::NUM_TYPES) or mapElementId < -1)
            {
                road.mapType = MapType::UNKNOWN;
            }
            else
            {
                road.mapType = static_cast<MapType>(mapElementId);
            }
        }

        for (int i = 0; i < road.numPoints; i++)
        {
            road.mean.x += (road.geometry[i].x - road.mean.x)/(i+1);
            road.mean.y += (road.geometry[i].y - road.mean.y)/(i+1);
        }

    }

    std::pair<float, float> calc_mean(const simdjson::dom::element &j)
    {
        std::pair<float, float> mean = {0, 0};
        int64_t numEntities = 0;

        // Process objects
        simdjson::dom::array objects;
        auto objects_error = j["objects"].get_array().get(objects);
        CHECK_JSON_ERROR(objects_error);

        for (auto obj : objects) {
            simdjson::dom::array positions;
            auto positions_error = obj["position"].get_array().get(positions);
            CHECK_JSON_ERROR(positions_error);

            simdjson::dom::array valid_array;
            auto valid_array_error = obj["valid"].get_array().get(valid_array);
            CHECK_JSON_ERROR(valid_array_error);

            int i = 0;
            for (auto pos : positions) {
                // Get valid flag
                bool is_valid;
                auto valid_val = valid_array.at(i);
                CHECK_JSON_ERROR(valid_val.get(is_valid));

                if (!is_valid) {
                    i++;
                    continue;
                }

                numEntities++;

                // Use the existing from_json function to parse the position
                double newX, newY;
                CHECK_JSON_ERROR(pos["x"].get(newX));
                CHECK_JSON_ERROR(pos["y"].get(newY));

                // Update mean incrementally
                mean.first += (newX - mean.first) / numEntities;
                mean.second += (newY - mean.second) / numEntities;

                i++;
            }
        }

        // Process roads
        simdjson::dom::array roads;
        auto roads_error = j["roads"].get_array().get(roads);
        CHECK_JSON_ERROR(roads_error);

        for (auto road : roads) {
            simdjson::dom::array geometry;
            auto geometry_error = road["geometry"].get_array().get(geometry);
            CHECK_JSON_ERROR(geometry_error);

            for (auto point : geometry) {
                numEntities++;
                double newX, newY;
                CHECK_JSON_ERROR(point["x"].get(newX));
                CHECK_JSON_ERROR(point["y"].get(newY));

                // Update mean incrementally
                mean.first += (newX - mean.first) / numEntities;
                mean.second += (newY - mean.second) / numEntities;
            }
        }

        return mean;
    }

    void from_json(const simdjson::dom::element &j, Map &map, float polylineReductionThreshold)
    {
        std::string_view name_view;
        CHECK_JSON_ERROR(j["name"].get(name_view));
        std::strncpy(map.mapName, name_view.data(), sizeof(map.mapName));

        std::string_view scenario_id_view;
        CHECK_JSON_ERROR(j["scenario_id"].get(scenario_id_view));
        std::strncpy(map.scenarioId, scenario_id_view.data(), sizeof(map.scenarioId));
        
        auto mean = calc_mean(j);
        map.mean = {mean.first, mean.second};

        simdjson::dom::array objects;
        CHECK_JSON_ERROR(j["objects"].get_array().get(objects));
        map.numObjects = std::min(objects.size(), static_cast<size_t>(MAX_OBJECTS));

        simdjson::dom::object metadata;
        CHECK_JSON_ERROR(j["metadata"].get(metadata));

        int64_t sdc_index;
        CHECK_JSON_ERROR(metadata["sdc_track_index"].get(sdc_index));
        
        // Create id to object index mapping
        std::unordered_map<int, size_t> idToObjIdx;
        size_t idx = 0;
        
        // Initialize SDC first if valid
        if (sdc_index >= 0 && sdc_index < objects.size()) {
            simdjson::dom::element sdc_obj;
            CHECK_JSON_ERROR(objects.at(sdc_index).get(sdc_obj));
            from_json(sdc_obj, map.objects[0]);
            map.objects[0].metadata.isSdc = 1;
            idToObjIdx[map.objects[0].id] = 0;
            idx = 1;
        }
        
        // Initialize all other objects
        for (size_t i = 0; i < objects.size(); i++) {
            if (i == sdc_index) continue; // Skip SDC as it's already initialized
            if (idx >= map.numObjects) break;
            simdjson::dom::element obj;
            CHECK_JSON_ERROR(objects.at(i).get(obj));
            from_json(obj, map.objects[idx]);
            idToObjIdx[map.objects[idx].id] = idx;
            idx++;
        }
        
        // Process objects_of_interest using the ID mapping
        simdjson::dom::array objects_of_interest;
        CHECK_JSON_ERROR(metadata["objects_of_interest"].get(objects_of_interest));
        for (const auto& obj_id : objects_of_interest) {
            int64_t interest_id;
            CHECK_JSON_ERROR(obj_id.get(interest_id));
            if (auto it = idToObjIdx.find(interest_id); it != idToObjIdx.end()) {
                map.objects[it->second].metadata.isObjectOfInterest = 1;
            }
        }

        // Process tracks_to_predict using the ID mapping
        simdjson::dom::array tracks_to_predict;
        CHECK_JSON_ERROR(metadata["tracks_to_predict"].get(tracks_to_predict));
        for (const auto& track : tracks_to_predict) {
            int64_t track_index;
            CHECK_JSON_ERROR(track["track_index"].get(track_index));
            if (track_index < 0 || track_index >= objects.size()) {
                std::cerr << "Warning: Invalid track_index " << track_index << " in scene " << name_view.data() << std::endl;
            } else {
                int64_t track_id;
                simdjson::dom::element track_obj;
                CHECK_JSON_ERROR(objects.at(track_index).get(track_obj));
                CHECK_JSON_ERROR(track_obj["id"].get(track_id));
                if (auto it = idToObjIdx.find(track_id); it != idToObjIdx.end()) {
                    map.objects[it->second].metadata.isTrackToPredict = 1;
                    int64_t difficulty;
                    CHECK_JSON_ERROR(track["difficulty"].get(difficulty));
                    map.objects[it->second].metadata.difficulty = difficulty;
                }
            }
        }
        
        // Process roads
        simdjson::dom::array roads;
        CHECK_JSON_ERROR(j["roads"].get_array().get(roads));
        map.numRoads = std::min(roads.size(), static_cast<size_t>(MAX_ROADS));
        size_t countRoadPoints = 0;
        idx = 0;
        for (const auto &road : roads) {
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
