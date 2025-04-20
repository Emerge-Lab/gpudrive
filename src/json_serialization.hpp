#pragma once

#include "init.hpp"
#include "types.hpp"
#include "consts.hpp"
#include <iostream>
#include <simdjson.h>
#include <unordered_set>

namespace madrona_gpudrive
{

    #define CHECK_JSON_ERROR(err) \
    if (err) { \
        std::cerr << "JSON error: " << simdjson::error_message(err) << std::endl; \
        abort(); \
    }

    template<typename T>
    T getValueOrDefault(const simdjson::dom::element &obj, 
                        std::string_view key, 
                        T defaultValue) {
        simdjson::dom::object json_obj;
        auto error = obj.get_object().get(json_obj);
        if (error) {
            printf("Warning: Failed to get object: %s\n", simdjson::error_message(error));
            abort();
        }

        simdjson::dom::element result;
        error = json_obj.at_key(key).get(result);
        
        if (error == simdjson::NO_SUCH_FIELD) {
            printf("Warning: Key %.*s not found in json\n", 
                   static_cast<int>(key.size()), key.data());
            abort();
        }
        CHECK_JSON_ERROR(error);
        
        // Handle different types appropriately
        if constexpr (std::is_same_v<T, float>) {
            double d;
            error = result.get_double().get(d);
            CHECK_JSON_ERROR(error);
            return static_cast<float>(d);
        } else if constexpr (std::is_same_v<T, int>) {
            int64_t i;
            error = result.get_int64().get(i);
            CHECK_JSON_ERROR(error);
            return static_cast<int>(i);
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            uint64_t u;
            error = result.get_uint64().get(u);
            CHECK_JSON_ERROR(error);
            return static_cast<uint32_t>(u);
        } else if constexpr (std::is_same_v<T, size_t>) {
            uint64_t u;
            error = result.get_uint64().get(u);
            CHECK_JSON_ERROR(error);
            return static_cast<size_t>(u);
        } else if constexpr (std::is_same_v<T, bool>) {
            bool b;
            error = result.get_bool().get(b);
            CHECK_JSON_ERROR(error);
            return b;
        } else if constexpr (std::is_same_v<T, std::string_view>) {
            std::string_view s;
            error = result.get_string().get(s);
            CHECK_JSON_ERROR(error);
            return s;
        } else if constexpr (std::is_same_v<T, simdjson::dom::array>) {
            simdjson::dom::array result_arr;
            error = result.get_array().get(result_arr);
            CHECK_JSON_ERROR(error);
            return result_arr;
        } else if constexpr (std::is_same_v<T, simdjson::dom::element>) {
            return result;
        } else {
            static_assert(!sizeof(T), "Unsupported type for getValueOrDefault");
        }
    }

    template<typename T>
    T getValueOrDefault(const simdjson::dom::array &arr, 
                        size_t idx, 
                        T defaultValue) {
        if (idx >= arr.size()) {
            printf("Warning: Index %zu not found in array\n", idx);
            return defaultValue;
        }
        
        simdjson::dom::element result;
        auto error = arr.at(idx).get(result);
        CHECK_JSON_ERROR(error);
        
        // Handle different types appropriately using the same pattern as above
        if constexpr (std::is_same_v<T, float>) {
            double d;
            error = result.get_double().get(d);
            CHECK_JSON_ERROR(error);
            return static_cast<float>(d);
        } else if constexpr (std::is_same_v<T, simdjson::dom::element>) {
            return result;
        } else if constexpr (std::is_same_v<T, int>) {
            int64_t i;
            error = result.get_int64().get(i);
            CHECK_JSON_ERROR(error);
            return static_cast<int>(i);
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            uint64_t u;
            error = result.get_uint64().get(u);
            CHECK_JSON_ERROR(error);
            return static_cast<uint32_t>(u);
        } else if constexpr (std::is_same_v<T, bool>) {
            bool b;
            error = result.get_bool().get(b);
            CHECK_JSON_ERROR(error);
            return b;
        } else if constexpr (std::is_same_v<T, std::string_view>) {
            std::string_view s;
            error = result.get_string().get(s);
            CHECK_JSON_ERROR(error);
            return s;
        } else if constexpr (std::is_same_v<T, simdjson::dom::array>) {
            simdjson::dom::array result_arr;
            error = result.get_array().get(result_arr);
            CHECK_JSON_ERROR(error);
            return result_arr;
        } else {
            static_assert(!sizeof(T), "Unsupported type for array getValueOrDefault");
        }
    }

    void from_dom(const simdjson::dom::element &j, MapVector2 &p) {
        p.x = getValueOrDefault<float>(j, "x", 0.0f);  // Provide a default value of 0.0f if missing
        p.y = getValueOrDefault<float>(j, "y", 0.0f);
    }

    void from_dom(const simdjson::dom::element &j, MapObject &obj)
    {
        obj.mean = {0,0};
        uint32_t i = 0;
        simdjson::dom::array positions = getValueOrDefault<simdjson::dom::array>(j, "position", {});

        for (const auto &pos : positions)
        { 
            if (i >= MAX_POSITIONS) break; // Avoid overflow
            from_dom(pos, obj.position[i]);
            obj.mean.x += (obj.position[i].x - obj.mean.x)/(i+1);
            obj.mean.y += (obj.position[i].y - obj.mean.y)/(i+1);
            ++i;
        }
        obj.numPositions = i;
        // Replace with default values (here zero) if the key is missing.
        obj.vehicle_size.width  = getValueOrDefault<float>(j, "width", 0.0f);
        obj.vehicle_size.length = getValueOrDefault<float>(j, "length", 0.0f);
        obj.vehicle_size.height = getValueOrDefault<float>(j, "height", 0.0f);
        obj.id = getValueOrDefault<int>(j, "id", 0);

        i = 0;
        simdjson::dom::array headings = getValueOrDefault<simdjson::dom::array>(j, "heading", {});
        for (const auto &h : headings)
        {
            if (i < MAX_POSITIONS)
            {
                double heading;
                CHECK_JSON_ERROR(h.get(heading));
                obj.heading[i] = static_cast<float>(heading);
                ++i;
            }
            else
            {
                break; // Avoid overflow
            }
        }
        obj.numHeadings = i;

        i = 0;
        simdjson::dom::array velocities = getValueOrDefault<simdjson::dom::array>(j, "velocity", {});
        for (const auto &v : velocities)
        {
            if (i>= MAX_POSITIONS) break; // Avoid overflow
            from_dom(v, obj.velocity[i]);
            ++i;
        }
        obj.numVelocities = i;

        i = 0;
        simdjson::dom::array valid = getValueOrDefault<simdjson::dom::array>(j, "valid", {});
        for (const auto &v : valid)
        {
            if (i >= MAX_POSITIONS) break; // Avoid overflow
            bool valid_val;
            CHECK_JSON_ERROR(v.get(valid_val));
            obj.valid[i] = valid_val;
            ++i;
        }
        obj.numValid = i;

        simdjson::dom::element goalPosition = getValueOrDefault<simdjson::dom::element>(j, "goalPosition", {});
        from_dom(goalPosition, obj.goalPosition);

        std::string_view type = getValueOrDefault<std::string_view>(j, "type", "");
        if(type == "vehicle")
            obj.type = EntityType::Vehicle;
        else if(type == "pedestrian")
            obj.type = EntityType::Pedestrian;
        else if(type == "cyclist")
            obj.type = EntityType::Cyclist;
        else
            obj.type = EntityType::None;

        obj.markAsExpert = getValueOrDefault<bool>(j, "mark_as_expert", false);

         // Initialize metadata fields to 0
        obj.metadata.isSdc = 0;
        obj.metadata.isObjectOfInterest = 0;
        obj.metadata.isTrackToPredict = 0;
        obj.metadata.difficulty = 0;
    }

    void from_dom(const simdjson::dom::element &j, MapRoad &road, float polylineReductionThreshold = 0.0)
    {
        road.mean = {0,0};
        std::string_view type = getValueOrDefault<std::string_view>(j, "type", "");
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
        simdjson::dom::array geometry = getValueOrDefault<simdjson::dom::array>(j, "geometry", {});
        for(const auto &point: geometry)
        {
            MapVector2 p;
            from_dom(point, p);
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

        road.id = getValueOrDefault<uint32_t>(j, "id", -1);
        
        int32_t mapElementId = getValueOrDefault<int32_t>(j, "map_element_id", static_cast<int32_t>(MapType::UNKNOWN));
        if(mapElementId == 4 or mapElementId >= static_cast<int32_t>(MapType::NUM_TYPES) or mapElementId < -1)
            road.mapType = MapType::UNKNOWN;
        else
            road.mapType = static_cast<MapType>(mapElementId);

        for (uint32_t i = 0; i < road.numPoints; i++)
        {
            road.mean.x += (road.geometry[i].x - road.mean.x)/(i+1);
            road.mean.y += (road.geometry[i].y - road.mean.y)/(i+1);
        }

    }

    std::pair<float, float> calc_mean(const simdjson::dom::element &j)
    {
        std::pair<float, float> mean = {0, 0};
        int64_t numEntities = 0;
        simdjson::dom::array objects = getValueOrDefault<simdjson::dom::array>(j, "objects", {});
        for (const auto &obj : objects)
        {
            int i = 0;
            simdjson::dom::array position = getValueOrDefault<simdjson::dom::array>(obj, "position", {});
            simdjson::dom::array valids = getValueOrDefault<simdjson::dom::array>(obj, "valid", {});
            for (const auto &pos : position)
            {
                bool valid_val;
                CHECK_JSON_ERROR(valids.at(i).get(valid_val));
                if(!valid_val)
                    continue;
                numEntities++;
                float newX = getValueOrDefault<float>(pos, "x", 0.0f);
                float newY = getValueOrDefault<float>(pos, "y", 0.0f);
                // Update mean incrementally
                mean.first += (newX - mean.first) / numEntities;
                mean.second += (newY - mean.second) / numEntities;
                i++;
            }
        }
        simdjson::dom::array roads = getValueOrDefault<simdjson::dom::array>(j, "roads", {});
        for (const auto &obj : roads)
        {
            simdjson::dom::array geometry = getValueOrDefault<simdjson::dom::array>(obj, "geometry", {});
            for (const auto &point : geometry)
            {
                numEntities++;
                float newX = getValueOrDefault<float>(point, "x", 0.0f);
                float newY = getValueOrDefault<float>(point, "y", 0.0f);

                // Update mean incrementally
                mean.first += (newX - mean.first) / numEntities;
                mean.second += (newY - mean.second) / numEntities;
            }
        }
        return mean;
    }

    void from_json(const simdjson::dom::element &j, Map &map, float polylineReductionThreshold)
    {
        std::string_view name = getValueOrDefault<std::string_view>(j, "name", "");
        std::strncpy(map.mapName, name.data(), sizeof(map.mapName));

        std::string_view scenario_id = getValueOrDefault<std::string_view>(j, "scenario_id", "");
        std::strncpy(map.scenarioId, scenario_id.data(), sizeof(map.scenarioId));
        
        auto mean = calc_mean(j);
        map.mean = {mean.first, mean.second};
        // map.numObjects = std::min(getValueOrDefault<size_t>(j, "objects", 0), static_cast<size_t>(MAX_OBJECTS));
        auto objects = getValueOrDefault<simdjson::dom::array>(j, "objects", {});
        map.numObjects = std::min(objects.size(), static_cast<size_t>(MAX_OBJECTS));

        simdjson::dom::element metadata = getValueOrDefault<simdjson::dom::element>(j, "metadata", {});
        int sdc_index = getValueOrDefault<int>(metadata, "sdc_track_index", -1);
        
        // Create id to object index mapping
        std::unordered_map<int, size_t> idToObjIdx;
        size_t idx = 0;

        // First, identify which objects are tracks_to_predict and objects_of_interest
        std::unordered_set<int> tracks_to_predict_indices;
        std::unordered_set<int> objects_of_interest_ids;

        // Collect tracks_to_predict indices
        simdjson::dom::array tracks_to_predict = getValueOrDefault<simdjson::dom::array>(metadata, "tracks_to_predict", {});
        for (const auto& track : tracks_to_predict) {
            int track_index = getValueOrDefault<int>(track, "track_index", -1);
            if (track_index >= 0 && static_cast<uint32_t>(track_index) < map.numObjects) {
                tracks_to_predict_indices.insert(track_index);
            } else {
                std::cerr << "Warning: Invalid track_index " << track_index << " in scene " << std::string(map.mapName) << std::endl;
            }
        }

        // Collect objects_of_interest IDs
        simdjson::dom::array objects_of_interest = getValueOrDefault<simdjson::dom::array>(metadata, "objects_of_interest", {});
        for (const auto& obj_id : objects_of_interest) {
            objects_of_interest_ids.insert(getValueOrDefault<int>(obj_id, "id", -1));
        }
        // Initialize SDC first if valid
        if (sdc_index >= 0 && static_cast<uint32_t>(sdc_index) < map.numObjects) {
            
            simdjson::dom::element sdc_element;
            auto error = objects.at(sdc_index).get(sdc_element);
            CHECK_JSON_ERROR(error);
            from_dom(sdc_element, map.objects[0]);
            map.objects[0].metadata.isSdc = 1;
            
            // Set additional metadata if needed
            int sdc_id = map.objects[0].id;
            if (tracks_to_predict_indices.find(sdc_index) != tracks_to_predict_indices.end()) {
                map.objects[0].metadata.isTrackToPredict = 1;
                // Find and set difficulty
                for (const auto& track : tracks_to_predict) {
                    if (getValueOrDefault<int>(track, "track_index", -1) == sdc_index) {
                        map.objects[0].metadata.difficulty = getValueOrDefault<int>(track, "difficulty", 0);
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
        for (size_t i = 0; i < map.numObjects && idx < map.numObjects; i++) {
            if (i == static_cast<size_t>(sdc_index)) continue; // Skip SDC as it's already initialized
            
            if (tracks_to_predict_indices.find(i) != tracks_to_predict_indices.end()) {
                from_dom(getValueOrDefault<simdjson::dom::element>(objects, i, simdjson::dom::element()), map.objects[idx]);
                map.objects[idx].metadata.isTrackToPredict = 1;
                
                // Find and set difficulty
                for (const auto& track : tracks_to_predict) {
                    if (getValueOrDefault<int>(track, "track_index", -1) == static_cast<int>(i)) {
                        map.objects[idx].metadata.difficulty = getValueOrDefault<int>(track, "difficulty", 0);
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
        for (size_t i = 0; i < map.numObjects && idx < map.numObjects; i++) {
            if (i == static_cast<size_t>(sdc_index)) continue;
            

            int obj_id = getValueOrDefault<int>(getValueOrDefault<simdjson::dom::element>(objects, i, simdjson::dom::element()), "id", -1);
            if (objects_of_interest_ids.find(obj_id) != objects_of_interest_ids.end()) {
                from_dom(getValueOrDefault<simdjson::dom::element>(objects, i, simdjson::dom::element()), map.objects[idx]);
                map.objects[idx].metadata.isObjectOfInterest = 1;
                
                idToObjIdx[map.objects[idx].id] = idx;
                idx++;
            }
        }

        // Initialize all remaining objects
        for (size_t i = 0; i < map.numObjects && idx < map.numObjects; i++) {
            if (i == static_cast<size_t>(sdc_index)) continue;
            
            int obj_id = getValueOrDefault<int>(getValueOrDefault<simdjson::dom::element>(objects, i, simdjson::dom::element()), "id", -1);
            if (idToObjIdx.find(obj_id) == idToObjIdx.end()) { // Check if not already processed
                from_dom(getValueOrDefault<simdjson::dom::element>(objects, i, simdjson::dom::element()), map.objects[idx]);
                idToObjIdx[map.objects[idx].id] = idx;
                idx++;
            }
        }
        
        // Process roads
        auto roads = getValueOrDefault<simdjson::dom::array>(j, "roads", {});
        map.numRoads = std::min(roads.size(), static_cast<size_t>(MAX_ROADS));
        size_t countRoadPoints = 0;
        idx = 0;
        for (const auto &road : roads) {
            if (idx >= map.numRoads)
                break;
            from_dom(road, map.roads[idx], polylineReductionThreshold);
            size_t roadPoints = map.roads[idx].numPoints;
            countRoadPoints += (map.roads[idx].type <= EntityType::RoadLane) ? (roadPoints - 1) : 1;
            ++idx;
        }
        map.numRoadSegments = countRoadPoints;
    }
}