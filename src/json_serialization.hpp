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

    

    void from_json(simdjson::dom::element &j, MapVector2 &p)
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
            } else {
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

        // Parse heading array
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
            } else {
                break; // Avoid overflow
            }
        }
        obj.numHeadings = i;

        // Parse velocity array
        i = 0;
        simdjson::dom::array velocities;
        auto velocities_error = j["velocity"].get_array().get(velocities);
        CHECK_JSON_ERROR(velocities_error);
        
        for (auto v : velocities) {
            if (i < MAX_POSITIONS) {
                from_json(v, obj.velocity[i]);
                ++i;
            } else {
                break; // Avoid overflow
            }
        }
        obj.numVelocities = i;

        // Continue with the rest of the parsing...
    }

    std::pair<float, float> calc_mean(simdjson::dom::element &doc)
    {
        std::pair<float, float> mean = {0, 0};
        int64_t numEntities = 0;
        
        // Process objects
        simdjson::dom::array objects;
        auto objects_error = doc["objects"].get_array().get(objects);
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
        auto roads_error = doc["roads"].get_array().get(roads);
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

    void from_json(simdjson::dom::element &doc, Map &map, float polylineReductionThreshold)
    {
        // Get map name
        std::string_view name_view;
        std::string name(name_view);
        std::strncpy(map.mapName, name.c_str(), sizeof(map.mapName));
        
        // Get scenario ID
        std::string_view scenario_id_view;
        std::string scenario_id(scenario_id_view);
        std::strncpy(map.scenarioId, scenario_id.c_str(), sizeof(map.scenarioId));
        
        // Calculate mean
        auto mean = calc_mean(doc);
        map.mean = {mean.first, mean.second};
        
        // Process objects
        simdjson::dom::array objects;
        auto objects_error = doc["objects"].get_array().get(objects);
        CHECK_JSON_ERROR(objects_error);
        
        // Count objects first to determine numObjects
        uint64_t objects_count = 0;
        map.numObjects = std::min(static_cast<size_t>(objects_count), static_cast<size_t>(MAX_OBJECTS));
        
        // Create id to object index mapping
        std::unordered_map<int, size_t> idToObjIdx;
        size_t idx = 0;
        
        for (auto obj : objects) {
            if (idx >= map.numObjects)
                break;
                
            // Parse object
            from_json(obj, map.objects[idx]);
            idToObjIdx[map.objects[idx].id] = idx;
            idx++;
        }
        
        // Process metadata
        auto metadata = doc["metadata"].get_object();
        
        // Set SDC
        int64_t sdc_index;
        
        if (sdc_index < 0 || sdc_index >= static_cast<int64_t>(objects_count)) {
            std::cerr << "Warning: Invalid sdc_track_index " << sdc_index << " in scene " << name << std::endl;
        } else {
            // Get the object at sdc_index
            auto sdc_obj = objects.at(sdc_index).get_object();
            
            int64_t sdc_id;
            
            if (auto it = idToObjIdx.find(sdc_id); it != idToObjIdx.end()) {
                map.objects[it->second].metadata.isSdc = 1;
            }
        }
        
        // Continue with the rest of the parsing...
    }
}