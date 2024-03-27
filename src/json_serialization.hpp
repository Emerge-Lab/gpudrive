#pragma once

#include "init.hpp"
#include <iostream>
#include <nlohmann/json.hpp>

namespace gpudrive
{
    void from_json(const nlohmann::json &j, MapVector2 &p)
    {
        p.x = j.at("x").get<float>();
        p.y = j.at("y").get<float>();
    }

    void from_json(const nlohmann::json &j, MapObject &obj)
    {
        const auto &valid = j.at("valid");

        obj.mean = {0,0};
        uint32_t i = 0;
        for (const auto &pos : j.at("position"))
        { 
            if (i < MAX_POSITIONS)
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
        j.at("width").get_to(obj.width);
        j.at("length").get_to(obj.length);

        i = 0;
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
        obj.numHeadings = i;

        i = 0;
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
        obj.numVelocities = i;

        i = 0;
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
        obj.numValid = i;


        from_json(j.at("goalPosition"), obj.goalPosition);
        std::string type = j.at("type");
        if(type == "vehicle")
            obj.type = MapObjectType::Vehicle;
        else if(type == "pedestrian")
            obj.type = MapObjectType::Pedestrian;
        else if(type == "cyclist")
            obj.type = MapObjectType::Cyclist;
        else
            obj.type = MapObjectType::Invalid;
    }

    void from_json(const nlohmann::json &j, MapRoad &road, float polylineReductionThreshold = 0.0)
    {
        road.mean = {0,0};
        std::string type = j.at("type");
        if(type == "road_edge")
            road.type = MapRoadType::RoadEdge;
        else if(type == "road_line")
            road.type = MapRoadType::RoadLine;
        else if(type == "lane")
            road.type = MapRoadType::Lane;
        else if(type == "crosswalk")
            road.type = MapRoadType::CrossWalk;
        else if(type == "speed_bump")
            road.type = MapRoadType::SpeedBump;
        else if(type == "stop_sign")
            road.type = MapRoadType::StopSign;
        else
            road.type = MapRoadType::Invalid;
        
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
        if (num_segments >= 10 && (road.type == MapRoadType::Lane || road.type == MapRoadType::RoadEdge || road.type == MapRoadType::RoadLine))
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
            for (int i = 0; i < new_geometry_points.size(); i++)
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

        for (int i = 0; i < road.numPoints; i++)
        {
            road.mean.x += (road.geometry[i].x - road.mean.x)/(i+1);
            road.mean.y += (road.geometry[i].y - road.mean.y)/(i+1);
        }

    }

    void from_json(const nlohmann::json &j, Map &map, float polylineReductionThreshold)
    {
        map.mean = {0,0};
        size_t totalPoints = 0; // Total count of points
        uint32_t i = 0;
        for (const auto &obj : j.at("objects"))
        {
            if (i < MAX_OBJECTS)
            {
                obj.get_to(map.objects[i]);    
                size_t objPoints = map.objects[i].numPositions;
                map.mean.x = ((map.mean.x * totalPoints) + (map.objects[i].mean.x * objPoints)) / (totalPoints + objPoints);
                map.mean.y = ((map.mean.y * totalPoints) + (map.objects[i].mean.y * objPoints)) / (totalPoints + objPoints);
                totalPoints += objPoints;
                ++i;
            }
            else
            {
                break; // Avoid overflow
            }
        }
        map.numObjects = i;

        i = 0;
        size_t count_road_points = 0;
        for (const auto &road : j.at("roads"))
        {
            if (i < MAX_ROADS)
            {
                from_json(road, map.roads[i], polylineReductionThreshold);
                size_t roadPoints = map.roads[i].numPoints;
                map.mean.x = ((map.mean.x * totalPoints) + (map.roads[i].mean.x * roadPoints)) / (totalPoints + roadPoints);
                map.mean.y = ((map.mean.y * totalPoints) + (map.roads[i].mean.y * roadPoints)) / (totalPoints + roadPoints);
                totalPoints += roadPoints;
                if(map.roads[i].type <= MapRoadType::Lane)
                    count_road_points += roadPoints - 1;
                else if(map.roads[i].type > MapRoadType::Lane)
                    count_road_points += 1;
                ++i;
            }
            else
            {
                break; // Avoid overflow
            }
        }
        map.numRoadSegments = count_road_points;
        map.numRoads = i;
        // tl_states is ignored as it"s always empty
    }
}