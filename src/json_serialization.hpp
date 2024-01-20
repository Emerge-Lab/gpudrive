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
                if(valid[i] == true)
                {
                    obj.mean.x += (obj.position[i].x - obj.mean.x)/(i+1);
                    obj.mean.y += (obj.position[i].y - obj.mean.y)/(i+1);
                }
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

        uint32_t i = 0;
        if (road.type == MapRoadType::RoadEdge || road.type == MapRoadType::RoadLine || road.type == MapRoadType::Lane)
        {
            for (const auto &geom : j.at("geometry"))
            {
                if (i < 2)
                {
                    from_json(geom, road.geometry[i]);
                    i++;
                    
                    continue;
                }
                float x1 = road.geometry[i - 2].x;
                float y1 = road.geometry[i - 2].y;
                float x2 = road.geometry[i - 1].x;
                float y2 = road.geometry[i - 1].y;
                float x3 = geom["x"];
                float y3 = geom["y"];
                float shoelace_area = 0.5 * abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1));
                if (shoelace_area > polylineReductionThreshold)
                {
                    from_json(geom, road.geometry[i]);
                    ++i;
                    if (i == MAX_GEOMETRY)
                        break; // Avoid overflow
                }
                else
                {
                    from_json(geom, road.geometry[i]);
                }
            }
            road.numPoints = i;
        } 
        else if (road.type == MapRoadType::CrossWalk || road.type == MapRoadType::SpeedBump || road.type == MapRoadType::StopSign)
        {
            for (const auto &geom : j.at("geometry"))
            {
                if (i < MAX_GEOMETRY)
                {
                    from_json(geom, road.geometry[i]);
                    ++i;
                }
                else
                {
                    break; // Avoid overflow
                }
            }
            road.numPoints = i;
        }

        for (size_t j = 0; j < road.numPoints; j++)
        {
            road.mean.x += (road.geometry[j].x - road.mean.x)/(j+1);
            road.mean.y += (road.geometry[j].y - road.mean.y)/(j+1);
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
                if(obj.at("type") != "vehicle")
                    continue;
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
                // road.get_to(map.roads[i]);
                from_json(road, map.roads[i], polylineReductionThreshold);
                size_t roadPoints = map.roads[i].numPoints;
                map.mean.x = ((map.mean.x * totalPoints) + (map.roads[i].mean.x * roadPoints)) / (totalPoints + roadPoints);
                map.mean.y = ((map.mean.y * totalPoints) + (map.roads[i].mean.y * roadPoints)) / (totalPoints + roadPoints);
                totalPoints += roadPoints;
                count_road_points += roadPoints;
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