#include "types.hpp"
#include <iostream>
#include <nlohmann/json.hpp>

namespace gpudrive
{
    void from_json(const nlohmann::json &j, MapPosition &p)
    {
        j.at("x").get_to(p.x);
        j.at("y").get_to(p.y);
    }

    void from_json(const nlohmann::json &j, MapObject &obj)
    {
        const auto &valid = j.at("valid");

        obj.meanx = 0;
        obj.meany = 0;
        int i = 0;
        for (const auto &pos : j.at("position"))
        { 
            if (i < MAX_POSITIONS)
            {
                pos.get_to(obj.position[i]);
                if(valid[i] == true)
                {
                    obj.meanx += (obj.position[i].x - obj.meanx)/(i+1);
                    obj.meany += (obj.position[i].y - obj.meany)/(i+1);
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
                v.get_to(obj.velocity[i]);
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

        j.at("goalPosition").get_to(obj.goalPosition);
        // j.at("type").get_to(obj.type);
        std::string type = j.at("type");
        if(type == "vehicle")
            obj.type = MapObjectType::vehicle;
        else if(type == "pedestrian")
            obj.type = MapObjectType::pedestrian;
        else if(type == "cyclist")
            obj.type = MapObjectType::cyclist;
        else
            obj.type = MapObjectType::invalid;
    }

    void from_json(const nlohmann::json &j, MapRoad &road)
    {
        road.meanx = 0;
        road.meany = 0;
        std::string type = j.at("type");
        if(type == "road_edge")
            road.type = MapRoadType::road_edge;
        else if(type == "road_line")
            road.type = MapRoadType::road_line;
        else if(type == "lane")
            road.type = MapRoadType::lane;
        else if(type == "crosswalk")
            road.type = MapRoadType::crosswalk;
        else if(type == "speed_bump")
            road.type = MapRoadType::speed_bump;
        else if(type == "stop_sign")
            road.type = MapRoadType::stop_sign;
        else
            road.type = MapRoadType::invalid;

        size_t i = 0;
        if (road.type == MapRoadType::road_edge || road.type == MapRoadType::road_line || road.type == MapRoadType::lane)
        {
            for (const auto &geom : j.at("geometry"))
            {
                if (i < 2)
                {
                    geom.get_to(road.geometry[i]);
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
                if (shoelace_area > 0.001)
                {
                    geom.get_to(road.geometry[i]);
                    ++i;
                    if (i == MAX_GEOMETRY)
                        break; // Avoid overflow
                }
                else
                {
                    geom.get_to(road.geometry[i - 1]);
                }
            }
            road.numPositions = i;
        } 
        else if (road.type == MapRoadType::crosswalk || road.type == MapRoadType::speed_bump || road.type == MapRoadType::stop_sign)
        {
            for (const auto &geom : j.at("geometry"))
            {
                if (i < MAX_GEOMETRY)
                {
                    geom.get_to(road.geometry[i]);
                    ++i;
                }
                else
                {
                    break; // Avoid overflow
                }
            }
            road.numPositions = i;
        }

        for (size_t j = 0; j < road.numPositions; j++)
        {
            road.meanx += (road.geometry[j].x - road.meanx)/(j+1);
            road.meany += (road.geometry[j].y - road.meany)/(j+1);
        }

    }

    void from_json(const nlohmann::json &j, Map &map)
    {
        map.meanx = 0;
        map.meany = 0;
        size_t totalPoints = 0; // Total count of points
        int i = 0;
        for (const auto &obj : j.at("objects"))
        {
            if (i < MAX_OBJECTS)
            {
                if(obj.at("type") != "vehicle")
                    continue;
                obj.get_to(map.objects[i]);    
                size_t objPoints = map.objects[i].numPositions;
                map.meanx = ((map.meanx * totalPoints) + (map.objects[i].meanx * objPoints)) / (totalPoints + objPoints);
                map.meany = ((map.meany * totalPoints) + (map.objects[i].meany * objPoints)) / (totalPoints + objPoints);
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
                road.get_to(map.roads[i]);
                size_t roadPoints = map.roads[i].numPositions;
                map.meanx = ((map.meanx * totalPoints) + (map.roads[i].meanx * roadPoints)) / (totalPoints + roadPoints);
                map.meany = ((map.meany * totalPoints) + (map.roads[i].meany * roadPoints)) / (totalPoints + roadPoints);
                totalPoints += roadPoints;
                count_road_points += roadPoints;
                ++i;
            }
            else
            {
                break; // Avoid overflow
            }
        }
        map.numRoads = i;
        // tl_states is ignored as it"s always empty
    }
}