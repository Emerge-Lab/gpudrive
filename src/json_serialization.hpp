#include "types.hpp"
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
        int i = 0;
        for (const auto &pos : j.at("position"))
        {
            if (i < MAX_POSITIONS)
            {
                pos.get_to(obj.position[i]);
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
            if (i < MAX_HEADINGS)
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
            if (i < MAX_VELOCITIES)
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
            if (i < MAX_VELOCITIES)
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
        int i = 0;
        for (const auto &geom : j.at("geometry"))
        {
            if (i < MAX_POSITIONS)
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
        // j.at("type").get_to(road.type);
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
    }

    void from_json(const nlohmann::json &j, Map &map)
    {
        int i = 0;
        for (const auto &obj : j.at("objects"))
        {
            if (i < MAX_OBJECTS)
            {
                obj.get_to(map.objects[i]);
                ++i;
            }
            else
            {
                break; // Avoid overflow
            }
        }
        map.numObjects = i;

        i = 0;
        for (const auto &road : j.at("roads"))
        {
            if (i < MAX_ROADS)
            {
                road.get_to(map.roads[i]);
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