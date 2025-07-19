#pragma once

#include "sim.hpp"
#include "utils.hpp"

namespace madrona_gpudrive
{

    /**
     * @brief Create all the entities that will exist in the simulation
     * for the duration of the episode. This includes agents, walls, and other
     * static elements of the environment.
     *
     * @param ctx The engine context.
     */
    void createPersistentEntities(Engine &ctx);

    /**
     * @brief This is called each time an episode begins. It resets the world
     * to its initial state, repositioning agents and other dynamic elements.
     *
     * @param ctx The engine context.
     */
    void resetWorld(Engine &ctx);

    /**
     * @brief Destroys all entities in the world. This is called when the
     * simulation is being shut down or a full reset is required.
     *
     * @param ctx The engine context.
     */
    void destroyWorld(Engine &ctx);

    /**
     * @brief Get a zero action for the given dynamics model.
     *
     * This function returns a zero action, which represents no change in the
     * vehicle's state. This is useful for initializing actions or for when
     * no action is specified.
     *
     * @param model The dynamics model to get the zero action for.
     * @return The zero action.
     */
    static inline Action getZeroAction(DynamicsModel model)
    {
        switch (model)
        {
        case DynamicsModel::Classic:
        case DynamicsModel::InvertibleBicycle:
        {
            return Action{.classic = {0, 0, 0}};
            break;
        }

        case DynamicsModel::DeltaLocal:
        {
            return Action{.delta{.dx = 0, .dy = 0, .dyaw = 0}};
        }
        case DynamicsModel::State:
        {
            return Action{.state = {.position = madrona::math::Vector3{0, 0, 1}, .yaw = 0, .velocity = {.linear = madrona::math::Vector3::zero(), .angular = madrona::math::Vector3::zero()}}};
        }
        default:
            return Action{.classic = {0, 0, 0}};
        }
    }

    /**
     * @brief Set the properties of a road entity.
     *
     * This function sets the position, rotation, scale, and other properties
     * of a road entity. It also updates the corresponding map observation.
     *
     * @param ctx The engine context.
     * @param road The road entity to set the properties for.
     * @param pos The position of the road entity.
     * @param rot The rotation of the road entity.
     * @param scale The scale of the road entity.
     * @param type The entity type of the road entity.
     * @param objId The object ID of the road entity.
     * @param responseType The response type of the road entity.
     * @param roadIdx The road index of the road entity.
     * @param mapType The map type of the road entity.
     */
    static inline void setRoadEntitiesProps(Engine &ctx,
                                            Entity road,
                                            madrona::math::Vector3 pos,
                                            madrona::math::Quat rot,
                                            Scale scale,
                                            EntityType type,
                                            ObjectID objId,
                                            ResponseType responseType, 
                                            uint32_t roadIdx,
                                            MapType mapType)
    {
        ctx.get<Position>(road) = pos;
        ctx.get<Rotation>(road) = rot;
        ctx.get<Velocity>(road) = Velocity{madrona::math::Vector3::zero(), madrona::math::Vector3::zero()};
        ctx.get<Scale>(road) = scale;
        ctx.get<EntityType>(road) = type;
        ctx.get<ObjectID>(road) = objId;
        ctx.get<ResponseType>(road) = responseType;
        ctx.get<RoadMapId>(road).id = roadIdx;
        ctx.get<MapType>(road) = mapType;
        ctx.get<MapObservation>(ctx.get<RoadInterfaceEntity>(road).e) = MapObservation{.position = pos.xy(),
                                                                                       .scale = scale,
                                                                                       .heading = utils::quatToYaw(rot),
                                                                                       .type = (float)type,
                                                                                       .id = static_cast<float>(roadIdx),
                                                                                       .mapType = static_cast<float>(mapType)};
    }

} // namespace madrona_gpudrive