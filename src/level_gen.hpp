#pragma once

#include "sim.hpp"
#include "utils.hpp"

namespace gpudrive
{

    void createPersistentEntities(Engine &ctx);

void resetWorld(Engine &ctx);

// Destroys all entities in the world
void destroyWorld(Engine &ctx);

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

    static inline void setRoadEntitiesProps(Engine &ctx,
                                            Entity road,
                                            madrona::math::Vector3 pos,
                                            madrona::math::Quat rot,
                                            Scale scale,
                                            EntityType type,
                                            ObjectID objId,
                                            ResponseType responseType)
    {
        ctx.get<Position>(road) = pos;
        ctx.get<Rotation>(road) = rot;
        ctx.get<Scale>(road) = scale;
        ctx.get<EntityType>(road) = type;
        ctx.get<ObjectID>(road) = objId;
        ctx.get<ResponseType>(road) = responseType;
        ctx.get<MapObservation>(ctx.get<RoadInterfaceEntity>(road).e) = MapObservation{.position = pos.xy(),
                                                                                       .scale = scale,
                                                                                       .heading = utils::quatToYaw(rot),
                                                                                       .type = (float)type};
    }

} // namespace gpudrive