#pragma once 

#include "rng.hpp"
#include "sim.hpp"

namespace GPUHideSeek {

#include "geo_gen.hpp"
inline madrona::Entity makeDynObject(
    Engine &ctx,
    madrona::math::Vector3 pos,
    madrona::math::Quat rot,
    int32_t obj_id,
    madrona::phys::ResponseType response_type = ResponseType::Dynamic,
    madrona::math::Diag3x3 scale = {1, 1, 1});

void populateStaticGeometry(Engine &ctx,
                            RNG &rng,
                            madrona::math::Vector2 level_scale,
                            CountT &srcRoom, CountT &dstRoom);

// Finds which room an entity is in
Room *containedRoom(madrona::math::Vector2 pos, Room *rooms);
// Finds whether an entity is pressing a button or not
bool isPressingButton(madrona::math::Vector2 pos, Room *room);

}

#include "geo_gen.inl"
