#pragma once 

#include "rng.hpp"
#include "sim.hpp"

namespace GPUHideSeek {

inline Entity makeDynObject(
    Engine &ctx,
    madrona::math::Vector3 pos,
    madrona::math::Quat rot,
    int32_t obj_id,
    madrona::phys::ResponseType response_type = ResponseType::Dynamic,
    OwnerTeam owner_team = OwnerTeam::None,
    madrona::math::Diag3x3 scale = {1, 1, 1});

CountT populateStaticGeometry(Engine &ctx,
                              RNG &rng,
                              madrona::math::Vector2 level_scale);

}

#include "geo_gen.inl"
