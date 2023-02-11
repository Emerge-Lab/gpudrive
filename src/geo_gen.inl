#pragma once

namespace GPUHideSeek {

madrona::Entity makeDynObject(Engine &ctx,
                     madrona::math::Vector3 pos,
                     madrona::math::Quat rot,
                     int32_t obj_id,
                     madrona::phys::ResponseType response_type,
                     OwnerTeam owner_team,
                     madrona::math::Diag3x3 scale)
{
    using namespace madrona;
    using namespace madrona::math;

    Entity e = ctx.makeEntityNow<DynamicObject>();
    ctx.getUnsafe<Position>(e) = pos;
    ctx.getUnsafe<Rotation>(e) = rot;
    ctx.getUnsafe<Scale>(e) = scale;
    ctx.getUnsafe<ObjectID>(e) = ObjectID { obj_id };
    ctx.getUnsafe<phys::broadphase::LeafID>(e) =
        phys::RigidBodyPhysicsSystem::registerEntity(ctx, e, ObjectID {obj_id});
    ctx.getUnsafe<Velocity>(e) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.getUnsafe<ResponseType>(e) = response_type;
    ctx.getUnsafe<OwnerTeam>(e) = owner_team;
    ctx.getUnsafe<ExternalForce>(e) = Vector3::zero();
    ctx.getUnsafe<ExternalTorque>(e) = Vector3::zero();

    return e;
}

}
