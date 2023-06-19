#pragma once


namespace GPUHideSeek {
madrona::Entity makeDynObject(Engine &ctx,
                     madrona::math::Vector3 pos,
                     madrona::math::Quat rot,
                     int32_t obj_id,
                     madrona::phys::ResponseType response_type,
                     madrona::math::Diag3x3 scale)
{
    using namespace madrona;
    using namespace madrona::math;

    Entity e = ctx.makeEntity<DynamicObject>();
    ctx.get<Position>(e) = pos;
    ctx.get<Rotation>(e) = rot;
    ctx.get<Scale>(e) = scale;
    ctx.get<ObjectID>(e) = ObjectID { obj_id };
    ctx.get<phys::broadphase::LeafID>(e) =
        phys::RigidBodyPhysicsSystem::registerEntity(ctx, e, ObjectID {obj_id});
    ctx.get<Velocity>(e) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.get<ResponseType>(e) = response_type;
    ctx.get<ExternalForce>(e) = Vector3::zero();
    ctx.get<ExternalTorque>(e) = Vector3::zero();

    return e;
}

}
