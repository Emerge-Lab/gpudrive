#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>
#include <madrona/physics.hpp>
#include <madrona/render.hpp>

#include "init.hpp"
#include "rng.hpp"

namespace GPUHideSeek {

using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::phys::Velocity;

class Engine;

struct WorldReset {
    int32_t resetNow;
};

struct DynamicObject : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    madrona::phys::CollisionAABB,
    madrona::phys::broadphase::LeafID
> {};

struct Action {
    int32_t action;
};

static_assert(sizeof(Action) == sizeof(int32_t));

struct Agent : public madrona::Archetype<
    Position,
    Rotation,
    madrona::render::ActiveView,
    Action
> {};

struct Sim : public madrona::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry);

    static void setupTasks(madrona::TaskGraph::Builder &builder);

    Sim(Engine &ctx, const WorldInit &init);

    EpisodeManager *episodeMgr;
    RNG rng;

    madrona::Entity agent;
    madrona::Entity staticEntity;
    madrona::Entity *dynObjects;
    madrona::CountT numObjects;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
