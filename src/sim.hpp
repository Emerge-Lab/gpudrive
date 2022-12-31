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
    int32_t resetLevel;
};

struct DynamicObject : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    madrona::phys::CollisionAABB,
    madrona::phys::broadphase::LeafID,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::SubstepStartState,
    madrona::phys::solver::SubstepVelocityState
> {};

struct StaticObject : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
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

    madrona::Entity *allEntities;
    madrona::CountT numEntities;
    madrona::Entity agent;
    madrona::CountT minEpisodeEntities;
    madrona::CountT maxEpisodeEntities;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
