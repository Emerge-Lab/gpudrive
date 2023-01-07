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

using madrona::Entity;
using madrona::CountT;
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
    madrona::phys::broadphase::LeafID,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity
> {};

struct StaticObject : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    madrona::phys::broadphase::LeafID
> {};

struct Action {
    int32_t action;
};

struct AgentImpl {
    madrona::Entity implEntity;
};

static_assert(sizeof(Action) == sizeof(int32_t));

struct AgentInterface : public madrona::Archetype<
    Action,
    AgentImpl
> {};

struct CameraAgent : public madrona::Archetype<
    Position,
    Rotation,
    madrona::render::ViewSettings,
    madrona::render::ViewID
> {};

struct DynAgent : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    Velocity,
    madrona::phys::broadphase::LeafID,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    madrona::render::ViewSettings,
    madrona::render::ViewID
> {};

struct Sim : public madrona::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry);

    static void setupTasks(madrona::TaskGraph::Builder &builder);

    Sim(Engine &ctx, const WorldInit &init);

    EpisodeManager *episodeMgr;
    RNG rng;

    Entity *allEntities;
    CountT numEntities;
    Entity agents[6];
    CountT numAgents;
    CountT minEpisodeEntities;
    CountT maxEpisodeEntities;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
