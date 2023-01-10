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

namespace consts {

static inline constexpr int32_t maxBoxes = 9;
static inline constexpr int32_t maxRamps = 2;
static inline constexpr int32_t maxAgents = 6;
static inline constexpr int32_t totalObservationsPerAgent = 
    maxBoxes + maxRamps + maxAgents - 1;

}

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

struct Action {
    int32_t x;
    int32_t y;
    int32_t r;
    int32_t g;
    int32_t l;
};

struct SimEntity {
    Entity e;
};

struct TeamReward {
    std::atomic<float> hidersReward;
};

struct Reward {
    float reward;
};

struct AgentType {
    bool isHider;
};

struct ObservationMask {
    float mask;
};

struct PositionObservation {
    madrona::math::Vector3 x;
};

struct VelocityObservation {
    madrona::math::Vector3 v;
};

struct VisibilityMasks {
    float visible[consts::totalObservationsPerAgent];
};

static_assert(sizeof(Action) == 5 * sizeof(int32_t));

struct AgentInterface : public madrona::Archetype<
    SimEntity,
    Action,
    Reward,
    AgentType,
    PositionObservation,
    VelocityObservation,
    ObservationMask,
    VisibilityMasks
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

struct BoxObservation : public madrona::Archetype<
    SimEntity,
    PositionObservation,
    VelocityObservation,
    ObservationMask
> {};

struct RampObservation : public madrona::Archetype<
    SimEntity,
    PositionObservation,
    VelocityObservation,
    ObservationMask
> {};

struct Sim : public madrona::WorldBase {

    static void registerTypes(madrona::ECSRegistry &registry);

    static void setupTasks(madrona::TaskGraph::Builder &builder);

    Sim(Engine &ctx, const WorldInit &init);

    EpisodeManager *episodeMgr;
    RNG rng;

    Entity *obstacles;
    CountT numObstacles;

    Entity hiders[3];
    CountT numHiders;
    Entity seekers[3];
    CountT numSeekers;

    Entity boxObservations[consts::maxBoxes];
    Entity rampObservations[consts::maxRamps];
    Entity agentObservations[consts::maxAgents];
    CountT numActiveAgents;

    CountT minEpisodeEntities;
    CountT maxEpisodeEntities;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
