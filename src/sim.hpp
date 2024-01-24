#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/custom_context.hpp>

#include "consts.hpp"
#include "types.hpp"
#include "init.hpp"
#include "rng.hpp"

namespace gpudrive {

class Engine;

// This enum is used by the Sim and Manager classes to track the export slots
// for each component exported to the training code.
enum class ExportID : uint32_t {
    Reset,
    Action,
    Reward,
    Done,
    SelfObservation,
    PartnerObservations,
    Lidar,
    StepsRemaining,
    BicycleModel,
    MapObservation,
    Shape,
    NumExports
};

// Stores values for the ObjectID component that links entities to
// render / physics assets.
enum class SimObject : uint32_t {
    Cube,
    Agent,
    StopSign,
    SpeedBump,
    Plane,
    NumObjects,
};

// The Sim class encapsulates the per-world state of the simulation.
// Sim is always available by calling ctx.data() given a reference
// to the Engine / Context object that is passed to each ECS system.
//
// Per-World state that is frequently accessed but only used by a few
// ECS systems should be put in a singleton component rather than
// in this class in order to ensure efficient access patterns.
struct Sim : public madrona::WorldBase {
    struct Config {
        bool enableViewer;
        bool autoReset;
    };

    // Sim::registerTypes is called during initialization
    // to register all components & archetypes with the ECS.
    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    // Sim::setupTasks is called during initialization to build
    // the system task graph that will be invoked by the 
    // Manager class (src/mgr.hpp) for each step.
    static void setupTasks(madrona::TaskGraphBuilder &builder,
                           const Config &cfg);

    // The constructor is called for each world during initialization.
    // Config is global across all worlds, while WorldInit (src/init.hpp)
    // can contain per-world initialization data, created in (src/mgr.cpp)
    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &init);

    float polylineReductionThreshold;

    // EpisodeManager globally tracks episode IDs with an atomic across the
    // simulation.
    EpisodeManager *episodeMgr;

    // Simple random number generator seeded with episode ID.
    RNG rng;

    // Floor plane entity, constant across all episodes.
    Entity floorPlane;

    // Border wall entities: 3 walls to the left, up and down that define
    // play area. These are constant across all episodes.
    Entity borders[3];

    // Agent entity references. This entities live across all episodes
    // and are just reset to the start of the level on reset.
    madrona::CountT numAgents;
    Entity agents[consts::kMaxAgentCount];
    madrona::CountT numRoads;
    Entity roads[consts::kMaxRoadEntityCount];

    madrona::math::Vector2 mean;

    // Episode ID number
    int32_t curEpisodeIdx;

    // Should the environment automatically reset (generate a new episode)
    // at the end of each episode?
    bool autoReset;

    // Are we visualizing the simulation in the viewer?
    bool enableVizRender;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
