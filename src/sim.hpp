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
    AgentMapObservations,
    Lidar,
    StepsRemaining,
    BicycleModel,
    MapObservation,
    Shape,
    ControlledState,
    AbsoluteSelfObservation,
    ValidState,
    Info,
    ResponseType,
    Trajectory,
    Map,
    ResetMap,
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

enum class TaskGraphID : uint32_t {
    Step,
    Reset,
    NumTaskGraphs,
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
        const madrona::render::RenderECSBridge *renderBridge;
        bool enableLidar = false;
    };

    // Sim::registerTypes is called during initialization
    // to register all components & archetypes with the ECS.
    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    // Sim::setupTasks is called during initialization to build
    // the system task graph that will be invoked by the 
    // Manager class (src/mgr.hpp) for each step.
    static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr,
                           const Config &cfg);

   const std::pair<EntityType,EntityType> collisionPairs[20] = {                                                                                                 
                                                              {EntityType::Pedestrian, EntityType::RoadEdge},                                                                                                          
                                                              {EntityType::Pedestrian, EntityType::RoadLine},                                                                                                                
                                                              {EntityType::Pedestrian, EntityType::RoadLane},                                                                                                                
                                                              {EntityType::Pedestrian, EntityType::CrossWalk},                                                                                                               
                                                              {EntityType::Pedestrian, EntityType::SpeedBump},                                                                                                                
                                                              {EntityType::Cyclist, EntityType::RoadEdge},                                                                                                                   
                                                              {EntityType::Cyclist, EntityType::RoadLine},                                                                                                                   
                                                              {EntityType::Cyclist, EntityType::RoadLane},                                                                                                                   
                                                              {EntityType::Cyclist, EntityType::CrossWalk},                                                                                                                  
                                                              {EntityType::Cyclist, EntityType::SpeedBump},                                                                                                                 
                                                              {EntityType::Vehicle, EntityType::CrossWalk},                                                                                                                  
                                                              {EntityType::Vehicle, EntityType::SpeedBump},                                                                                                                  
                                                              {EntityType::Vehicle, EntityType::RoadLine},                                                                                                                   
                                                              {EntityType::Vehicle, EntityType::RoadLane}};                   

    // The constructor is called for each world during initialization.
    // Config is global across all worlds, while WorldInit (src/init.hpp)
    // can contain per-world initialization data, created in (src/mgr.cpp)
    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &init);

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

    Entity agent_ifaces[consts::kMaxAgentCount];
    Entity road_ifaces[consts::kMaxRoadEntityCount];

    Entity camera_agent;

    madrona::CountT numControlledAgents;

    madrona::math::Vector2 mean;

    Parameters params;

    // Episode ID number
    int32_t curEpisodeIdx;

    // Are we visualizing the simulation in the viewer?
    bool enableRender;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;

    // These are convenience helpers for creating renderable
    // entities when rendering isn't necessarily enabled
    template <typename ArchetypeT>
    inline madrona::Entity makeRenderableEntity();
    inline void destroyRenderableEntity(Entity e);
};

}

#include "sim.inl"
