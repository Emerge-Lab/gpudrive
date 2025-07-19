#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/custom_context.hpp>

#include "consts.hpp"
#include "types.hpp"
#include "init.hpp"
#include "rng.hpp"

namespace madrona_gpudrive {

class Engine;

/**
 * @brief This enum is used by the Sim and Manager classes to track the export slots
 * for each component exported to the training code.
 * It defines a unique identifier for each piece of data that is shared
 * between the simulation and the training environment.
 */
enum class ExportID : uint32_t {
    Reset,
    Action,
    Reward,
    Done,
    SelfObservation,
    PartnerObservations,
    AgentMapObservations,
    Lidar,
    BevObservations,
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
    WorldMeans,
    MetaData,
    DeletedAgents,
    MapName,
    ScenarioId,
    NumExports
};

/**
 * @brief Stores values for the ObjectID component that links entities to
 * render / physics assets. This helps in identifying the type of
 * simulation object, which is crucial for rendering and physics interactions.
 */
enum class SimObject : uint32_t {
    Cube,
    Agent,
    StopSign,
    SpeedBump,
    Plane,
    NumObjects,
};

/**
 * @brief Defines identifiers for different task graphs used in the simulation.
 * Task graphs are used to manage the execution of different systems in a structured
 * and efficient manner.
 */
enum class TaskGraphID : uint32_t {
    Step,
    Reset,
    NumTaskGraphs,
};

/**
 * @brief The Sim class encapsulates the per-world state of the simulation.
 * Sim is always available by calling ctx.data() given a reference
 * to the Engine / Context object that is passed to each ECS system.
 *
 * Per-World state that is frequently accessed but only used by a few
 * ECS systems should be put in a singleton component rather than
 * in this class in order to ensure efficient access patterns.
 */
struct Sim : public madrona::WorldBase {
    /**
     * @brief Configuration for the simulation.
     * This struct holds settings that are global across all worlds.
     */
    struct Config {
        const madrona::render::RenderECSBridge *renderBridge;
        bool enableLidar = false;
    };

    /**
     * @brief Sim::registerTypes is called during initialization
     * to register all components & archetypes with the ECS.
     * @param registry The ECS registry to register components and archetypes with.
     * @param cfg The simulation configuration.
     */
    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    /**
     * @brief Sim::setupTasks is called during initialization to build
     * the system task graph that will be invoked by the
     * Manager class (src/mgr.hpp) for each step.
     * @param taskgraph_mgr The task graph manager to build the task graph with.
     * @param cfg The simulation configuration.
     */
    static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr,
                           const Config &cfg);

    /**
     * @brief Defines pairs of entity types that should not be checked for collision.
     * This helps to optimize collision detection by ignoring irrelevant interactions.
     */
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

    /**
     * @brief The constructor is called for each world during initialization.
     * Config is global across all worlds, while WorldInit (src/init.hpp)
     * can contain per-world initialization data, created in (src/mgr.cpp)
     * @param ctx The engine context.
     * @param cfg The simulation configuration.
     * @param init Per-world initialization data.
     */
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

    Parameters params;

    // Episode ID number
    int32_t curEpisodeIdx;

    // Are we visualizing the simulation in the viewer?
    bool enableRender;
};

/**
 * @brief The Engine class extends the Madrona CustomContext to provide
 * simulation-specific functionalities. It serves as the main entry point
* for interacting with the simulation world.
 */
class Engine : public ::madrona::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;

    /**
     * @brief These are convenience helpers for creating renderable
     * entities when rendering isn't necessarily enabled.
     * @tparam ArchetypeT The archetype of the entity to create.
     * @return The created entity.
     */
    template <typename ArchetypeT>
    inline madrona::Entity makeRenderableEntity();

    /**
     * @brief Destroys a renderable entity.
     * @param e The entity to destroy.
     */
    inline void destroyRenderableEntity(Entity e);
};

}

#include "sim.inl"
