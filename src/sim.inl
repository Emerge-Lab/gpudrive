namespace madrona_gpudrive {

template <typename ArchetypeT>
madrona::Entity Engine::makeRenderableEntity()
{
    Entity e = makeEntity<ArchetypeT>();
    if (data().enableRender) {
        madrona::render::RenderingSystem::makeEntityRenderable(*this, e);
    }
    
    return e;
}

inline void Engine::destroyRenderableEntity(Entity e)
{
    if (data().enableRender) {
        madrona::render::RenderingSystem::cleanupRenderableEntity(*this, e);
    }
    destroyEntity(e);
}

}
