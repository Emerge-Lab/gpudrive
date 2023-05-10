#include "geo_gen.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace GPUHideSeek {

template <typename T>
struct TmpArray {
public:
    void alloc(Context &ctx, CountT maxSize) {
        mMaxSize = maxSize;
        // mItems = new T[mMaxSize];
        // mItems = (T *)TmpAllocator::get(sizeof(T) * mMaxSize);
        mItems = (T *)ctx.tmpAlloc(sizeof(T) * mMaxSize);

        mCurrentSize = 0;
    }

    void push_back(T item) {
        assert(mCurrentSize < mMaxSize);
        mItems[mCurrentSize++] = item;
    }

    T &operator[](CountT i) {
        return mItems[i];
    }

    const T &operator[](CountT i) const {
        return mItems[i];
    }

    CountT size() const {
        return mCurrentSize;
    }

private:
    T *mItems;
    CountT mMaxSize;
    CountT mCurrentSize;
};


static inline float randomFloat(RNG &rng)
{
    // return (float)rand() / (float)RAND_MAX;
    return rng.rand();
}

struct Wall {
    // p1 < p2
    math::Vector2 p1, p2;

    Wall() = default;

    // Room 0 is on left,down
    // Room 1 i s on top,right
    Wall(math::Vector2 p1_in, math::Vector2 p2_in) 
    : p1(p1_in), p2(p2_in) {
        if (p1_in.x > p2_in.x || p1_in.y > p2_in.y) {
            p1 = p2_in, p2 = p1_in;
        }
    }

    bool isHorizontal() const {
        return (fabs(p1.y - p2.y) < 0.000001f);
    }

    void resort() {
        math::Vector2 oldP1 = p1;
        math::Vector2 oldP2 = p2;
        if (oldP1.x > oldP2.x || oldP1.y > oldP2.y) {
            p1 = oldP2, p2 = oldP1;
        }
    }

    float length() const {
        if (isHorizontal()) {
            return p2.x - p1.x;
        }
        else {
            return p2.y - p1.y;
        }
    }
};

enum WallOperation {
    // Connect two walls with a perpendicular wall
    WallConnectAndAddDoor,
    WallAddDoor,
    WallMaxEnum
};

struct WallOperationSelection {
    int selectionSize;
    WallOperation operations[WallMaxEnum];

    template <typename ...T>
    WallOperationSelection(int *, T ...selections)
        : selectionSize(sizeof...(T)),
          operations{selections...}
    {}

    WallOperationSelection(int *counts) 
    : selectionSize(WallMaxEnum) {
        for (int i = 0; i < WallMaxEnum; ++i) {
            operations[i] = (WallOperation)i;
        }

        for (int i = 0; i < selectionSize; ++i) {
            if (counts[operations[i]] == 0) {
                --selectionSize;
                operations[i] = operations[selectionSize];
                --i;
            }
        }
    }

    // How many can we chose left
    WallOperation select(int *counts, RNG &rng) {
        int opIdx = rng.u32Rand() % selectionSize;
        WallOperation op = operations[opIdx];

        --counts[op];
        
        if (counts[op] == 0) {
            // Yank out this selection
            --selectionSize;
            
            operations[opIdx] = operations[selectionSize];
        }

        return op;
    }
};

struct Walls {
    TmpArray<Wall> walls;
    TmpArray<uint8_t> horizontal;
    TmpArray<uint8_t> vertical;

    inline Walls(Context &ctx, uint32_t maxAddDoors, uint32_t maxConnect) {
        walls.alloc(ctx, maxAddDoors * 3 + maxConnect * 2);
        horizontal.alloc(ctx, maxAddDoors * 3 + maxConnect * 2);
        vertical.alloc(ctx, maxAddDoors * 3 + maxConnect * 2);
    }

    inline int addWall(Wall wall) {
        if (wall.isHorizontal()) {
            // This is a horizontal wall
            horizontal.push_back((uint8_t)walls.size());
        }
        else {
            vertical.push_back((uint8_t)walls.size());
        }

        walls.push_back(wall);
        return walls.size()-1;
    }

    inline void scale(Vector2 min, Vector2 max) {
        // Scale all walls and rooms
        auto doScale = [&min, &max] (const Vector2 &v) { // v: [0,1]
            Vector2 range = max - min;
            return min + Vector2 { range.x * v.x, range.y * v.y };
        };

        for (int i = 0; i < walls.size(); ++i) {
            walls[i].p1 = doScale(walls[i].p1);
            walls[i].p2 = doScale(walls[i].p2);
        }
    }
};

int findAnotherWall(
    const Walls &walls,
    const TmpArray<uint8_t> &list, int chosenIndirectIdx,
    RNG &rng) {
    const Wall &chosen = walls.walls[list[chosenIndirectIdx]];

    if (chosen.isHorizontal()) {
        // Search for a wall
        int startIndirectIdx = chosenIndirectIdx + 1 + rng.u32Rand() % (list.size()-1);
        for (int i = 0; i < list.size() - 1; ++i) {
            int currentIndirectIdx = (startIndirectIdx + i) % list.size();
            if (currentIndirectIdx == chosenIndirectIdx) {
                currentIndirectIdx = (currentIndirectIdx + 1) % list.size();
            }

            // Make sure the walls can even be connected in the first place and have enough distance between them
            const Wall &otherWall = walls.walls[list[currentIndirectIdx]];
            if (!(chosen.p1.x >= otherWall.p2.x || chosen.p2.x <= otherWall.p1.x) && 
                chosen.length() >= 0.3f && otherWall.length() >= 0.3f) {

                // Make sure there aren't any walls between these
                float high = std::min(chosen.p2.x, otherWall.p2.x);
                float low = std::max(chosen.p1.x, otherWall.p1.x);

                bool works = true;
                for (int j = 0; j < list.size(); ++j) {
                    if (j != currentIndirectIdx) {
                        // Check that this wall isn't in between
                        float inbetweenLow = std::max(walls.walls[list[j]].p1.x, low - 0.1f);
                        float inbetweenHigh = std::min(walls.walls[list[j]].p2.x, high + 0.1f);
                        if (inbetweenLow < inbetweenHigh) {
                            float y = walls.walls[list[j]].p1.y;
                            float yMin = std::min(chosen.p1.y, otherWall.p1.y);
                            float yMax = std::max(chosen.p1.y, otherWall.p1.y);
                            if (y > yMin && y < yMax) {
                                works = false;
                                // printf("Wall in between!\n");
                                break;
                            }
                        }
                    }
                }

                if (works) return currentIndirectIdx;
            }
        }

        return -1;
    }
    else {
        // Search for a wall
        int startIndirectIdx = chosenIndirectIdx + 1 + rng.u32Rand() % (list.size()-1);
        for (int i = 0; i < list.size() - 1; ++i) {
            int currentIndirectIdx = (startIndirectIdx + i) % list.size();
            if (currentIndirectIdx == chosenIndirectIdx) {
                currentIndirectIdx = (currentIndirectIdx + 1) % list.size();
            }

            // Make sure the walls can even be connected in the first place
            const Wall &otherWall = walls.walls[list[currentIndirectIdx]];
            if (!(chosen.p1.y >= otherWall.p2.y || chosen.p2.y <= otherWall.p1.y) &&
                chosen.length() >= 0.5f && otherWall.length() >= 0.5f) {
                // Make sure there aren't any walls between these
                float high = std::min(chosen.p2.y, otherWall.p2.y);
                float low = std::max(chosen.p1.y, otherWall.p1.y);

                bool works = true;
                for (int j = 0; j < list.size(); ++j) {
                    if (j != currentIndirectIdx) {
                        // Check that this wall isn't in between
                        float inbetweenLow = std::max(walls.walls[list[j]].p1.y, low - 0.1f);
                        float inbetweenHigh = std::min(walls.walls[list[j]].p2.y, high + 0.1f);
                        if (inbetweenLow < inbetweenHigh) {
                            float x = walls.walls[list[j]].p1.x;
                            float xMin = std::min(chosen.p1.x, otherWall.p1.x);
                            float xMax = std::max(chosen.p1.x, otherWall.p1.x);
                            if (x > xMin && x < xMax) {
                                works = false;
                                // printf("Wall in between!\n");
                                break;
                            }
                        }
                    }
                }

                if (works) return currentIndirectIdx;
            }
        }

        return -1;
    }
}

static constexpr float kDoorSize = 0.1f;

// Wall has to at least be length of 3 * kDoorSize
void addDoor(Walls &walls, Wall &wall, float doorSize, RNG &rng) {
    // printf("Added door!\n");

    if (wall.isHorizontal()) {
        float low = wall.p1.x + doorSize;
        float high = wall.p2.x - doorSize;
        float rat = 0.3f + randomFloat(rng) * 0.4f;
        // Position of door
        float x = low + rat * (high-low);

        float oldP2x = wall.p2.x;
        wall.p2.x = x - doorSize*0.5f;
        wall.resort();

        // This will have the same walls as the old bigger wall
        Wall newSplit = Wall({x + doorSize * 0.5f, wall.p1.y}, {oldP2x, wall.p1.y});
        walls.addWall(newSplit);
    }
    else {
        float low = wall.p1.y + doorSize;
        float high = wall.p2.y - doorSize;
        float rat = 0.3f + randomFloat(rng) * 0.4f;
        // Position of door
        float y = low + rat * (high-low);

        float oldP2y = wall.p2.y;
        wall.p2.y = y - doorSize*0.5f;
        wall.resort();

        Wall newSplit = Wall({wall.p1.x, y + doorSize * 0.5f}, {wall.p1.x, oldP2y});
        walls.addWall(newSplit);
    }
}

void applyWallOperation(WallOperation op, Walls &walls, RNG &rng) {
    switch (op) {
        case WallConnectAndAddDoor: {
            // printf("Connected!\n");

            // First choose a random wall
            bool isHorizontal = (bool)(rng.u32Rand() % 2);
            auto *list = [&walls, &isHorizontal] () -> TmpArray<uint8_t> * {
                return isHorizontal ? &walls.horizontal : &walls.vertical;
            }();

            // Current wall
            int wallIndirectIdx = rng.u32Rand() % list->size();
            int otherWallIndirectIdx;

            int counter = 0;
            while ((otherWallIndirectIdx = findAnotherWall(walls, *list, wallIndirectIdx, rng)) == -1) {
                // Find another wall
                isHorizontal = (bool)(rng.u32Rand() % 2);
                list = [&walls, &isHorizontal] () -> TmpArray<uint8_t> * {
                    return isHorizontal ? &walls.horizontal : &walls.vertical;
                }();

                wallIndirectIdx = rng.u32Rand() % list->size();

                if (counter++ > 4) return;
            }

            Wall *first = &walls.walls[(*list)[wallIndirectIdx]];
            Wall *second = &walls.walls[(*list)[otherWallIndirectIdx]];

            if (isHorizontal) {
                // Get range
                float high = std::min(first->p2.x, second->p2.x);
                float low = std::max(first->p1.x, second->p1.x);

                // Make sure that first has a lower y coordinate
                if (first->p1.y > second->p1.y) { std::swap(first, second); std::swap(wallIndirectIdx, otherWallIndirectIdx); }

                float rat = 0.4f + randomFloat(rng) * 0.2f;
                float x = low + rat * (high-low);

                // printf("%f - %f - %f\n", low, x, high);

                // y value in p1&p2 is gonna be the same
                int newWallIdx = walls.addWall(Wall({x, first->p1.y}, {x, second->p1.y}));
                first = &walls.walls[(*list)[wallIndirectIdx]];
                second = &walls.walls[(*list)[otherWallIndirectIdx]];

                // We have to split the previous walls (adds 2 new walls)
                float firstOldP2x = first->p2.x;
                float secondOldP2x = second->p2.x;

                // the original walls have to be split - shorten first and second and add new walls for the other splits
                first->p2.x = x;
                first->resort();
                second->p2.x = x;
                second->resort();

                Wall new0 = Wall({x, first->p1.y}, {firstOldP2x, first->p1.y});
                Wall new1 = Wall({x, second->p1.y}, {secondOldP2x, second->p1.y});

                walls.addWall(new0);
                walls.addWall(new1);

                addDoor(walls, walls.walls[newWallIdx], kDoorSize, rng);
            }
            else {
                // Get range
                float high = std::min(first->p2.y, second->p2.y);
                float low = std::max(first->p1.y, second->p1.y);

                if (first->p1.x > second->p1.x) { std::swap(first, second); std::swap(wallIndirectIdx, otherWallIndirectIdx); }

                float rat = 0.4f + randomFloat(rng) * 0.2f;
                float y = low + rat * (high-low);

                // printf("%f - %f - %f\n", low, y, high);

                int newWallIdx = walls.addWall(Wall({first->p1.x, y}, {second->p1.x, y}));
                first = &walls.walls[(*list)[wallIndirectIdx]];
                second = &walls.walls[(*list)[otherWallIndirectIdx]];

                // We have to split the previous walls (adds 2 new walls)
                float firstOldP2y = first->p2.y;
                float secondOldP2y = second->p2.y;

                first->p2.y = y;
                first->resort();
                second->p2.y = y;
                second->resort();

                Wall new0 = Wall({first->p1.x, y}, {first->p1.x, firstOldP2y});
                Wall new1 = Wall({second->p1.x, y}, {second->p1.x, secondOldP2y});

                walls.addWall(new0);
                walls.addWall(new1);

                addDoor(walls, walls.walls[newWallIdx], kDoorSize, rng);
            }
        } break;

        case WallAddDoor: {
            float doorSize = kDoorSize * 2.0f;

            // Choose a random wall
            int randomWallIdx = rng.u32Rand() % walls.walls.size();
            Wall &wall = walls.walls[randomWallIdx];
            
            if (wall.length() > 3.0f * doorSize) {
                addDoor(walls, wall, doorSize, rng);
            }
        } break;

        case WallMaxEnum: {

        } break;
    }
}

Walls makeWalls(Context &ctx, RNG &rng) {
    const uint32_t maxAddDoors = 7;
    const uint32_t maxConnect = 6;

    Walls walls(ctx, maxAddDoors, maxConnect);
    walls.addWall(Wall({0.0f,0.0f}, {1.0f,0.0f}));
    walls.addWall(Wall({0.0f,0.0f}, {0.0f,1.0f}));
    walls.addWall(Wall({0.0f,1.0f}, {1.0f,1.0f}));
    walls.addWall(Wall({1.0f,1.0f}, {1.0f,0.0f}));

    int wallConnectAndAddDoorCount = 1 + rng.u32Rand() % (maxConnect-1);
    int wallAddDoorCount = 4 + rng.u32Rand() % (maxAddDoors - 4);

    int maxCounts[WallMaxEnum] = {};
    maxCounts[WallConnectAndAddDoor] = wallConnectAndAddDoorCount;
    maxCounts[WallAddDoor] = wallAddDoorCount;

    auto isFinished = [&maxCounts] () -> bool {
        bool ret = true;
        for (int i = 0; i < WallMaxEnum; ++i) {
            ret &= (maxCounts[i] == 0);
        }
        return ret;
    };

    // First selection
    WallOperationSelection selector = WallOperationSelection(maxCounts);
    WallOperation firstOp = selector.select(maxCounts, rng);
    applyWallOperation(firstOp, walls, rng);

    while (!isFinished()) {
        WallOperation op = selector.select(maxCounts, rng);
        applyWallOperation(op, walls, rng);
    }

    return walls;
}

CountT populateStaticGeometry(Engine &ctx,
                              RNG &rng,
                              Vector2 level_scale)
{
    Entity *obstacles = ctx.data().obstacles;

    Walls walls = makeWalls(ctx, rng);
    walls.scale(-level_scale, level_scale);

    // Add walls
    for (int i = 0; i < walls.walls.size(); ++i) {
        Wall &wall = walls.walls[i];

        // Wall center
        Vector3 position {
            0.5f * (wall.p1.x + wall.p2.x),
            0.5f * (wall.p1.y + wall.p2.y),
            0.f,
        };

        Diag3x3 scale;

        if (wall.isHorizontal()) {
            scale = {
                wall.p2.x - position.x, 0.2f, 1.0f
            };
        } else {
            scale = {
                0.2f, wall.p2.y - position.y, 1.0f
            };
        }

        obstacles[i] = makeDynObject(
            ctx, position, Quat::angleAxis(0, {1, 0, 0}), 3, 
            ResponseType::Static, OwnerTeam::Unownable, scale);
    }

    return walls.walls.size();
}

}
