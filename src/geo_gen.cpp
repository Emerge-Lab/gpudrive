#include "level_gen.hpp"
#include "geo_gen.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace GPUHideSeek {

static constexpr float EPS = 0.000001f;

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

enum class Orientation { VERTICAL, HORIZONTAL };

struct ConnectingDoor {
    // 0 -> vertical, 1 -> horizontal
    Orientation orientation;

    Vector2 start;
    Vector2 end;
};

struct EnvironmentRooms {
    TmpArray<int8_t> leafs;
    TmpArray<Room> rooms;

    uint32_t srcRoom;
    uint32_t dstRoom;
};

// This will generate all the rooms
EnvironmentRooms makeRooms(Engine &ctx, RNG &rng)
{
    Room root = {
        { 0.0f, 0.0f },
        { 1.0f, 1.0f },

        // Leaf right now with no parent
        -1, -1, -1,
        0,

        0,
        { -1, -1, -1, -1 },
        false,

        false,
        0.0f,
        {}
    };

    EnvironmentRooms data;

    data.rooms.alloc(ctx, consts::maxRooms);
    data.leafs.alloc(ctx, consts::maxRooms);

    data.rooms.push_back(root);
    data.leafs.push_back(0);

    while (data.rooms.size() < consts::maxRooms-1) {
        // Select a random room to split
        int8_t parentLeafIdx = data.leafs[rng.u32Rand() % data.leafs.size()];
        Room &parent = data.rooms[parentLeafIdx];

        // Get a split factor which determines how deep the split is (0.4-0.6)
        float splitFactor = rng.rand() * 0.2 + 0.4;

        // Select horizontal or vertical split
        bool isHorizontal = rng.u32Rand() % 2;

        parent.splitHorizontal = isHorizontal;
        parent.splitFactor = splitFactor;

        if (isHorizontal) {
            Room r0 = parent;
            Room r1 = parent;
            r0.parent = parentLeafIdx;
            r1.parent = parentLeafIdx;

            r0.offset = parent.offset;
            r0.extent = {parent.extent.x * splitFactor, parent.extent.y};
            // Replace parent's leaf index with r0
            r0.leafIdx = parent.leafIdx;
            parent.leafIdx = -1;

            // Make sure the leaf idx now points to the new room
            data.leafs[r0.leafIdx] = data.rooms.size();
            parent.splitNeg = data.rooms.size();
            data.rooms.push_back(r0);


            // For r1, we need to push a new leaf idx
            r1.offset = { r0.offset.x + r0.extent.x, r0.offset.y };
            r1.extent = { parent.extent.x - r0.extent.x, r0.extent.y };
            r1.leafIdx = data.leafs.size();
            data.leafs.push_back(data.rooms.size());
            parent.splitPlus = data.rooms.size();
            data.rooms.push_back(r1);
        }
        else {
            Room r0 = parent;
            Room r1 = parent;
            r0.parent = parentLeafIdx;
            r1.parent = parentLeafIdx;

            r0.offset = parent.offset;
            r0.extent = {parent.extent.x, parent.extent.y * splitFactor};
            // Replace parent's leaf index with r0
            r0.leafIdx = parent.leafIdx;
            parent.leafIdx = -1;

            // Make sure the leaf idx now points to the new room
            data.leafs[r0.leafIdx] = data.rooms.size();
            parent.splitNeg = data.rooms.size();
            data.rooms.push_back(r0);


            // For r1, we need to push a new leaf idx
            r1.offset = { r0.offset.x, r0.offset.y + r0.extent.y };
            r1.extent = { r0.extent.x, parent.extent.y - r0.extent.y };
            r1.leafIdx = data.leafs.size();
            data.leafs.push_back(data.rooms.size());
            parent.splitPlus = data.rooms.size();
            data.rooms.push_back(r1);
        }
    }

    // Source room is just the first one in the list
    data.srcRoom = 0;

    // Find destination room
    Room *room = &data.rooms[0];
    int8_t roomIdx = 0;
    while (room->splitPlus != -1) {
        roomIdx = room->splitPlus;
        room = &data.rooms[roomIdx];
    }

    assert(roomIdx != -1);
    data.dstRoom = (uint32_t)roomIdx;

    return data;
}

TmpArray<uint32_t> findEligibleRoomsForDoors(Engine &ctx, EnvironmentRooms &rooms)
{
    TmpArray<uint32_t> eligibleRooms;
    eligibleRooms.alloc(ctx, consts::maxRooms);

    for (int i = 0; i < rooms.leafs.size(); ++i) {
        int8_t roomIdx = rooms.leafs[i];
        assert(roomIdx != -1);
        Room &room = rooms.rooms[(uint32_t)roomIdx];
        bool isEligible = true;

        // Check that this room is eligible - this means checking that the walls are
        // not interrupted by breaks (they are contiguous).
        for (int j = 0; j < rooms.leafs.size(); ++j) {
            int8_t otherRoomIdx = rooms.leafs[j];
            assert(otherRoomIdx != -1);
            Room &other = rooms.rooms[(uint32_t)otherRoomIdx];

            // Perform comparison
            if ((other.offset.y > room.offset.y && other.offset.y < room.offset.y+room.extent.y) ||
                (other.offset.y+other.extent.y > room.offset.y && other.offset.y+other.extent.y < room.offset.y+room.extent.y)) {
                // Check if this room is to the right or left of this one
                if (fabs(other.offset.x - (room.offset.x + room.extent.x)) < EPS || 
                    fabs((other.offset.x + other.extent.x) - room.offset.x) < EPS) {
                    // The room is not eligible!
                    isEligible = false;
                    break;
                }
            }
            else if ((other.offset.x > room.offset.x && other.offset.x < room.offset.y+room.extent.x) ||
                    (other.offset.x+other.extent.x > room.offset.x && other.offset.x+other.extent.x < room.offset.x+room.extent.x)) {
                // Check if this room is on top or on the bottom of this one
                if (fabs(other.offset.y - (room.offset.y + room.extent.y)) < EPS || 
                    fabs((other.offset.y + other.extent.y) - room.offset.y) < EPS) {
                    // The room is not eligible!
                    isEligible = false;
                    break;
                }
            }
        }

        if (isEligible) {
            rooms.rooms[roomIdx].isEligible = true;
            eligibleRooms.push_back((uint32_t)roomIdx);
        }
    }
    
    return eligibleRooms;
}

void makeRoomsAwareOfDoor(EnvironmentRooms &rooms, Room &room, ConnectingDoor &door, uint32_t doorIdx) {
    for (int i = 0; i < rooms.rooms.size(); ++i) {
        Room &other = rooms.rooms[i];

        if (door.orientation == Orientation::HORIZONTAL) {
            // Check if the top or bottom of the room touch the door
            Vector2 bottomStart = room.offset;
            Vector2 bottomEnd = room.offset + Vector2{ room.extent.x, 0.0f };
            Vector2 topStart = room.offset + Vector2{ 0.0f, room.extent.y };
            Vector2 topEnd = room.offset + room.extent;

            if (fabs(bottomStart.y - door.start.y) < EPS) {
                if (bottomEnd.x >= door.start.x && door.end.x >= bottomStart.x) {
                    other.doors[other.doorCount++] = doorIdx;
                }
            }
            else if (fabs(topStart.y - door.start.y) < EPS) {
                if (topEnd.x >= door.start.x && door.end.x >= topStart.x) {
                    other.doors[other.doorCount++] = doorIdx;
                }
            }
        }
        else if (door.orientation == Orientation::VERTICAL) {
            // Check if the top or bottom of the room touch the door
            Vector2 leftStart = room.offset;
            Vector2 leftEnd = room.offset + Vector2{ 0.0f, room.extent.y };
            Vector2 rightStart = room.offset + Vector2{ room.extent.x, 0.0f };
            Vector2 rightEnd = room.offset + room.extent;

            if (fabs(leftStart.x - door.start.x) < EPS) {
                if (leftEnd.y >= door.start.y && door.end.y >= leftStart.y) {
                    other.doors[other.doorCount++] = doorIdx;
                }
            }
            else if (fabs(rightStart.x - door.start.x) < EPS) {
                if (rightEnd.y >= door.start.y && door.end.y >= rightStart.y) {
                    other.doors[other.doorCount++] = doorIdx;
                }
            }
        }
        else {
            assert(false);
        }
    }
}

void placeDoors(TmpArray<uint32_t> &eligibleRooms, EnvironmentRooms &rooms, TmpArray<ConnectingDoor> &doors)
{
    // Place the doors in the rooms that are eligible for doors
    // also make adjacent rooms aware of these doors.
    for (int i = 0; i < eligibleRooms.size(); ++i) {
        Room &room = rooms.rooms[eligibleRooms[i]];
        ConnectingDoor newDoors[4] = {
            // Left
            { Orientation::VERTICAL, room.offset, room.offset + Vector2{ 0.0f, room.extent.y } },
            // Right
            { Orientation::VERTICAL, room.offset + Vector2{room.extent.x, 0}, room.offset + room.extent },
            // Bottom
            { Orientation::HORIZONTAL, room.offset, room.offset + Vector2{ room.extent.x, 0.0f } },
            // Top
            { Orientation::HORIZONTAL, room.offset + Vector2{0, room.extent.y}, room.offset + room.extent }
        };

        // Only add these doors if they are not at the boundaries of the world
        for (int j = 0; j < 4; ++j) {
            if (newDoors[j].orientation == Orientation::VERTICAL) {
                if (!(fabs(newDoors[j].start.x - 0.0f) < EPS || fabs(newDoors[j].start.x - 1.0f) < EPS)) {
                    // Now, let's make the relevant rooms aware of this new door
                    room.doors[j] = doors.size();
                    // Make adjacent rooms aware of this door
                    makeRoomsAwareOfDoor(rooms, room, newDoors[j], doors.size());

                    doors.push_back(newDoors[j]);
                }
            }
            else {
                if (!(fabs(newDoors[j].start.y - 0.0f) < EPS || fabs(newDoors[j].start.y - 1.0f) < EPS)) {
                    // Now, let's make the relevant rooms aware of this new door
                    room.doors[j] = doors.size();
                    // Make adjacent rooms aware of this door
                    makeRoomsAwareOfDoor(rooms, room, newDoors[j], doors.size());

                    doors.push_back(newDoors[j]);
                }
            }
        }
    }
}

struct WallData {
    Vector2 start;
    Vector2 end;

    // -1 if there is no door on this wall
    int32_t doorIdx;
};

void addEligibleRoomWalls(TmpArray<uint32_t> &eligibleRooms, EnvironmentRooms &rooms, TmpArray<WallData> &verticalWalls, TmpArray<WallData> &horizontalWalls)
{
    // First, add the walls from eligible rooms (for doors)
    for (int i = 0; i < eligibleRooms.size(); ++i) {
        Room &room = rooms.rooms[eligibleRooms[i]];

        // Add left wall
        {
            // Make sure that this wall isn't contained in any other
            WallData newWall = {
                { room.offset },
                { room.offset + Vector2{0.0f, room.extent.y} },
                room.doors[0] // May be -1 or an actual door index
            };

            verticalWalls.push_back(newWall);
        }

        // Add right wall
        {
            // Make sure that this wall isn't contained in any other
            WallData newWall = {
                { room.offset + Vector2{ room.extent.x, 0.0f } },
                { room.offset + room.extent },
                room.doors[1]
            };

            verticalWalls.push_back(newWall);
        }

        // Add bottom wall
        {
            // Make sure that this wall isn't contained in any other
            WallData newWall = {
                { room.offset },
                { room.offset + Vector2{room.extent.x, 0.0f} },
                room.doors[2]
            };

            horizontalWalls.push_back(newWall);
        }

        // Add top wall
        {
            // Make sure that this wall isn't contained in any other
            WallData newWall = {
                { room.offset + Vector2{0.0f, room.extent.y} },
                { room.offset + Vector2{room.extent.x, room.extent.y} },
                room.doors[3]
            };

            horizontalWalls.push_back(newWall);
        }

        // Now order the doors in the room's door array properly
        room.doorCount = 0;
        for (int d = 0; d < consts::maxDoorsPerRoom; ++d) {
            if (room.doors[d] != -1)
                room.doors[room.doorCount++] = room.doors[d];
        }
    }
}

// If the wall shouldn't exist (duplicate), returns false, otherwise true
bool cropVerticalWall(WallData &wall, TmpArray<WallData> &verticalWalls)
{
    for (int i = 0; i < verticalWalls.size(); ++i) {
        WallData &other = verticalWalls[i];

        if (fabs(other.start.x - wall.start.x) < EPS) {
            // These are duplicates/this wall is contained in another one's
            if (wall.start.y >= other.start.y && wall.end.y <= other.end.y)
                return false;

            // Crop from bottom
            if (other.start.y >= wall.start.y && other.start.y <= wall.end.y)
                wall.start.y = other.start.y;

            if (other.end.y <= wall.end.y && other.end.y >= wall.start.y)
                wall.end.y = other.end.y;
        }
    }

    return true;
}

bool cropHorizontalWall(WallData &wall, TmpArray<WallData> &horizontalWalls)
{
    for (int i = 0; i < horizontalWalls.size(); ++i) {
        WallData &other = horizontalWalls[i];

        if (fabs(other.start.y - wall.start.y) < EPS) {
            // These are duplicates/this wall is contained in another one's
            if (wall.start.x >= other.start.x && wall.end.x <= other.end.x)
                return false;

            // Crop from bottom
            if (other.start.x >= wall.start.x && other.start.x <= wall.end.x)
                wall.start.x = other.start.x;

            if (other.end.x <= wall.end.x && other.end.x >= wall.start.x)
                wall.end.x = other.end.x;
        }
    }

    return true;
}

void addOtherRoomsWalls(EnvironmentRooms &rooms, TmpArray<WallData> &verticalWalls, TmpArray<WallData> &horizontalWalls)
{
    for (int i = 0; i < rooms.leafs.size(); ++i) {
        // Add walls but make sure not to duplicate
        Room &room = rooms.rooms[rooms.leafs[i]];

        // Add left wall
        {
            // Make sure that this wall isn't contained in any other
            WallData newWall = {
                { room.offset },
                { room.offset + Vector2{0.0f, room.extent.y} },
                -1,
            };

            if (cropVerticalWall(newWall, verticalWalls))
                verticalWalls.push_back(newWall);
        }

        // Add right wall
        {
            // Make sure that this wall isn't contained in any other
            WallData newWall = {
                { room.offset + Vector2{ room.extent.x, 0.0f } },
                { room.offset + room.extent },
                -1,
            };

            if (cropVerticalWall(newWall, verticalWalls))
                verticalWalls.push_back(newWall);
        }

        // Add bottom wall
        {
            // Make sure that this wall isn't contained in any other
            WallData newWall = {
                { room.offset },
                { room.offset + Vector2{room.extent.x, 0.0f} },
                -1,
            };

            if (cropHorizontalWall(newWall, horizontalWalls))
                horizontalWalls.push_back(newWall);
        }

        // Add top wall
        {
            // Make sure that this wall isn't contained in any other
            WallData newWall = {
                { room.offset + Vector2{0.0f, room.extent.y} },
                { room.offset + Vector2{room.extent.x, room.extent.y} },
                -1,
            };

            if (cropHorizontalWall(newWall, horizontalWalls))
                horizontalWalls.push_back(newWall);
        }
    }
}

void placeButtons(Engine &ctx, RNG &rng, EnvironmentRooms &rooms)
{
    for (int i = 0; i < rooms.leafs.size(); ++i) {
        Room &room = rooms.rooms[rooms.leafs[i]];
        
        // Find the range of where the button center could be
        float xStart = room.offset.x + BUTTON_WIDTH/2.0f;
        float yStart = room.offset.y + BUTTON_WIDTH/2.0f;

        float xEnd = room.offset.x+room.extent.x - BUTTON_WIDTH/2.0f;
        float yEnd = room.offset.y+room.extent.y - BUTTON_WIDTH/2.0f;

        float x = xStart + rng.rand() * (xEnd - xStart);
        float y = yStart + rng.rand() * (yEnd - yStart);

        room.button.start = { x - BUTTON_WIDTH/2.0f, y - BUTTON_WIDTH/2.0f };
        room.button.end = room.button.start + Vector2{BUTTON_WIDTH, BUTTON_WIDTH};
    }
}

madrona::Entity makeWallObject(Engine &ctx,
                     madrona::math::Vector3 pos,
                     madrona::math::Quat rot,
                     int32_t obj_id,
                     madrona::phys::ResponseType response_type,
                     madrona::math::Diag3x3 scale)
{
    using namespace madrona;
    using namespace madrona::math;

    Entity e = ctx.makeEntity<WallObject>();
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
    ctx.get<OpenState>(e) = {false};

    return e;
}

void populateStaticGeometry(Engine &ctx,
        RNG &rng,
        Vector2 level_scale,
        uint32_t &srcRoom,
        uint32_t &dstRoom)
{
    (void)level_scale;
    // This generated the rooms data structure as well as determine what the
    // source room and destination room is
    EnvironmentRooms rooms = makeRooms(ctx, rng);
    srcRoom = rooms.srcRoom;
    dstRoom = rooms.dstRoom;

    placeButtons(ctx, rng, rooms);

    // Get the rooms that are eligible for doors
    TmpArray<uint32_t> eligibleRooms = findEligibleRoomsForDoors(ctx, rooms);

    // Allocate doors
    TmpArray<ConnectingDoor> doors;
    doors.alloc(ctx, eligibleRooms.size() * consts::maxDoorsPerRoom);

    placeDoors(eligibleRooms, rooms, doors);

    // For now, let's split the walls into horizontal and vertical walls
    TmpArray<WallData> horizontalWalls;
    horizontalWalls.alloc(ctx, consts::maxRooms * 2);

    TmpArray<WallData> verticalWalls;
    verticalWalls.alloc(ctx, consts::maxRooms * 2);

    // The eligible rooms' walls have been added
    addEligibleRoomWalls(eligibleRooms, rooms, verticalWalls, horizontalWalls);
    // Other rooms' walls have been added
    addOtherRoomsWalls(rooms, verticalWalls, horizontalWalls);

    auto doScale = [] (const Vector2 &v, const Vector2 &min, const Vector2 &max) { // v: [0,1]
        Vector2 range = max - min;
        return min + Vector2 { range.x * v.x, range.y * v.y };
    };

    auto doDirScale = [] (const Vector2 &v, const Vector2 &min, const Vector2 &max) { // v: [0,1]
        Vector2 range = max - min;
        return Vector2 { range.x * v.x, range.y * v.y };
    };

    // Allocate rooms
    ctx.data().roomCount = rooms.rooms.size();

    // Copy the rooms from the temporary allocator into the environment context
    for (int i = 0; i < rooms.rooms.size(); ++i) {
        ctx.data().rooms[i] = rooms.rooms[i];

        // Apply scale
        ctx.data().rooms[i].offset = doScale(ctx.data().rooms[i].offset, -level_scale, level_scale);
        ctx.data().rooms[i].extent = doDirScale(ctx.data().rooms[i].extent, -level_scale, level_scale);

        ctx.data().rooms[i].button.start = doScale(ctx.data().rooms[i].button.start, -level_scale, level_scale);
        ctx.data().rooms[i].button.end = doScale(ctx.data().rooms[i].button.end, -level_scale, level_scale);
    }

    ctx.data().leafCount = rooms.leafs.size();
    for (int i = 0; i < rooms.leafs.size(); ++i) {
        ctx.data().leafs[i] = rooms.leafs[i];
    }

    // Allocate walls
    ctx.data().numWalls = horizontalWalls.size() + verticalWalls.size();

    // Allocate doors
    ctx.data().numDoors = doors.size();

    uint32_t wallCount = 0;
    // Now, we can add the walls' geometry to the scene
    for (int i = 0; i < horizontalWalls.size(); ++i) {
        WallData &wall = horizontalWalls[i];

        // Perform scale for the wall
        Vector2 start = doScale(wall.start, -level_scale, level_scale);
        Vector2 end = doScale(wall.end, -level_scale, level_scale);

        // Wall center
        Vector3 position {
            0.5f * (start.x + end.x),
            0.5f * (start.y + end.y),
            0.f,
        };

        Diag3x3 scale;
        scale = { end.x - position.x, 0.2f, 1.0f };

        CountT wall_idx = wallCount++;

        ctx.data().walls[wall_idx] = makeWallObject(
            ctx, position, Quat::angleAxis(0, {1, 0, 0}), 3, 
            ResponseType::Static, scale);

        // This wall is a door
        if (wall.doorIdx != -1) {
            // Preserve the order of the doors in the array of doors in SIM struct
            ctx.data().doors[wall.doorIdx] = ctx.data().walls[wall_idx];
        }
    }

    // Vertical walls
    for (int i = 0; i < verticalWalls.size(); ++i) {
        WallData &wall = verticalWalls[i];

        // Perform scale for the wall
        Vector2 start = doScale(wall.start, -level_scale, level_scale);
        Vector2 end = doScale(wall.end, -level_scale, level_scale);

        // Wall center
        Vector3 position {
            0.5f * (start.x + end.x),
            0.5f * (start.y + end.y),
            0.f,
        };

        Diag3x3 scale;
        scale = { end.x - position.x, 0.2f, 1.0f };

        CountT wall_idx = wallCount++;

        ctx.data().walls[wall_idx] = makeWallObject(
            ctx, position, Quat::angleAxis(0, {1, 0, 0}), 3, 
            ResponseType::Static, scale);

        if (wall.doorIdx != -1) {
            // Preserve the order of the doors in the array of doors in SIM struct
            ctx.data().doors[wall.doorIdx] = ctx.data().walls[wall_idx];
        }
    }
}

Room *containedRoom(madrona::math::Vector2 pos, Room *rooms)
{
    Room *parent = rooms;

    while (parent->splitNeg != -1) {
        if (parent->splitHorizontal) {
            if (pos.x < rooms[parent->splitPlus].offset.x) {
                // The position is contained in the left node
                parent = &rooms[parent->splitNeg];
            }
            else {
                parent = &rooms[parent->splitPlus];
            }
        }
        else {
            if (pos.y < rooms[parent->splitPlus].offset.y) {
                // The position is contained in the left node
                parent = &rooms[parent->splitNeg];
            }
            else {
                parent = &rooms[parent->splitPlus];
            }
        }
    }

    return parent;
}

bool isPressingButton(madrona::math::Vector2 pos, Room *room)
{
    return (room->button.start.x <= pos.x && pos.x <= room->button.end.x &&
        room->button.start.y <= pos.y && pos.y <= room->button.end.y);
}

}
