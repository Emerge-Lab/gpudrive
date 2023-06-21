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

    CountT complement;
};

struct EnvironmentRooms {
    TmpArray<CountT> leafs;
    TmpArray<Room> rooms;

    CountT srcRoom;
    CountT dstRoom;
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
        0,
        { },
        false,

        false,
        0.0f,
        {}
    };

    for (CountT i = 0; i < consts::maxDoorsPerRoom + Room::kTmpPadSpace; ++i)
        root.doors[i] = -1;

    EnvironmentRooms data;

    data.rooms.alloc(ctx, consts::maxRooms);
    data.leafs.alloc(ctx, consts::maxRooms);

    data.rooms.push_back(root);
    data.leafs.push_back(0);

    while (data.rooms.size() < consts::maxRooms-1) {
        // Select a random room to split
        CountT parentLeafIdx = data.leafs[rng.u32Rand() % data.leafs.size()];
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

    { // Find destination room
        Room *room = &data.rooms[0];
        CountT roomIdx = 0;
        while (room->splitNeg != -1) {
            roomIdx = room->splitNeg;
            room = &data.rooms[roomIdx];
        }

        assert(roomIdx != -1);
        data.srcRoom = roomIdx;
    }

    { // Find destination room
        Room *room = &data.rooms[0];
        CountT roomIdx = 0;
        while (room->splitPlus != -1) {
            roomIdx = room->splitPlus;
            room = &data.rooms[roomIdx];
        }

        assert(roomIdx != -1);
        data.dstRoom = roomIdx;
    }

    return data;
}

void makeDoors(Engine &ctx, EnvironmentRooms &rooms, TmpArray<ConnectingDoor> &doors)
{
    for (CountT leafIdx = 0; leafIdx < rooms.leafs.size(); ++leafIdx) {
        Room &room = rooms.rooms[rooms.leafs[leafIdx]];

        ConnectingDoor newDoors[4] = {
            // Left
            { Orientation::VERTICAL, room.offset, room.offset + Vector2{ 0.0f, room.extent.y }, 1 },
            // Right
            { Orientation::VERTICAL, room.offset + Vector2{room.extent.x, 0}, room.offset + room.extent, 0 },
            // Bottom
            { Orientation::HORIZONTAL, room.offset, room.offset + Vector2{ room.extent.x, 0.0f }, 3 },
            // Top
            { Orientation::HORIZONTAL, room.offset + Vector2{0, room.extent.y}, room.offset + room.extent, 2 }
        };

        for (CountT localDoorIdx = 0; localDoorIdx < 4; ++localDoorIdx) {
            ConnectingDoor &newDoor = newDoors[localDoorIdx];

            
            for (CountT otherLeafIdx = 0; otherLeafIdx < rooms.leafs.size(); ++otherLeafIdx) {
                if (otherLeafIdx == leafIdx) continue;

                Room &otherRoom = rooms.rooms[rooms.leafs[otherLeafIdx]];

            }
        }
    }
}

TmpArray<CountT> findEligibleRoomsForDoors(Engine &ctx, EnvironmentRooms &rooms)
{
    TmpArray<CountT> eligibleRooms;
    eligibleRooms.alloc(ctx, consts::maxRooms);

    for (int i = 0; i < rooms.leafs.size(); ++i)
        eligibleRooms.push_back(rooms.leafs[i]);

    return eligibleRooms;
}

void makeRoomsAwareOfDoor(EnvironmentRooms &rooms, CountT roomIdx, Room &room, ConnectingDoor &door, CountT doorIdx) {
    for (CountT i = 0; i < rooms.leafs.size(); ++i) {
        if (rooms.leafs[i] == roomIdx)
            continue;

        Room &other = rooms.rooms[rooms.leafs[i]];

        if (door.orientation == Orientation::HORIZONTAL) {
            // Check if the top or bottom of the room touch the door
            Vector2 bottomStart = other.offset;
            Vector2 bottomEnd = other.offset + Vector2{ other.extent.x, 0.0f };
            Vector2 topStart = other.offset + Vector2{ 0.0f, other.extent.y };
            Vector2 topEnd = other.offset + other.extent;

            if (fabs(bottomStart.y - door.start.y) < EPS) {
                if (bottomEnd.x > door.start.x && door.end.x > bottomStart.x) {
                    other.addDoor(door.complement, doorIdx);
                }
            }
            else if (fabs(topStart.y - door.start.y) < EPS) {
                if (topEnd.x > door.start.x && door.end.x > topStart.x) {
                    other.addDoor(door.complement, doorIdx);
                }
            }
        }
        else if (door.orientation == Orientation::VERTICAL) {
            // Check if the top or bottom of the room touch the door
            Vector2 leftStart = other.offset;
            Vector2 leftEnd = other.offset + Vector2{ 0.0f, other.extent.y };
            Vector2 rightStart = other.offset + Vector2{ other.extent.x, 0.0f };
            Vector2 rightEnd = other.offset + other.extent;

            if (fabs(leftStart.x - door.start.x) < EPS) {
                if (leftEnd.y > door.start.y && door.end.y > leftStart.y) {
                    other.addDoor(door.complement, doorIdx);
                }
            }
            else if (fabs(rightStart.x - door.start.x) < EPS) {
                if (rightEnd.y > door.start.y && door.end.y > rightStart.y) {
                    other.addDoor(door.complement, doorIdx);
                }
            }
        }
        else {
            assert(false);
        }
    }
}

CountT numRoomsSharingBoundary(EnvironmentRooms &rooms, Vector2 start, Vector2 end, Orientation orientation)
{
    CountT sharedCount = 0;
    for (CountT i = 0; i < rooms.leafs.size(); ++i) {
        Room &room = rooms.rooms[rooms.leafs[i]];

        ConnectingDoor newDoors[4] = {
            // Left
            { Orientation::VERTICAL, room.offset, room.offset + Vector2{ 0.0f, room.extent.y }, 1 },
            // Right
            { Orientation::VERTICAL, room.offset + Vector2{room.extent.x, 0}, room.offset + room.extent, 0 },
            // Bottom
            { Orientation::HORIZONTAL, room.offset, room.offset + Vector2{ room.extent.x, 0.0f }, 3 },
            // Top
            { Orientation::HORIZONTAL, room.offset + Vector2{0, room.extent.y}, room.offset + room.extent, 2 }
        };

        for (CountT j = 0; j < 4; ++j) {
            if (orientation == Orientation::HORIZONTAL) {
                if (fabs(start.y - newDoors[j].start.y) < EPS && end.x > newDoors[j].start.x && newDoors[j].end.x > start.x) {
                    sharedCount++;
                    break;
                }
            }
            else {
                if (fabs(start.x - newDoors[j].start.x) < EPS && end.y > newDoors[j].start.y && newDoors[j].end.y > start.y) {
                    sharedCount++;
                    break;
                }
            }
        }
    }

    return sharedCount;
}

void placeDoors(TmpArray<CountT> &eligibleRooms, EnvironmentRooms &rooms, TmpArray<ConnectingDoor> &doors)
{
    // Place the doors in the rooms that are eligible for doors
    // also make adjacent rooms aware of these doors.
    for (CountT i = 0; i < eligibleRooms.size(); ++i) {
        Room &room = rooms.rooms[eligibleRooms[i]];
        ConnectingDoor newDoors[4] = {
            // Left
            { Orientation::VERTICAL, room.offset, room.offset + Vector2{ 0.0f, room.extent.y }, 1 },
            // Right
            { Orientation::VERTICAL, room.offset + Vector2{room.extent.x, 0}, room.offset + room.extent, 0 },
            // Bottom
            { Orientation::HORIZONTAL, room.offset, room.offset + Vector2{ room.extent.x, 0.0f }, 3 },
            // Top
            { Orientation::HORIZONTAL, room.offset + Vector2{0, room.extent.y}, room.offset + room.extent, 2 }
        };

        // Only add these doors if they are not at the boundaries of the world and if they don't already exist
        for (CountT j = 0; j < 4; ++j) {
            if (room.doors[j] != -1)
                continue;

            bool alreadyExists = false;
            for (CountT d = 0; d < doors.size(); ++d) {
                ConnectingDoor &existing = doors[d];
                ConnectingDoor &newDoor = newDoors[j];

                if ((existing.start - newDoor.start).length2() < EPS && (existing.end - newDoor.end).length2() < EPS) {
                    room.addDoor(j, d);
                    alreadyExists = true;
                }
            }

            if (alreadyExists) 
                continue;

            if (numRoomsSharingBoundary(rooms, newDoors[j].start, newDoors[j].end, newDoors[j].orientation) > 2)
                continue;

            if (newDoors[j].orientation == Orientation::VERTICAL) {
                if (!(fabs(newDoors[j].start.x - 0.0f) < EPS || fabs(newDoors[j].start.x - 1.0f) < EPS)) {
                    // Now, letos make the relevant rooms aware of this new door
                    room.addDoor(j, doors.size());
                    // Make adjacent rooms aware of this door
                    makeRoomsAwareOfDoor(rooms, eligibleRooms[i], room, newDoors[j], doors.size());

                    doors.push_back(newDoors[j]);
                }
            }
            else {
                if (!(fabs(newDoors[j].start.y - 0.0f) < EPS || fabs(newDoors[j].start.y - 1.0f) < EPS)) {
                    // Now, let's make the relevant rooms aware of this new door
                    room.addDoor(j, doors.size());
                    // Make adjacent rooms aware of this door
                    makeRoomsAwareOfDoor(rooms, eligibleRooms[i], room, newDoors[j], doors.size());

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
    CountT doorIdx;
};

// If the wall shouldn't exist (duplicate), returns false, otherwise true
bool cropVerticalWall(WallData &wall, TmpArray<WallData> &verticalWalls, TmpArray<ConnectingDoor> &doors)
{
    for (CountT i = 0; i < verticalWalls.size(); ++i) {
        WallData &other = verticalWalls[i];

        if (fabs(other.start.x - wall.start.x) < EPS) {
            // These are duplicates/this wall is contained in another one's
            if (wall.start.y >= other.start.y && wall.end.y <= other.end.y)
                return false;

            // Crop from bottom
            if (other.start.y > wall.start.y && other.start.y < wall.end.y)
                wall.end.y = other.start.y;

            if (other.end.y < wall.end.y && other.end.y > wall.start.y)
                wall.start.y = other.end.y;
        }
    }

    for (CountT i = 0; i < doors.size(); ++i) {
        ConnectingDoor &other = doors[i];
        if (other.orientation == Orientation::HORIZONTAL)
            continue;

        if (fabs(other.start.x - wall.start.x) < EPS) {
            // These are duplicates/this wall is contained in another one's
            if (wall.start.y >= other.start.y && wall.end.y <= other.end.y)
                return false;

            // Crop from bottom
            if (other.start.y > wall.start.y && other.start.y < wall.end.y)
                wall.end.y = other.start.y;

            if (other.end.y < wall.end.y && other.end.y > wall.start.y)
                wall.start.y = other.end.y;
        }
    }

    return true;
}

bool cropHorizontalWall(WallData &wall, TmpArray<WallData> &horizontalWalls, TmpArray<ConnectingDoor> &doors)
{
    for (CountT i = 0; i < horizontalWalls.size(); ++i) {
        WallData &other = horizontalWalls[i];

        if (fabs(other.start.y - wall.start.y) < EPS) {
            // These are duplicates/this wall is contained in another one's
            if (wall.start.x >= other.start.x && wall.end.x <= other.end.x)
                return false;

            // Crop from bottom
            if (other.start.x > wall.start.x && other.start.x < wall.end.x)
                wall.end.x = other.start.x;

            if (other.end.x < wall.end.x && other.end.x > wall.start.x)
                wall.start.x = other.end.x;
        }
    }

    for (CountT i = 0; i < doors.size(); ++i) {
        ConnectingDoor &other = doors[i];
        if (other.orientation == Orientation::VERTICAL)
            continue;

        if (fabs(other.start.x - wall.start.x) < EPS) {
            // These are duplicates/this wall is contained in another one's
            if (wall.start.y >= other.start.y && wall.end.y <= other.end.y)
                return false;

            // Crop from bottom
            if (other.start.y > wall.start.y && other.start.y < wall.end.y)
                wall.end.y = other.start.y;

            if (other.end.y < wall.end.y && other.end.y > wall.start.y)
                wall.start.y = other.end.y;
        }
    }

    return true;
}

void addEligibleRoomWalls(TmpArray<CountT> &eligibleRooms, EnvironmentRooms &rooms, 
                          TmpArray<WallData> &verticalWalls, TmpArray<WallData> &horizontalWalls,
                          TmpArray<ConnectingDoor> &doors)
{
    // First, add the walls from eligible rooms (for doors)
    for (CountT i = 0; i < eligibleRooms.size(); ++i) {
        Room &room = rooms.rooms[eligibleRooms[i]];

        // Add left wall
        {
            // Make sure that this wall isn't contained in any other
            WallData newWall = {
                { room.offset },
                { room.offset + Vector2{0.0f, room.extent.y} },
                room.doors[0] // May be -1 or an actual door index
            };

            if (cropVerticalWall(newWall, verticalWalls, doors))
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

            if (cropVerticalWall(newWall, verticalWalls, doors))
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

            if (cropHorizontalWall(newWall, horizontalWalls, doors))
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

            if (cropHorizontalWall(newWall, horizontalWalls, doors))
                horizontalWalls.push_back(newWall);
        }

        // Now order the doors in the room's door array properly
        room.doorCount = 0;
        for (CountT d = 0; d < consts::maxDoorsPerRoom + Room::kTmpPadSpace; ++d) {
            if (room.doors[d] != -1)
                room.doors[room.doorCount++] = room.doors[d];
        }
    }
}

void placeButtons(Engine &ctx, RNG &rng, EnvironmentRooms &rooms)
{
    for (CountT i = 0; i < rooms.leafs.size(); ++i) {
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
        CountT &srcRoom,
        CountT &dstRoom)
{
    (void)level_scale;
    // This generated the rooms data structure as well as determine what the
    // source room and destination room is
    EnvironmentRooms rooms = makeRooms(ctx, rng);
    srcRoom = rooms.srcRoom;
    dstRoom = rooms.dstRoom;

    placeButtons(ctx, rng, rooms);

    // Get the rooms that are eligible for doors
    TmpArray<CountT> eligibleRooms = findEligibleRoomsForDoors(ctx, rooms);

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
    addEligibleRoomWalls(eligibleRooms, rooms, verticalWalls, horizontalWalls, doors);

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
    for (CountT i = 0; i < rooms.rooms.size(); ++i) {
        ctx.data().rooms[i] = rooms.rooms[i];

        // Apply scale
        ctx.data().rooms[i].offset = doScale(ctx.data().rooms[i].offset, -level_scale, level_scale);
        ctx.data().rooms[i].extent = doDirScale(ctx.data().rooms[i].extent, -level_scale, level_scale);

        ctx.data().rooms[i].button.start = doScale(ctx.data().rooms[i].button.start, -level_scale, level_scale);
        ctx.data().rooms[i].button.end = doScale(ctx.data().rooms[i].button.end, -level_scale, level_scale);
    }

    ctx.data().leafCount = rooms.leafs.size();
    for (CountT i = 0; i < rooms.leafs.size(); ++i) {
        ctx.data().leafs[i] = rooms.leafs[i];
    }

    // Allocate walls
    ctx.data().numWalls = horizontalWalls.size() + verticalWalls.size();

    // Allocate doors
    ctx.data().numDoors = doors.size();

    CountT wallCount = 0;
    
    // Now, we can add the walls' geometry to the scene
    for (CountT i = 0; i < horizontalWalls.size(); ++i) {
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
    }

    // Vertical walls
    for (CountT i = 0; i < verticalWalls.size(); ++i) {
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
        scale = { 0.2f, end.y - position.y, 1.0f };

        CountT wall_idx = wallCount++;

        ctx.data().walls[wall_idx] = makeWallObject(
            ctx, position, Quat::angleAxis(0, {1, 0, 0}), 3, 
            ResponseType::Static, scale);
    }

    // Doors
    for (CountT i = 0; i < doors.size(); ++i) {
        ConnectingDoor &door = doors[i];

        // Perform scale for the wall
        Vector2 start = doScale(door.start, -level_scale, level_scale);
        Vector2 end = doScale(door.end, -level_scale, level_scale);

        // Wall center
        Vector3 position {
            0.5f * (start.x + end.x),
            0.5f * (start.y + end.y),
            0.f,
        };

        Diag3x3 scale;
        if (door.orientation == Orientation::VERTICAL)
            scale = { 0.2f, end.y - position.y, 1.0f };
        else
            scale = { end.x - position.x, 0.2f, 1.0f };

        ctx.data().doors[i] = makeWallObject(
            ctx, position, Quat::angleAxis(0, {1, 0, 0}), 3, 
            ResponseType::Static, scale);
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
