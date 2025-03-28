#pragma once

#include "types.hpp"
#include <cmath>
#include <madrona/components.hpp>
#include <madrona/mw_gpu_entry.hpp>

namespace madrona_gpudrive {
namespace rasterizer {

// Convert ego-relative position (meters) to grid coordinates
inline std::pair<int, int> toGridCoords(
    const madrona::math::Vector2 &rel_pos,
    float radius,
    int resolution)
{
    float scale_m = resolution / (2 * radius);
    int x_grid = static_cast<int>((rel_pos.x + radius) * scale_m);
    int y_grid = static_cast<int>((rel_pos.y + radius) * scale_m);
    return {
        std::min(std::max(0, x_grid), resolution - 1),
        std::min(std::max(0, y_grid), resolution - 1)
    };
}

// Rasterize rectangle into grid
inline void rasterizeRotatedRectangle(
    BevObservations &grid,
    const madrona::math::Vector2 &center,
    float yaw,
    float length,
    float width,
    size_t type,
    float radius,
    int resolution)
{
    float half_w = width / 2.0f;
    float half_l = length / 2.0f;

    float scale_px = (2 * radius) / resolution;

    auto [gx, gy] = toGridCoords(center, radius, resolution);

    // Sub-optimal method which finds maximum bounding box of object
    float max_side = std::max(half_w, half_l);
    int box_radius = static_cast<int>(std::ceil(
        std::sqrt(2 * (max_side * max_side)) / scale_px));
    
    float cos_yaw = std::cos(-yaw);
    float sin_yaw = std::sin(-yaw);
    
    for (int dy = -box_radius; dy <= box_radius; dy++) {
        for (int dx = -box_radius; dx <= box_radius; dx++) {
            int x = gx + dx;
            int y = gy + dy;

            if (x < 0 || x >= resolution || y < 0 || y >= resolution)
                continue;
            
            float px = x * scale_px - radius;
            float py = y * scale_px - radius;

            float local_dx = px - center.x;
            float local_dy = py - center.y;
        
            float local_x = local_dx * cos_yaw - local_dy * sin_yaw;
            float local_y = local_dx * sin_yaw + local_dy * cos_yaw;

            constexpr float epsilon = 1e-3f;

            if (std::abs(local_x) <= half_l + epsilon &&
                std::abs(local_y) <= half_w + epsilon)
            {
                grid.obs[y][x].type = type;
            }
        }
    }
}
}
}