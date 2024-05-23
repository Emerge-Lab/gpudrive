#pragma once

#include <madrona/math.hpp>
#include <madrona/types.hpp>

using namespace madrona;
using namespace madrona::math;
namespace gpudrive
{

    void forwardKinematics(Action &action, BicycleModel &model, VehicleSize &size, Rotation &rotation, Position &position, Velocity &velocity)
    {
        const float maxSpeed{std::numeric_limits<float>::max()};
        const float dt{0.1};

        auto clipSpeed = [maxSpeed](float speed)
        {
            return std::max(std::min(speed, maxSpeed), -maxSpeed);
        };
        // TODO(samk): hoist into Vector2::PolarToVector2D
        auto polarToVector2D = [](float r, float theta)
        {
            return math::Vector2{r * cosf(theta), r * sinf(theta)};
        };

        // Average speed
        const float v{clipSpeed(model.speed + 0.5f * action.acceleration * dt)};
        const float tanDelta{tanf(action.steering)};
        // Assume center of mass lies at the middle of length, then l / L == 0.5.
        const float beta{std::atan(0.5f * tanDelta)};
        const math::Vector2 d{polarToVector2D(v, model.heading + beta)};
        const float w{v * std::cos(beta) * tanDelta / size.length};

        model.position += d * dt;
        model.heading = utils::AngleAdd(model.heading, w * dt);
        model.speed = clipSpeed(model.speed + action.acceleration * dt);

        position = madrona::base::Position({.x = model.position.x, .y = model.position.y, .z = 1});
        rotation = Quat::angleAxis(model.heading, madrona::math::up);
        velocity.linear.x = model.speed * cosf(model.heading);
        velocity.linear.y = model.speed * sinf(model.heading);
        velocity.linear.z = 0;
        velocity.angular = Vector3::zero();
        velocity.angular.z = w;
    }

    void forwardWaymaxModel(Action &action, BicycleModel &model, VehicleSize &size, Rotation &rotation, Position &position, Velocity &velocity)
    {
        const float dt{0.1};
        //new_x = x + vel_x * t + 0.5 * accel * jnp.cos(yaw) * t**2
        model.position.x = model.position.x + model.velocity.linear.x * dt + 0.5 * action.acceleration * cosf(model.heading) * dt * dt;
        // new_y = y + vel_y * t + 0.5 * accel * jnp.sin(yaw) * t**2
        model.position.y = model.position.y + model.velocity.linear.y * dt + 0.5 * action.acceleration * sinf(model.heading) * dt * dt;
        // delta_yaw = steering * (speed * t + 0.5 * accel * t**2)
        float delta_yaw = action.steering * (model.speed * dt + 0.5 * action.acceleration * dt * dt);
        // new_yaw = geometry.wrap_yaws(yaw + delta_yaw)
        model.heading = utils::AngleAdd(model.heading, delta_yaw);
        // new_vel = speed + accel * t
        model.speed = model.speed + action.acceleration * dt;
        // new_vel_x = new_vel * jnp.cos(new_yaw)
        velocity.linear.x = model.speed * cosf(new_yaw);
        // new_vel_y = new_vel * jnp.sin(new_yaw)
        velocity.linear.y = model.speed * sinf(new_yaw);
        velocity.linear.z = 0;

        velocity.angular = Vector3::zero();
        velocity.angular.z = delta_yaw / dt; // Is this correct ? 

        position = madrona::base::Position({.x = model.position.x, .y = model.position.y, .z = 1});
        rotation = Quat::angleAxis(model.heading, madrona::math::up);
    }

}