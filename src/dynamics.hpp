#pragma once

#include <madrona/math.hpp>
#include <madrona/types.hpp>

using namespace madrona;
using namespace madrona::math;
namespace gpudrive
{

    inline void forwardKinematics(Action &action, VehicleSize &size, Rotation &rotation, Position &position, Velocity &velocity)
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

        float speed = velocity.linear.length();
        float yaw = utils::quatToYaw(rotation);
        // Average speed
        const float v{clipSpeed(speed + 0.5f * action.acceleration * dt)};
        const float tanDelta{tanf(action.steering)};
        // Assume center of mass lies at the middle of length, then l / L == 0.5.
        const float beta{std::atan(0.5f * tanDelta)};
        const math::Vector2 d{polarToVector2D(v, yaw + beta)};
        const float w{v * std::cos(beta) * tanDelta / size.length};

        // model.position += d * dt;
        // model.heading = utils::AngleAdd(model.heading, w * dt);
        // model.speed = clipSpeed(model.speed + action.acceleration * dt);
        float new_yaw = utils::AngleAdd(yaw, w * dt);
        float new_speed = clipSpeed(speed + action.acceleration * dt);
        position.x += d.x * dt;
        position.y += d.y * dt;
        position.z = 1;
        rotation = Quat::angleAxis(new_yaw, madrona::math::up);
        velocity.linear.x = new_speed * cosf(new_yaw);
        velocity.linear.y = new_speed * sinf(new_yaw);
        velocity.linear.z = 0;
        velocity.angular = Vector3::zero();
        velocity.angular.z = w;
    }

    inline void forwardBicycleModel(Action &action, Rotation &rotation, Position &position, Velocity &velocity)
    {
        // Clip acceleration and steering
        action.acceleration = fmaxf(-6.0, fminf(action.acceleration, 6.0));
        action.steering = fmaxf(-3.0, fminf(action.steering, 3.0));

        const float dt{0.1};
        float yaw = utils::quatToYaw(rotation);
        float speed = velocity.linear.length();
        //new_x = x + vel_x * t + 0.5 * accel * jnp.cos(yaw) * t**2
        position.x = position.x + velocity.linear.x * dt + 0.5 * action.acceleration * cosf(yaw) * dt * dt;
        // new_y = y + vel_y * t + 0.5 * accel * jnp.sin(yaw) * t**2
        position.y = position.y + velocity.linear.y * dt + 0.5 * action.acceleration * sinf(yaw) * dt * dt;
        // delta_yaw = steering * (speed * t + 0.5 * accel * t**2)
        float delta_yaw = action.steering * (speed * dt + 0.5 * action.acceleration * dt * dt);
        // new_yaw = geometry.wrap_yaws(yaw + delta_yaw)
        float new_yaw = utils::AngleAdd(yaw, delta_yaw);
        // new_vel = speed + accel * t
        float new_speed = speed + action.acceleration * dt;
        // new_vel_x = new_vel * jnp.cos(new_yaw)
        velocity.linear.x = new_speed * cosf(new_yaw);
        // new_vel_y = new_vel * jnp.sin(new_yaw)
        velocity.linear.y = new_speed * sinf(new_yaw);
        velocity.linear.z = 0;

        velocity.angular = Vector3::zero();
        velocity.angular.z = delta_yaw / dt; // Is this correct ? 

        rotation = Quat::angleAxis(new_yaw, madrona::math::up);
    }

    inline void forwardDeltaModel(DeltaAction &action, Rotation &rotation, Position &position, Velocity &velocity)
    {
        // start delta model
            // start DeltaLocal
        const float dt{0.1};
        float yaw = utils::quatToYaw(rotation);
        // rotated_xy = jnp.matmul(rotation_mat, action.data[..., :2, jnp.newaxis], precision='float32')[..., 0]
        // From https://en.wikipedia.org/wiki/Rotation_matrix
        float cos = std::cos(yaw);
        float sin = std::sin(yaw);
        float speed = velocity.linear.length();
        // x = c * x - s * y
        // y = s * x + c * y
        float dx = action.dx * cos - action.dy * sin;
        float dy = action.dx * sin + action.dy * cos;
            // end DeltaLocal
            // start DeltaGlobal
        position.x = position.x + dx;
        position.y = position.y + dy;


        velocity.linear.x = dx / dt;
        velocity.linear.y = dy / dt;
        velocity.linear.z = 0;
        velocity.angular = Vector3::zero();
        velocity.angular.z = action.dyaw / dt; // Is this correct ?
            // end DeltaGlobal
        float new_yaw = utils::AngleAdd(yaw, action.dyaw);
        // end delta model

        rotation = Quat::angleAxis(new_yaw, madrona::math::up);

    }

    inline Action inverseBicycleModel(const Rotation &rotation, const Velocity &velocity, const Rotation &targetRotation, const Velocity &targetVelocity)
    {
        const float dt{0.1};

        Action action = {0, 0, 0};
        float speed = velocity.linear.length();
        float target_speed = targetVelocity.linear.length();

        // accel = (new_vel - vel) / dt
        action.acceleration = (target_speed - speed) / dt;

        float yaw = utils::NormalizeAngle<float>(utils::quatToYaw(rotation));
        float target_yaw = utils::NormalizeAngle<float>(utils::quatToYaw(targetRotation));

        if(consts::useEstimatedYaw)
        {
            target_yaw = atan2f(targetVelocity.linear.y, targetVelocity.linear.x);
        }

        // steering = (new_yaw - yaw) / (speed * dt + 1/2 * accel * dt ** 2)
        float denominator = speed * dt + 0.5 * action.acceleration * dt * dt;
        if (denominator != 0)
        {
            action.steering = (target_yaw - yaw) / denominator;
        }
        else
        {
            action.steering = 0;
        }
        
        return action;

    }

    inline DeltaAction inverseDeltaModel(const Rotation &rotation, const Position &position, const Rotation &targetRotation, const Position &targetPosition)
    {
        const float dt{0.1};

        DeltaAction action = {0, 0, 0};
        float yaw = utils::quatToYaw(rotation);
        float target_yaw = utils::quatToYaw(targetRotation);
        // start delta model
            // start DeltaGlobal
        action.dx = targetPosition.x - position.x;
        action.dy = targetPosition.y - position.y;
        action.dyaw = target_yaw - yaw;

        action.dx  = fmaxf(-6.0, fminf(action.dx, 6.0));
        action.dy = fmaxf(-6.0, fminf(action.dy, 6.0));
            // end DeltaGlobal
            // start DeltaLocal

        // rotated_xy = jnp.matmul(rotation_mat, action.data[..., :2, jnp.newaxis], precision='float32')[..., 0]
        // From https://en.wikipedia.org/wiki/Rotation_matrix
        float cos = std::cos(-yaw);
        float sin = std::sin(-yaw);
        // x = c * x - s * y
        // y = s * x + c * y
        float local_dx= action.dx * cos - action.dy * sin;
        float local_dy = action.dx * sin + action.dy * cos;

        action.dx = fmaxf(-6.0, fminf(local_dx, 6.0));
        action.dy = fmaxf(-6.0, fminf(local_dy, 6.0));
        action.dyaw = fmaxf(-3.14, fminf(action.dyaw, 3.14));
            // end DeltaLocal
        // end delta model

        return action;

    }

}