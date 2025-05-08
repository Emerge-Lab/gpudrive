#pragma once

#include <madrona/math.hpp>
#include <madrona/types.hpp>

using namespace madrona;
using namespace madrona::math;
namespace madrona_gpudrive
{

    inline void forwardKinematics(const Action &action, VehicleSize &size, Rotation &rotation, Position &position, Velocity &velocity)
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
        const float v{clipSpeed(speed + 0.5f * action.classic.acceleration * dt)};
        const float tanDelta{tanf(action.classic.steering)};
        // Assume center of mass lies at the middle of length, then l / L == 0.5.
        const float beta{std::atan(0.5f * tanDelta)};
        const math::Vector2 d{polarToVector2D(v, yaw + beta)};
        const float w{v * std::cos(beta) * tanDelta / size.length};

        // model.position += d * dt;
        // model.heading = utils::AngleAdd(model.heading, w * dt);
        // model.speed = clipSpeed(model.speed + action.acceleration * dt);
        float new_yaw = utils::AngleAdd(yaw, w * dt);
        float new_speed = clipSpeed(speed + action.classic.acceleration * dt);
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

    inline void forwardJerkModel(Action &action, VehicleSize &size, Rotation &rotation, Position &position, Velocity &velocity)
    {
        // babs don't hardcode this
        const float dt{0.1};
        // We are going to call them accel and steer but it is really lateral and longitudinal jerk
        // TODO(ev) add missing coefficients C
        new_accel_long = accel.longitudinal + action.longitudinal_jerk * dt;
        new_acc_later = accel.lateral + action.lateral_jerk * dt;
        // clip the accels to reasonable values
        // TODO(ev) add missing coefficients C
        new_accel_long = fmaxf(-5.0, fminf(new_accel_long, 2.5 * 1.5));
        new_accel_lateral = fmaxf(-4.0, fminf(new_accel_lateral, 4.0));
        // TODO(ev) velocity clipping?
        // update the speed
        float new_speed = velocity.linear.length() + (accel.longitudinal + new_accel_long) * dt;
        new_speed = fmaxf(-2.0, fminf(new_speed, 30.0));

        // rho_inv = a_lat / max(v^2, 1e-5)
        // TODO(ev) add squared value don't just multiply like this
        rho_inv = new_accel_lateral / fmaxf(new_speed * new_speed, 1e-5);
        // rho_inv = sign(rho_inv) * fmaxf(fabs(rho_inv), 1e-5)
        // TODO(ev) is this safe? What does copysign do at zero?
        rho_inv = copysignf(1.0, rho_inv) * fmaxf(fabsf(rho_inv), 1e-5);
        // phi = arctan(rho_inv * length of wheelbase)
        phi = atan2f(rho_inv * size.length, 1.0);

        // Lets also compute phi_prev
        rho_inv_prev = accel.lateral / fmaxf(velocity.linear.length() * velocity.linear.length(), 1e-5);
        rho_inv_prev = copysignf(1.0, rho_inv_prev) * fmaxf(fabsf(rho_inv_prev), 1e-5);
        phi_prev = atan2f(rho_inv_prev * size.length, 1.0);

        // delta_phi = clip(\phi - phi_prev, -0.6 delta t, 0.6 delta t)
        // TODO(ev) keep track of phi_prev
        delta_phi = fmaxf(-0.6 * dt, fminf(phi - phi_prev, 0.6 * dt));
        // phi_t = clip(\phi_prev + delta_phi, -0.55, 0.55)
        phi_t = fmaxf(-0.55, fminf(phi_prev + delta_phi, 0.55));

        // rho_inv = tan(phi_t) / length of wheelbase
        rho_inv = tanf(phi_t) / size.length;
        // a_lat = v(t)^2 * rho_inv
        accel.lateral = new_speed * new_speed * rho_inv;
        // store the previous velocity as well
        d = 0.5 * (velocity.linear.length() + new_speed) * dt;
        theta = d * rho_inv;
        // Update yaw
        float yaw = utils::quatToYaw(rotation);
        float new_yaw = utils::AngleAdd(theta, yaw);
        // end delta model
        rotation = Quat::angleAxis(new_yaw, madrona::math::up);
        // Increment delta x by rho sin(\theta)
        position.x += rho_inv * sinf(theta);
        // Increment delta y by rho cos(\theta)
        position.y += rho_inv * cosf(theta);
        // Update the speed
        // new_vel_x = new_vel * jnp.cos(new_yaw)
        velocity.linear.x = new_speed * cosf(new_yaw);
        // new_vel_y = new_vel * jnp.sin(new_yaw)
        velocity.linear.y = new_speed * sinf(new_yaw);
        velocity.linear.z = 0;
        // And now we actually update the accels too
        accel.longitudinal = new_accel_long;
        accel.lateral = new_accel_lateral;

    }

    inline void forwardBicycleModel(Action &action, Rotation &rotation, Position &position, Velocity &velocity)
    {
        // Clip acceleration and steering
        action.classic.acceleration = fmaxf(-6.0, fminf(action.classic.acceleration, 6.0));
        action.classic.steering = fmaxf(-3.0, fminf(action.classic.steering, 3.0));

        const float dt{0.1};
        float yaw = utils::quatToYaw(rotation);
        float speed = velocity.linear.length();
        //new_x = x + vel_x * t + 0.5 * accel * jnp.cos(yaw) * t**2
        position.x = position.x + velocity.linear.x * dt + 0.5 * action.classic.acceleration * cosf(yaw) * dt * dt;
        // new_y = y + vel_y * t + 0.5 * accel * jnp.sin(yaw) * t**2
        position.y = position.y + velocity.linear.y * dt + 0.5 * action.classic.acceleration * sinf(yaw) * dt * dt;
        // delta_yaw = steering * (speed * t + 0.5 * accel * t**2)
        float delta_yaw = action.classic.steering * (speed * dt + 0.5 * action.classic.acceleration * dt * dt);
        // new_yaw = geometry.wrap_yaws(yaw + delta_yaw)
        float new_yaw = utils::AngleAdd(yaw, delta_yaw);
        // new_vel = speed + accel * t
        float new_speed = speed + action.classic.acceleration * dt;
        // new_vel_x = new_vel * jnp.cos(new_yaw)
        velocity.linear.x = new_speed * cosf(new_yaw);
        // new_vel_y = new_vel * jnp.sin(new_yaw)
        velocity.linear.y = new_speed * sinf(new_yaw);
        velocity.linear.z = 0;

        velocity.angular = Vector3::zero();
        velocity.angular.z = delta_yaw / dt; // Is this correct ? 

        rotation = Quat::angleAxis(new_yaw, madrona::math::up);
    }

    inline void forwardDeltaModel(Action &action, Rotation &rotation, Position &position, Velocity &velocity)
    {
        // start delta model
            // start DeltaLocal
        const float dt{0.1};
        float yaw = utils::quatToYaw(rotation);
        // rotated_xy = jnp.matmul(rotation_mat, action.data[..., :2, jnp.newaxis], precision='float32')[..., 0]
        // From https://en.wikipedia.org/wiki/Rotation_matrix
        float cos = std::cos(yaw);
        float sin = std::sin(yaw);

        // x = c * x - s * y
        // y = s * x + c * y
        float dx = action.delta.dx * cos - action.delta.dy * sin;
        float dy = action.delta.dx * sin + action.delta.dy * cos;
            // end DeltaLocal
            // start DeltaGlobal
        position.x = position.x + dx;
        position.y = position.y + dy;


        velocity.linear.x = dx / dt;
        velocity.linear.y = dy / dt;
        velocity.linear.z = 0;
        velocity.angular = Vector3::zero();
        velocity.angular.z = action.delta.dyaw / dt; // Is this correct ?
            // end DeltaGlobal
        float new_yaw = utils::AngleAdd(yaw, action.delta.dyaw);
        // end delta model

        rotation = Quat::angleAxis(new_yaw, madrona::math::up);

    }

    inline Action inverseBicycleModel(const Rotation &rotation, const Velocity &velocity, const Rotation &targetRotation, const Velocity &targetVelocity)
    {
        const float dt{0.1};

        Action action = {.classic = {0, 0, 0}};
        float speed = velocity.linear.length();
        float target_speed = targetVelocity.linear.length();

        // accel = (new_vel - vel) / dt
        action.classic.acceleration = (target_speed - speed) / dt;

        float yaw = utils::NormalizeAngle<float>(utils::quatToYaw(rotation));
        float target_yaw = utils::NormalizeAngle<float>(utils::quatToYaw(targetRotation));

        if(consts::useEstimatedYaw)
        {
            target_yaw = atan2f(targetVelocity.linear.y, targetVelocity.linear.x);
        }

        // steering = (new_yaw - yaw) / (speed * dt + 1/2 * accel * dt ** 2)
        float denominator = speed * dt + 0.5 * action.classic.acceleration * dt * dt;
        if (denominator != 0)
        {
            action.classic.steering = (target_yaw - yaw) / denominator;
        }
        else
        {
            action.classic.steering = 0;
        }
        
        return action;

    }

    inline Action inverseDeltaModel(const Rotation &rotation, const Position &position, const Rotation &targetRotation, const Position &targetPosition)
    {
        Action action{.delta = {0, 0, 0}};
        float yaw = utils::quatToYaw(rotation);
        float target_yaw = utils::quatToYaw(targetRotation);
        // start delta model
            // start DeltaGlobal
        action.delta.dx = targetPosition.x - position.x;
        action.delta.dy = targetPosition.y - position.y;
        action.delta.dyaw = target_yaw - yaw;

        action.delta.dx  = fmaxf(-6.0, fminf(action.delta.dx, 6.0));
        action.delta.dy = fmaxf(-6.0, fminf(action.delta.dy, 6.0));
            // end DeltaGlobal
            // start DeltaLocal

        // rotated_xy = jnp.matmul(rotation_mat, action.data[..., :2, jnp.newaxis], precision='float32')[..., 0]
        // From https://en.wikipedia.org/wiki/Rotation_matrix
        float cos = std::cos(-yaw);
        float sin = std::sin(-yaw);
        // x = c * x - s * y
        // y = s * x + c * y
        float local_dx= action.delta.dx * cos - action.delta.dy * sin;
        float local_dy = action.delta.dx * sin + action.delta.dy * cos;

        action.delta.dx = fmaxf(-6.0, fminf(local_dx, 6.0));
        action.delta.dy = fmaxf(-6.0, fminf(local_dy, 6.0));
        action.delta.dyaw = utils::NormalizeAngle<float>(action.delta.dyaw);
            // end DeltaLocal
        // end delta model

        return action;

    }

    inline void forwardStateModel(Action &action, Rotation &rotation, Position &position, Velocity &velocity)
    {        
        // No clipping happening here. 
        // This can go out of bounds with invalid actions
        position = action.state.position;
        velocity = action.state.velocity;

        rotation = Quat::angleAxis(action.state.yaw, madrona::math::up);
    }
}