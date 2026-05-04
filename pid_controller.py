"""
Cascaded PID Controller for Quadrotor
Outer loop: Position error -> desired thrust + desired roll/pitch
Inner loop: Attitude error -> desired torques
"""
import numpy as np
from config import PID
from dynamics import quat_to_euler, quat_to_rotation_matrix


class PIDController:

    def __init__(self, quad_dynamics):
        self.quad = quad_dynamics

        self.pos_P = PID['pos_P'].copy()
        self.pos_I = PID['pos_I'].copy()
        self.pos_D = PID['pos_D'].copy()
        self.pos_I_max = PID['pos_I_max'].copy()

        self.att_P = PID['att_P'].copy()
        self.att_I = PID['att_I'].copy()
        self.att_D = PID['att_D'].copy()
        self.att_I_max = PID['att_I_max'].copy()

        self.max_tilt = PID.get('max_tilt_cmd', 0.35)

        self.pos_integral = np.zeros(3)
        self.att_integral = np.zeros(3)
        self.first_step = True

    def reset(self):
        self.pos_integral = np.zeros(3)
        self.att_integral = np.zeros(3)
        self.first_step = True

    def compute(self, state, target_pos, target_yaw, dt):
        # ──── Outer Loop: Position -> Desired Acceleration ────
        pos_error = target_pos - state.pos
        vel = state.vel

        self.pos_integral += pos_error * dt
        self.pos_integral = np.clip(self.pos_integral, -self.pos_I_max, self.pos_I_max)

        acc_des = (self.pos_P * pos_error
                   + self.pos_I * self.pos_integral
                   - self.pos_D * vel)

        # Gravity compensation (from actual mass, not hardcoded)
        acc_des[2] += self.quad.g

        # ──── Thrust: project desired acceleration onto body z-axis ────
        R = quat_to_rotation_matrix(state.quat)
        body_z_world = R @ np.array([0.0, 0.0, 1.0])
        thrust = self.quad.mass * np.dot(acc_des, body_z_world)
        thrust = max(thrust, 0.0)

        # ──── Desired attitude from desired acceleration direction ────
        acc_norm = np.linalg.norm(acc_des)
        if acc_norm < 1e-6:
            z_des = np.array([0.0, 0.0, 1.0])
        else:
            z_des = acc_des / acc_norm

        pitch_des = np.arcsin(np.clip(z_des[0], -1.0, 1.0))
        roll_des = np.arctan2(-z_des[1], z_des[2])

        roll_des = np.clip(roll_des, -self.max_tilt, self.max_tilt)
        pitch_des = np.clip(pitch_des, -self.max_tilt, self.max_tilt)

        # ──── Inner Loop: Attitude -> Torques ────
        euler = quat_to_euler(state.quat)
        att_error = np.array([
            roll_des - euler[0],
            pitch_des - euler[1],
            target_yaw - euler[2],
        ])
        # Wrap yaw error
        att_error[2] = (att_error[2] + np.pi) % (2 * np.pi) - np.pi

        self.att_integral += att_error * dt
        self.att_integral = np.clip(self.att_integral, -self.att_I_max, self.att_I_max)

        torques = (self.att_P * att_error
                   + self.att_I * self.att_integral
                   - self.att_D * state.omega)

        self.first_step = False

        # ──── Mixer: wrench -> RPM ────
        wrench = np.array([thrust, torques[0], torques[1], torques[2]])
        rpm_commands = self.quad.rpm_from_wrench(wrench)

        return rpm_commands, wrench