"""
Gymnasium Environment for Quadrotor Fault-Tolerant Control

RL agent outputs compensation signals ADDED to PID output.
PID handles baseline flight. RL handles fault compensation.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import QUAD, SIM, RL, REWARD
from dynamics import QuadrotorDynamics, gravity_in_body, quat_to_euler, euler_to_quat
from pid_controller import PIDController
from fault_model import FaultModel


class QuadrotorFTCEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, curriculum_level=None, render_mode=None):
        super().__init__()

        obs_high = np.ones(RL['obs_dim']) * 10.0
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(RL['act_dim'],), dtype=np.float32)

        self.quad = None
        self.pid = None
        self.fault = FaultModel()

        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.target_yaw = 0.0
        self.current_step = 0
        self.sim_time = 0.0
        self.prev_rl_action = np.zeros(4)
        self.prev_pid_wrench = np.zeros(4)
        self.time_at_goal = 0.0
        self.episode_reward = 0.0

        self.curriculum_config = curriculum_level
        self.rng = np.random.default_rng()

        self.history = {
            'pos': [], 'vel': [], 'euler': [], 'omega': [],
            'rpm_cmd': [], 'rpm_actual': [], 'reward': [], 'time': [],
        }
        # Build quad immediately so step() never sees None
        self._build_quad()
        self.pid.reset()

    def _build_quad(self):
        params = QUAD.copy()
        cfg = self.curriculum_config

        if cfg is not None and cfg.get('domain_rand', False):
            mass_scale = self.rng.uniform(*cfg.get('mass_range', (1.0, 1.0)))
            inertia_scale = self.rng.uniform(*cfg.get('inertia_range', (1.0, 1.0)))
            kf_scale = self.rng.uniform(*cfg.get('kf_range', (1.0, 1.0)))
            params['mass'] = QUAD['mass'] * mass_scale
            params['I'] = QUAD['I'] * inertia_scale
            params['k_f'] = QUAD['k_f'] * kf_scale
            params['hover_rpm'] = np.sqrt(params['mass'] * params['g'] / (4 * params['k_f']))

        self.quad = QuadrotorDynamics(params)
        self.pid = PIDController(self.quad)

    def _get_init_conditions(self):
        cfg = self.curriculum_config
        r = cfg['init_pos_range'] if cfg else 0.3
        att_r = np.radians(cfg['init_att_range_deg'] if cfg else 5.0)

        pos = self.target_pos + self.rng.uniform(-r, r, size=3)
        pos[2] = np.clip(pos[2], 0.3, 2.0)

        rpy = self.rng.uniform(-att_r, att_r, size=3)
        quat = euler_to_quat(rpy)

        vel = self.rng.uniform(-0.1, 0.1, size=3)
        omega = self.rng.uniform(-0.2, 0.2, size=3)

        return pos, vel, quat, omega

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._build_quad()
        self.pid.reset()

        pos, vel, quat, omega = self._get_init_conditions()
        self.quad.reset(pos=pos, vel=vel, quat=quat, omega=omega)

        cfg = self.curriculum_config
        fault_range = cfg['fault_range'] if cfg else (1.0, 1.0)
        inject_mid = cfg.get('inject_mid_episode', False) if cfg else False
        self.fault.reset(
            fault_range=fault_range,
            inject_mid_episode=inject_mid,
            episode_duration=SIM['episode_duration'],
            rng=self.rng,
        )

        self.current_step = 0
        self.sim_time = 0.0
        self.prev_rl_action = np.zeros(4)
        self.prev_pid_wrench = np.zeros(4)
        self.time_at_goal = 0.0
        self.episode_reward = 0.0

        for k in self.history:
            self.history[k] = []

        obs = self._get_obs()
        info = self.fault.get_info()
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        max_compensation = RL['compensation_fraction'] * self.quad.max_rpm
        rl_rpm = action * max_compensation

        for _ in range(SIM['ctrl_steps_per_physics']):
            self.fault.update(self.sim_time)

            pid_rpm, pid_wrench = self.pid.compute(
                self.quad.state, self.target_pos, self.target_yaw, SIM['dt']
            )
            self.prev_pid_wrench = pid_wrench

            combined_rpm = pid_rpm + rl_rpm
            combined_rpm = np.clip(combined_rpm, self.quad.min_rpm, self.quad.max_rpm)

            actual_rpm = self.fault.apply(combined_rpm)
            self.quad.step(actual_rpm, SIM['dt'])
            self.sim_time += SIM['dt']

        reward, done, info = self._compute_reward(action)
        self.episode_reward += reward

        s = self.quad.state
        self.history['pos'].append(s.pos.copy())
        self.history['vel'].append(s.vel.copy())
        self.history['euler'].append(s.euler.copy())
        self.history['omega'].append(s.omega.copy())
        self.history['rpm_cmd'].append((pid_rpm + rl_rpm).copy())
        self.history['rpm_actual'].append(actual_rpm.copy())
        self.history['reward'].append(reward)
        self.history['time'].append(self.sim_time)

        self.prev_rl_action = action.copy()
        self.current_step += 1

        truncated = self.current_step >= SIM['max_steps']
        info.update(self.fault.get_info())
        info['episode_reward'] = self.episode_reward
        info['time'] = self.sim_time

        obs = self._get_obs()
        return obs, reward, done, truncated, info

    def _get_obs(self):
        s = self.quad.state

        pos_error = s.pos - self.target_pos
        vel = s.vel
        g_body = gravity_in_body(s.quat)
        omega = s.omega
        prev_act = self.prev_rl_action

        pid_norm = self.prev_pid_wrench.copy()
        pid_norm[0] /= (self.quad.mass * self.quad.g * 2.0 + 1e-6)
        pid_norm[1:] /= (0.01 + 1e-8)
        pid_norm = np.clip(pid_norm, -5.0, 5.0)

        obs = np.concatenate([
            np.clip(pos_error, -5.0, 5.0),
            np.clip(vel, -5.0, 5.0),
            g_body,
            np.clip(omega, -10.0, 10.0),
            prev_act,
            pid_norm,
        ]).astype(np.float32)

        # Kill any NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        obs = np.clip(obs, -10.0, 10.0)
        return obs

    def _compute_reward(self, action):
        s = self.quad.state
        R = REWARD
        # NaN protection - if dynamics blew up, crash immediately
        if (np.any(np.isnan(s.pos)) or np.any(np.isnan(s.vel)) or
            np.any(np.isnan(s.quat)) or np.any(np.isnan(s.omega))):
            return R['crash_penalty'], True, {'crash_reason': 'nan'}

        pos_error = np.linalg.norm(s.pos - self.target_pos)
        vel_mag = np.linalg.norm(s.vel)
        euler = s.euler
        tilt = np.sqrt(euler[0]**2 + euler[1]**2)
        omega_mag = np.linalg.norm(s.omega)
        action_rate = np.linalg.norm(action - self.prev_rl_action)
        action_mag = np.linalg.norm(action)

        done = False
        if np.degrees(tilt) > R['max_tilt_deg']:
            done = True
            return R['crash_penalty'], done, {'crash_reason': 'tilt'}
        if pos_error > R['max_pos_error']:
            done = True
            return R['crash_penalty'], done, {'crash_reason': 'pos_error'}
        if s.pos[2] < 0.05:
            done = True
            return R['crash_penalty'], done, {'crash_reason': 'ground'}

        reward = 0.0
        reward -= R['k_pos'] * pos_error
        reward -= R['k_vel'] * vel_mag**2
        reward -= R['k_att'] * tilt
        reward -= R['k_omega'] * omega_mag**2
        reward -= R['k_action_rate'] * action_rate**2
        reward -= R['k_action_mag'] * action_mag**2
        reward += R['alive_bonus']

        if pos_error < R['goal_radius']:
            self.time_at_goal += SIM['ctrl_dt']
            reward += R['near_bonus']
            if self.time_at_goal >= R['sustained_time']:
                reward += R['goal_bonus']
        elif pos_error < R['near_radius']:
            reward += R['near_bonus'] * 0.5
            self.time_at_goal = max(0, self.time_at_goal - SIM['ctrl_dt'])
        else:
            self.time_at_goal = max(0, self.time_at_goal - SIM['ctrl_dt'] * 2)

        return reward, done, {}

    def set_curriculum(self, level_config):
        self.curriculum_config = level_config

    def get_history(self):
        return {k: np.array(v) for k, v in self.history.items() if len(v) > 0}