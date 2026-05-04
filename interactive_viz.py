"""
Interactive Real-Time Quadrotor Visualization

Controls:
    1-4     : Toggle fault on Motor 1/2/3/4
    UP/DOWN : Increase/decrease fault severity by 5%
    R       : Reset episode
    SPACE   : Pause/Resume
    Q/ESC   : Quit

Shows:
    - Top-down view (XY) and side view (XZ) of quad position
    - Real-time state plots
    - Fault status display
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from collections import deque
import time

from stable_baselines3 import PPO
import torch
torch.distributions.Distribution.set_default_validate_args(False)

from config import QUAD, SIM, RL, REWARD, CURRICULUM
from dynamics import QuadrotorDynamics, quat_to_euler, euler_to_quat, quat_to_rotation_matrix
from pid_controller import PIDController
from fault_model import FaultModel


class InteractiveQuadViz:

    def __init__(self, model_path):
        self.model = PPO.load(model_path, device='cpu')

        # Build quad and PID
        self.quad = QuadrotorDynamics()
        self.pid = PIDController(self.quad)
        self.fault = FaultModel()

        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.target_yaw = 0.0

        # Fault control state
        self.faulted_motor = None       # None = healthy, 0-3 = faulted
        self.fault_lambda = 0.80        # current fault severity
        self.paused = False

        # History buffers (last 500 steps = 10 seconds)
        self.max_history = 500
        self.time_hist = deque(maxlen=self.max_history)
        self.pos_hist = deque(maxlen=self.max_history)
        self.vel_hist = deque(maxlen=self.max_history)
        self.euler_hist = deque(maxlen=self.max_history)
        self.error_hist = deque(maxlen=self.max_history)
        self.rpm_hist = deque(maxlen=self.max_history)
        self.reward_hist = deque(maxlen=self.max_history)

        self.sim_time = 0.0
        self.prev_rl_action = np.zeros(4)
        self.prev_pid_wrench = np.zeros(4)
        self.cumulative_reward = 0.0

        # Setup plot
        self._setup_plot()
        self._reset()

    def _reset(self):
        """Reset quad to hover position."""
        self.quad.reset(pos=self.target_pos.copy())
        self.pid.reset()
        self.sim_time = 0.0
        self.prev_rl_action = np.zeros(4)
        self.prev_pid_wrench = np.zeros(4)
        self.cumulative_reward = 0.0

        self.time_hist.clear()
        self.pos_hist.clear()
        self.vel_hist.clear()
        self.euler_hist.clear()
        self.error_hist.clear()
        self.rpm_hist.clear()
        self.reward_hist.clear()

        # Apply current fault setting
        self.fault.lambdas = np.ones(4)
        self.fault.active = True
        self.fault.faulted_motor = self.faulted_motor
        if self.faulted_motor is not None:
            self.fault.lambdas[self.faulted_motor] = self.fault_lambda
            self.fault.fault_magnitude = self.fault_lambda

    def _setup_plot(self):
        """Create the matplotlib figure with all subplots."""
        plt.ion()
        self.fig = plt.figure(figsize=(18, 11))
        self.fig.patch.set_facecolor('#1a1a2e')
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.4, wspace=0.35)

        # Top-down view (XY)
        self.ax_xy = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_xy.set_facecolor('#16213e')
        self.ax_xy.set_xlim(-1.5, 1.5)
        self.ax_xy.set_ylim(-1.5, 1.5)
        self.ax_xy.set_aspect('equal')
        self.ax_xy.set_title('Top View (XY)', color='white', fontsize=12)
        self.ax_xy.tick_params(colors='gray')
        self.ax_xy.grid(True, alpha=0.2, color='gray')

        # Position error plot
        self.ax_err = self.fig.add_subplot(gs[0, 2:4])
        self.ax_err.set_facecolor('#16213e')
        self.ax_err.set_title('Position Error', color='white', fontsize=11)
        self.ax_err.set_ylabel('Error (m)', color='gray')
        self.ax_err.tick_params(colors='gray')
        self.ax_err.grid(True, alpha=0.2, color='gray')

        # Attitude plot
        self.ax_att = self.fig.add_subplot(gs[1, 2:4])
        self.ax_att.set_facecolor('#16213e')
        self.ax_att.set_title('Attitude', color='white', fontsize=11)
        self.ax_att.set_ylabel('Angle (deg)', color='gray')
        self.ax_att.tick_params(colors='gray')
        self.ax_att.grid(True, alpha=0.2, color='gray')

        # Motor Thrust plot
        self.ax_rpm = self.fig.add_subplot(gs[2, 0:2])
        self.ax_rpm.set_facecolor('#16213e')
        self.ax_rpm.set_title('Motor Thrust', color='white', fontsize=11)
        self.ax_rpm.set_ylabel('Thrust (mN)', color='gray')
        self.ax_rpm.set_xlabel('Time (s)', color='gray')
        self.ax_rpm.tick_params(colors='gray')
        self.ax_rpm.grid(True, alpha=0.2, color='gray')

        # Status panel
        self.ax_status = self.fig.add_subplot(gs[2, 2:4])
        self.ax_status.set_facecolor('#16213e')
        self.ax_status.axis('off')

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    def _on_key(self, event):
        """Handle keyboard input."""
        if event.key in ['1', '2', '3', '4']:
            motor = int(event.key) - 1
            if self.faulted_motor == motor:
                # Toggle off
                self.faulted_motor = None
                self.fault.lambdas = np.ones(4)
                self.fault.faulted_motor = None
                self.fault.active = True
            else:
                # Fault this motor
                self.faulted_motor = motor
                self.fault.lambdas = np.ones(4)
                self.fault.lambdas[motor] = self.fault_lambda
                self.fault.faulted_motor = motor
                self.fault.fault_magnitude = self.fault_lambda
                self.fault.active = True

        elif event.key == 'up':
            self.fault_lambda = min(0.95, self.fault_lambda + 0.05)
            if self.faulted_motor is not None:
                self.fault.lambdas[self.faulted_motor] = self.fault_lambda
                self.fault.fault_magnitude = self.fault_lambda

        elif event.key == 'down':
            self.fault_lambda = max(0.30, self.fault_lambda - 0.05)
            if self.faulted_motor is not None:
                self.fault.lambdas[self.faulted_motor] = self.fault_lambda
                self.fault.fault_magnitude = self.fault_lambda

        elif event.key == 'r':
            self._reset()

        elif event.key == ' ':
            self.paused = not self.paused

        elif event.key in ['q', 'escape']:
            plt.close('all')

    def _get_obs(self):
        """Build observation for the RL agent."""
        s = self.quad.state
        pos_error = s.pos - self.target_pos
        vel = s.vel
        from dynamics import gravity_in_body
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
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        obs = np.clip(obs, -10.0, 10.0)
        return obs

    def _step_sim(self):
        """Run one RL control step (multiple physics substeps)."""
        obs = self._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        max_comp = RL['compensation_fraction'] * self.quad.max_rpm
        rl_rpm = action * max_comp

        for _ in range(SIM['ctrl_steps_per_physics']):
            pid_rpm, pid_wrench = self.pid.compute(
                self.quad.state, self.target_pos, self.target_yaw, SIM['dt']
            )
            self.prev_pid_wrench = pid_wrench

            combined_rpm = np.clip(pid_rpm + rl_rpm, self.quad.min_rpm, self.quad.max_rpm)
            actual_rpm = self.fault.apply(combined_rpm)
            self.quad.step(actual_rpm, SIM['dt'])
            self.sim_time += SIM['dt']

        self.prev_rl_action = action.copy()

        # Log
        s = self.quad.state
        self.time_hist.append(self.sim_time)
        self.pos_hist.append(s.pos.copy())
        self.vel_hist.append(s.vel.copy())
        self.euler_hist.append(np.degrees(s.euler))
        self.error_hist.append(np.linalg.norm(s.pos - self.target_pos))
        self.rpm_hist.append(actual_rpm.copy())

        # Simple reward calc
        err = np.linalg.norm(s.pos - self.target_pos)
        r = -2.0 * err + 0.05
        if err < 0.15:
            r += 0.1
        self.cumulative_reward += r
        self.reward_hist.append(r)

    def _draw_quad_topview(self):
        """Draw quadrotor in top-down XY view."""
        ax = self.ax_xy
        ax.cla()
        ax.set_facecolor('#16213e')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title('Top View (XY)  |  Z = {:.3f}m'.format(
            self.quad.state.pos[2]), color='white', fontsize=12)
        ax.tick_params(colors='gray')
        ax.grid(True, alpha=0.2, color='gray')

        # Target
        target_circle = Circle((self.target_pos[0], self.target_pos[1]),
                               REWARD['goal_radius'], fill=False,
                               color='lime', linestyle='--', alpha=0.6)
        ax.add_patch(target_circle)
        ax.plot(self.target_pos[0], self.target_pos[1], '+', color='lime',
                markersize=15, markeredgewidth=2)

        # Trajectory trail
        if len(self.pos_hist) > 1:
            positions = np.array(self.pos_hist)
            n = len(positions)
            for i in range(max(0, n-100), n-1):
                alpha = 0.1 + 0.9 * (i - max(0, n-100)) / min(n, 100)
                ax.plot(positions[i:i+2, 0], positions[i:i+2, 1],
                        color='cyan', alpha=alpha, linewidth=1.5)

        # Quad body
        s = self.quad.state
        R = quat_to_rotation_matrix(s.quat)
        arm = self.quad.arm_length * 15  # scale up for visibility

        # Motor positions in body frame (X-config)
        motor_body = np.array([
            [ arm,  arm, 0],   # M1 front-left
            [ arm, -arm, 0],   # M2 front-right
            [-arm, -arm, 0],   # M3 rear-right
            [-arm,  arm, 0],   # M4 rear-left
        ])

        motor_colors = ['#00ff88', '#00ff88', '#00ff88', '#00ff88']
        if self.faulted_motor is not None:
            motor_colors[self.faulted_motor] = '#ff4444'

        for i in range(4):
            mpos_world = s.pos[:2] + (R @ motor_body[i])[:2]
            ax.plot(mpos_world[0], mpos_world[1], 'o', color=motor_colors[i],
                    markersize=8, markeredgecolor='white', markeredgewidth=1)

            # Draw arm
            ax.plot([s.pos[0], mpos_world[0]], [s.pos[1], mpos_world[1]],
                    color='white', linewidth=2, alpha=0.7)

        # Center dot
        ax.plot(s.pos[0], s.pos[1], 'o', color='white', markersize=5)

        # Heading arrow
        heading = R @ np.array([arm * 1.5, 0, 0])
        ax.annotate('', xy=(s.pos[0] + heading[0], s.pos[1] + heading[1]),
                     xytext=(s.pos[0], s.pos[1]),
                     arrowprops=dict(arrowstyle='->', color='yellow', lw=2))

    def _draw_plots(self):
        """Update time-series plots."""
        if len(self.time_hist) < 2:
            return

        t = np.array(self.time_hist)
        errors = np.array(self.error_hist)
        eulers = np.array(self.euler_hist)
        rpms = np.array(self.rpm_hist)

        # Error plot
        ax = self.ax_err
        ax.cla()
        ax.set_facecolor('#16213e')
        ax.plot(t, errors, color='cyan', linewidth=1.5)
        ax.axhline(y=REWARD['goal_radius'], color='lime', linestyle='--', alpha=0.5)
        ax.set_title('Position Error', color='white', fontsize=11)
        ax.set_ylabel('Error (m)', color='gray')
        ax.tick_params(colors='gray')
        ax.grid(True, alpha=0.2, color='gray')
        ax.set_ylim(0, max(0.5, errors.max() * 1.1))

        # Attitude plot
        ax = self.ax_att
        ax.cla()
        ax.set_facecolor('#16213e')
        ax.plot(t, eulers[:, 0], color='#ff6b6b', linewidth=1.2, label='roll')
        ax.plot(t, eulers[:, 1], color='#ffd93d', linewidth=1.2, label='pitch')
        ax.plot(t, eulers[:, 2], color='#6bcb77', linewidth=1.2, label='yaw')
        ax.set_title('Attitude', color='white', fontsize=11)
        ax.set_ylabel('Angle (deg)', color='gray')
        ax.tick_params(colors='gray')
        ax.grid(True, alpha=0.2, color='gray')
        ax.legend(fontsize=8, facecolor='#16213e', edgecolor='gray', labelcolor='white')

        # Thrust plot
        ax = self.ax_rpm
        ax.cla()
        ax.set_facecolor('#16213e')
        colors = ['#00ff88', '#4dabf7', '#ffd93d', '#ff6b6b']
        thrust_mN = QUAD['k_f'] * rpms**2 * 1000  # N -> mN
        hover_thrust_mN = QUAD['k_f'] * QUAD['hover_rpm']**2 * 1000
        for i in range(4):
            style = '--' if i == self.faulted_motor else '-'
            ax.plot(t, thrust_mN[:, i], color=colors[i], linewidth=1.2,
                    linestyle=style, label=f'M{i+1}')
        ax.axhline(y=hover_thrust_mN, color='white', linestyle=':', alpha=0.4, linewidth=1)
        ax.set_title('Motor Thrust (dashed=faulted)', color='white', fontsize=11)
        ax.set_ylabel('Thrust (mN)', color='gray')
        ax.set_xlabel('Time (s)', color='gray')
        ax.tick_params(colors='gray')
        ax.grid(True, alpha=0.2, color='gray')
        ax.legend(fontsize=8, facecolor='#16213e', edgecolor='gray',
                  labelcolor='white', ncol=4)

    def _draw_status(self):
        """Draw status panel with controls info."""
        ax = self.ax_status
        ax.cla()
        ax.set_facecolor('#16213e')
        ax.axis('off')

        s = self.quad.state
        err = np.linalg.norm(s.pos - self.target_pos)
        thrust_loss = (1 - self.fault_lambda**2) * 100 if self.faulted_motor is not None else 0

        lines = [
            f"QUADROTOR FTC - INTERACTIVE",
            f"",
            f"Position:  [{s.pos[0]:+.3f}, {s.pos[1]:+.3f}, {s.pos[2]:+.3f}]",
            f"Error:     {err:.4f} m",
            f"Time:      {self.sim_time:.1f} s",
            f"Reward:    {self.cumulative_reward:.1f}",
            f"",
            f"FAULT STATUS:",
            f"  Motor:     {'M' + str(self.faulted_motor+1) if self.faulted_motor is not None else 'HEALTHY'}",
            f"  Lambda:    {self.fault_lambda:.2f}  (thrust loss: {thrust_loss:.0f}%)",
            f"",
            f"CONTROLS:",
            f"  1-4:       Fault Motor 1/2/3/4 (fault introduced only when a motor is selected)",
            f"  UP/DOWN:   Adjust fault %",
            f"  R:         Reset",
            f"  SPACE:     Pause",
            f"  Q:         Quit",
        ]

        status_color = '#ff4444' if self.faulted_motor is not None else '#00ff88'
        for i, line in enumerate(lines):
            color = 'white'
            if 'HEALTHY' in line:
                color = '#00ff88'
            elif 'M' in line and 'Motor' not in line and 'CONTROLS' not in line and self.faulted_motor is not None:
                color = '#ff4444'
            if i == 0:
                color = '#4dabf7'
                ax.text(0.05, 0.95 - i * 0.055, line, transform=ax.transAxes,
                        fontsize=11, color=color, fontweight='bold',
                        fontfamily='monospace', verticalalignment='top')
            else:
                ax.text(0.05, 0.95 - i * 0.055, line, transform=ax.transAxes,
                        fontsize=9, color=color, fontfamily='monospace',
                        verticalalignment='top')

    def run(self):
        """Main loop."""
        print("\n" + "="*50)
        print("INTERACTIVE QUADROTOR FTC VISUALIZATION")
        print("="*50)
        print("Press 1-4 to inject fault on a motor")
        print("UP/DOWN to adjust fault severity")
        print("R to reset, SPACE to pause, Q to quit")
        print("="*50 + "\n")

        target_dt = SIM['ctrl_dt']  # 50 Hz target

        while plt.fignum_exists(self.fig.number):
            t_start = time.time()

            if not self.paused:
                self._step_sim()

            self._draw_quad_topview()
            self._draw_plots()
            self._draw_status()

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            # Frame rate control
            elapsed = time.time() - t_start
            sleep_time = max(0, target_dt - elapsed)
            if sleep_time > 0:
                plt.pause(sleep_time)
            else:
                plt.pause(0.001)

        print("Visualization closed.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./checkpoints/level_3_severe_fault.zip')
    args = parser.parse_args()

    viz = InteractiveQuadViz(args.model)
    viz.run()

    #python interactive_viz.py --model ./checkpoints/level_3_severe_fault.zip