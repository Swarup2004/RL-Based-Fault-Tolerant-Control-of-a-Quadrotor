"""
Visualization for trained agent.
Usage:
    python visualize.py
    python visualize.py --model checkpoints/level_3_severe_fault.zip --level 3
    python visualize.py --compare
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO
import torch
torch.distributions.Distribution.set_default_validate_args(False)

from config import CURRICULUM, SIM, REWARD, QUAD
from env import QuadrotorFTCEnv


def run_episode(model, env, use_model=True):
    obs, info = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        if use_model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(4)
        obs, reward, done, truncated, info = env.step(action)
    return env.get_history(), info


def plot_episode(history, target, title="Episode", fault_info=None):
    t = history['time']
    pos = history['pos']
    vel = history['vel']
    euler = np.degrees(history['euler'])
    omega = history['omega']
    rpm_actual = history['rpm_actual']
    rewards = history['reward']

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Position
    ax1 = fig.add_subplot(gs[0, 0])
    for i, label in enumerate(['x', 'y', 'z']):
        ax1.plot(t, pos[:, i], label=label)
        ax1.axhline(y=target[i], color=f'C{i}', linestyle='--', alpha=0.4)
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Position Error
    ax2 = fig.add_subplot(gs[0, 1])
    errors = np.linalg.norm(pos - target, axis=1)
    ax2.plot(t, errors, 'k-')
    ax2.axhline(y=REWARD['goal_radius'], color='g', linestyle='--', alpha=0.7, label=f"Goal ({REWARD['goal_radius']}m)")
    ax2.axhline(y=REWARD['near_radius'], color='orange', linestyle='--', alpha=0.7, label=f"Near ({REWARD['near_radius']}m)")
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Position Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Velocity
    ax3 = fig.add_subplot(gs[1, 0])
    for i, label in enumerate(['vx', 'vy', 'vz']):
        ax3.plot(t, vel[:, i], label=label)
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Attitude
    ax4 = fig.add_subplot(gs[1, 1])
    for i, label in enumerate(['roll', 'pitch', 'yaw']):
        ax4.plot(t, euler[:, i], label=label)
    ax4.set_ylabel('Angle (deg)')
    ax4.set_title('Euler Angles')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Angular Velocity
    ax5 = fig.add_subplot(gs[2, 0])
    for i, label in enumerate(['p', 'q', 'r']):
        ax5.plot(t, omega[:, i], label=label)
    ax5.set_ylabel('Angular Vel (rad/s)')
    ax5.set_title('Angular Velocity')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Motor Thrust
    ax6 = fig.add_subplot(gs[2, 1])
    thrust_mN = QUAD['k_f'] * rpm_actual**2 * 1000  # Newtons -> millinewtons
    hover_thrust_mN = QUAD['k_f'] * QUAD['hover_rpm']**2 * 1000
    for i in range(4):
        style = '--' if (fault_info and fault_info.get('faulted_motor') == i) else '-'
        ax6.plot(t, thrust_mN[:, i], label=f'M{i+1}', linestyle=style)
    ax6.axhline(y=hover_thrust_mN, color='gray', linestyle=':', alpha=0.6, label='Hover')
    ax6.set_ylabel('Thrust (mN)')
    ax6.set_title('Motor Thrust (dashed=faulted)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Cumulative Reward
    ax7 = fig.add_subplot(gs[3, :])
    ax7.plot(t, np.cumsum(rewards), 'b-')
    ax7.set_ylabel('Cumulative Reward')
    ax7.set_xlabel('Time (s)')
    ax7.set_title('Cumulative Reward')
    ax7.grid(True, alpha=0.3)

    if fault_info and fault_info.get('faulted_motor') is not None:
        fig.text(0.5, 0.01,
                 f"Fault: Motor {fault_info['faulted_motor']+1}, lambda={fault_info['fault_magnitude']:.2f}",
                 ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_3d(history, target, title="3D Trajectory"):
    pos = history['pos']
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    n = len(pos)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    for i in range(n - 1):
        ax.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2], color=colors[i], linewidth=1.5)

    ax.scatter(*pos[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*pos[-1], color='red', s=100, marker='x', label='End')
    ax.scatter(*target, color='blue', s=150, marker='*', label='Target')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    return fig


def compare_ftc(model_path, level=3):
    model = PPO.load(model_path, device='cpu')
    cfg = CURRICULUM[min(level, len(CURRICULUM)-1)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Blue = With RL FTC  |  Red = PID Only (no RL)', fontsize=14)

    for color, label, use_model in [('blue', 'With RL', True), ('red', 'PID Only', False)]:
        env = QuadrotorFTCEnv(curriculum_level=cfg)
        history, info = run_episode(model, env, use_model=use_model)
        t = history['time']
        pos = history['pos']
        euler = np.degrees(history['euler'])
        errors = np.linalg.norm(pos - env.target_pos, axis=1)

        axes[0, 0].plot(t, errors, color=color, label=label)
        axes[0, 1].plot(t, pos[:, 2], color=color, label=label)
        axes[1, 0].plot(t, euler[:, 0], color=color, label=f'{label} roll')
        axes[1, 0].plot(t, euler[:, 1], color=color, linestyle='--', alpha=0.7)
        axes[1, 1].plot(t, euler[:, 2], color=color, label=label)

    axes[0, 0].set_title('Position Error')
    axes[0, 0].set_ylabel('Error (m)')
    axes[0, 0].axhline(y=REWARD['goal_radius'], color='g', linestyle=':', alpha=0.5)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Altitude (Z)')
    axes[0, 1].set_ylabel('Z (m)')
    axes[0, 1].axhline(y=1.0, color='g', linestyle=':', alpha=0.5)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Roll & Pitch')
    axes[1, 0].set_ylabel('Angle (deg)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Yaw')
    axes[1, 1].set_ylabel('Angle (deg)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./checkpoints/level_3_severe_fault.zip')
    parser.add_argument('--level', type=int, default=3)
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--save', action='store_true', help='Save figures to PNG instead of showing')
    args = parser.parse_args()

    import os
    headless = os.environ.get('MPLBACKEND', '').lower() == 'agg' or args.save

    if args.compare:
        compare_ftc(args.model, args.level)
        if headless:
            plt.savefig('trajectory_compare.png', dpi=150, bbox_inches='tight')
            print("Saved: trajectory_compare.png")
        else:
            plt.show()
    else:
        model = PPO.load(args.model, device='cpu')
        cfg = CURRICULUM[min(args.level, len(CURRICULUM)-1)]
        env = QuadrotorFTCEnv(curriculum_level=cfg)
        history, info = run_episode(model, env)

        fault_info = {'faulted_motor': info.get('faulted_motor'),
                      'fault_magnitude': info.get('fault_magnitude', 1.0)}

        pos = history['pos']
        errors = np.linalg.norm(pos - env.target_pos, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        crashed = info.get('crash_reason') is not None

        print(f"Motor {info.get('faulted_motor')}, lambda={info.get('fault_magnitude', 1.0):.2f}")
        print(f"RMSE: {rmse:.4f} m")
        print(f"Max error: {errors.max():.4f} m")
        print(f"Crashed: {crashed}")

        plot_episode(history, env.target_pos, f"RMSE={rmse:.3f}m | Motor {info.get('faulted_motor')} lambda={info.get('fault_magnitude',1):.2f}", fault_info)
        plot_3d(history, env.target_pos)

        if headless:
            figs = [plt.figure(i) for i in plt.get_fignums()]
            for j, fig in enumerate(figs):
                fname = f'trajectory_episode_{j+1}.png'
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                print(f"Saved: {fname}")
        else:
            plt.show()