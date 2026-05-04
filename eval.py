"""
Proper evaluation: run 50 episodes, report stats.
Also compare RL vs PID-only on SAME scenarios.
"""
import numpy as np
from stable_baselines3 import PPO
import torch
torch.distributions.Distribution.set_default_validate_args(False)

from config import CURRICULUM, SIM, REWARD
from env import QuadrotorFTCEnv
import matplotlib.pyplot as plt


def evaluate(model_path, level=3, n_episodes=50):
    model = PPO.load(model_path, device='cpu')
    cfg = CURRICULUM[min(level, len(CURRICULUM)-1)]

    rl_results = []
    pid_results = []

    for i in range(n_episodes):
        seed = 1000 + i  # same seed for both

        # Run with RL
        env_rl = QuadrotorFTCEnv(curriculum_level=cfg)
        obs, info = env_rl.reset(seed=seed)
        fault_motor = info.get('faulted_motor')
        fault_lambda = info.get('fault_magnitude', 1.0)
        done = False
        truncated = False
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env_rl.step(action)
        h_rl = env_rl.get_history()
        errors_rl = np.linalg.norm(h_rl['pos'] - env_rl.target_pos, axis=1)
        crashed_rl = info.get('crash_reason') is not None

        # Run PID only (same seed = same fault)
        env_pid = QuadrotorFTCEnv(curriculum_level=cfg)
        obs, _ = env_pid.reset(seed=seed)
        done = False
        truncated = False
        while not done and not truncated:
            obs, reward, done, truncated, info = env_pid.step(np.zeros(4))
        h_pid = env_pid.get_history()
        errors_pid = np.linalg.norm(h_pid['pos'] - env_pid.target_pos, axis=1)
        crashed_pid = info.get('crash_reason') is not None

        rmse_rl = np.sqrt(np.mean(errors_rl**2))
        rmse_pid = np.sqrt(np.mean(errors_pid**2))

        rl_results.append({'rmse': rmse_rl, 'max_err': errors_rl.max(), 'crashed': crashed_rl,
                           'motor': fault_motor, 'lambda': fault_lambda})
        pid_results.append({'rmse': rmse_pid, 'max_err': errors_pid.max(), 'crashed': crashed_pid})

        status_rl = "CRASH" if crashed_rl else f"RMSE={rmse_rl:.3f}"
        status_pid = "CRASH" if crashed_pid else f"RMSE={rmse_pid:.3f}"
        print(f"  Ep {i+1:3d}: M{fault_motor} lam={fault_lambda:.2f}  "
              f"RL: {status_rl:12s}  PID: {status_pid:12s}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS over {n_episodes} episodes (Level {level}: {cfg['name']})")
    print(f"{'='*60}")

    rl_rmses = [r['rmse'] for r in rl_results if not r['crashed']]
    pid_rmses = [r['rmse'] for r in pid_results if not r['crashed']]
    rl_crashes = sum(1 for r in rl_results if r['crashed'])
    pid_crashes = sum(1 for r in pid_results if r['crashed'])

    print(f"\n{'':20s} {'RL+PID':>12s} {'PID Only':>12s}")
    print(f"  {'Crash rate':20s} {rl_crashes/n_episodes*100:10.1f}% {pid_crashes/n_episodes*100:10.1f}%")
    if rl_rmses:
        print(f"  {'Mean RMSE (m)':20s} {np.mean(rl_rmses):12.4f} {np.mean(pid_rmses):12.4f}")
        print(f"  {'Std RMSE (m)':20s} {np.std(rl_rmses):12.4f} {np.std(pid_rmses):12.4f}")
    if [r['max_err'] for r in rl_results if not r['crashed']]:
        rl_maxes = [r['max_err'] for r in rl_results if not r['crashed']]
        pid_maxes = [r['max_err'] for r in pid_results if not r['crashed']]
        print(f"  {'Mean Max Error (m)':20s} {np.mean(rl_maxes):12.4f} {np.mean(pid_maxes):12.4f}")

    if rl_rmses:
        mean_rmse = np.mean(rl_rmses)
        lambdas = [r['lambda'] for r in rl_results if not r['crashed']]
        print(f"\n  Lambda range tested: {min(lambdas):.2f} - {max(lambdas):.2f}")
        print(f"\n  Benchmark comparison:")
        print(f"  Our RMSE:        {mean_rmse:.4f}m")
        print(f"  Kim et al.:      0.129m  (lambda=0.7-0.9, same notation)")

    # Plot RMSE comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = range(n_episodes)
    rl_r = [r['rmse'] if not r['crashed'] else 2.0 for r in rl_results]
    pid_r = [r['rmse'] if not r['crashed'] else 2.0 for r in pid_results]

    axes[0].bar(np.array(list(x)) - 0.2, rl_r, 0.4, label='RL+PID', color='blue', alpha=0.7)
    axes[0].bar(np.array(list(x)) + 0.2, pid_r, 0.4, label='PID Only', color='red', alpha=0.7)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('RMSE (m)')
    axes[0].set_title('RMSE per Episode (2.0 = crashed)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter: lambda vs RMSE
    lambdas = [r['lambda'] for r in rl_results]
    axes[1].scatter(lambdas, rl_r, c='blue', alpha=0.6, label='RL+PID')
    axes[1].scatter(lambdas, pid_r, c='red', alpha=0.6, label='PID Only')
    axes[1].set_xlabel('Fault Lambda (lower = worse)')
    axes[1].set_ylabel('RMSE (m)')
    axes[1].set_title('Fault Severity vs RMSE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./eval_results.png', dpi=150, bbox_inches='tight')
    print("\n  Plot saved to ./eval_results.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./checkpoints/level_3_severe_fault.zip')
    parser.add_argument('--level', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()

    evaluate(args.model, args.level, args.episodes)