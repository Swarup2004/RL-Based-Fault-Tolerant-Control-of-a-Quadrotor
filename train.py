"""
Main training entry point.
"""
import argparse
import torch
# Fix for PyTorch 2.x + SB3 compatibility
torch.distributions.Distribution.set_default_validate_args(False)

from curriculum import CurriculumManager


def main():
    parser = argparse.ArgumentParser(description='Train Quadrotor FTC Agent')
    parser.add_argument('--level', type=int, default=0, help='Starting curriculum level')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--n-envs', type=int, default=4, help='Parallel environments')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Checkpoint dir')
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("Training on CPU")

    manager = CurriculumManager(n_envs=args.n_envs, save_dir=args.save_dir)
    manager.current_level = args.level

    if args.resume:
        from stable_baselines3 import PPO
        print(f"Resuming from {args.resume}")
        manager.build_envs()
        manager.model = PPO.load(args.resume, env=manager.envs, device='cpu')

    manager.train()


if __name__ == '__main__':
    main()