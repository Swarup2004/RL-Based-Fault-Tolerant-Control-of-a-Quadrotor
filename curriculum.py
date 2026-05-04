"""
Curriculum Training Manager
Progressively increases fault severity.
Promotes to next level when performance criteria are met.
"""
import torch
torch.distributions.Distribution.set_default_validate_args(False)
import os
import numpy as np
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from config import CURRICULUM, RL, SIM
from env import QuadrotorFTCEnv


class CurriculumCallback(BaseCallback):

    def __init__(self, manager, verbose=1):
        super().__init__(verbose)
        self.manager = manager
        self.episode_rewards = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        self.episode_count = 0
        self.last_check_count = 0

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_reward = info['episode']['r']
                self.episode_rewards.append(ep_reward)
                crashed = info.get('crash_reason', None) is not None
                self.episode_successes.append(0.0 if crashed else 1.0)
                self.episode_count += 1

        # Only check every 50 NEW episodes
        if (self.episode_count >= self.last_check_count + 50 and
                len(self.episode_rewards) >= 50):
            self.last_check_count = self.episode_count

            avg_reward = np.mean(self.episode_rewards)
            success_rate = np.mean(self.episode_successes)
            level = self.manager.current_level
            cfg = CURRICULUM[level]

            if self.verbose:
                print(f"\n[Curriculum] Level {level} ({cfg['name']}): "
                      f"avg_reward={avg_reward:.1f}, "
                      f"success_rate={success_rate:.2f}, "
                      f"episodes={self.episode_count}")

            if (avg_reward >= cfg['promotion_reward'] and
                    success_rate >= cfg['promotion_success_rate']):
                promoted = self.manager.promote()
                if promoted:
                    self.episode_rewards.clear()
                    self.episode_successes.clear()
                    self.episode_count = 0
                    self.last_check_count = 0

        return True

class CurriculumManager:

    def __init__(self, n_envs=8, save_dir='./checkpoints'):
        self.current_level = 0
        self.n_envs = n_envs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.envs = None
        self.model = None

    def make_env(self, rank, level_config):
        def _init():
            env = QuadrotorFTCEnv(curriculum_level=level_config)
            return env
        return _init

    def build_envs(self):
        cfg = CURRICULUM[self.current_level]
        env_fns = [self.make_env(i, cfg) for i in range(self.n_envs)]
        self.envs = VecMonitor(DummyVecEnv(env_fns))
        return self.envs

    def build_model(self):
        envs = self.build_envs()

        if self.model is None:
            self.model = PPO(
                'MlpPolicy',
                envs,
                learning_rate=RL['learning_rate'],
                n_steps=RL['n_steps'],
                batch_size=RL['batch_size'],
                n_epochs=RL['n_epochs'],
                gamma=RL['gamma'],
                gae_lambda=RL['gae_lambda'],
                clip_range=RL['clip_range'],
                ent_coef=RL['ent_coef'],
                vf_coef=RL['vf_coef'],
                max_grad_norm=RL['max_grad_norm'],
                policy_kwargs=dict(
                    net_arch=dict(
                        pi=RL['policy_layers'],
                        vf=RL['policy_layers'],
                    ),
                ),
                verbose=1,
                device='auto',
                tensorboard_log='./tb_logs',
            )
        else:
            self.model.set_env(envs)

        return self.model

    def promote(self):
        if self.current_level >= len(CURRICULUM) - 1:
            print(f"\n Already at max curriculum level {self.current_level}!")
            return False

        save_path = os.path.join(
            self.save_dir,
            f"level_{self.current_level}_{CURRICULUM[self.current_level]['name']}"
        )
        self.model.save(save_path)
        print(f"\n Saved checkpoint: {save_path}")

        self.current_level += 1
        cfg = CURRICULUM[self.current_level]
        print(f"\n PROMOTING to Level {self.current_level}: {cfg['name']}")
        print(f"   Fault range: {cfg['fault_range']}")
        print(f"   Init pos range: {cfg['init_pos_range']}m")
        print(f"   Domain rand: {cfg['domain_rand']}")
        print(f"   Mid-episode inject: {cfg.get('inject_mid_episode', False)}")

        self.build_model()
        return True

    def train(self):
        self.build_model()
        callback = CurriculumCallback(self)

        for level_idx in range(self.current_level, len(CURRICULUM)):
            cfg = CURRICULUM[level_idx]
            print(f"\n{'='*60}")
            print(f"Training Level {level_idx}: {cfg['name']}")
            print(f"{'='*60}")

            self.model.learn(
                total_timesteps=RL['total_timesteps_per_level'],
                callback=callback,
                reset_num_timesteps=False,
                tb_log_name=f"level_{level_idx}_{cfg['name']}",
            )

            if self.current_level == level_idx:
                self.promote()

        final_path = os.path.join(self.save_dir, 'final_model')
        self.model.save(final_path)
        print(f"\n Training complete! Final model saved to {final_path}")
        return self.model