#!/usr/bin/env python3
"""
Chess Reinforcement Learning using Ray RLlib.
This implementation uses RLlib's built-in distributed training capabilities.
"""
# Standard library imports
import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Union

# Third-party imports
import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.tune.utils import merge_dicts

# Local imports
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper

# Framework-specific imports
torch, nn = try_import_torch()


class ChessMetricsCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeV2, object],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional["RLModule"] = None,
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[str, Policy]] = None,
        **kwargs,
    ) -> None:
        metrics = {
            "white_win": 0.0,
            "black_win": 0.0,
            "draw": 0.0,
            "checkmate": 0.0,
            "stalemate": 0.0,
        }

        try:
            if hasattr(episode, "get_last_info"):
                info = episode.get_last_info() or {}
            elif hasattr(episode, "infos") and episode.infos:
                info = episode.infos[-1]
            else:
                info = {}

            if "outcome" in info:
                outcome = info["outcome"]
                if outcome == "white_win":
                    metrics["white_win"] = 1.0
                elif outcome == "black_win":
                    metrics["black_win"] = 1.0
                elif outcome == "draw":
                    metrics["draw"] = 1.0

            if "termination_reason" in info:
                reason = info["termination_reason"]
                if reason == "checkmate":
                    metrics["checkmate"] = 1.0
                elif reason == "stalemate":
                    metrics["stalemate"] = 1.0

            if metrics_logger:
                for key, value in metrics.items():
                    metrics_logger.log_value(key, value)

        except Exception as e:
            print(f"Error in on_episode_end callback: {e}")
            import traceback
            traceback.print_exc()


class ChessMaskingRLModule(TorchRLModule):
    def __init__(self, observation_space=None, action_space=None, model_config=None, **kwargs):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config or {}
        )
        
        self.board_shape = observation_space["board"].shape
        self.board_channels = self.board_shape[0]
        self.action_space_n = action_space.n

        self.features_extractor = nn.Sequential(
            nn.Conv2d(self.board_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 832),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Linear(832, self.action_space_n)
        self.value_head = nn.Linear(832, 1)
        
    def forward(self, batch):
        obs = batch.get("obs", batch)
        board = obs["board"]
        action_mask = obs["action_mask"]
        
        if len(board.shape) == 3:
            board = board.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
            
        batch_size = board.shape[0]
        features = self.features_extractor(board) + 1e-8
        action_logits = self.policy_head(features)
        
        # Apply action masking
        action_mask = action_mask.to(action_logits.device)
        masked_logits = action_logits + (action_mask - 1) * 1e9  # Large negative for invalid actions
        value = self.value_head(features).view(batch_size, 1)
        
        return {
            "action_dist_inputs": masked_logits,
            "vf_preds": value
        }


def create_rllib_chess_env(config):
    env = ChessEnv()
    env = ActionMaskWrapper(env)
    
    class DictObsWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Dict({
                "board": gym.spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(env.action_space.n,), dtype=np.float32),
                "white_to_move": gym.spaces.Discrete(2)
            })
            self.action_space = env.action_space

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._wrap_observation(obs), info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            return self._wrap_observation(obs), reward, terminated, truncated, info

        def _wrap_observation(self, obs):
            if not isinstance(obs, dict) or "board" not in obs or "action_mask" not in obs:
                return {
                    "board": np.zeros((13, 8, 8), dtype=np.float32),
                    "action_mask": np.ones(self.action_space.n, dtype=np.float32),
                    "white_to_move": 1
                }
            return {
                "board": np.asarray(obs["board"], dtype=np.float32),
                "action_mask": np.asarray(obs["action_mask"], dtype=np.float32),
                "white_to_move": int(obs.get("white_to_move", 1))
            }

    return DictObsWrapper(env)


def train(args):
    if args.device == "cuda" and not args.force_cpu:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

    ray.init(
        address=args.head_address if args.distributed else None,
        ignore_reinit_error=True,
        include_dashboard=args.dashboard,
        _redis_password=args.redis_password,
        num_cpus=118,
    )

    driver_gpus = 0.99999
    worker_gpus = 3
    num_workers = 12
    cpus_per_worker = 4
    driver_cpus = 8
    num_envs = 4
    gpus_per_worker = round(worker_gpus / num_workers, 6)
    
    tune.register_env("chess_env", create_rllib_chess_env)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    observation_space = gym.spaces.Dict({
        "board": gym.spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(20480,), dtype=np.float32),
        "white_to_move": gym.spaces.Discrete(2)
    })
    action_space = gym.spaces.Discrete(20480)

    rl_module_spec = RLModuleSpec(
        module_class=ChessMaskingRLModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config={"fcnet_hiddens": []},
    )

    config = (
        PPOConfig()
        .environment("chess_env")
        .framework("torch")
        .resources(num_gpus=driver_gpus)
        .learners(num_learners=4, num_gpus_per_learner=driver_gpus / 4)
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=num_envs,
            num_cpus_per_env_runner=cpus_per_worker,
            num_gpus_per_env_runner=gpus_per_worker,
        )
        .training(
            train_batch_size_per_learner=4096,
            minibatch_size=256,
            num_epochs=10,
            lr=5e-5,
            grad_clip=1.0,
            gamma=0.99,
            use_gae=True,
            lambda_=0.95,
            vf_loss_coeff=0.5,
            entropy_coeff=args.entropy_coeff,
            clip_param=0.2,
            kl_coeff=0.2,
            vf_share_layers=False,
        )
        .callbacks(ChessMetricsCallback)
        .rl_module(rl_module_spec=rl_module_spec)  # Removed action_distribution_config
        .experimental(_enable_new_api_stack=True)
    )

    analysis = tune.run(
        "PPO",
        stop={"training_iteration": args.max_iterations},
        checkpoint_freq=25,
        checkpoint_at_end=True,
        storage_path=checkpoint_dir,
        verbose=2,
        config=config,
        resume="AUTO",
        restore=args.checkpoint if args.checkpoint and os.path.exists(args.checkpoint) else None,
    )

    return analysis.best_checkpoint


def evaluate(args):
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f"Error: Valid checkpoint path required for evaluation, got: {args.checkpoint}")
        return

    ray.init(ignore_reinit_error=True, include_dashboard=args.dashboard)
    tune.register_env("chess_env", create_rllib_chess_env)

    observation_space = gym.spaces.Dict({
        "board": gym.spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(20480,), dtype=np.float32),
        "white_to_move": gym.spaces.Discrete(2)
    })
    action_space = gym.spaces.Discrete(20480)

    rl_module_spec = RLModuleSpec(
        module_class=ChessMaskingRLModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config={"fcnet_hiddens": []},
    )

    config = (
        PPOConfig()
        .environment("chess_env")
        .framework("torch")
        .resources(num_gpus=1 if args.device == "cuda" else 0)
        .env_runners(num_env_runners=0, num_cpus_per_env_runner=1)
        .rl_module(rl_module_spec=rl_module_spec)  # Removed action_distribution_config
        .exploration(explore=False)
        .training(lambda_=0.95, num_epochs=0)
        .experimental(_enable_new_api_stack=True)
    )

    algo = PPO(config=config)
    algo.restore(args.checkpoint)

    env = create_rllib_chess_env({})
    num_episodes = 50
    total_rewards = []
    outcomes = {"white_win": 0, "black_win": 0, "draw": 0, "unknown": 0}

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0

        while not (done or truncated):
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if args.render:
                env.render()

        outcomes[info.get("outcome", "unknown")] += 1
        total_rewards.append(episode_reward)

    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Outcomes: {outcomes}")
    env.close()
    ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Chess RL training using RLlib")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--head_address", type=str, default=None)
    parser.add_argument("--redis_password", type=str, default=None)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--checkpoint_dir", type=str, default="rllib_checkpoints")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--entropy_coeff", type=float, default=0.05)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()