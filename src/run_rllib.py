#!/usr/bin/env python3
"""
Chess Reinforcement Learning using Ray RLlib.
This implementation uses RLlib's built-in distributed training capabilities with environment-based reward shaping.
"""
# Standard library imports
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Third-party imports
import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import PolicyID
from ray.tune.utils import merge_dicts
from ray.rllib.core.columns import Columns

# Local imports
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper
import gc
import os
import platform
import tracemalloc
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import gymnasium as gym

from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import (
    OldAPIStack,
    override,
    OverrideToImplementCustomLogic,
    PublicAPI,
)
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import AgentID, EnvType, EpisodeType, PolicyID
from ray.tune.callback import _CallbackMeta

# Framework-specific imports
torch, nn = try_import_torch()

# Metrics-only callback
class ChessCombinedCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        # Extract episode info from the last step
        infos = episode.get_infos()
        info = infos[-1] if infos else {}
        
        # Calculate metrics based on episode outcome
        white_win = 1.0 if info.get("outcome") == "white_win" else 0.0
        black_win = 1.0 if info.get("outcome") == "black_win" else 0.0
        draw = 1.0 if info.get("outcome") == "draw" else 0.0
        checkmate = 1.0 if info.get("termination_reason") == "checkmate" else 0.0
        stalemate = 1.0 if info.get("termination_reason") == "stalemate" else 0.0
        
        # Use the metrics_logger to properly record these metrics with a sliding window
        if metrics_logger is not None:
            # Each metric is logged with a sliding window of 100 episodes
            metrics_logger.log_value("white_win", white_win, window=100)
            metrics_logger.log_value("black_win", black_win, window=100)
            metrics_logger.log_value("draw", draw, window=100)
            metrics_logger.log_value("checkmate", checkmate, window=100)
            metrics_logger.log_value("stalemate", stalemate, window=100)
        else:
            # Fallback to print if no metrics_logger available (shouldn't happen with newer RLlib)
            print(f"Game outcome metrics (no metrics_logger available): white_win={white_win}, black_win={black_win}, draw={draw}, checkmate={checkmate}, stalemate={stalemate}")

class ChessMaskingRLModule(TorchRLModule):
    def __init__(self, observation_space=None, action_space=None, model_config=None, inference_only=False, catalog_class=None, **kwargs):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config or {},
            inference_only=inference_only,
            catalog_class=catalog_class,
            **kwargs
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
        
    def _forward(self, batch, **kwargs):
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
            Columns.ACTIONS: torch.distributions.Categorical(logits=masked_logits).sample(),
            Columns.ACTION_DIST_INPUTS: masked_logits,
            Columns.ACTION_LOGP: torch.distributions.Categorical(logits=masked_logits).log_prob(
                torch.distributions.Categorical(logits=masked_logits).sample()
            ),
            Columns.VF_PREDS: value
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
            self.steps = 0
            print(f"DictObsWrapper initialized with action space: {self.action_space}")

        def reset(self, **kwargs):
            self.steps = 0
            print(f"Environment reset")
            obs, info = self.env.reset(**kwargs)
            wrapped_obs = self._wrap_observation(obs)
            print(f"Reset obs: {type(wrapped_obs)}, info: {info}")
            return wrapped_obs, info

        def step(self, action):
            self.steps += 1
            #print(f"Step {self.steps}, action: {action}")
            obs, reward, terminated, truncated, info = self.env.step(action)
            #print(f"Raw step result - reward: {reward}, terminated: {terminated}, info: {info}")
            
            # Terminal-only rewards with player perspective flipping
            shaped_reward = 0.0  # No intermediate rewards
            
            # Only assign rewards at termination
            if terminated and "outcome" in info:
                outcome = info["outcome"]
                last_player = (self.steps - 1) % 2  # 0 for White, 1 for Black
                
                if outcome == "white_win":
                    shaped_reward = 1.0 if last_player == 0 else -1.0
                elif outcome == "black_win":
                    shaped_reward = 1.0 if last_player == 1 else -1.0
                elif outcome == "draw":
                    shaped_reward = 0.0
            
            #print(f"Shaped reward: {shaped_reward} (original: {reward})")
            wrapped_obs = self._wrap_observation(obs)
            #print(f"Wrapped obs type: {type(wrapped_obs)}")
            return wrapped_obs, shaped_reward, terminated, truncated, info

        def _wrap_observation(self, obs):
            if not isinstance(obs, dict) or "board" not in obs or "action_mask" not in obs:
                #print(f"Warning: Invalid observation format: {type(obs)}")
                raise ValueError(f"Invalid observation format: {type(obs)}")
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
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPUs")
    else:
        gpu_count = 0

    ray.init(
        address=args.head_address if args.distributed else None,
        ignore_reinit_error=True,
        include_dashboard=args.dashboard,
        _redis_password=args.redis_password,
        num_cpus=118,
        num_gpus=gpu_count,
    )

    # Resource configuration
    num_env_runners = 1
    driver_gpus = 0.5
    num_cpus_per_env_runner = 4
    num_gpus_per_env_runner = 0.5
    num_envs_per_env_runner = 4
    num_learners = 3
    num_gpus_per_learner = 1

    # Register the environment
    tune.register_env("chess_env", create_rllib_chess_env)

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    observation_space = gym.spaces.Dict({
        "board": gym.spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(20480,), dtype=np.float32),
        "white_to_move": gym.spaces.Discrete(2)
    })
    action_space = gym.spaces.Discrete(20480)

    # In Ray 2.31.0, we use the module_class directly in the config.rl_module() call
    config = (
        PPOConfig()
        .environment("chess_env")
        .framework("torch")
        .resources(
            num_gpus=driver_gpus,
            num_cpus_for_main_process=8,
        )
        .learners(
            num_learners=num_learners,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=num_cpus_per_env_runner,
            num_gpus_per_env_runner=num_gpus_per_env_runner,
            sample_timeout_s=None,
        )
        .training(
            train_batch_size_per_learner=4096,
            train_batch_size=256,
            num_epochs=10,
            lr=5e-5,
            grad_clip=1.0,
            gamma=1.0,            # No discounting - equal weight for all moves
            use_gae=True,         # Enable GAE
            lambda_=0.95,         # GAE lambda parameter
            vf_loss_coeff=0.5,    # Value function loss coefficient
            entropy_coeff=args.entropy_coeff,
            clip_param=0.2,       # PPO clipping parameter
            kl_coeff=0.2,         # KL divergence coefficient
            vf_share_layers=False,
        )
        .callbacks(ChessCombinedCallback)
        # Ray 2.31.0 style RL module configuration
        .rl_module(
            module_class=ChessMaskingRLModule,
            observation_space=observation_space,
            action_space=action_space,
            model_config={},
        )
        # Use the new API stack fully
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        # Disable config validation to avoid version-specific issues
        .experimental(_validate_config=False)
    )

    print(f"Training with {num_learners} learners, {num_env_runners} env runners")

    analysis = tune.run(
        "PPO",  # Use default PPO since CustomPPO was for old API debugging
        stop={"training_iteration": args.max_iterations},
        checkpoint_freq=25,
        checkpoint_at_end=True,
        storage_path=checkpoint_dir,
        verbose=3,
        config=config.to_dict(),  # Convert to dict for tune.run
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

    # Ray 2.31.0 style configuration
    config = (
        PPOConfig()
        .environment("chess_env")
        .framework("torch")
        .resources(num_gpus=1 if args.device == "cuda" else 0)
        .env_runners(num_env_runners=0, num_cpus_per_env_runner=1)
        .rl_module(
            module_class=ChessMaskingRLModule,
            observation_space=observation_space,
            action_space=action_space,
            model_config={},
        )
        .exploration(explore=False)
        .training(lambda_=0.95, num_epochs=0)
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .experimental(_validate_config=False)
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