#!/usr/bin/env python3
"""
Chess Reinforcement Learning using Ray RLlib.
This implementation uses RLlib's built-in distributed training capabilities with environment-based reward shaping.
"""
# Standard library imports
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple

# Third-party imports
import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import PolicyID
from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

# Local imports
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper
import gc
import os

# Framework-specific imports
torch, nn = try_import_torch()

# Metrics-only callback
class ChessCombinedCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeV2, "EpisodeType"],
        env_runner: Optional[Any] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
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
    """RLModule that implements action masking for chess environment.
    
    This RLModule handles action masking by using the action_mask provided
    by the environment to mask out invalid actions.
    """
    
    @override(TorchRLModule)
    def setup(self):
        """Initialize neural network modules."""
        self.board_shape = self.observation_space["board"].shape
        self.board_channels = self.board_shape[0]
        self.action_space_n = self.action_space.n

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
        
    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for inference (evaluation).
        
        Args:
            batch: Input batch of observations with action masks.
            
        Returns:
            Dict with action distribution inputs and value predictions.
        """
        return self._masked_forward(batch, explore=False)
    
    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for exploration (training data collection).
        
        Args:
            batch: Input batch of observations with action masks.
            
        Returns:
            Dict with action distribution inputs and value predictions.
        """
        return self._masked_forward(batch, explore=True)
        
    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for training.
        
        Args:
            batch: Input batch containing observations, action masks, actions, etc.
            
        Returns:
            Dict with value function predictions for training.
        """
        # Extract observations from batch
        obs = batch.get(Columns.OBS)
        
        # Handle observations during training (may have been processed by connectors)
        if isinstance(obs, dict) and "board" in obs:
            board = obs["board"]
            # During training, action_mask might be in a different format due to connectors
            if "action_mask" in obs:
                action_mask = obs["action_mask"]
            else:
                # If action mask not directly available, we still need value predictions
                action_mask = None
        else:
            # If observations have been flattened, we still need to compute values
            board = obs
            action_mask = None
            
        if len(board.shape) == 3:
            board = board.unsqueeze(0)
        
        # Extract features from board state
        features = self.features_extractor(board)
        
        # Compute value predictions (required for PPO training)
        value = self.value_head(features).view(-1, 1)
        
        # If we don't have action masks, we just need to return value predictions
        if action_mask is None:
            return {
                Columns.VF_PREDS: value
            }
            
        # Otherwise, we compute policy outputs with masking
        if len(action_mask.shape) == 1:
            action_mask = action_mask.unsqueeze(0)
            
        action_mask = action_mask.to(features.device)
        action_logits = self.policy_head(features)
        
        # Apply action masking
        masked_logits = action_logits + (action_mask - 1) * 1e9
        
        return {
            Columns.VF_PREDS: value,
            Columns.ACTION_DIST_INPUTS: masked_logits
        }
    
    def _masked_forward(self, batch: Dict[str, Any], explore: bool) -> Dict[str, Any]:
        """Common forward pass implementation for both inference and exploration.
        
        Args:
            batch: Input batch containing observations and action masks.
            explore: Whether to use exploration or deterministic actions.
            
        Returns:
            Dict with actions, action distribution inputs, action log probs, and value predictions.
        """
        # Extract observations
        obs = batch.get(Columns.OBS, batch)
        
        # Extract board and action mask
        board = obs["board"]
        action_mask = obs["action_mask"]
        
        # Handle single observations vs batches
        if len(board.shape) == 3:
            board = board.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
            
        # Extract features from board state
        features = self.features_extractor(board)
        
        # Get logits and apply action masking
        action_logits = self.policy_head(features)
        action_mask = action_mask.to(action_logits.device)
        masked_logits = action_logits + (action_mask - 1) * 1e9  # Large negative for invalid actions
        
        # Create categorical distribution for action sampling
        dist = torch.distributions.Categorical(logits=masked_logits)
        
        # Sample actions or take deterministic actions based on exploration flag
        if explore:
            actions = dist.sample()
        else:
            actions = torch.argmax(masked_logits, dim=-1)
            
        # Compute value predictions
        value = self.value_head(features).view(-1, 1)
        
        return {
            Columns.ACTIONS: actions,
            Columns.ACTION_DIST_INPUTS: masked_logits,
            Columns.ACTION_LOGP: dist.log_prob(actions),
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

    # Create an RLModuleSpec instance
    module_spec = RLModuleSpec(
        module_class=ChessMaskingRLModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config={},
    )

    # Configure PPO with Ray 2.43.0 API conventions
    config = (
        PPOConfig()
        .environment("chess_env")
        .framework("torch")
        # Resources configuration
        .resources(
            num_gpus=driver_gpus,
            num_cpus_for_main_process=8,
        )
        # Learner configuration
        .learners(
            num_learners=num_learners,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        # Environment runners configuration
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=num_cpus_per_env_runner,
            num_gpus_per_env_runner=num_gpus_per_env_runner,
            add_default_connectors_to_env_to_module_pipeline=True,
            add_default_connectors_to_module_to_env_pipeline=True,
            remote_env_batch_wait_ms=0,
            sample_timeout_s=None,
        )
        # Training configuration
        .training(
            train_batch_size_per_learner=4096,
            minibatch_size=256,
            num_epochs=1,
            lr=5e-5,
            grad_clip=1.0,
            gamma=1.0,
            use_gae=True,
            lambda_=0.95,
            vf_loss_coeff=0.5,
            entropy_coeff=args.entropy_coeff,
            clip_param=0.2,
            kl_coeff=0.2,
            vf_share_layers=False,
        )
        # Callback configuration
        .callbacks(ChessCombinedCallback)
        # Custom RL module configuration
        .rl_module(
            model_config_dict={},
            rl_module_spec=module_spec,
        )
    )

    print(f"Training with {num_learners} learners, {num_env_runners} env runners")

    analysis = tune.run(
        "PPO",
        stop={"training_iteration": args.max_iterations},
        checkpoint_freq=25,
        checkpoint_at_end=True,
        storage_path=checkpoint_dir,
        verbose=3,
        config=config,
        resume="AUTO",
        restore=args.checkpoint if args.checkpoint and os.path.exists(args.checkpoint) else None,
    )

    return analysis.best_checkpoint

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
if __name__ == "__main__":
    main()