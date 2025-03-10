#!/usr/bin/env python3
"""
Chess Reinforcement Learning using Ray RLlib.
This implementation uses RLlib's built-in distributed training capabilities.
"""

import os
import argparse
import json
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX

# Import custom environment
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper

torch, nn = try_import_torch()

class ChessMaskedModel(TorchModelV2, nn.Module):
    """Custom model for Chess that supports action masking"""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Get feature dimensions from observation space
        assert isinstance(obs_space, gym.spaces.Dict)
        self.board_shape = obs_space["board"].shape
        self.action_mask_shape = obs_space["action_mask"].shape
        self.board_channels = self.board_shape[0]  # 13 channels (6 piece types x 2 colors + empty)
        self.board_size = self.board_shape[1]  # 8x8 board
        
        # Feature extractor CNN
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
            nn.ReLU()
        )
        
        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(832, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        )
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(832, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Current value estimate (needed for RLlib's TorchModelV2 API)
        self._cur_value = None
    
    def forward(self, input_dict, state, seq_lens):
        """Forward pass of the model"""
        # Extract observations
        obs = input_dict["obs"]
        board = obs["board"].float()
        action_mask = obs["action_mask"].float()
        
        # Process the board through the CNN
        batch_size = board.shape[0]
        board_3d = board.reshape(batch_size, self.board_channels, self.board_size, self.board_size)
        features = self.features_extractor(board_3d)
        
        # Get policy logits
        logits = self.policy_net(features)
        
        # Apply action mask: assign -inf to unavailable actions
        inf_mask = torch.clamp(torch.log(action_mask), -FLOAT_MAX, FLOAT_MAX)
        masked_logits = logits + inf_mask
        
        # Save value function output
        self._cur_value = self.value_net(features).squeeze(1)
        
        return masked_logits, state
    
    def value_function(self):
        """Return the current value function estimate"""
        assert self._cur_value is not None, "value function called before forward pass"
        return self._cur_value


def create_rllib_chess_env(config):
    """Factory function to create chess environment for RLlib"""
    env = ChessEnv()
    env = ActionMaskWrapper(env)
    return env


def train(args):
    """Main training function using RLlib PPO"""
    # Initialize Ray
    if args.redis_password:
        ray.init(
            address=args.head_address,
            ignore_reinit_error=True, 
            include_dashboard=args.dashboard,
            _redis_password=args.redis_password
        )
    else:
        ray.init(
            address="auto" if args.distributed else None,
            ignore_reinit_error=True,
            include_dashboard=args.dashboard
        )
    
    # Register the custom model
    ModelCatalog.register_custom_model("chess_masked_model", ChessMaskedModel)
    
    # Configure the algorithm - use AlgorithmConfig directly
    config = (
        AlgorithmConfig()
        .environment(create_rllib_chess_env)
        .framework("torch")
        .env_runners(num_env_runners=args.num_workers)
        .training(
            model={
                "custom_model": "chess_masked_model",
                "custom_model_config": {}
            },
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5
        )
        .resources(
            num_gpus=1 if args.device == "cuda" and not args.force_cpu else 0,
            num_cpus_per_worker=1
        )
    )
    
    # Add checkpoint config
    config = config.checkpointing(
        checkpoint_frequency=args.checkpoint_interval,
        checkpoint_at_end=True,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Run training
    algorithm = PPO(config=config)
    
    # Train for the specified number of iterations
    for i in range(args.max_iterations):
        result = algorithm.train()
        
        # Log some useful metrics
        if i % 5 == 0:  # Log every 5 iterations
            episode_reward_mean = result.get("episode_reward_mean", 0)
            episode_len_mean = result.get("episode_len_mean", 0)
            total_timesteps = result.get("timesteps_total", 0)
            
            print(f"Iteration {i}: reward={episode_reward_mean:.3f}, " +
                  f"length={episode_len_mean:.1f}, total_steps={total_timesteps}")
        
        # Save checkpoint periodically
        if i % args.checkpoint_interval == 0 or i == args.max_iterations - 1:
            checkpoint_path = algorithm.save(args.checkpoint_dir)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint = algorithm.save(args.checkpoint_dir)
    print(f"Final checkpoint saved to {final_checkpoint}")
    
    return final_checkpoint


def evaluate(args):
    """Evaluate a trained policy"""
    raise NotImplementedError("Evaluation mode not implemented yet")


def main():
    """Parse arguments and run appropriate mode"""
    parser = argparse.ArgumentParser(description="Chess RL training using RLlib")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", 
                        help="Mode: train or evaluate")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", 
                        help="Device to use (cpu or cuda)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint to load")
    parser.add_argument("--force_cpu", action="store_true", 
                        help="Force CPU use even if CUDA is available")
    parser.add_argument("--distributed", action="store_true", 
                        help="Enable distributed training")
    parser.add_argument("--head", action="store_true", 
                        help="This node is the head node")
    parser.add_argument("--head_address", type=str, default=None, 
                        help="Ray head node address (for worker nodes)")
    parser.add_argument("--redis_password", type=str, default=None, 
                        help="Redis password for Ray cluster")
    parser.add_argument("--dashboard", action="store_true", 
                        help="Enable Ray dashboard")
    parser.add_argument("--num_workers", type=int, default=2, 
                        help="Number of worker processes")
    parser.add_argument("--max_iterations", type=int, default=5000, 
                        help="Maximum number of training iterations")
    parser.add_argument("--checkpoint_dir", type=str, default="rllib_checkpoints", 
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=10, 
                        help="How often to save checkpoints (iterations)")
    parser.add_argument("--render", action="store_true", 
                        help="Render environment during evaluation")
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Run in appropriate mode
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main() 