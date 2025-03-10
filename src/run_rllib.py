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
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.models.torch.torch_distributions import TorchCategorical
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX

# Import custom environment
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper

torch, nn = try_import_torch()

class ChessRLModule(RLModule):
    """Custom RLModule for Chess that supports action masking"""
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize action dist class
        self.action_dist_cls = TorchCategorical
        
        # Get feature dimensions
        self.board_channels = 13  # Number of channels in chess board representation
        self.board_size = 8  # Size of chess board (8x8)
        
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
            nn.Linear(512, 20480)  # Match the large action space we defined
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
    
    def _forward_inference(self, batch):
        """Forward pass for inference"""
        return self._common_forward(batch)
    
    def _forward_exploration(self, batch):
        """Forward pass for exploration"""
        return self._common_forward(batch)
    
    def _forward_train(self, batch):
        """Forward pass for training"""
        return self._common_forward(batch)
    
    def _common_forward(self, batch):
        """Common forward pass logic for all modes"""
        # Extract observations
        obs = batch["obs"]
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
        
        # Get value function estimate
        values = self.value_net(features).squeeze(-1)
        
        # Return both outputs
        return {"action_dist_inputs": masked_logits, "values": values}


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
    
    # Create a config dictionary
    config = {
        "env": create_rllib_chess_env,
        "framework": "torch",
        "num_workers": args.num_workers,
        "num_gpus": 1 if args.device == "cuda" and not args.force_cpu else 0,
        # RL Module configuration
        "rl_module": {
            "_disable_preprocessor_api": False,
            "rl_module_spec": {
                "module_class": ChessRLModule,
                "config": {},
            },
        },
        # PPO specific configs
        "gamma": 0.99,
        "lambda": 0.95,
        "kl_coeff": 0.2,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "lr": 3e-4,
        "clip_param": 0.2,
        "vf_clip_param": 10.0,
        "entropy_coeff": 0.01,
        "vf_loss_coeff": 0.5,
        # Checkpointing
        "checkpoint_freq": args.checkpoint_interval,
        "checkpoint_at_end": True,
        "local_dir": args.checkpoint_dir
    }
    
    # Create the algorithm with the config
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