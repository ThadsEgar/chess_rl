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
    
    # First wrap with ActionMaskWrapper to add the action mask to observation
    env = ActionMaskWrapper(env)
    
    # Verify the observation space is a Dict with the correct structure
    if not isinstance(env.observation_space, gym.spaces.Dict) or 'board' not in env.observation_space.spaces or 'action_mask' not in env.observation_space.spaces:
        # If not, add ObservationDictWrapper to ensure proper Dict format
        from src.ray_a3c import ObservationDictWrapper
        env = ObservationDictWrapper(env)
    
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
    
    # Register the environment
    tune.register_env("chess_env", create_rllib_chess_env)
    
    # Create an absolute path for checkpoint directory
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Using checkpoint directory: {checkpoint_dir}")
    
    # Configure RLlib using Ray Tune directly to bypass API version issues
    analysis = tune.run(
        "PPO",
        stop={"training_iteration": args.max_iterations},
        checkpoint_freq=args.checkpoint_interval,
        checkpoint_at_end=True,
        storage_path=checkpoint_dir,
        verbose=1,
        config={
            "env": "chess_env",
            "framework": "torch",
            "num_workers": args.num_workers,
            "num_gpus": 1 if args.device == "cuda" and not args.force_cpu else 0,
            "model": {
                "custom_model": "chess_masked_model",
            },
            # PPO specific configs
            "gamma": 0.99,
            "lambda": 0.95,
            "kl_coeff": 0.2,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 30,
            "lr": 3e-4,
            "clip_param": 0.2,
            "vf_clip_param": 10.0,
            "entropy_coeff": 0.01,
            "vf_loss_coeff": 0.5,
            # Completely bypass validation
            "_enable_new_api_stack": False,
            "_experimental_enable_new_api_stack": False,
            "_disable_execution_plan_api": True,
            "_skip_validate_config": True,
            "enable_rl_module_and_learner": False
        },
    )
    
    # Get best checkpoint
    best_checkpoint = analysis.best_checkpoint
    print(f"Best checkpoint: {best_checkpoint}")
    
    return best_checkpoint


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
    
    # Run in appropriate mode
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main() 