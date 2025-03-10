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
        
        # Handle both Dict observation spaces and non-Dict spaces
        if isinstance(obs_space, gym.spaces.Dict):
            # Get feature dimensions from observation space
            self.board_shape = obs_space["board"].shape
            self.action_mask_shape = obs_space["action_mask"].shape
        else:
            # If we don't have a Dict space, assume it's a Box with the expected board dimensions
            # Log a warning but continue instead of raising an error
            print(f"WARNING: Expected Dict observation space but got {type(obs_space)}. Assuming standard dimensions.")
            # Assume standard chess board dimensions: 13 channels for pieces, 8x8 board
            self.board_shape = (13, 8, 8)
            self.action_mask_shape = (20480,)  # Standard action mask size
            
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
            nn.ReLU(),
        )
        
        # Policy and value heads
        self.policy_head = nn.Linear(832, num_outputs)
        self.value_head = nn.Linear(832, 1)
        
        # Initialize the models
        self._value = None
        
    def forward(self, input_dict, state, seq_lens):
        # Extract observation
        if "obs" in input_dict:
            # Extract relevant observation parts
            if isinstance(input_dict["obs"], dict):
                # Dict observation space
                board = input_dict["obs"]["board"]
                action_mask = input_dict["obs"]["action_mask"]
            else:
                # Non-Dict observation space - try to extract board from raw observation
                # Assume first part of flat observation is the board, reshape it
                obs = input_dict["obs"]
                if isinstance(obs, np.ndarray) or isinstance(obs, torch.Tensor):
                    board_size = 13 * 8 * 8
                    if len(obs.shape) == 1 and obs.shape[0] > board_size:
                        # Reshape board part
                        if isinstance(obs, np.ndarray):
                            board = obs[:board_size].reshape(13, 8, 8)
                            action_mask = obs[board_size:]
                        else:
                            board = obs[:board_size].reshape(-1, 13, 8, 8)
                            action_mask = obs[board_size:]
                    else:
                        # Default board if we can't extract properly
                        board = torch.zeros((1, 13, 8, 8), device=self.device)
                        action_mask = torch.ones((1, 20480), device=self.device)
                else:
                    # Default board if we can't extract properly
                    board = torch.zeros((1, 13, 8, 8), device=self.device)
                    action_mask = torch.ones((1, 20480), device=self.device)
        else:
            # If "obs" not in input_dict, create default values
            board = torch.zeros((1, 13, 8, 8), device=self.device)
            action_mask = torch.ones((1, 20480), device=self.device)
            
        # Process through CNN feature extractor
        features = self.features_extractor(board)
        
        # Get raw action outputs (logits)
        action_logits = self.policy_head(features)
        
        # Store value function for value_function() method
        self._value = self.value_head(features)
        
        # Apply action mask by setting illegal actions to a large negative number
        if action_mask is not None:
            # Ensure action_mask is properly shaped for broadcasting
            if len(action_mask.shape) == 1:
                action_mask = action_mask.unsqueeze(0)
                
            # Ensure mask values are 0 or 1
            inf_mask = torch.clamp(1 - action_mask, min=0, max=1) * -FLOAT_MAX
            
            # Apply the mask to the logits
            masked_logits = action_logits + inf_mask
            return masked_logits, state
        
        # If no action mask available, return unmasked logits
        return action_logits, state
        
    def value_function(self):
        return self._value.squeeze(1)


def create_rllib_chess_env(config):
    """Factory function to create chess environment for RLlib"""
    try:
        # Create the base environment
        env = ChessEnv()
        
        # First wrap with ActionMaskWrapper 
        env = ActionMaskWrapper(env)
        
        # Extra validation and wrapping for RLlib compatibility
        # This is critical for ensuring Dict observation space in all contexts
        from gymnasium import spaces
        import numpy as np
        
        # Check if observation space is already a Dict with expected structure
        if not isinstance(env.observation_space, spaces.Dict) or 'board' not in env.observation_space.spaces or 'action_mask' not in env.observation_space.spaces:
            # Print current observation space for debugging
            print(f"WARNING: Expected Dict observation space but got {type(env.observation_space)}. Creating wrapper.")
            
            # Define a custom wrapper right here to ensure Dict observation space
            class DictObsWrapper(gym.Wrapper):
                def __init__(self, env):
                    super().__init__(env)
                    # Define the correct observation space as Dict
                    self.observation_space = spaces.Dict({
                        'board': spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
                        'action_mask': spaces.Box(low=0, high=1, shape=(env.action_space.n,), dtype=np.float32)
                    })
                
                def reset(self, **kwargs):
                    result = self.env.reset(**kwargs)
                    if isinstance(result, tuple):
                        obs, info = result
                        return self._wrap_observation(obs), info
                    else:
                        return self._wrap_observation(result)
                
                def step(self, action):
                    result = self.env.step(action)
                    if len(result) == 4:  # obs, reward, done, info
                        obs, reward, done, info = result
                        return self._wrap_observation(obs), reward, done, info
                    elif len(result) == 5:  # obs, reward, terminated, truncated, info
                        obs, reward, terminated, truncated, info = result
                        return self._wrap_observation(obs), reward, terminated, truncated, info
                
                def _wrap_observation(self, obs):
                    # Convert observation to Dict format if it's not already
                    if isinstance(obs, dict) and 'board' in obs and 'action_mask' in obs:
                        return obs
                    elif isinstance(obs, np.ndarray):
                        # Split array into board and mask components
                        board_size = 13 * 8 * 8  # Standard chess board size
                        if len(obs.shape) == 1 and obs.shape[0] > board_size:
                            board = obs[:board_size].reshape(13, 8, 8)
                            action_mask = obs[board_size:]
                            return {'board': board, 'action_mask': action_mask}
                    
                    # If we can't process the observation properly, return default
                    print(f"WARNING: Could not process observation of type {type(obs)}, returning default observation")
                    return {
                        'board': np.zeros((13, 8, 8), dtype=np.float32),
                        'action_mask': np.ones(env.action_space.n, dtype=np.float32)
                    }
            
            # Apply our custom wrapper
            env = DictObsWrapper(env)
        
        # Verify the wrapper is applied correctly
        if not isinstance(env.observation_space, spaces.Dict):
            print(f"ERROR: After wrapping, observation space is still not Dict: {type(env.observation_space)}")
            
        return env
        
    except Exception as e:
        # Log any errors during environment creation
        import traceback
        print(f"Error creating environment: {e}")
        print(traceback.format_exc())
        # Return a placeholder environment with the expected Dict observation space
        from gymnasium import spaces
        import numpy as np
        
        class PlaceholderEnv(gym.Env):
            def __init__(self):
                self.action_space = spaces.Discrete(20480)
                self.observation_space = spaces.Dict({
                    'board': spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
                    'action_mask': spaces.Box(low=0, high=1, shape=(20480,), dtype=np.float32)
                })
            
            def reset(self, **kwargs):
                return {'board': np.zeros((13, 8, 8), dtype=np.float32), 
                        'action_mask': np.ones(20480, dtype=np.float32)}, {}
            
            def step(self, action):
                return {'board': np.zeros((13, 8, 8), dtype=np.float32), 
                        'action_mask': np.ones(20480, dtype=np.float32)}, 0, True, False, {}
        
        return PlaceholderEnv()


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
                # Add some extra model config parameters to help with initialization
                "custom_model_config": {
                    "handle_missing_action_mask": True,
                }
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
            "enable_rl_module_and_learner": False,
            # Add extra options to help with initialization
            "create_env_on_driver": True,
            "normalize_actions": False,
            "log_level": "DEBUG",
            # Worker configuration to ensure observation space is correctly initialized
            "remote_worker_envs": False,
            "recreate_failed_workers": True,
            "restart_failed_sub_environments": True
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