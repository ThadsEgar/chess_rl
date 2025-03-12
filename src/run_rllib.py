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
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Import custom environment
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper

torch, nn = try_import_torch()

# Define a custom callback to track chess game statistics
class ChessMetricsCallback(DefaultCallbacks):
    """Callback to track chess-specific metrics during training."""
    
    def __init__(self):
        super().__init__()
        # Import numpy for array operations 
        import numpy as np
        self.np = np
        
        self.win_rates = {"white": [], "black": [], "draw": []}
        self.num_episodes = 0
        self.total_rewards = []
        self.episode_lengths = []
        self.is_white_to_move = {}  # Track which color is to move for each episode
    
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # Initialize tracking for which player is to move in this episode
        episode_id = episode.episode_id
        self.is_white_to_move[episode_id] = True  # Start with White to move
    
    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        # Track player turns - flip after each action
        episode_id = episode.episode_id
        if episode_id in self.is_white_to_move:
            self.is_white_to_move[episode_id] = not self.is_white_to_move[episode_id]
    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # Get the final outcome (from White's perspective)
        last_info = episode.last_info_for(env_index)
        
        # Handle the case where last_info is None (which can happen with some environment errors)
        if last_info is None:
            print(f"Warning: last_info is None for episode {episode.episode_id}, env_index {env_index}")
            # Try to get some info from the episode
            outcome = "unknown"
            
            # See if we can get termination info from the episode length
            if episode.length >= 400:  # Check if we reached move limit
                print(f"   Episode reached max length ({episode.length}), likely terminated due to move limit")
                outcome = "draw"  # Assume draw if we reach the move limit
                
            # Try to infer outcome from reward (since rewards are only given at the end)
            elif abs(episode.total_reward) > 0.9:  # If we got a substantial reward
                if episode.total_reward > 0:
                    outcome = "white_win"
                    print(f"   Inferred white win from positive reward: {episode.total_reward}")
                else:
                    outcome = "black_win"
                    print(f"   Inferred black win from negative reward: {episode.total_reward}")
        else:
            # Extract outcome from info, defaulting to "unknown" if not present
            outcome = last_info.get("outcome", "unknown")
            
            # If outcome is still unknown, try other fields
            if outcome == "unknown" and "game_outcome" in last_info:
                outcome = last_info.get("game_outcome")
            
            # If still unknown but we know it's a draw, set to draw
            if outcome == "unknown" and last_info.get("draw", False):
                outcome = "draw"
        
        # Print diagnostic info for debugging
        if outcome == "unknown":
            print(f"Unknown outcome for episode {episode.episode_id}, length: {episode.length}, reward: {episode.total_reward}")
            if last_info is not None:
                print(f"   Available info keys: {list(last_info.keys())}")
        
        episode_id = episode.episode_id
        
        # Update local counters
        white_win = 0
        black_win = 0
        draw = 0
        game_completed = 1  # By default, consider every episode as completed
        
        # Set the appropriate counter based on outcome
        if outcome == "white_win":
            white_win = 1
            self.win_rates["white"].append(1)
            self.win_rates["black"].append(0)
            self.win_rates["draw"].append(0)
        elif outcome == "black_win":
            black_win = 1
            self.win_rates["white"].append(0)
            self.win_rates["black"].append(1)
            self.win_rates["draw"].append(0)
        elif outcome == "draw":
            draw = 1
            self.win_rates["white"].append(0)
            self.win_rates["black"].append(0)
            self.win_rates["draw"].append(1)
        else:
            # Unknown outcome
            self.win_rates["white"].append(0)
            self.win_rates["black"].append(0)
            self.win_rates["draw"].append(0)
            
        # IMPORTANT: Register metrics with RLlib's reporting system
        episode.custom_metrics["white_win"] = white_win
        episode.custom_metrics["black_win"] = black_win
        episode.custom_metrics["draw"] = draw
        episode.custom_metrics["game_completed"] = game_completed
        episode.custom_metrics["game_length"] = episode.length
            
        # Clean up tracking
        if episode_id in self.is_white_to_move:
            del self.is_white_to_move[episode_id]
            
        # Record other episode stats
        self.num_episodes += 1
        self.total_rewards.append(episode.total_reward)
        self.episode_lengths.append(episode.length)

    def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        """Critical method for correct advantage calculation in zero-sum chess games.
        
        This ensures value targets are properly flipped based on which player (White or Black) is to move.
        """
        # Get trajectory data
        dones = postprocessed_batch["dones"]
        obs = postprocessed_batch["obs"]
        rewards = postprocessed_batch["rewards"]
        
        # Extract turn information from observations
        is_white_turn = []
        for observation in obs:
            # Use the white_to_move field we added to the observation
            if isinstance(observation, dict) and "white_to_move" in observation:
                # Convert to boolean (0=False, 1=True)
                is_white_turn.append(bool(observation["white_to_move"]))
            else:
                # Fallback to alternating turns if field not available
                is_white_turn.append(len(is_white_turn) % 2 == 0)
        
        # Convert is_white_turn to numpy array
        is_white_turn = self.np.array(is_white_turn)
        
        # Find the final outcome (z) - the reward at terminal state
        # In chess, rewards are sparse and only given at the end of the game
        terminal_reward = None
        if any(dones):
            # Find the first done step
            for i, done in enumerate(dones):
                if done:
                    terminal_reward = rewards[i]  # This is the final reward from White's perspective
                    break
        
        # If no terminal reward in this batch, we can't properly set value targets
        if terminal_reward is None:
            # Skip processing if there's no terminal state
            return postprocessed_batch
        
        # Calculate value targets for each state in the trajectory
        # Convert to numpy array to match RLlib's expected format
        value_targets = self.np.zeros_like(rewards)
        
        for i, white_turn in enumerate(is_white_turn):
            # Apply correct zero-sum value target:
            # White's turn: value target = terminal_reward (outcome from White's perspective)
            # Black's turn: value target = -terminal_reward (outcome from Black's perspective)
            if white_turn:
                value_targets[i] = terminal_reward  # White's perspective
            else:
                value_targets[i] = -terminal_reward  # Black's perspective flipped
        
        # Update batch with corrected value targets
        if "value_targets" in postprocessed_batch:
            postprocessed_batch["value_targets"] = value_targets
        
        return postprocessed_batch

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Called after each training iteration, to log chess statistics"""
        import gc
        
        # Helper to find metrics recursively in the result structure
        def find_metrics_recursively(data, prefix=""):
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in ["white_win_mean", "black_win_mean", "draw_mean", "game_completed_mean"]:
                        return True
                    if isinstance(v, (dict, list)):
                        if find_metrics_recursively(v, prefix=f"{prefix}.{k}" if prefix else k):
                            return True
            return False
        
        # Check for metrics in different possible locations (RLlib structure changed in versions)
        custom_metrics = {}
        episodes_this_iter = result.get("episodes_this_iter", 0)
        
        # Try the env_runners structure (newer RLlib versions)
        if "env_runners" in result:
            env_runners = result["env_runners"]
            if "custom_metrics" in env_runners:
                custom_metrics = env_runners["custom_metrics"]
            episodes_this_iter = env_runners.get("episodes_this_iter", episodes_this_iter)
        
        # If nothing found, check the top-level custom_metrics
        if not custom_metrics and "custom_metrics" in result:
            custom_metrics = result["custom_metrics"]
        
        # Extract metrics for the current iteration
        white_wins = custom_metrics.get("white_win_mean", 0) * episodes_this_iter
        black_wins = custom_metrics.get("black_win_mean", 0) * episodes_this_iter
        draws = custom_metrics.get("draw_mean", 0) * episodes_this_iter
        completed = custom_metrics.get("game_completed_mean", 0) * episodes_this_iter
        
        # Add to total wins
        if not hasattr(algorithm, "total_white_wins"):
            algorithm.total_white_wins = 0
            algorithm.total_black_wins = 0
            algorithm.total_draws = 0
            algorithm.total_completed_games = 0
            algorithm.total_episodes = 0
        
        algorithm.total_white_wins += int(white_wins)
        algorithm.total_black_wins += int(black_wins)
        algorithm.total_draws += int(draws)
        algorithm.total_completed_games += int(completed)
        algorithm.total_episodes += episodes_this_iter
        
        # Print a summary
        print("\n----- Chess Stats -----")
        print(f"White Wins: {algorithm.total_white_wins}")
        print(f"Black Wins: {algorithm.total_black_wins}")
        print(f"Draws:      {algorithm.total_draws}")
        print(f"Total Completed Games: {algorithm.total_completed_games}")
        
        # Calculate win percentages (if any games have been completed)
        if algorithm.total_completed_games > 0:
            white_win_pct = algorithm.total_white_wins / algorithm.total_completed_games * 100
            black_win_pct = algorithm.total_black_wins / algorithm.total_completed_games * 100
            draw_pct = algorithm.total_draws / algorithm.total_completed_games * 100
            print(f"Win %: White={white_win_pct:.1f}%, Black={black_win_pct:.1f}%, Draw={draw_pct:.1f}%")
        
        # Calculate completion rate
        if algorithm.total_episodes > 0:
            completion_rate = algorithm.total_completed_games / algorithm.total_episodes * 100
            print(f"Completion Rate: {completion_rate:.1f}%")
        print("----------------------\n")
        
        # Explicitly run garbage collection to recover memory
        gc.collect()
        
        # Clear any large temporary variables that might be held
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory cache


class ChessMaskedModel(TorchModelV2, nn.Module):
    """Custom model for Chess that supports action masking"""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Print observation space details for debugging
        print(f"Initializing ChessMaskedModel with observation space: {obs_space}")
        if isinstance(obs_space, gym.spaces.Dict):
            print(f"Dict keys: {obs_space.spaces.keys()}")
            for key, space in obs_space.spaces.items():
                print(f"  {key}: {space}")
        
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
        
        # Epsilon-greedy exploration parameters
        self.initial_epsilon = 0.1  # 10% random action probability
        self.final_epsilon = 0.02   # 2% random action probability
        self.epsilon_timesteps = 1_000_000  # Decay over 1M timesteps
        self.current_epsilon = self.initial_epsilon
        self.random_exploration = True  # Set to False to disable exploration
        self.timesteps = 0  # Track timesteps for epsilon decay
        
        # Check if we're in evaluation mode from model config
        model_config_dict = model_config.get("custom_model_config", {})
        if model_config_dict.get("evaluation_mode", False):
            self.random_exploration = False
            print("Model initialized in evaluation mode, exploration disabled")
        else:
            print(f"Initialized epsilon-greedy exploration with initial_epsilon={self.initial_epsilon}, final_epsilon={self.final_epsilon}")
    
    def set_evaluation_mode(self, evaluation_mode=True):
        """Explicitly set whether this model is in evaluation mode (disables exploration)"""
        previous_mode = not self.random_exploration
        if evaluation_mode:
            self.random_exploration = False
            print("Model set to evaluation mode, exploration disabled")
        else:
            self.random_exploration = True
            print("Model set to training mode, exploration enabled")
        return previous_mode
    
    def update_epsilon(self, num_steps=None):
        """Update exploration epsilon based on timesteps or the provided step count"""
        if num_steps is not None:
            self.timesteps = num_steps
        else:
            self.timesteps += 1
            
        # Linear decay from initial_epsilon to final_epsilon over epsilon_timesteps
        if self.timesteps < self.epsilon_timesteps:
            self.current_epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (self.timesteps / self.epsilon_timesteps)
        else:
            self.current_epsilon = self.final_epsilon
            
        return self.current_epsilon
        
    def forward(self, input_dict, state, seq_lens):
        # Check if we're in inference mode
        # RLlib sets deterministic=True during evaluation/inference
        inference_mode = input_dict.get("deterministic", False)
        if inference_mode:
            # Temporarily disable exploration during this forward pass
            original_exploration = self.random_exploration
            self.random_exploration = False
        
        # Get the device from the input tensors
        if isinstance(input_dict.get("obs", {}), dict) and "board" in input_dict["obs"] and hasattr(input_dict["obs"]["board"], "device"):
            device = input_dict["obs"]["board"].device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
                        board = torch.zeros((1, 13, 8, 8), device=device)
                        action_mask = torch.ones((1, 20480), device=device)
                else:
                    # Default board if we can't extract properly
                    board = torch.zeros((1, 13, 8, 8), device=device)
                    action_mask = torch.ones((1, 20480), device=device)
        else:
            # If "obs" not in input_dict, create default values
            board = torch.zeros((1, 13, 8, 8), device=device)
            action_mask = torch.ones((1, 20480), device=device)
            
        # Process through CNN feature extractor
        features = self.features_extractor(board)
        
        # Add small epsilon to features to avoid NaN issues
        features = features + 1e-8
        
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
            if isinstance(action_mask, np.ndarray):
                # Convert numpy array to torch tensor
                action_mask_tensor = torch.FloatTensor(action_mask).to(device)
                # Use -1000 instead of -FLOAT_MAX for better numerical stability
                inf_mask = torch.clamp(1 - action_mask_tensor, min=0, max=1) * -1000.0
            else:
                # Already a torch tensor
                inf_mask = torch.clamp(1 - action_mask, min=0, max=1) * -1000.0
            
            # Apply the mask to the logits
            masked_logits = action_logits + inf_mask
            
            # Simple epsilon-greedy exploration - only apply during training
            if self.random_exploration:
                # Decay epsilon over time
                self.update_epsilon()
                batch_size = masked_logits.shape[0]
                
                # Generate random numbers to decide which samples get random actions
                random_samples = torch.rand(batch_size, device=device) < self.current_epsilon
                
                # For samples selected for random exploration
                for i in range(batch_size):
                    if random_samples[i]:
                        # Find legal actions (mask value == 1 or equivalently, where inf_mask == 0)
                        legal_actions = torch.where(action_mask[i] > 0)[0]
                        
                        # Only proceed if there are legal actions
                        if len(legal_actions) > 0:
                            # Choose a random legal action
                            random_action_idx = legal_actions[torch.randint(0, len(legal_actions), (1,), device=device)].item()
                            
                            # Use smaller values for more numerical stability (5.0 instead of 10.0)
                            masked_logits[i] = torch.full_like(masked_logits[i], -5.0)
                            masked_logits[i, random_action_idx] = 5.0
            
            # Apply gradient clipping to logits
            masked_logits = torch.clamp(masked_logits, min=-50.0, max=50.0)
            
            # If we temporarily disabled exploration, restore it
            if inference_mode:
                self.random_exploration = original_exploration
                
            return masked_logits, state
        
        # If no action mask available, return unmasked logits (with clipping)
        action_logits = torch.clamp(action_logits, min=-50.0, max=50.0)
        
        # If we temporarily disabled exploration, restore it
        if inference_mode:
            self.random_exploration = original_exploration
            
        return action_logits, state
        
    def value_function(self):
        """Return the current value function estimate for the last processed batch."""
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
        
        # ALWAYS use the Dict wrapper regardless of current observation space
        # Print current observation space for debugging
        print(f"Wrapping environment with DictObsWrapper, original obs space: {type(env.observation_space)}")
        
        # Define a custom wrapper right here to ensure Dict observation space
        class DictObsWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                # Define the correct observation space as Dict
                self.observation_space = spaces.Dict({
                    'board': spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
                    'action_mask': spaces.Box(low=0, high=1, shape=(env.action_space.n,), dtype=np.float32),
                    'white_to_move': spaces.Discrete(2)  # Boolean flag: 0=False (Black's turn), 1=True (White's turn)
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
                if len(result) == 4:  # obs, reward, done, info (old style)
                    obs, reward, done, info = result
                    # Convert to new Gymnasium API (obs, reward, terminated, truncated, info)
                    info = self._ensure_info(info)
                    return self._wrap_observation(obs), reward, done, False, info
                elif len(result) == 5:  # obs, reward, terminated, truncated, info (new style)
                    obs, reward, terminated, truncated, info = result
                    info = self._ensure_info(info)
                    return self._wrap_observation(obs), reward, terminated, truncated, info
                else:
                    # Handle unexpected result format
                    print(f"WARNING: Unexpected step result format with {len(result)} elements")
                    # Return a safe default with the correct format
                    return self._wrap_observation(None), 0.0, True, False, {"outcome": "unknown"}
            
            def _wrap_observation(self, obs):
                # Check board shape and fix if needed
                if isinstance(obs, dict) and 'board' in obs and (not isinstance(obs['board'], np.ndarray) or obs['board'].shape != (13, 8, 8)):
                    print(f"Warning: Fixing board shape from {getattr(obs['board'], 'shape', 'unknown')} to (13, 8, 8)")
                    # Create a proper shape board with zeros
                    proper_board = np.zeros((13, 8, 8), dtype=np.float32)
                    
                    # If it's a 2D board, try to convert it to channel format
                    if isinstance(obs['board'], np.ndarray) and len(obs['board'].shape) == 2 and obs['board'].shape == (8, 8):
                        # Set channel values based on piece values
                        board_2d = obs['board']
                        for i in range(8):
                            for j in range(8):
                                piece_val = board_2d[i, j]
                                if piece_val > 0:  # White pieces (1-6)
                                    proper_board[int(piece_val)-1, i, j] = 1.0
                                elif piece_val < 0:  # Black pieces (-1 to -6)
                                    proper_board[int(abs(piece_val))+5, i, j] = 1.0
                                else:  # Empty squares
                                    proper_board[12, i, j] = 1.0
                        obs['board'] = proper_board
                    else:
                        # Default handling for other cases
                        proper_board[12, :, :] = 1.0  # Set empty squares
                        obs['board'] = proper_board
                                
                # Convert observation to Dict format if it's not already
                if isinstance(obs, dict) and 'board' in obs and 'action_mask' in obs:
                    # If observation already has white_to_move field, use it
                    if 'white_to_move' in obs:
                        return {
                            'board': obs['board'],
                            'action_mask': obs['action_mask'],
                            'white_to_move': int(obs['white_to_move'])  # Convert to int for Discrete space
                        }
                    else:
                        # Add a default white_to_move (assuming White's turn)
                        return {
                            'board': obs['board'],
                            'action_mask': obs['action_mask'],
                            'white_to_move': 1  # Default to White's turn if not specified
                        }
                elif obs is None:
                    # Create a placeholder observation when None is passed
                    return {
                        'board': np.zeros((13, 8, 8), dtype=np.float32),
                        'action_mask': np.ones(self.env.action_space.n, dtype=np.float32),
                        'white_to_move': 1  # Default to White's turn
                    }
                else:
                    # Create a placeholder observation with reasonable defaults
                    print(f"WARNING: Unexpected observation format: {type(obs)}, creating placeholder.")
                    return {
                        'board': np.zeros((13, 8, 8), dtype=np.float32),
                        'action_mask': np.ones(self.env.action_space.n, dtype=np.float32),
                        'white_to_move': 1  # Default to White's turn
                    }
                    
            def _ensure_info(self, info):
                """Ensure the info dictionary has the required fields"""
                if info is None:
                    info = {}
                
                # Ensure outcome is always present
                if "outcome" not in info:
                    # Try to extract info from the environment if possible
                    if hasattr(self.env, 'env') and hasattr(self.env.env, 'board'):
                        # Get direct access to the chess board
                        board = self.env.env.board
                        
                        # Check game state conditions
                        if board.is_checkmate():
                            # Who won?
                            if board.turn:  # Black's turn now, so White won
                                info["outcome"] = "white_win"
                                info["game_outcome"] = "white_win"
                            elif not board.turn:  # White's turn now, so Black won
                                info["outcome"] = "black_win"
                                info["game_outcome"] = "black_win"
                        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
                            info["outcome"] = "draw"
                            info["game_outcome"] = "draw"
                        elif hasattr(self.env.env, 'move_count') and hasattr(self.env.env, 'max_moves') and self.env.env.move_count >= self.env.env.max_moves:
                            info["outcome"] = "draw"
                            info["game_outcome"] = "draw"
                            info["termination_reason"] = "move_limit_exceeded"
                        # If game_outcome exists, use that
                        elif "game_outcome" in info:
                            info["outcome"] = info["game_outcome"]
                        else:
                            info["outcome"] = "unknown"
                    else:
                        # If outcome exists, use that
                        if "game_outcome" in info:
                            info["outcome"] = info["game_outcome"]
                        else:
                            info["outcome"] = "unknown"
                
                # Ensure game_outcome is always present
                if "game_outcome" not in info:
                    # If outcome exists, use that
                    if "outcome" in info:
                        info["game_outcome"] = info["outcome"]
                    else:
                        info["game_outcome"] = "unknown"
                
                # Add termination_reason if not present
                if "termination_reason" not in info:
                    info["termination_reason"] = "unknown"
                
                return info
                    
        # Always apply the wrapper to ensure consistent observation format
        env = DictObsWrapper(env)
        
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
                super().__init__()
                # Define action and observation spaces
                self.action_space = spaces.Discrete(20480)  # Same as ChessEnv
                
                # Define observation space with same structure as ChessEnv and DictObsWrapper
                self.observation_space = spaces.Dict({
                    'board': spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
                    'action_mask': spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.float32),
                    'white_to_move': spaces.Discrete(2)  # Boolean flag: 0=False (Black's turn), 1=True (White's turn)
                })
            
            def reset(self, **kwargs):
                # Return a default observation with the correct structure
                obs = {
                    'board': np.zeros((13, 8, 8), dtype=np.float32),
                    'action_mask': np.ones(self.action_space.n, dtype=np.float32),
                    'white_to_move': 1  # Default to White's turn
                }
                # Set empty squares in board representation (channel 12)
                obs['board'][12, :, :] = 1.0
                
                return obs, {}
            
            def step(self, action):
                # Return a default observation with the correct Gymnasium API format
                # (obs, reward, terminated, truncated, info)
                obs = {
                    'board': np.zeros((13, 8, 8), dtype=np.float32),
                    'action_mask': np.ones(self.action_space.n, dtype=np.float32),
                    'white_to_move': 1  # Default to White's turn
                }
                # Set empty squares in board representation (channel 12)
                obs['board'][12, :, :] = 1.0
                
                return obs, 0.0, True, False, {}
        
        return PlaceholderEnv()


def train(args):
    """Main training function using RLlib PPO"""
    # Import required modules
    import os
    
    # Optimize PyTorch memory usage for CUDA
    if args.device == "cuda" and not args.force_cpu:
        # Enable TensorFloat32 for faster computation on Ampere GPUs
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # Set memory allocation strategy to reduce fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
        print("🚀 Enabled PyTorch CUDA optimizations")
    
    # Initialize Ray with fixed number of CPUs and GPUs
    if args.redis_password:
        ray.init(
            address=args.head_address,
            ignore_reinit_error=True, 
            include_dashboard=args.dashboard,
            _redis_password=args.redis_password,
            num_cpus=120,
        )
    else:
        ray.init(
            address="auto" if args.distributed else None,
            ignore_reinit_error=True,
            include_dashboard=args.dashboard,
            num_cpus=120,
        )
    
    # Hardware configuration - using fixed allocation as requested
    print("\n===== Hardware Configuration =====")
    print(f"Ray runtime resources: {ray.available_resources()}")
    
    # Fixed resource allocation as requested:
    # - 3:3 GPU split
    # - 20 workers with 4 CPUs each
    driver_gpus = 4          # Fixed at 3 GPUs for driver
    worker_gpus = 1.999         # Fixed at 3 GPUs for ivworkers
    num_workers = 20          # Fixed at 20 workers
    cpus_per_worker = 4       # Fixed at 4 CPUs per worker
    driver_cpus = 16       # Fixed at 8 CPUs for driver
    num_envs = 8            # Environments per worker
    
    # Calculate exact GPU allocation per worker
    if num_workers > 0 and worker_gpus > 0:
        # If workers > GPUs, give each worker a fixed fraction (no remainder)
        if num_workers >= worker_gpus:
            # Integer workers per GPU
            workers_per_gpu = num_workers // worker_gpus
            
            # Distribute GPUs evenly, ensuring each worker gets the same amount
            # (e.g., with 3 GPUs and 20 workers, each worker gets 3/20 = 0.15 GPU)
            gpus_per_worker = worker_gpus / num_workers
            
            # Round to 2 decimal places to avoid floating point issues
            gpus_per_worker = round(worker_gpus / num_workers, 6)
        else:
            # If GPUs > workers, each worker gets at least 1 GPU
            gpus_per_worker = worker_gpus // num_workers
    else:
        gpus_per_worker = 0
    
    # Use fixed batch sizes
    train_batch_size = 32768 * 4
    sgd_minibatch_size = 4096 * 4
    
    total_cpu_request = driver_cpus + (cpus_per_worker * num_workers)
    print(f"CPU allocation: {driver_cpus} (driver) + {cpus_per_worker}*{num_workers} (workers) = {total_cpu_request}")
    
    # Print GPU allocation
    print(f"GPU allocation: {driver_gpus} (driver) + {gpus_per_worker}*{num_workers} (workers) = {driver_gpus + (gpus_per_worker * num_workers)}")
    print(f"Batch size: {train_batch_size} (train) / {sgd_minibatch_size} (SGD)")
    print("==================================\n")
    
    # Test environment creation to diagnose observation space issues
    print("\n===== Environment Test =====")
    env = create_rllib_chess_env({})
    print(f"Environment observation space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")
    obs, _ = env.reset()
    print(f"Observation structure: {type(obs)}")
    if isinstance(obs, dict):
        print(f"Observation keys: {obs.keys()}")
        for key, value in obs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {type(value)} with shape {value.shape}")
            else:
                print(f"  {key}: {type(value)} with value {value}")
    print("============================\n")
    
    # Register the custom model and environment
    ModelCatalog.register_custom_model("chess_masked_model", ChessMaskedModel)
    tune.register_env("chess_env", create_rllib_chess_env)
    
    # Create an absolute path for checkpoint directory
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Using checkpoint directory: {checkpoint_dir}")
    
    # Print exploration strategy
    print(f"\n===== Exploration Strategy =====")
    print(f"Initial entropy coefficient: {args.entropy_coeff}")
    print(f"Annealing schedule: 100% → 50% → 10% over {args.max_iterations} iterations")
    print(f"This will encourage the agent to explore diverse moves early and exploit its knowledge later.")
    print(f"================================\n")
    
    # Optimize configuration with fixed resource allocation
    config = {
        "env": "chess_env",
        "framework": "torch",
        "disable_env_checking": True,
        "_enable_rl_module_api": True,
        "_enable_learner_api": True,
        "enable_rl_module_and_learner": False,
        
        # Resource allocation - use whole numbers as requested
        "num_cpus_for_driver": driver_cpus,
        "num_workers": num_workers,
        "num_cpus_per_env_runner": cpus_per_worker,
        "num_gpus": driver_gpus,
        "num_gpus_per_env_runner": gpus_per_worker,
        "num_envs_per_env_runner": num_envs,
        
        # Model configuration
        "model": {
            "custom_model": "chess_masked_model",
            "custom_model_config": {"handle_missing_action_mask": True},
            "no_final_linear": True  # Prevent RLlib from adding extra layers
        },
        
        # Disable preprocessors that are causing the dimension mismatch
        "preprocessor_pref": None,
        "_disable_preprocessor_api": True,
        "observation_filter": "NoFilter",
        "compress_observations": False,
        
        # Training parameters
        "train_batch_size": int(train_batch_size),  # Ensure integer
        "sgd_minibatch_size": int(sgd_minibatch_size),  # Ensure integer
        "mini_batch_size": int(sgd_minibatch_size),
        "num_sgd_iter": 5,
        "num_epochs": 5,
        "lr": 5e-5,  # Reduced for numerical stability
        "callbacks": ChessMetricsCallback,
        "create_env_on_driver": False,  # Disable environment on driver to save resources
        "log_level": "INFO",
        
        # Gradient clipping to prevent NaN issues
        "grad_clip": 1.0,
        
        # Training Optimization - use GPU-optimized experience batches
        "batch_mode": "truncate_episodes",  # Process data in fixed-size chunks for better GPU utilization
        "rollout_fragment_length": "auto",  # Increased for better GPU utilization with more CPUs
        "_use_trajectory_view_api": True,
        "shuffle_buffer_size": 0,  # Disable shuffle buffer to save memory
        
        # Zero-sum game specific settings
        "gamma": 1.0,  # No temporal discounting for chess (outcome only matters at end)
        "lambda": 1.0,  # Use Monte Carlo estimate for advantage
        "use_critic": True,
        "use_gae": True,
        "vf_loss_coeff": 1.0,  # Balance policy and value function losses
        "postprocess_inputs": True,  # Allow our callback to process trajectories
        "normalize_actions": False,  # Chess actions are discrete and masked
        
        # Entropy settings to encourage exploration
        "entropy_coeff": args.entropy_coeff,  # Add entropy bonus for exploration
        "entropy_coeff_schedule": [
            [0, args.entropy_coeff],  # Start with the specified coefficient
            [args.max_iterations * 0.5, args.entropy_coeff * 0.5],  # Halfway through, halve the coefficient
            [args.max_iterations, args.entropy_coeff * 0.1],  # By the end, reduce to 10% of original
        ],
    }

    # Set up checkpoint restoration if provided
    restore_path = None
    if args.checkpoint:
        # Validate checkpoint path
        if os.path.exists(args.checkpoint):
            restore_path = args.checkpoint
            print(f"Will restore from checkpoint: {restore_path}")
        else:
            print(f"Warning: Checkpoint path {args.checkpoint} does not exist. Starting fresh.")

    # Enable true multi-GPU training with the learner API
    config.update({
        # Set up dedicated learners
        "num_learners": 5,                          # Use 5 dedicated learner processes
        "num_gpus_per_learner": 1.0,                # Each learner gets a full GPU
        "num_gpus": 1.0,                            # Driver only needs 1 GPU
        "num_gpus_per_env_runner": 0.0,             # Don't allocate GPUs to workers with learner API
        
        # Optimize data flow to learners
        "num_aggregator_actors_per_learner": 3,     # More aggregators = faster data feeding
        "max_requests_in_flight_per_learner": 3,    # Allow multiple batches in flight
        
        # Reduce CPU overhead
        "torch_compile_learner": True,              # Use torch.compile() for faster training
        "torch_compile_learner_dynamo_backend": "inductor",
        
        # Optimized GPU batch sizes
        "train_batch_size": 131072,                 # Larger batch for more GPU utilization
        "sgd_minibatch_size": 16384,                # Larger minibatch for better GPU saturation
        "num_sgd_iter": 3,                          # Fewer SGD iterations per batch 
        "num_epochs": 3,                            # Fewer epochs for faster training
        
        # Memory optimization
        "simple_optimizer": True,                   # Use simple optimizer for better memory usage
    })
    
    print("\n===== Multi-GPU Learner Configuration =====")
    print(f"Distributed training across {config['num_learners']} dedicated learners")
    print(f"GPUs: 1 (driver) + {config['num_learners']}*1.0 (learners) = {1 + config['num_learners']}")
    print(f"Each learner gets 1 full GPU for maximum throughput")
    print(f"Workers will focus on data collection (no GPUs allocated)")
    print("===========================================\n")
    
    analysis = tune.run(
        "PPO",
        stop={"training_iteration": args.max_iterations},
        checkpoint_freq=25,  # Save checkpoint every 25 iterations (increased frequency)
        checkpoint_at_end=True,
        storage_path=checkpoint_dir,
        verbose=2,  # Detailed output
        metric="env_runners/episode_reward_mean",
        mode="max",
        resume="AUTO",  # AUTO mode: resume if checkpoint exists, otherwise start fresh
        restore=restore_path,  # Add this for restoring from specific checkpoint
        config=config,
    )
    
    # Get best checkpoint
    best_checkpoint = analysis.best_checkpoint
    print(f"Best checkpoint: {best_checkpoint}")
    
    return best_checkpoint


def evaluate(args):
    """Evaluate a trained policy"""
    if not args.checkpoint:
        print("Error: Checkpoint path required for evaluation mode")
        return

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint path '{args.checkpoint}' does not exist")
        return
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, include_dashboard=args.dashboard)
    
    # Register environment and model same as in training
    ModelCatalog.register_custom_model("chess_masked_model", ChessMaskedModel)
    tune.register_env("chess_env", create_rllib_chess_env)
    
    # Create evaluation configuration
    config = {
        "env": "chess_env",
        "framework": "torch",
        "num_workers": 0,  # Single worker for evaluation
        "num_gpus": 1 if args.device == "cuda" else 0,  # Use exact integers for GPU allocation
        "model": {
            "custom_model": "chess_masked_model",
            "custom_model_config": {
                "handle_missing_action_mask": True,
                "evaluation_mode": True  # Explicitly use evaluation mode (disables exploration)
            }
        },
        # Critical: Ensure we're using deterministic actions (no exploration)
        "explore": False,
    }
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    # Create algorithm and restore from checkpoint
    algo = PPO(config=config)
    algo.restore(args.checkpoint)
    
    # Create environment for evaluation
    env = create_rllib_chess_env({})
    
    # Run evaluation
    num_episodes = 50  # Number of games to evaluate
    total_rewards = []
    outcomes = {"white_win": 0, "black_win": 0, "draw": 0, "unknown": 0}
    
    print(f"Evaluating model over {num_episodes} episodes...")
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (done or truncated):
            # Get action from policy, using deterministic=True to disable exploration
            action = algo.compute_single_action(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Optionally render if requested
            if args.render:
                env.render()
        
        # Record outcome
        outcome = info.get("outcome", "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward}, Steps = {steps}, Outcome = {outcome}")
    
    # Print evaluation results
    print("\n----- Evaluation Results -----")
    print(f"Average Episode Reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Win %: White={outcomes['white_win']/num_episodes*100:.1f}%, Black={outcomes['black_win']/num_episodes*100:.1f}%, Draw={outcomes['draw']/num_episodes*100:.1f}%")
    print(f"Total Outcomes: White Wins: {outcomes['white_win']}, Black Wins: {outcomes['black_win']}, Draws: {outcomes['draw']}")
    print("-----------------------------\n")
    
    # Close environment
    env.close()
    
    # Shutdown Ray
    ray.shutdown()


def main():
    """Parse arguments and run appropriate mode"""
    parser = argparse.ArgumentParser(description="Chess RL training using RLlib")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", 
                        help="Mode: train or evaluate")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", 
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
    parser.add_argument("--max_iterations", type=int, default=10000, 
                        help="Maximum number of training iterations")
    parser.add_argument("--checkpoint_dir", type=str, default="rllib_checkpoints", 
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=10, 
                        help="How often to save checkpoints (iterations)")
    parser.add_argument("--render", action="store_true", 
                        help="Render environment during evaluation")
    parser.add_argument("--inference_mode", choices=["cpu", "gpu"], default="gpu", 
                        help="Inference mode: cpu or gpu")
    parser.add_argument("--entropy_coeff", type=float, default=0.05, 
                        help="Entropy coefficient for PPO")
    
    args = parser.parse_args()
    
    # Run in appropriate mode
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main() 