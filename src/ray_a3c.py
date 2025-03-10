import ray
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import json
import argparse
from collections import deque
import logging
from datetime import datetime
import chess
import random
from pathlib import Path
import socket
import platform
import gym
from gym import spaces

# Import the PyTorch chess model
from src.chess_model import ChessCNN, MCTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ray_a3c.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ray_a3c")

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


@ray.remote
class ParameterServer:
    """
    A parameter server that holds the current model weights and distributes them to workers.
    """
    def __init__(self, model_config=None, device='cpu', checkpoint_path=None):
        # Validate the requested device is actually available
        self.device = self._validate_device(device)
        
        # Create model
        if model_config:
            self.model = model_config['model_class'](**model_config['model_kwargs'])
        else:
            from src.chess_model import ChessCNN
            self.model = ChessCNN()
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Initialize metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': [],
            'entropy_losses': [],
            'total_losses': [],
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'games': 0,
            'total_moves': 0,  # Track total moves across all environments
        }
        
        # Current model version
        self.version = 0
        
        # Training hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        
        logger.info(f"ParameterServer initialized with device: {self.device}")
    
    def _validate_device(self, requested_device):
        """Validate and possibly adjust the requested device based on availability."""
        if requested_device == 'cuda':
            # Verify CUDA is actually available
            try:
                if torch.cuda.is_available():
                    # Test if CUDA can be initialized
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    return 'cuda'
                else:
                    logger.warning("CUDA requested but torch.cuda.is_available() is False")
                    return 'cpu'
            except Exception as e:
                logger.warning(f"CUDA initialization failed: {str(e)}")
                return 'cpu'
        elif requested_device == 'mps':
            # Verify MPS is available (Apple Silicon)
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # Test if MPS can be initialized
                    test_tensor = torch.zeros(1).to('mps')
                    del test_tensor
                    return 'mps'
                else:
                    logger.warning("MPS requested but not available")
                    return 'cpu'
            except Exception as e:
                logger.warning(f"MPS initialization failed: {str(e)}")
                return 'cpu'
        
        # For 'cpu' or any other device, just return as is
        return requested_device
    
    def get_weights(self):
        """Return current model weights"""
        self.version += 1
        return {
            'weights': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'version': self.version,
            'device': self.device
        }
    
    def apply_gradients(self, gradients, metrics=None):
        """Apply gradients from a worker to update the model"""
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Set gradients
        for g, (name, param) in zip(gradients, self.model.named_parameters()):
            if g is not None:
                param.grad = g.to(self.device)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Update weights
        self.optimizer.step()
        
        # Record metrics if provided
        if metrics:
            # Update episode metrics
            if 'episode_reward' in metrics:
                self.metrics['episode_rewards'].append(metrics['episode_reward'])
            if 'episode_length' in metrics:
                self.metrics['episode_lengths'].append(metrics['episode_length'])
            
            # Update loss metrics
            if 'value_loss' in metrics:
                self.metrics['value_losses'].append(metrics['value_loss'])
            if 'policy_loss' in metrics:
                self.metrics['policy_losses'].append(metrics['policy_loss'])
            if 'entropy_loss' in metrics:
                self.metrics['entropy_losses'].append(metrics['entropy_loss'])
            if 'total_loss' in metrics:
                self.metrics['total_losses'].append(metrics['total_loss'])
                
            # Update game outcome metrics
            if 'white_win' in metrics and metrics['white_win']:
                self.metrics['white_wins'] += 1
                self.metrics['games'] += 1
            elif 'black_win' in metrics and metrics['black_win']:
                self.metrics['black_wins'] += 1
                self.metrics['games'] += 1
            elif 'draw' in metrics and metrics['draw']:
                self.metrics['draws'] += 1
                self.metrics['games'] += 1
        
        # Return new weights
        return self.get_weights()
    
    def get_metrics(self):
        """Return current training metrics"""
        # Calculate derived metrics
        metrics = {**self.metrics}
        
        # Add win rates
        if metrics['games'] > 0:
            metrics['white_win_rate'] = metrics['white_wins'] / metrics['games']
            metrics['black_win_rate'] = metrics['black_wins'] / metrics['games']
            metrics['draw_rate'] = metrics['draws'] / metrics['games']
        
        # Add recent average metrics
        window_size = min(100, len(metrics['episode_rewards']))
        if window_size > 0:
            metrics['recent_avg_reward'] = np.mean(metrics['episode_rewards'][-window_size:])
            metrics['recent_avg_length'] = np.mean(metrics['episode_lengths'][-window_size:])
        
        return metrics
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'version': self.version,
            'metrics': self.metrics
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.version = checkpoint.get('version', 0)
        
        # Load metrics if available
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']
        
        logger.info(f"Checkpoint loaded from {path} (version {self.version})")

    def increment_total_moves(self, num_moves):
        """Increment the total moves counter"""
        self.metrics['total_moves'] += num_moves
        return self.metrics['total_moves']


@ray.remote
class RolloutWorker:
    """
    A worker that collects experience by playing games and computes gradients.
    """
    def __init__(self, worker_id, env_creator, param_server, device='cpu', mcts_sims=0):
        self.worker_id = worker_id
        self.env = env_creator()
        self.param_server = param_server
        
        # Validate device
        self.requested_device = device
        self.device = self._validate_device(device)
        
        # Initialize model
        try:
            self.model = ChessCNN().to(self.device)
            self.model.eval()  # Start in eval mode
        except Exception as e:
            logger.error(f"Worker {worker_id}: Error initializing model on {device}: {str(e)}")
            # Fall back to CPU if device initialization fails
            if self.device != 'cpu':
                logger.info(f"Worker {worker_id}: Falling back to CPU")
                self.device = 'cpu'
                self.model = ChessCNN().to(self.device)
                self.model.eval()
        
        # Initialize MCTS if needed
        self.mcts_sims = mcts_sims
        if self.mcts_sims > 0:
            try:
                self.model.init_mcts(num_simulations=self.mcts_sims)
                logger.info(f"Worker {worker_id}: MCTS initialized with {mcts_sims} simulations")
            except Exception as e:
                logger.error(f"Worker {worker_id}: Error initializing MCTS: {str(e)}")
                self.mcts_sims = 0
        
        # Get initial weights
        self.update_weights()
        
        # Experience buffer
        self.buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # Episode tracking
        self.current_obs = None
        self.episode_reward = 0
        self.episode_length = 0
        self.player_color = None  # None=random, 1=white, 0=black
        
        # Metrics for complete episodes
        self.episode_metrics = {
            'episode_reward': [],
            'episode_length': [],
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'games': 0
        }
        
        # Initialize environment
        self.reset_env()
        
        logger.info(f"Worker {worker_id} initialized with device: {self.device}, MCTS sims: {self.mcts_sims}")
    
    def _validate_device(self, requested_device):
        """Validate and possibly adjust the requested device based on availability."""
        if requested_device == 'cuda':
            # Verify CUDA is actually available
            try:
                if torch.cuda.is_available():
                    # Test if CUDA can be initialized
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    return 'cuda'
                else:
                    logger.warning(f"Worker {self.worker_id}: CUDA requested but torch.cuda.is_available() is False")
                    return 'cpu'
            except Exception as e:
                logger.warning(f"Worker {self.worker_id}: CUDA initialization failed: {str(e)}")
                return 'cpu'
        elif requested_device == 'mps':
            # Verify MPS is available (Apple Silicon)
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # Test if MPS can be initialized
                    test_tensor = torch.zeros(1).to('mps')
                    del test_tensor
                    return 'mps'
                else:
                    logger.warning(f"Worker {self.worker_id}: MPS requested but not available")
                    return 'cpu'
            except Exception as e:
                logger.warning(f"Worker {self.worker_id}: MPS initialization failed: {str(e)}")
                return 'cpu'
        
        # For 'cpu' or any other device, just return as is
        return requested_device
    
    def update_weights(self):
        """Get latest weights from parameter server"""
        try:
            weights_info = ray.get(self.param_server.get_weights.remote())
            self.model.load_state_dict({k: v.to(self.device) for k, v in weights_info['weights'].items()})
            self.model_version = weights_info['version']
            
            # Update device if parameter server is using a different device
            if 'device' in weights_info and weights_info['device'] != self.device:
                logger.info(f"Worker {self.worker_id}: Parameter server using {weights_info['device']}, " 
                          f"worker using {self.device}")
                
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error updating weights: {str(e)}")
            raise
    
    def reset_env(self):
        """Reset the environment and update the current observation"""
        try:
            result = self.env.reset()
            
            # Handle different return formats
            if isinstance(result, tuple) and len(result) >= 1:
                # Modern Gym API: (obs, info)
                self.current_obs = result[0]
            else:
                self.current_obs = result

            # Reset episode trackers
            self.episode_reward = 0
            self.episode_length = 0
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error in reset_env: {str(e)}", exc_info=True)
            # Create a fallback observation
            self.current_obs = {'board': np.zeros((13, 8, 8), dtype=np.float32), 'action_mask': np.ones(20480, dtype=np.float32)}
            self.episode_reward = 0
            self.episode_length = 0
    
    def select_action(self, obs):
        """Select an action using the model or MCTS"""
        try:
            # Handle different observation formats
            if isinstance(obs, tuple) and len(obs) >= 1:
                # Tuple format: (obs, info)
                actual_obs = obs[0]
            else:
                # Direct observation
                actual_obs = obs
            
            # Convert observation to tensor format expected by the model
            if isinstance(actual_obs, dict) and 'board' in actual_obs and 'action_mask' in actual_obs:
                # Dict format with board and action_mask
                # Create copies of the arrays to ensure contiguous memory layout
                board_data = np.array(actual_obs['board']).copy() if isinstance(actual_obs['board'], np.ndarray) else actual_obs['board']
                mask_data = np.array(actual_obs['action_mask']).copy() if isinstance(actual_obs['action_mask'], np.ndarray) else actual_obs['action_mask']
                
                obs_dict = {
                    'board': torch.FloatTensor(board_data).unsqueeze(0).to(self.device),
                    'action_mask': torch.FloatTensor(mask_data).unsqueeze(0).to(self.device)
                }
            elif isinstance(actual_obs, np.ndarray):
                # Numpy array format - split into board and mask components
                # Create a copy to ensure contiguous memory layout
                actual_obs_copy = actual_obs.copy()
                
                # Assuming board is first 832 elements (13*8*8) and mask is the rest
                board_size = 13 * 8 * 8  # Board size
                if len(actual_obs_copy.shape) == 1 and actual_obs_copy.shape[0] > board_size:
                    obs_dict = {
                        'board': torch.FloatTensor(actual_obs_copy[:board_size]).unsqueeze(0).to(self.device),
                        'action_mask': torch.FloatTensor(actual_obs_copy[board_size:]).unsqueeze(0).to(self.device)
                    }
                else:
                    # Just treat the whole thing as a board
                    obs_dict = {
                        'board': torch.FloatTensor(actual_obs_copy).unsqueeze(0).to(self.device),
                        'action_mask': torch.ones(1, 20480, device=self.device)  # Updated from 4672 to 20480
                    }
            else:
                # Unknown format - log error and create a dummy observation
                logger.error(f"Worker {self.worker_id}: Unknown observation format: {type(actual_obs)}")
                obs_dict = {
                    'board': torch.zeros(1, 13 * 8 * 8, device=self.device),
                    'action_mask': torch.ones(1, 20480, device=self.device)  # Updated from 4672 to 20480
                }
            
            # Use MCTS if enabled and we're playing as the current player
            # Determine current player from environment state if possible
            if hasattr(self.env, 'get_current_player'):
                current_player = self.env.get_current_player()
            elif hasattr(self.env, 'board') and hasattr(self.env.board, 'turn'):
                current_player = 1 if self.env.board.turn else 0  # 1=white, 0=black
            else:
                # Fallback - assume it's white's turn
                current_player = 1
            
            use_mcts = (self.mcts_sims > 0 and current_player == self.player_color)
            
            if use_mcts:
                # Use MCTS to select action
                with torch.no_grad():
                    # Get a copy of the game state
                    if hasattr(self.env, 'board'):
                        state = self.env.board.copy()
                    elif hasattr(self.env, 'state') and hasattr(self.env.state, 'board'):
                        state = self.env.state.board.copy()
                    else:
                        logger.error(f"Worker {self.worker_id}: Could not get board state for MCTS")
                        # Fall back to using the model directly
                        result = self.model(obs_dict)
                        return result['actions'][0].item(), result['values'][0].item(), result['log_probs'][0].item()
                    
                    # Run MCTS search
                    action, _ = self.model.mcts.search(state)
                    
                    # Get value and log_prob for the selected action
                    result = self.model(obs_dict)
                    values = result['values']
                    
                    # Get log probability for the MCTS action
                    actions_tensor = torch.tensor([action], device=self.device)
                    eval_result = self.model.evaluate_actions(obs_dict, actions_tensor)
                    log_probs = eval_result['log_probs']
            else:
                # Use model directly
                with torch.no_grad():
                    result = self.model(obs_dict)
                    action = result['actions'][0].item()
                    values = result['values']
                    log_probs = result['log_probs']
            
            return action, values[0].item(), log_probs[0].item()
        
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error in select_action: {str(e)}", exc_info=True)
            # Return a safe fallback
            return 0, 0.0, 0.0
    
    def collect_experience(self, num_steps=200):
        """Collect experience for a number of steps"""
        # Initialize buffer
        self.buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # Run until buffer is filled or episode ends
        complete_episode = False
        num_moves = 0  # Count moves in this collection
        
        try:
            for _ in range(num_steps):
                # Save current observation
                self.buffer['obs'].append(self.current_obs)
                
                # Select action
                try:
                    action, value, log_prob = self.select_action(self.current_obs)
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error selecting action: {str(e)}", exc_info=True)
                    # Use a safe default action
                    action, value, log_prob = 0, 0.0, 0.0
                
                # Step the environment
                try:
                    step_result = self.env.step(action)
                    num_moves += 1  # Count valid move
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error stepping environment: {str(e)}", exc_info=True)
                    # Create a safe fallback
                    if isinstance(self.current_obs, dict) and 'board' in self.current_obs:
                        dummy_obs = {
                            'board': self.current_obs['board'],
                            'action_mask': np.ones(20480, dtype=np.float32)
                        }
                    else:
                        dummy_obs = {'board': np.zeros((13, 8, 8), dtype=np.float32), 'action_mask': np.ones(20480, dtype=np.float32)}
                    
                    step_result = (dummy_obs, 0.0, True, {'error': str(e)})
                
                # Handle different return formats from environment step
                if isinstance(step_result, tuple):
                    if len(step_result) >= 4:  # Modern Gym/Gymnasium API: (obs, reward, done, truncated, info)
                        next_obs = step_result[0]
                        reward = step_result[1]
                        done = step_result[2]
                        # May also have truncated and info
                        if len(step_result) >= 5:
                            info = step_result[4]
                        else:
                            info = {}
                    else:
                        # Unexpected format
                        logger.error(f"Worker {self.worker_id}: Unexpected step result format with {len(step_result)} elements")
                        next_obs, reward, done, info = {}, 0, True, {}
                else:
                    # Unexpected format
                    logger.error(f"Worker {self.worker_id}: Unexpected step result type: {type(step_result)}")
                    next_obs, reward, done, info = {}, 0, True, {}
                
                # Save transition
                self.buffer['actions'].append(action)
                self.buffer['values'].append(value)
                self.buffer['log_probs'].append(log_prob)
                self.buffer['rewards'].append(reward)
                self.buffer['dones'].append(done)
                
                # Update episode tracking
                self.episode_reward += reward
                self.episode_length += 1
                
                # Handle episode termination
                if done:
                    # Record episode metrics
                    self.episode_metrics['episode_reward'].append(self.episode_reward)
                    self.episode_metrics['episode_length'].append(self.episode_length)
                    
                    # Track game outcomes
                    self.episode_metrics['games'] += 1
                    if isinstance(info, dict):
                        if info.get('white_won', False):
                            self.episode_metrics['white_wins'] += 1
                        elif info.get('black_won', False):
                            self.episode_metrics['black_wins'] += 1
                        else:
                            self.episode_metrics['draws'] += 1
                    else:
                        # Default to draw if info is not available
                        self.episode_metrics['draws'] += 1
                    
                    # Reset environment for next episode
                    self.reset_env()
                    complete_episode = True
                else:
                    # Continue episode
                    self.current_obs = next_obs
                
                # Break if steps exceed limit (to prevent overly long episodes)
                if self.episode_length >= 1000:
                    # Force termination and reset
                    self.episode_metrics['episode_reward'].append(self.episode_reward)
                    self.episode_metrics['episode_length'].append(self.episode_length)
                    self.episode_metrics['draws'] += 1
                    self.episode_metrics['games'] += 1
                    self.reset_env()
                    complete_episode = True
                    break
            
            # Process the buffer for training
            processed_buffer = self.process_buffer()
            
            return processed_buffer, complete_episode, self.get_current_metrics(), num_moves
        
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error in collect_experience: {str(e)}", exc_info=True)
            # Return empty results as a fallback
            return {
                'obs': [],
                'actions': np.array([]),
                'returns': np.array([]),
                'advantages': np.array([]),
                'log_probs': np.array([]),
                'values': np.array([])
            }, True, {}, 0  # Added 0 for num_moves
    
    def process_buffer(self):
        """Process buffer for training (calculate returns and advantages)"""
        try:
            # Check if buffer is empty or has no observations
            if not self.buffer or 'obs' not in self.buffer or len(self.buffer['obs']) == 0:
                logger.warning(f"Worker {self.worker_id}: Empty buffer in process_buffer")
                return {
                    'obs': [],
                    'actions': np.array([]),
                    'returns': np.array([]),
                    'advantages': np.array([]),
                    'log_probs': np.array([]),
                    'values': np.array([])
                }
            
            # Convert to numpy arrays
            obs = self.buffer['obs']
            actions = np.array(self.buffer['actions'])
            rewards = np.array(self.buffer['rewards'])
            values = np.array(self.buffer['values'])
            log_probs = np.array(self.buffer['log_probs'])
            dones = np.array(self.buffer['dones'])
            
            # Calculate returns and advantages using GAE
            advantages = np.zeros_like(rewards)
            returns = np.zeros_like(rewards)
            
            # Get next value (0 if last observation led to done)
            if dones[-1]:
                next_value = 0
            else:
                # Get value for the last observation
                try:
                    # Only try to predict a value if we have a valid observation
                    if isinstance(self.current_obs, dict) and 'board' in self.current_obs and 'action_mask' in self.current_obs:
                        # Create copies to avoid stride issues
                        board_data = np.array(self.current_obs['board']).copy() if isinstance(self.current_obs['board'], np.ndarray) else self.current_obs['board']
                        mask_data = np.array(self.current_obs['action_mask']).copy() if isinstance(self.current_obs['action_mask'], np.ndarray) else self.current_obs['action_mask']
                        
                        obs_dict = {
                            'board': torch.FloatTensor(board_data).unsqueeze(0).to(self.device),
                            'action_mask': torch.FloatTensor(mask_data).unsqueeze(0).to(self.device)
                        }
                        
                        with torch.no_grad():
                            next_value = self.model.predict_values(obs_dict).item()
                    else:
                        # Fallback if observation is invalid
                        next_value = 0
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error getting next value: {str(e)}")
                    next_value = 0
            
            # Calculate GAE
            gae = 0
            gamma = 0.99
            gae_lambda = 0.95
            
            for t in reversed(range(len(rewards))):
                # For terminal states, the next value is 0
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_values = values[t + 1]
                
                # Calculate delta and GAE
                delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
                gae = delta + gamma * gae_lambda * next_non_terminal * gae
                
                # Store return and advantage
                advantages[t] = gae
                returns[t] = gae + values[t]
            
            # Return processed buffer
            return {
                'obs': obs,
                'actions': actions,
                'returns': returns,
                'advantages': advantages,
                'log_probs': log_probs,
                'values': values
            }
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error in process_buffer: {str(e)}", exc_info=True)
            # Return empty buffer as fallback
            return {
                'obs': [],
                'actions': np.array([]),
                'returns': np.array([]),
                'advantages': np.array([]),
                'log_probs': np.array([]),
                'values': np.array([])
            }
    
    def compute_gradients(self, processed_buffer):
        """Compute gradients from the processed buffer"""
        # Unpack buffer
        obs_list = processed_buffer['obs']
        actions = torch.tensor(processed_buffer['actions'], dtype=torch.int64, device=self.device)
        returns = torch.tensor(processed_buffer['returns'], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(processed_buffer['advantages'], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(processed_buffer['log_probs'], dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert observations to tensors
        obs_boards = [torch.FloatTensor(o['board']).to(self.device) for o in obs_list]
        obs_masks = [torch.FloatTensor(o['action_mask']).to(self.device) for o in obs_list]
        
        # Stack tensors
        obs_dict = {
            'board': torch.stack(obs_boards),
            'action_mask': torch.stack(obs_masks)
        }
        
        # Switch to training mode
        self.model.train()
        
        # Zero gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.zero_()
        
        # Forward pass
        result = self.model.evaluate_actions(obs_dict, actions)
        values = result['values']
        log_probs = result['log_probs']
        entropy = result['entropy'].mean()
        
        # Calculate losses
        value_loss = F.mse_loss(values, returns)
        policy_loss = -torch.mean(advantages * log_probs)
        entropy_loss = -entropy
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        # Backward pass
        loss.backward()
        
        # Get gradients
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone().detach())
            else:
                gradients.append(None)
        
        # Switch back to eval mode
        self.model.eval()
        
        # Return gradients and metrics
        metrics = {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': loss.item()
        }
        
        return gradients, metrics
    
    def get_current_metrics(self):
        """Get current metrics for the worker"""
        metrics = {}
        
        # Add episode metrics if available
        if len(self.episode_metrics['episode_reward']) > 0:
            metrics['episode_reward'] = self.episode_metrics['episode_reward'][-1]
            metrics['episode_length'] = self.episode_metrics['episode_length'][-1]
        
        # Add game outcome for the last game
        if self.episode_metrics['games'] > 0:
            games = self.episode_metrics['games']
            white_wins = self.episode_metrics['white_wins']
            black_wins = self.episode_metrics['black_wins']
            draws = self.episode_metrics['draws']
            
            # Add outcome metrics for the most recent game
            recent_white_win = white_wins > 0 and white_wins == self.episode_metrics.get('last_white_wins', 0) + 1
            recent_black_win = black_wins > 0 and black_wins == self.episode_metrics.get('last_black_wins', 0) + 1
            recent_draw = draws > 0 and draws == self.episode_metrics.get('last_draws', 0) + 1
            
            metrics['white_win'] = recent_white_win
            metrics['black_win'] = recent_black_win
            metrics['draw'] = recent_draw
            
            # Update last counts
            self.episode_metrics['last_white_wins'] = white_wins
            self.episode_metrics['last_black_wins'] = black_wins
            self.episode_metrics['last_draws'] = draws
        
        return metrics


def create_chess_env():
    """Create a chess environment"""
    try:
        from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper
        
        # Create base environment
        env = ChessEnv()
        
        # Wrap with action mask
        env = ActionMaskWrapper(env)
        
        # Add a wrapper to ensure consistent observation format
        env = ObservationDictWrapper(env)
        
        return env
    except Exception as e:
        logger.error(f"Error creating chess environment: {e}", exc_info=True)
        raise


class ObservationDictWrapper(gym.Wrapper):
    """Ensure observations are in dictionary format with board and action_mask keys"""
    
    def __init__(self, env):
        super(ObservationDictWrapper, self).__init__(env)
        # Store original observation space
        self.orig_observation_space = env.observation_space
        
        # Check if already a Dict space
        if isinstance(env.observation_space, spaces.Dict):
            self.observation_space = env.observation_space
        else:
            # We need to wrap it in a Dict
            self.observation_space = spaces.Dict({
                'board': spaces.Box(0, 1, shape=(13, 8, 8), dtype=np.float32),
                'action_mask': spaces.Box(0, 1, shape=(20480,), dtype=np.float32)
            })
        
    def reset(self, **kwargs):
        """Reset environment and ensure observation is a proper dict"""
        try:
            result = self.env.reset(**kwargs)
            
            # Handle different return formats
            if isinstance(result, tuple):
                # Unpack the tuple - could be (obs, info) or other format
                if len(result) >= 2:
                    obs, info = result[0], result[1]
                else:
                    # Single item tuple
                    obs, info = result[0], {}
            else:
                # Direct observation
                obs, info = result, {}
            
            # Ensure observation is in the correct format
            if isinstance(obs, dict) and 'board' in obs and 'action_mask' in obs:
                # Already in the right format
                fixed_obs = obs
            else:
                # Need to convert
                if isinstance(obs, np.ndarray):
                    # Try to split into board and mask components
                    # Assuming first part is board, rest is mask
                    board_size = 13 * 8 * 8  # Board representation size
                    obs_copy = obs.copy()  # Create a copy
                    
                    if len(obs_copy.shape) == 1 and obs_copy.shape[0] > board_size:
                        # Split flat array into board and mask
                        fixed_obs = {
                            'board': np.array(obs_copy[:board_size]).reshape(13, 8, 8),
                            'action_mask': np.ones(20480, dtype=np.float32)  # Updated from 4672 to 20480
                        }
                    else:
                        # Just treat the whole thing as a board
                        fixed_obs = {
                            'board': np.array(obs_copy),
                            'action_mask': np.ones(20480, dtype=np.float32)  # Updated from 4672 to 20480
                        }
                else:
                    # Unknown format, create dummy observation
                    fixed_obs = {
                        'board': np.zeros((13, 8, 8), dtype=np.float32),
                        'action_mask': np.ones(20480, dtype=np.float32)  # Updated from 4672 to 20480
                    }
            
            # Ensure all required keys are present
            if 'board' not in fixed_obs:
                fixed_obs['board'] = np.zeros((13, 8, 8), dtype=np.float32)
            if 'action_mask' not in fixed_obs:
                fixed_obs['action_mask'] = np.ones(20480, dtype=np.float32)  # Updated from 4672 to 20480
            
            # Return in the appropriate format based on what was received
            if isinstance(result, tuple) and len(result) >= 2:
                return fixed_obs, info
            else:
                return fixed_obs
        except Exception as e:
            logger.error(f"ObservationDictWrapper.reset: Error: {e}", exc_info=True)
            # Return a safe fallback
            dummy_obs = {
                'board': np.zeros((13, 8, 8), dtype=np.float32),
                'action_mask': np.ones(20480, dtype=np.float32)  # Updated from 4672 to 20480
            }
            return dummy_obs, {}
    
    def step(self, action):
        """Take a step and ensure the observation is a proper dict"""
        try:
            step_result = self.env.step(action)
            
            # Handle different return formats
            if isinstance(step_result, tuple):
                if len(step_result) == 4:  # (obs, reward, done, info)
                    obs, reward, done, info = step_result
                    truncated = False
                elif len(step_result) == 5:  # (obs, reward, terminated, truncated, info)
                    obs, reward, done, truncated, info = step_result
                else:
                    # Unexpected format
                    logger.error(f"ObservationDictWrapper.step: Unexpected step result format with {len(step_result)} elements")
                    obs, reward, done, truncated, info = None, 0, True, False, {}
            else:
                # Unexpected format
                logger.error(f"ObservationDictWrapper.step: Unexpected step result type: {type(step_result)}")
                obs, reward, done, truncated, info = None, 0, True, False, {}
            
            # Ensure observation is in the correct format
            if isinstance(obs, dict) and 'board' in obs and 'action_mask' in obs:
                # Already in the right format
                fixed_obs = obs
            else:
                # Need to convert
                if isinstance(obs, np.ndarray):
                    # Try to split into board and mask components
                    # Assuming first part is board, rest is mask
                    board_size = 13 * 8 * 8  # Board representation size
                    obs_copy = obs.copy()  # Create a copy
                    
                    if len(obs_copy.shape) == 1 and obs_copy.shape[0] > board_size:
                        # Split flat array into board and mask
                        fixed_obs = {
                            'board': np.array(obs_copy[:board_size]).reshape(13, 8, 8),
                            'action_mask': np.ones(20480, dtype=np.float32)  # Updated from 4672 to 20480
                        }
                    else:
                        # Just treat the whole thing as a board
                        fixed_obs = {
                            'board': np.array(obs_copy),
                            'action_mask': np.ones(20480, dtype=np.float32)  # Updated from 4672 to 20480
                        }
                else:
                    # Unknown format, create dummy observation
                    fixed_obs = {
                        'board': np.zeros((13, 8, 8), dtype=np.float32),
                        'action_mask': np.ones(20480, dtype=np.float32)  # Updated from 4672 to 20480
                    }
            
            # Ensure all required keys are present
            if 'board' not in fixed_obs:
                fixed_obs['board'] = np.zeros((13, 8, 8), dtype=np.float32)
            if 'action_mask' not in fixed_obs:
                fixed_obs['action_mask'] = np.ones(20480, dtype=np.float32)  # Updated from 4672 to 20480
            
            # Return in the appropriate format based on what we received
            if len(step_result) == 5:  # Modern Gymnasium API
                return fixed_obs, reward, done, truncated, info
            else:  # Classic Gym API
                return fixed_obs, reward, done, info
        except Exception as e:
            logger.error(f"ObservationDictWrapper.step: Error: {e}", exc_info=True)
            # Return a safe fallback
            dummy_obs = {
                'board': np.zeros((13, 8, 8), dtype=np.float32),
                'action_mask': np.ones(20480, dtype=np.float32)  # Updated from 4672 to 20480
            }
            return dummy_obs, 0, True, {}


def train(args):
    """
    Main training function for Ray A3C.
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Set up directory for checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Determine device to use
    device = args.device
    if args.force_cpu:
        logger.info("Forcing CPU use as requested")
        device = 'cpu'

    logger.info(f"Starting Ray A3C in train mode on {device}")
    
    # Create parameter server
    param_server = ParameterServer.remote(device=device, checkpoint_path=args.checkpoint)
    
    # Create workers
    workers = []
    logger.info(f"Creating {args.num_workers} workers")
    
    for i in range(args.num_workers):
        # Set worker device
        if device == 'cuda' and torch.cuda.is_available():
            worker_device = f'cuda:{i % torch.cuda.device_count()}' if torch.cuda.device_count() > 1 else 'cuda:0'
        else:
            worker_device = device
        
        # Set up worker resource requirements
        worker_kwargs = {}
        if device == 'cuda' and torch.cuda.device_count() > 1:
            worker_kwargs['num_gpus'] = 0.5  # Each worker uses 0.5 GPU
        elif device == 'cuda':
            worker_kwargs['num_gpus'] = 0.25  # Share GPU
        elif device == 'mps':  # Apple Silicon GPU
            worker_kwargs['num_cpus'] = 1
        else:
            worker_kwargs['num_cpus'] = 1
        
        # If we're on the Lambda machine with high-end GPUs, we can use MCTS more aggressively
        use_mcts = i < args.mcts_workers
        
        # Create worker with appropriate resource specifications
        worker_options = {k: v for k, v in worker_kwargs.items() if v is not None}
        if worker_options:
            worker = RolloutWorker.options(**worker_options).remote(
                worker_id=i,
                env_creator=create_chess_env,
                param_server=param_server,
                device=worker_device,
                mcts_sims=args.mcts_sims if use_mcts else 0
            )
        else:
            worker = RolloutWorker.remote(
                worker_id=i,
                env_creator=create_chess_env,
                param_server=param_server,
                device=worker_device,
                mcts_sims=args.mcts_sims if use_mcts else 0
            )
        workers.append(worker)
    
    logger.info(f"Created {len(workers)} workers")
    
    # Training loop
    iteration = 0
    start_time = time.time()
    keep_training = True
    
    try:
        while keep_training:
            # Start collecting experience and computing gradients in parallel
            worker_tasks = [worker.collect_experience.remote(args.steps_per_update) for worker in workers]
            
            # Process results as they come in
            for _ in range(len(workers)):
                ready_ids, remaining_tasks = ray.wait(worker_tasks, num_returns=1)
                
                # Find the index of the completed task
                completed_task_id = ready_ids[0]
                
                # Calculate the worker index
                worker_index = None
                for i, task in enumerate(worker_tasks):
                    if task == completed_task_id:
                        worker_index = i
                        break
                
                # If we can't find the worker directly in the list, find by comparing string representations
                if worker_index is None:
                    task_id_str = str(completed_task_id)
                    for i, task in enumerate(worker_tasks):
                        if str(task) == task_id_str:
                            worker_index = i
                            break
                
                # Fallback: just use a valid worker index
                if worker_index is None:
                    worker_index = 0
                    logger.warning(f"Could not find worker for task {completed_task_id}, using fallback worker")
                
                # Get the worker reference
                worker_ref = workers[worker_index]
                
                # Update the worker tasks list with the remaining tasks
                worker_tasks = remaining_tasks
                
                # Get results from the completed task
                try:
                    processed_buffer, complete_episode, metrics, num_moves = ray.get(ready_ids[0])
                except Exception as e:
                    logger.error(f"Error getting results from worker: {str(e)}", exc_info=True)
                    # Create dummy results
                    processed_buffer = {
                        'obs': [],
                        'actions': np.array([]),
                        'returns': np.array([]),
                        'advantages': np.array([]),
                        'log_probs': np.array([]),
                        'values': np.array([])
                    }
                    complete_episode = True
                    metrics = {}
                    num_moves = 0
                
                # Skip if buffer is empty
                if not processed_buffer or 'obs' not in processed_buffer or len(processed_buffer['obs']) == 0:
                    logger.warning(f"Empty buffer received from worker")
                    continue
                
                # Compute gradients using the same worker that collected the experience
                try:
                    gradients, loss_metrics = ray.get(worker_ref.compute_gradients.remote(processed_buffer))
                except Exception as e:
                    logger.error(f"Error computing gradients: {str(e)}", exc_info=True)
                    # Skip this update
                    continue
                
                # Update metrics with loss metrics
                metrics.update(loss_metrics)
                
                # Apply gradients to parameter server and update move counter
                ray.get(param_server.apply_gradients.remote(gradients, metrics))
                ray.get(param_server.increment_total_moves.remote(num_moves))
            
            # Launch weight updates asynchronously but don't wait for them
            update_tasks = [worker.update_weights.remote() for worker in workers]
            
            # Save checkpoint periodically
            iteration += 1
            if iteration % args.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"model_iter_{iteration}.pt"
                ray.get(param_server.save_checkpoint.remote(str(checkpoint_path)))
            
            # Log metrics periodically
            if iteration % args.log_interval == 0:
                metrics = ray.get(param_server.get_metrics.remote())
                elapsed_time = time.time() - start_time
                
                # Calculate summary metrics
                summary = {
                    'iteration': iteration,
                    'elapsed_time': elapsed_time,
                    'total_moves': metrics.get('total_moves', 0),
                    'games_played': metrics.get('games', 0),
                    'white_win_rate': metrics.get('white_win_rate', 0),
                    'black_win_rate': metrics.get('black_win_rate', 0),
                    'draw_rate': metrics.get('draw_rate', 0),
                    'avg_reward': np.mean(metrics.get('episode_rewards', [0])[-100:]) if metrics.get('episode_rewards') else 0,
                    'avg_length': np.mean(metrics.get('episode_lengths', [0])[-100:]) if metrics.get('episode_lengths') else 0,
                    'value_loss': np.mean(metrics.get('value_losses', [0])[-100:]) if metrics.get('value_losses') else 0,
                    'policy_loss': np.mean(metrics.get('policy_losses', [0])[-100:]) if metrics.get('policy_losses') else 0
                }
                
                # Log summary
                logger.info(f"Iteration {iteration}: {summary}")
            
            # Stop if max iterations reached
            if args.max_iterations and iteration >= args.max_iterations:
                keep_training = False
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
    finally:
        # Save final checkpoint
        final_checkpoint_path = checkpoint_dir / "model_final.pt"
        ray.get(param_server.save_checkpoint.remote(str(final_checkpoint_path)))
        logger.info(f"Final checkpoint saved to {final_checkpoint_path}")
        
        # Don't shutdown Ray if we're in distributed mode (it was started externally)
        if not (hasattr(args, 'distributed') and args.distributed):
            ray.shutdown()
            logger.info("Ray shutdown complete")


def evaluate(args):
    """
    Evaluate a trained model by playing games.
    """
    # Set device
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Evaluating model on device: {device}")
    
    # Load model
    if not args.checkpoint:
        logger.error("No checkpoint provided for evaluation")
        return
    
    model = ChessCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from {args.checkpoint}")
    
    # Initialize MCTS if needed
    if args.mcts_sims > 0:
        model.init_mcts(num_simulations=args.mcts_sims)
        logger.info(f"MCTS initialized with {args.mcts_sims} simulations")
    
    # Create environment
    env = create_chess_env()
    
    # Run evaluation games
    results = {
        'white_wins': 0,
        'black_wins': 0,
        'draws': 0,
        'total_games': 0,
        'white_win_rate': 0,
        'black_win_rate': 0,
        'draw_rate': 0,
        'avg_game_length': 0
    }
    
    game_lengths = []
    
    for game_id in range(args.num_games):
        # Reset environment
        obs = env.reset()
        done = False
        step = 0
        
        # Display start of game
        logger.info(f"Starting game {game_id+1}/{args.num_games}")
        if args.render:
            print("\nNew game started:")
            print(env.board)
        
        # Play game until done
        while not done:
            # Get current player (1=white, 0=black)
            current_player = 1 if env.board.turn else 0
            
            # Select action
            obs_dict = {
                'board': torch.FloatTensor(obs['board']).unsqueeze(0).to(device),
                'action_mask': torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(device)
            }
            
            if args.mcts_sims > 0:
                # Use MCTS for action selection
                state = env.board.copy()
                action, _ = model.mcts.search(state)
            else:
                # Use model directly
                with torch.no_grad():
                    result = model(obs_dict, deterministic=True)
                    action = result['actions'][0].item()
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Render if needed
            if args.render and step % args.render_interval == 0:
                move = env.board.peek() if env.board.move_stack else None
                move_str = move.uci() if move else "None"
                player = "White" if current_player == 1 else "Black"
                print(f"\nStep {step}: {player} played {move_str}")
                print(env.board)
            
            # Update for next step
            obs = next_obs
            step += 1
            
            # Break if game is too long
            if step >= 1000:
                logger.warning(f"Game {game_id+1} reached 1000 steps, forcing draw")
                done = True
                break
        
        # Update results
        results['total_games'] += 1
        game_lengths.append(step)
        
        if info.get('white_won', False):
            results['white_wins'] += 1
            outcome = "White won"
        elif info.get('black_won', False):
            results['black_wins'] += 1
            outcome = "Black won"
        else:
            results['draws'] += 1
            outcome = "Draw"
        
        # Display end of game
        logger.info(f"Game {game_id+1} ended after {step} steps: {outcome}")
        if args.render:
            print(f"\nGame over: {outcome} after {step} moves")
            print(env.board)
    
    # Calculate summary statistics
    total_games = results['total_games']
    if total_games > 0:
        results['white_win_rate'] = results['white_wins'] / total_games
        results['black_win_rate'] = results['black_wins'] / total_games
        results['draw_rate'] = results['draws'] / total_games
        results['avg_game_length'] = sum(game_lengths) / total_games
    
    # Display summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Total games: {total_games}")
    logger.info(f"White wins: {results['white_wins']} ({results['white_win_rate']:.2%})")
    logger.info(f"Black wins: {results['black_wins']} ({results['black_win_rate']:.2%})")
    logger.info(f"Draws: {results['draws']} ({results['draw_rate']:.2%})")
    logger.info(f"Average game length: {results['avg_game_length']:.1f} moves")
    
    return results


def self_play(args):
    """
    Generate self-play data for supervised learning.
    """
    # Set device
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Generating self-play data on device: {device}")
    
    # Load model if specified
    model = ChessCNN().to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {args.checkpoint}")
    
    # Always use MCTS for self-play
    mcts_sims = args.mcts_sims if args.mcts_sims > 0 else 800
    model.init_mcts(num_simulations=mcts_sims)
    logger.info(f"MCTS initialized with {mcts_sims} simulations")
    
    # Create environment
    env = create_chess_env()
    
    # Set up directory for self-play data
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate games
    for game_id in range(args.num_games):
        # Reset environment
        obs = env.reset()
        done = False
        step = 0
        
        # Initialize game data
        game_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'board_fens': [],
            'result': None
        }
        
        # Play game until done
        while not done:
            # Get current player (1=white, 0=black)
            current_player = 1 if env.board.turn else 0
            
            # Record state
            game_data['states'].append({
                'board': obs['board'].tolist(),
                'action_mask': obs['action_mask'].tolist()
            })
            game_data['board_fens'].append(env.board.fen())
            
            # Select action using MCTS
            state = env.board.copy()
            action, value = model.mcts.search(state)
            
            # Record action
            game_data['actions'].append({
                'action': int(action),
                'value': float(value)
            })
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Record reward
            game_data['rewards'].append(float(reward))
            
            # Update for next step
            obs = next_obs
            step += 1
            
            # Break if game is too long
            if step >= 1000:
                logger.warning(f"Game {game_id+1} reached 1000 steps, forcing draw")
                done = True
                break
        
        # Record game result
        if info.get('white_won', False):
            game_data['result'] = 'white_win'
            outcome = "White won"
        elif info.get('black_won', False):
            game_data['result'] = 'black_win'
            outcome = "Black won"
        else:
            game_data['result'] = 'draw'
            outcome = "Draw"
        
        # Save game data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_path = data_dir / f"game_{game_id}_{timestamp}.json"
        with open(game_path, 'w') as f:
            json.dump(game_data, f)
        
        logger.info(f"Game {game_id+1}/{args.num_games} completed: {outcome} after {step} moves. Saved to {game_path}")
    
    logger.info(f"Generated {args.num_games} self-play games in {data_dir}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Ray A3C Chess Training')
    
    # General arguments
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'self_play'],
                        help='Mode: train, evaluate, or self_play')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (cpu, cuda, auto)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for loading')
    
    # Training arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for parallel training')
    parser.add_argument('--mcts_workers', type=int, default=1,
                        help='Number of workers that use MCTS (0 to disable)')
    parser.add_argument('--mcts_sims', type=int, default=20,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--steps_per_update', type=int, default=200,
                        help='Number of steps per gradient update')
    parser.add_argument('--max_iterations', type=int, default=None,
                        help='Maximum number of training iterations')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Iterations between checkpoints')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Iterations between logging')
    
    # Evaluation arguments
    parser.add_argument('--num_games', type=int, default=10,
                        help='Number of games to play during evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Render games during evaluation')
    parser.add_argument('--render_interval', type=int, default=1,
                        help='Render every N steps')
    
    # Self-play arguments
    parser.add_argument('--data_dir', type=str, default='self_play_data',
                        help='Directory to save self-play data')
    
    args = parser.parse_args()
    
    # Run in the selected mode
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'self_play':
        self_play(args)


if __name__ == '__main__':
    main() 