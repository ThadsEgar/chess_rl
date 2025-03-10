from custom_gym.chess_gym import ActionMaskWrapper, ChessEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
import torch
import numpy as np
import os
from collections import defaultdict, deque
from src.cnn import create_cnn_mcts_ppo
from src.mcts import create_mcts_ppo
import multiprocessing
import re
import sys
import argparse

def make_env(rank, seed=0):
    def _init():
        env = ChessEnv()
        env = ActionMaskWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def get_latest_checkpoints(folder_path, num_checkpoints=10, pattern=r'.*\.zip'):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and re.match(pattern, f)]
    files.sort()  # Sort alphabetically
    latest_files = files[-num_checkpoints:]
    return [os.path.join(folder_path, f) for f in latest_files]

class OpponentPoolCallback(BaseCallback):
    def __init__(self, model, checkpoint_folder, update_interval, verbose=0, opponent_pool_prob=0.8):
        super().__init__(verbose)
        self.model = model
        self.checkpoint_folder = checkpoint_folder
        self.update_interval = update_interval
        self.current_opponent_paths = []  # Tracks the current set of model paths
        self.init_load = False
        self.opponent_pool_prob = opponent_pool_prob  # Probability of using opponent pool
        self.random_policy = None  # Will store reference to random policy if found

    def _on_step(self):
        # Preserve random policy if it exists
        if self.random_policy is None and hasattr(self.model, 'opponent_policies') and self.model.opponent_policies:
            # Check if any policy is a RandomPolicy
            for policy in self.model.opponent_policies:
                if policy.__class__.__name__ == 'RandomPolicy':
                    self.random_policy = policy
                    if self.verbose > 0:
                        print("Found and preserved random policy")
                    break
        
        if self.num_timesteps % self.update_interval == 0 or self.init_load is False:
            latest_paths = get_latest_checkpoints(self.checkpoint_folder)
            if latest_paths != self.current_opponent_paths:
                opponent_policies = []
                
                # Add random policy if we have one
                if self.random_policy is not None:
                    opponent_policies.append(self.random_policy)
                    if self.verbose > 0:
                        print("Added preserved random policy to opponent pool")
                
                for path in latest_paths:
                    try:
                        # Load the policy from the checkpoint
                        opponent_model = self.model.__class__.load(
                            path,
                            env=self.model.env,
                            device=self.model.device
                        )
                        
                        # Make sure the policy is initialized
                        if not hasattr(opponent_model, 'policy') or opponent_model.policy is None:
                            if self.verbose > 0:
                                print(f"Warning: Model loaded from {path} has no policy attribute. Skipping.")
                            continue
                            
                        opponent_policy = opponent_model.policy
                        opponent_policy.set_training_mode(False)  # Disable training mode
                        opponent_policies.append(opponent_policy)
                        
                        if self.verbose > 0:
                            print(f"Added opponent policy from {path}")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"Error loading model from {path}: {e}")
                        continue
                
                # Only update if we successfully loaded at least one policy
                if opponent_policies:
                    self.model.opponent_policies = opponent_policies
                    self.current_opponent_paths = latest_paths
                    
                    # Update opponent pool probability
                    self.model.opponent_pool_prob = self.opponent_pool_prob
                    
                    if self.verbose > 0:
                        print(f"Updated opponent pool with {len(opponent_policies)} policies")
                        print(f"Set opponent pool probability to {self.opponent_pool_prob}")
                elif not self.model.opponent_policies:
                    # If no policies were loaded and the model doesn't have any yet,
                    # use the current policy as a fallback
                    self.model.opponent_policies = [self.model.policy]
                    if self.verbose > 0:
                        print("Using current policy as opponent")
        
        return True  # Continue training

from collections import defaultdict, deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ChessMetricsCallback(BaseCallback):
    def __init__(self, verbose=0, log_freq=1_000):
        super().__init__(verbose)
        self.log_freq = log_freq  # How often to log metrics
        self.game_outcomes = defaultdict(int)
        self.move_counts = deque(maxlen=10_000)
        self.games_played = 0
        self.step_count = 0
        self.step_rewards = deque(maxlen=10_000)
    def _on_step(self) -> bool:
        self.step_count += 1

        rewards = self.locals['rewards']  # Array of rewards for each environment
        mean_step_reward = np.mean(rewards)
        self.step_rewards.append(mean_step_reward)

        # Existing game outcome tracking
        for info in self.locals['infos']:
            if info.get('is_terminal') or info.get('is_truncated'):
                self.games_played += 1
                if info.get('white_won'):
                    self.game_outcomes['white_wins'] += 1
                elif info.get('black_won'):
                    self.game_outcomes['black_wins'] += 1
                else:
                    self.game_outcomes['draws'] += 1
                self.move_counts.append(info.get('move_count', 0))

        # Log metrics only every log_freq steps
        if self.step_count % self.log_freq != 0:
            return True

        # Log existing game metrics
        if self.games_played > 0:
            white_win_rate = self.game_outcomes['white_wins'] / self.games_played
            black_win_rate = self.game_outcomes['black_wins'] / self.games_played
            draw_rate = self.game_outcomes['draws'] / self.games_played
            avg_moves = np.mean(list(self.move_counts)) if self.move_counts else 0.0

            self.logger.record('win_rates/white_win_rate', white_win_rate)
            self.logger.record('win_rates/black_win_rate', black_win_rate)
            self.logger.record('win_rates/draw_rate', draw_rate)
            self.logger.record('game_stats/average_moves_per_game', avg_moves)
            self.logger.record('game_outcomes/white_wins', self.game_outcomes['white_wins'])
            self.logger.record('game_outcomes/black_wins', self.game_outcomes['black_wins'])
            self.logger.record('game_outcomes/draws', self.game_outcomes['draws'])
            self.logger.record('game_stats/legal_moves_available', info.get('legal_moves_count', 0))
            self.logger.record('game_stats/position_repetitions', info.get('position_repetition_count', 0))

        # Log the mean step reward over the logging period
        if self.step_rewards:
            mean_step_reward_over_period = np.mean(self.step_rewards)
            self.logger.record('rollout/mean_step_reward', mean_step_reward_over_period)
            self.step_rewards = []  # Reset for the next logging period

        return True

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a chess reinforcement learning model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint to continue training from')
    parser.add_argument('--n_envs', type=int, default=48,
                        help='Number of parallel environments to use')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu). Default: auto-detect')
    parser.add_argument('--total_timesteps', type=int, default=int(1e8),
                        help='Total number of timesteps to train for')
    parser.add_argument('--save_freq', type=int, default=50000,
                        help='Frequency (in timesteps) to save model checkpoints')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Initial learning rate for the optimizer (default: 3e-5)')
    parser.add_argument('--n_epochs', type=int, default=4,
                        help='Number of epochs when optimizing the surrogate loss (default: 4)')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for training (default: 4096 for GPU efficiency)')
    parser.add_argument('--clip_range', type=float, default=0.2,
                        help='Clipping parameter for PPO (default: 0.2 for more conservative updates)')
    parser.add_argument('--max_grad_norm', type=float, default=1,
                        help='Maximum norm for gradients (default: 1 for more stable updates)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging directory
    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Create vectorized environment
    n_envs = args.n_envs
    print(f"Using {n_envs} parallel environments")
    
    # Set up multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Create environment functions
    envs = [make_env(i) for i in range(n_envs)]
    
    # Create vectorized environment
    env = SubprocVecEnv(envs)
    env = VecMonitor(env)
    
    # Get checkpoint path
    checkpoint = args.checkpoint
    
    # If no checkpoint specified via args but provided as positional argument (for backward compatibility)
    if checkpoint is None and len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        checkpoint = sys.argv[1]
        
    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        # Check if the checkpoint exists
        if not os.path.exists(checkpoint):
            # Try adding .zip extension if not present
            if not checkpoint.endswith('.zip'):
                checkpoint_with_ext = f"{checkpoint}.zip"
                if os.path.exists(checkpoint_with_ext):
                    checkpoint = checkpoint_with_ext
                    print(f"Using checkpoint with .zip extension: {checkpoint}")
                else:
                    # Try looking in data/models directory
                    models_dir_path = os.path.join("data", "models", checkpoint)
                    if os.path.exists(models_dir_path):
                        checkpoint = models_dir_path
                        print(f"Found checkpoint in data/models: {checkpoint}")
                    else:
                        models_dir_path_with_ext = f"{models_dir_path}.zip"
                        if os.path.exists(models_dir_path_with_ext):
                            checkpoint = models_dir_path_with_ext
                            print(f"Found checkpoint in data/models with .zip extension: {checkpoint}")
                        else:
                            print(f"Warning: Checkpoint {checkpoint} not found. Checking for available checkpoints...")
                            # List available checkpoints
                            available_checkpoints = []
                            for file in os.listdir("data/models"):
                                if file.endswith(".zip"):
                                    available_checkpoints.append(os.path.join("data", "models", file))
                            if available_checkpoints:
                                print("Available checkpoints:")
                                for cp in available_checkpoints:
                                    print(f"  - {cp}")
                                # Use the latest checkpoint
                                latest_checkpoint = max(available_checkpoints, key=os.path.getmtime)
                                print(f"Using latest checkpoint: {latest_checkpoint}")
                                checkpoint = latest_checkpoint
                            else:
                                print("No checkpoints found. Starting training from scratch.")
                                checkpoint = None
    
    # Use CNN-based policy
    model = create_cnn_mcts_ppo(
        env=env,
        tensorboard_log="data/logs",
        device=device,
        checkpoint=checkpoint,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        clip_range=args.clip_range,
        max_grad_norm=args.max_grad_norm
    )
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path="data/models/",
        name_prefix="chess_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Add metrics callback
    metrics_callback = ChessMetricsCallback(verbose=1, log_freq=10000)
    
    # Add opponent pool callback with higher probability of using opponent pool
    opponent_callback = OpponentPoolCallback(
        model=model,
        checkpoint_folder="data/models",
        update_interval=1000000,
        verbose=1,
        opponent_pool_prob=0.8
    )
    
    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, metrics_callback, opponent_callback],
        progress_bar=True
    )

    model.save("data/models/chess_model_complete")

    # Evaluation
    print("\nStarting evaluation...")
    eval_env = SubprocVecEnv([make_env(i) for i in range(4)])  # Use fewer envs for evaluation
    eval_env = VecMonitor(eval_env)
    
    obs = eval_env.reset()
    eval_games = 0
    eval_stats = defaultdict(int)

    try:
        while eval_games < 100:  # Evaluate 100 games
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            for env_info in info:
                if env_info.get('is_terminal', False) or env_info.get('is_truncated', False):
                    eval_games += 1
                    if env_info.get('white_won', False):
                        eval_stats['white_wins'] += 1
                    elif env_info.get('black_won', False):
                        eval_stats['black_wins'] += 1
                    else:
                        eval_stats['draws'] += 1
            
            if done.any():
                obs = eval_env.reset()
    finally:
        eval_env.close()

    print("\nEvaluation Results:")
    print(f"Total games played: {eval_games}")
    print(f"White wins: {eval_stats['white_wins']} ({eval_stats['white_wins']/eval_games*100:.1f}%)")
    print(f"Black wins: {eval_stats['black_wins']} ({eval_stats['black_wins']/eval_games*100:.1f}%)")
    print(f"Draws: {eval_stats['draws']} ({eval_stats['draws']/eval_games*100:.1f}%)")

    print("\nTraining complete and model saved.")

if __name__ == '__main__':
    main()