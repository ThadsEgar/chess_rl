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

def make_env(rank, seed=0):
    def _init():
        env = ChessEnv()
        env = ActionMaskWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def get_latest_checkpoints(folder_path, num_checkpoints=2, pattern=r'.*\.zip'):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and re.match(pattern, f)]
    files.sort()  # Sort alphabetically; assumes this reflects save order
    latest_files = files[-num_checkpoints:]
    return [os.path.join(folder_path, f) for f in latest_files]

class OpponentPoolCallback(BaseCallback):
    def __init__(self, model, checkpoint_folder, update_interval, verbose=0):
        super().__init__(verbose)
        self.model = model
        self.checkpoint_folder = checkpoint_folder
        self.update_interval = update_interval
        self.current_opponent_paths = []  # Tracks the current set of model paths
        self.init_load = False

    def _on_step(self):
        if self.num_timesteps % self.update_interval == 0 or self.init_load is False:
            latest_paths = get_latest_checkpoints(self.checkpoint_folder)
            if latest_paths != self.current_opponent_paths:
                opponent_policies = []
                for path in latest_paths:
                    # Load the policy from the checkpoint
                    opponent_model = self.model.__class__.load(
                        path,
                        env=self.model.env,
                        device=self.model.device
                    )
                    opponent_policy = opponent_model.policy
                    opponent_policy.set_training_mode(False)  # Disable training mode
                    opponent_policies.append(opponent_policy)
                # Update the model's opponent pool
                self.model.opponent_policies = opponent_policies
                self.current_opponent_paths = latest_paths
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

def main():
    # Set up logging directory
    TENSORBOARD_LOG = 'data/logs/'
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    os.makedirs("data/models", exist_ok=True)

    NUM_ENVS = 2  # Match to your number of CPU cores
    
    envs = [make_env(i) for i in range(NUM_ENVS)]
    
    multiprocessing.set_start_method('spawn', force=True)
    
    env = SubprocVecEnv(envs)
    env = VecMonitor(env)
    save_freq = 10_000_000 // NUM_ENVS

    model = create_cnn_mcts_ppo(
        env=env,
        tensorboard_log=TENSORBOARD_LOG,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint=None
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="data/models",
        name_prefix="chess_model"
    )

    metrics_callback = ChessMetricsCallback()
    opponent_callback = OpponentPoolCallback(
        model=model,
        checkpoint_folder="data/models",
        update_interval=10_000_000,
        verbose=1
    )

    # Train the model with both callbacks
    total_timesteps = 1_000_000_000
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, metrics_callback, opponent_callback]
        )
    finally:
        # Make sure to close the environments!
        env.close()

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