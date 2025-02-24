from custom_gym.chess_gym import ChessEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
import torch
import numpy as np
import os
from collections import defaultdict, deque
from src.mcts import create_mcts_ppo
import multiprocessing

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = ChessEnv()
        env.reset(seed=seed + rank)
        return env
    return _init

class ChessMetricsCallback(BaseCallback):
    def __init__(self, verbose=0, log_freq=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.game_outcomes = defaultdict(int)
        self.move_counts = deque(maxlen=100)
        self.games_played = 0
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Count games every step
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

        # Log only every log_freq steps
        if self.step_count % self.log_freq != 0:
            return True
        
        #print(f"Step {self.step_count}", flush=True)
        #for info in self.locals['infos']:
            #print(f"Info: {info}", flush=True)
        
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
        
        return True

def main():
    # Set up logging directory
    TENSORBOARD_LOG = 'data/logs/'
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    os.makedirs("data/models", exist_ok=True)

    # Number of parallel environments
    NUM_ENVS = 24  # Match to your number of CPU cores
    
    # Create environments with proper multiprocessing start method
    envs = [make_env(i) for i in range(NUM_ENVS)]
    
    # Set the start method to 'spawn' (more stable than 'fork' for PyTorch)
    multiprocessing.set_start_method('spawn', force=True)
    
    env = SubprocVecEnv(envs)
    env = VecMonitor(env)
    save_freq = 10_000_000 // NUM_ENVS

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="data/models",
        name_prefix="chess_model"
    )

    metrics_callback = ChessMetricsCallback()

    # Create the MCTS-PPO model
    model = create_mcts_ppo(
        env=env,
        tensorboard_log=TENSORBOARD_LOG,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint='data/models/chess_model_checkpoint1'
    )

    # Train the model with both callbacks
    total_timesteps = 500_000_000
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, metrics_callback]
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