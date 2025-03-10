import numpy as np
import os
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from src.train import make_env
from src.cnn import MCTSPPO, CNNMCTSActorCriticPolicy
import argparse
from collections import defaultdict
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--n_games', type=int, default=100,
                        help='Number of games to evaluate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation')
    return parser.parse_args()

def evaluate_vs_random(model_path, n_games=100, device='cuda'):
    """Evaluate a model against random play to verify learning."""
    print(f"Evaluating model {model_path} against random play...")
    
    # Create environments for evaluation
    n_envs = min(8, n_games)  # Use at most 8 parallel environments
    envs = [make_env(i) for i in range(n_envs)]
    env = SubprocVecEnv(envs)
    env = VecMonitor(env)
    
    # Load the model
    try:
        model = MCTSPPO.load(model_path, env=env, device=device)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Statistics
    stats = defaultdict(int)
    games_played = 0
    
    # Run evaluation
    obs = env.reset()
    while games_played < n_games:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, _, done, info = env.step(action)
        
        # Check for game completions
        for i, d in enumerate(done):
            if d and games_played < n_games:
                games_played += 1
                
                # Record outcome
                if info[i].get('white_won', False):
                    stats['white_wins'] += 1
                elif info[i].get('black_won', False):
                    stats['black_wins'] += 1
                else:
                    stats['draws'] += 1
                
                # Log progress
                if games_played % 10 == 0:
                    print(f"Completed {games_played}/{n_games} games")
    
    # Calculate win rates
    white_win_rate = stats['white_wins'] / n_games * 100
    black_win_rate = stats['black_wins'] / n_games * 100
    draw_rate = stats['draws'] / n_games * 100
    
    print("\n===== EVALUATION RESULTS =====")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Games played: {n_games}")
    print(f"White wins: {stats['white_wins']} ({white_win_rate:.1f}%)")
    print(f"Black wins: {stats['black_wins']} ({black_win_rate:.1f}%)")
    print(f"Draws: {stats['draws']} ({draw_rate:.1f}%)")
    print(f"Total win rate: {(white_win_rate + black_win_rate):.1f}%")
    
    # Compare to random baseline (~2.5% win rate for each side)
    if white_win_rate + black_win_rate > 5.0:
        print("✅ Model is performing better than random (>5% total win rate)")
    else:
        print("❌ Model is not clearly outperforming random play (<=5% total win rate)")
    
    return stats

if __name__ == "__main__":
    args = parse_args()
    evaluate_vs_random(args.checkpoint, args.n_games, args.device) 