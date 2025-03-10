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
    n_envs = min(8, n_games // 2)  # Use at most 8 parallel environments
    
    # Create separate environments for testing as white and black
    white_envs = [make_env(i) for i in range(n_envs)]
    black_envs = [make_env(i + n_envs) for i in range(n_envs)]
    
    white_env = SubprocVecEnv(white_envs)
    black_env = SubprocVecEnv(black_envs)
    
    white_env = VecMonitor(white_env)
    black_env = VecMonitor(black_env)
    
    # Load the model
    try:
        # We'll load the model twice, once for each set of environments
        white_model = MCTSPPO.load(model_path, env=white_env, device=device)
        black_model = MCTSPPO.load(model_path, env=black_env, device=device)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Statistics
    white_stats = defaultdict(int)
    black_stats = defaultdict(int)
    
    # Test model playing as WHITE against random BLACK
    print("\n=== Testing model as WHITE against random BLACK ===")
    white_games_played = 0
    white_obs = white_env.reset()
    
    # Create a random agent for black
    def random_action(obs):
        actions = []
        for o in obs:
            action_mask = o['action_mask']
            legal_actions = np.where(action_mask > 0.5)[0]
            actions.append(np.random.choice(legal_actions))
        return np.array(actions)
    
    # Play games with model as white
    while white_games_played < n_games // 2:
        # White's turn (model)
        white_action, _ = white_model.predict(white_obs, deterministic=True)
        white_obs, _, white_done, white_info = white_env.step(white_action)
        
        # If black's turn and not done, make random move
        for env_idx, done in enumerate(white_done):
            if not done and white_info[env_idx].get('current_player') == 0:  # Black's turn
                # Get observation for this environment
                black_action = random_action([white_obs[env_idx]])[0]
                # Make black's move
                single_obs, _, single_done, single_info = white_env.step(
                    np.array([black_action if i == env_idx else 0 for i in range(n_envs)])
                )
                # Update observation
                white_obs[env_idx] = single_obs[env_idx]
                white_done[env_idx] = single_done[env_idx]
                white_info[env_idx] = single_info[env_idx]
        
        # Check for game completions
        for i, d in enumerate(white_done):
            if d and white_games_played < n_games // 2:
                white_games_played += 1
                
                # Record outcome (from model's perspective as WHITE)
                if white_info[i].get('white_won', False):
                    white_stats['wins'] += 1
                elif white_info[i].get('black_won', False):
                    white_stats['losses'] += 1
                else:
                    white_stats['draws'] += 1
                
                # Reset this environment
                single_obs = white_env.reset(indices=[i])
                white_obs[i] = single_obs[i]
                white_done[i] = False
                
                # Log progress
                if white_games_played % 10 == 0:
                    print(f"Completed {white_games_played}/{n_games//2} games as WHITE")
    
    # Test model playing as BLACK against random WHITE
    print("\n=== Testing model as BLACK against random WHITE ===")
    black_games_played = 0
    black_obs = black_env.reset()
    
    # Play games with model as black
    while black_games_played < n_games // 2:
        # First we need to make a random move as white
        for env_idx in range(n_envs):
            if not black_done[env_idx] if 'black_done' in locals() else True:
                if black_info[env_idx].get('current_player', 1) == 1:  # White's turn
                    # Get random action for white
                    white_action = random_action([black_obs[env_idx]])[0]
                    # Make white's move
                    single_obs, _, single_done, single_info = black_env.step(
                        np.array([white_action if i == env_idx else 0 for i in range(n_envs)])
                    )
                    # Update observation
                    black_obs[env_idx] = single_obs[env_idx]
                    if 'black_done' in locals():
                        black_done[env_idx] = single_done[env_idx]
                    black_info[env_idx] = single_info[env_idx]
        
        # Black's turn (model)
        black_action, _ = black_model.predict(black_obs, deterministic=True)
        black_obs, _, black_done, black_info = black_env.step(black_action)
        
        # Check for game completions
        for i, d in enumerate(black_done):
            if d and black_games_played < n_games // 2:
                black_games_played += 1
                
                # Record outcome (from model's perspective as BLACK)
                if black_info[i].get('black_won', False):
                    black_stats['wins'] += 1
                elif black_info[i].get('white_won', False):
                    black_stats['losses'] += 1
                else:
                    black_stats['draws'] += 1
                
                # Reset this environment
                single_obs = black_env.reset(indices=[i])
                black_obs[i] = single_obs[i]
                black_done[i] = False
                
                # Log progress
                if black_games_played % 10 == 0:
                    print(f"Completed {black_games_played}/{n_games//2} games as BLACK")
    
    # Calculate win rates
    white_win_rate = white_stats['wins'] / (n_games // 2) * 100
    white_draw_rate = white_stats['draws'] / (n_games // 2) * 100
    white_loss_rate = white_stats['losses'] / (n_games // 2) * 100
    
    black_win_rate = black_stats['wins'] / (n_games // 2) * 100
    black_draw_rate = black_stats['draws'] / (n_games // 2) * 100
    black_loss_rate = black_stats['losses'] / (n_games // 2) * 100
    
    overall_win_rate = (white_stats['wins'] + black_stats['wins']) / n_games * 100
    
    print("\n===== EVALUATION RESULTS =====")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Total games played: {n_games}")
    
    print("\nAs WHITE:")
    print(f"Wins: {white_stats['wins']} ({white_win_rate:.1f}%)")
    print(f"Draws: {white_stats['draws']} ({white_draw_rate:.1f}%)")
    print(f"Losses: {white_stats['losses']} ({white_loss_rate:.1f}%)")
    
    print("\nAs BLACK:")
    print(f"Wins: {black_stats['wins']} ({black_win_rate:.1f}%)")
    print(f"Draws: {black_stats['draws']} ({black_draw_rate:.1f}%)")
    print(f"Losses: {black_stats['losses']} ({black_loss_rate:.1f}%)")
    
    print(f"\nOverall win rate: {overall_win_rate:.1f}%")
    
    # Compare to random baseline (~2.5% win rate for each side)
    if overall_win_rate > 5.0:
        print("✅ Model is performing better than random (>5% total win rate)")
    else:
        print("❌ Model is not clearly outperforming random play (<=5% total win rate)")
    
    # Check for balance between white and black play
    win_rate_diff = abs(white_win_rate - black_win_rate)
    if win_rate_diff < 5.0:
        print("✅ Model is balanced between white and black play (<5% win rate difference)")
    else:
        print(f"⚠️ Model favors one side ({win_rate_diff:.1f}% win rate difference)")
        if white_win_rate > black_win_rate:
            print("   Model performs better as WHITE")
        else:
            print("   Model performs better as BLACK")
    
    return {'white': white_stats, 'black': black_stats}

if __name__ == "__main__":
    args = parse_args()
    evaluate_vs_random(args.checkpoint, args.n_games, args.device) 