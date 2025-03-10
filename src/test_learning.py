import numpy as np
import os
import argparse
from collections import defaultdict
import torch
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper
from src.chess_model import ChessCNN

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained chess model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--n_games', type=int, default=100, help='Number of games to play against random')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run evaluation on (cuda or cpu)')
    return parser.parse_args()

def evaluate_vs_random(model_path, n_games=100, device='cuda'):
    """
    Evaluate a trained model against a random agent.
    """
    # Create environment
    env = ChessEnv()
    env = ActionMaskWrapper(env)
    
    # Load model
    model = ChessCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    
    # Initialize MCTS if using it for evaluation
    use_mcts = True
    if use_mcts:
        model.init_mcts(num_simulations=100)
        print("Using MCTS for evaluation")
    
    # Tracking statistics
    stats = {
        'white_wins': 0,
        'black_wins': 0,
        'draws': 0,
        'game_lengths': []
    }
    
    for game in range(n_games):
        # Reset environment
        obs = env.reset()
        done = False
        step = 0
        
        # Choose who plays as model (random)
        model_plays_white = np.random.choice([True, False])
        current_player = 1  # White starts
        
        while not done:
            # Determine if it's the model's turn
            model_turn = (model_plays_white and current_player == 1) or (not model_plays_white and current_player == 0)
            
            if model_turn:
                # Model's turn
                obs_dict = {
                    'board': torch.FloatTensor(obs['board']).unsqueeze(0).to(device),
                    'action_mask': torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(device)
                }
                
                if use_mcts:
                    # Use MCTS for action selection
                    state = env.board.copy()
                    action, _ = model.mcts.search(state)
                else:
                    # Use model directly
                    with torch.no_grad():
                        result = model(obs_dict, deterministic=True)
                        action = result['actions'][0].item()
            else:
                # Random agent's turn
                action = random_action(obs)
            
            # Take action
            obs, reward, done, info = env.step(action)
            step += 1
            
            # Update current player
            current_player = 1 - current_player  # Switch between 0 and 1
            
            # Break if game is too long
            if step >= 200:
                info['draw'] = True
                done = True
            
        # Record game result
        stats['game_lengths'].append(step)
        if info.get('white_won', False):
            stats['white_wins'] += 1
            result = "White won"
            model_won = model_plays_white
        elif info.get('black_won', False):
            stats['black_wins'] += 1
            result = "Black won"
            model_won = not model_plays_white
        else:
            stats['draws'] += 1
            result = "Draw"
            model_won = None
        
        print(f"Game {game+1}/{n_games}: {result} after {step} steps. Model played as {'White' if model_plays_white else 'Black'}" +
              (f" and {'won' if model_won else 'lost'}" if model_won is not None else ""))
    
    # Calculate statistics
    print("\nEvaluation Results:")
    print(f"Total games: {n_games}")
    
    # Calculate model win rate properly
    model_wins = 0
    model_games = 0
    for i in range(n_games):
        if np.random.choice([True, False]):  # Model plays white in this simulation
            if stats['white_wins'] > 0:
                model_wins += 1
                stats['white_wins'] -= 1
        else:  # Model plays black
            if stats['black_wins'] > 0:
                model_wins += 1
                stats['black_wins'] -= 1
        model_games += 1
    
    print(f"Model win rate: {model_wins / model_games:.2%}")
    print(f"White wins: {stats['white_wins']} ({stats['white_wins'] / n_games:.2%})")
    print(f"Black wins: {stats['black_wins']} ({stats['black_wins'] / n_games:.2%})")
    print(f"Draws: {stats['draws']} ({stats['draws'] / n_games:.2%})")
    print(f"Average game length: {sum(stats['game_lengths']) / len(stats['game_lengths']):.1f} moves")
    
    return stats

def random_action(obs):
    """Select a random legal action."""
    action_mask = obs['action_mask']
    legal_actions = np.where(action_mask == 1)[0]
    if len(legal_actions) > 0:
        return np.random.choice(legal_actions)
    return 0  # Fallback (shouldn't happen)

if __name__ == "__main__":
    args = parse_args()
    evaluate_vs_random(args.model_path, args.n_games, args.device) 