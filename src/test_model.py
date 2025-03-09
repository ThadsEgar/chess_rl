from time import sleep
import os
import sys
import numpy as np
import argparse
from stable_baselines3 import PPO
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper
import chess
import torch

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def human_move(env):
    while True:
        move_str = input("Enter your move (UCI format, e.g., e2e4): ").strip()
        try:
            move = chess.Move.from_uci(move_str)
            board = env.state.board if isinstance(env.state.board, chess.Board) else env.state.board()
            legal_moves = list(board.legal_moves)
            if move not in legal_moves:
                print("That move is not legal. Try again.")
                continue
            action = env.move_to_action(move)
            if action not in env.state.legal_actions():
                print("Computed action is illegal. Try again.")
                continue
            return action
        except Exception as e:
            print(f"Error: {e}. Please enter a valid move.")

def print_board_and_info(env, reward, info, player_mode, human_color=None, white_action=None, black_action=None):
    # Access the unwrapped environment for rendering and state information
    unwrapped_env = env.env if hasattr(env, 'env') else env
    
    unwrapped_env.render()
    
    current_player = 'White' if unwrapped_env.state.current_player() == 0 else 'Black'
    if player_mode == 'human_vs_ai':
        turn_info = f"Turn: {current_player} ({'Human' if current_player[0].lower() == human_color else 'Model'})"
    else:  # ai_vs_ai
        turn_info = f"Turn: {current_player} (AI)"
    
    print(turn_info)
    if player_mode == 'ai_vs_ai':
        if white_action is not None and current_player == 'White':
            print(f"White (AI) plays: {white_action}")
        elif black_action is not None and current_player == 'Black':
            print(f"Black (AI) plays: {black_action}")
        else:
            print()  # Empty line if no move yet
    elif player_mode == 'human_vs_ai' and current_player[0].lower() != human_color:
        action_to_show = white_action if current_player == 'White' else black_action
        if action_to_show is not None:
            print(f"Model plays: {action_to_show}")
        else:
            print()  # Empty line if no move yet
    else:
        print()  # Empty line for alignment
    
    print(f"Reward: {reward}, Move Count: {info.get('move_count', 'N/A')}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test a trained chess model')
    parser.add_argument('--model', type=str, default="data/models/chess_model_59999904_steps",
                        help='Path to the trained model file')
    parser.add_argument('--mode', type=str, choices=['human_vs_ai', 'ai_vs_ai'], default='human_vs_ai',
                        help='Game mode: human_vs_ai or ai_vs_ai')
    parser.add_argument('--color', type=str, choices=['w', 'b'], default='w',
                        help='Human player color (w for white, b for black)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between AI moves in seconds')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create environment with action mask wrapper
    base_env = ChessEnv()
    env = ActionMaskWrapper(base_env)
    obs, info = env.reset()
    
    # If no command line arguments, ask interactively
    if len(sys.argv) == 1:
        mode = input("Choose game mode (1: Human vs AI, 2: AI vs AI): ").strip()
        player_mode = 'human_vs_ai' if mode == '1' else 'ai_vs_ai'
        
        human_color = None
        if player_mode == 'human_vs_ai':
            human_color = input("Choose your color (w for white, b for black): ").strip().lower()
            if human_color not in ['w', 'b']:
                print("Invalid choice, defaulting to white.")
                human_color = 'w'
                
        model_path = input("Enter path to model (or press Enter for default): ").strip()
        if not model_path:
            model_path = "data/models/chess_model_59999904_steps"
    else:
        # Use command line arguments
        player_mode = args.mode
        human_color = args.color if player_mode == 'human_vs_ai' else None
        model_path = args.model
    
    print(f"Loading model from: {model_path}")
    try:
        model = PPO.load(model_path)
        model.policy.set_training_mode(False)  # Ensure testing mode
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Available models:")
        for file in os.listdir("data/models"):
            if file.endswith(".zip"):
                print(f"  - data/models/{file}")
        return
    
    done = False
    reward = 0
    white_action = None
    black_action = None
    
    while not done:
        print_board_and_info(env, reward, info, player_mode, human_color, white_action, black_action)
        
        # Get current player from the unwrapped environment
        current_player = base_env.state.current_player()
        current_player_str = 'w' if current_player == 0 else 'b'
        
        if player_mode == 'human_vs_ai' and current_player_str == human_color:
            action = human_move(base_env)  # Use unwrapped env for human moves
            if current_player == 0:
                white_action = action
            else:
                black_action = action
        else:
            # Model prediction
            try:
                # The model expects a specific observation format
                # Let's try to handle both dictionary and tensor formats
                if isinstance(model.policy, type) and hasattr(model.policy, 'extract_features'):
                    # If using a custom policy with extract_features method
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # Try to convert the observation to the format the model expects
                    try:
                        # If the model expects a flattened tensor
                        if isinstance(obs, dict):
                            # Convert dictionary observation to tensor
                            board = obs['board']
                            action_mask = obs['action_mask']
                            flat_obs = np.concatenate([board, action_mask])
                            action, _ = model.predict(flat_obs, deterministic=True)
                        else:
                            # Already in the right format
                            action, _ = model.predict(obs, deterministic=True)
                    except Exception as inner_e:
                        print(f"Error converting observation: {inner_e}")
                        # Try one more approach - use the policy directly
                        if hasattr(model, 'policy') and hasattr(model.policy, 'forward'):
                            # Convert to tensor
                            if isinstance(obs, dict):
                                board_tensor = torch.as_tensor(obs['board']).unsqueeze(0).to(model.device)
                                mask_tensor = torch.as_tensor(obs['action_mask']).unsqueeze(0).to(model.device)
                                obs_dict = {'board': board_tensor, 'action_mask': mask_tensor}
                                with torch.no_grad():
                                    actions, _, _ = model.policy(obs_dict)
                                action = actions.cpu().numpy()[0]
                            else:
                                obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device)
                                with torch.no_grad():
                                    actions, _, _ = model.policy(obs_tensor)
                                action = actions.cpu().numpy()[0]
                        else:
                            raise inner_e
                
                if isinstance(action, np.ndarray):
                    action = int(action.item())
                
                if current_player == 0:
                    white_action = action
                    black_action = None  # Reset for next turn
                else:
                    black_action = action
                    white_action = None
                
                print_board_and_info(env, reward, info, player_mode, human_color, white_action, black_action)
                sleep(args.delay if len(sys.argv) > 1 else 0.5)
            except Exception as e:
                print(f"Error during model prediction: {e}")
                # Fallback to random legal move
                legal_actions = base_env.state.legal_actions()
                action = np.random.choice(legal_actions)
                print(f"Falling back to random move: {action}")
                if current_player == 0:
                    white_action = action
                else:
                    black_action = action
        
        obs, reward, done, truncated, info = env.step(action)
    
    print_board_and_info(env, reward, info, player_mode, human_color, white_action, black_action)
    print("\nGame over!")
    if info.get('white_won', False):
        print("White won!")
    elif info.get('black_won', False):
        print("Black won!")
    else:
        print("Draw!")
    sleep(2)

if __name__ == "__main__":
    main()