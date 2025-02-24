from time import sleep
import os
import sys
import chess
import numpy as np
from stable_baselines3 import PPO
from custom_gym.chess_gym import ChessEnv

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def human_move(env):
    """Prompt human for a move in UCI format."""
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
    """Print the board and info in place using ANSI codes."""
    sys.stdout.write('\033[H')
    sys.stdout.flush()
    
    env.render()
    
    current_player = 'White' if env.state.current_player() == 0 else 'Black'
    if player_mode == 'human_vs_ai':
        turn_info = f"Turn: {current_player} ({'Human' if current_player[0].lower() == human_color else 'Model'})"
    else:  # ai_vs_ai
        turn_info = f"Turn: {current_player} (AI)"
    
    print(turn_info)
    if player_mode == 'ai_vs_ai':
        if white_action is not None and current_player == 'Black':  # White just moved
            print(f"White (AI) plays: {white_action}")
        elif black_action is not None and current_player == 'White':  # Black just moved
            print(f"Black (AI) plays: {black_action}")
        else:
            print()  # Empty line if no move yet
    elif player_mode == 'human_vs_ai' and white_action is not None and current_player[0].lower() != human_color:
        print(f"Model plays: {white_action}")
    else:
        print()  # Empty line for alignment
    
    print(f"Reward: {reward}, Move Count: {info.get('move_count', 'N/A')}")
    sys.stdout.write('\n')
    sys.stdout.flush()

def main():
    env = ChessEnv()
    obs, info = env.reset()
    
    clear_screen()
    mode = input("Choose game mode (1: Human vs AI, 2: AI vs AI): ").strip()
    if mode not in ['1', '2']:
        print("Invalid choice, defaulting to Human vs AI.")
        mode = '1'
    
    player_mode = 'human_vs_ai' if mode == '1' else 'ai_vs_ai'
    human_color = None
    if player_mode == 'human_vs_ai':
        human_color = input("Choose your color (w for white, b for black): ").strip().lower()
        if human_color not in ['w', 'b']:
            print("Invalid choice, defaulting to white.")
            human_color = 'w'
    
    model = PPO.load("data/models/chess_model_54999120_steps")
    model.policy.set_training_mode(False)  # Ensure testing mode
    
    done = False
    reward = 0
    white_action = None
    black_action = None
    
    while not done:
        print_board_and_info(env, reward, info, player_mode, human_color, white_action, black_action)
        
        current_player = env.state.current_player()
        current_player_str = 'w' if current_player == 0 else 'b'
        
        if player_mode == 'human_vs_ai' and current_player_str == human_color:
            action = human_move(env)
            if current_player == 0:
                white_action = action
            else:
                black_action = action
        else:
            action, _ = model.predict(obs, state=env.state, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            if current_player == 0:
                white_action = action
                black_action = None  # Reset for next turn
            else:
                black_action = action
                white_action = None
            print_board_and_info(env, reward, info, player_mode, human_color, white_action, black_action)
            sleep(0.5)
        
        obs, reward, done, truncated, info = env.step(action)
    
    print_board_and_info(env, reward, info, player_mode, human_color, white_action, black_action)
    print("\nGame over!")
    sleep(2)

if __name__ == "__main__":
    main()