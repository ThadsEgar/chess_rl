from time import sleep
import os
import sys
import numpy as np
import argparse
import torch
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper
import chess

from src.chess_model import ChessCNN

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def human_move(env):
    while True:
        move_str = input("Enter your move (UCI format, e.g., e2e4): ").strip()
        try:
            move = chess.Move.from_uci(move_str)
            board = env.state.board if isinstance(env.state.board, chess.Board) else env.state.board()
            legal_moves = list(board.legal_moves)
            
            if move in legal_moves:
                action = env.move_to_action(move)
                return action
            else:
                print("Illegal move. Please try again.")
        except Exception as e:
            print(f"Invalid move format. Use UCI format (e.g., e2e4). Error: {e}")

def print_board_and_info(env, reward, info, player_mode, human_color=None, white_action=None, black_action=None):
    clear_screen()
    
    # Print the board
    board_str = str(env.board)
    
    # Determine which player is to move
    current_player = "White" if env.board.turn else "Black"
    
    # Print header with information
    print("\n===== CHESS RL =====")
    print(f"Mode: {player_mode}")
    if human_color is not None:
        print(f"Human playing as: {'White' if human_color == 'white' else 'Black'}")
    print(f"Turn: {current_player} to move")
    
    # Format the last move information
    last_move_info = ""
    if white_action is not None:
        try:
            white_move = env.action_to_move(white_action)
            last_move_info += f"\nWhite's last move: {white_move.uci()}"
        except:
            last_move_info += f"\nWhite's last action: {white_action}"
    
    if black_action is not None:
        try:
            black_move = env.action_to_move(black_action)
            last_move_info += f"\nBlack's last move: {black_move.uci()}"
        except:
            last_move_info += f"\nBlack's last action: {black_action}"
    
    print(last_move_info)
    
    # Print reward if applicable
    if reward is not None:
        print(f"Reward: {reward}")
    
    # Print game outcome if available
    if info.get('white_won', False):
        print("\n### White won! ###")
    elif info.get('black_won', False):
        print("\n### Black won! ###")
    elif info.get('draw', False):
        print("\n### Draw! ###")
    
    # Print the chess board
    print("\n" + board_str)
    
    # Print legal moves for better user experience
    legal_moves = list(env.board.legal_moves)
    print("\nLegal moves:", " ".join([move.uci() for move in legal_moves]))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test a trained chess model")
    parser.add_argument("--model", type=str, help="Path to the model checkpoint")
    parser.add_argument("--mode", choices=["ai_vs_ai", "human_vs_ai", "human_vs_human"], default="human_vs_ai", 
                       help="Game mode (ai_vs_ai, human_vs_ai, human_vs_human)")
    parser.add_argument("--color", choices=["white", "black"], default="white",
                       help="Human player color when playing against AI")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between AI moves in seconds")
    parser.add_argument("--mcts_sims", type=int, default=100,
                       help="Number of MCTS simulations for AI moves")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda or cpu)")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create the environment
    env = ChessEnv()
    env = ActionMaskWrapper(env)
    obs = env.reset()
    
    # Load the model if needed for AI players
    model = None
    if args.mode != "human_vs_human" and args.model:
        # Load the PyTorch model
        model = ChessCNN().to(args.device)
        checkpoint = torch.load(args.model, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Initialize MCTS
        if args.mcts_sims > 0:
            model.init_mcts(num_simulations=args.mcts_sims)
        
        print(f"Model loaded from {args.model}")
    
    # Game loop
    done = False
    reward = 0
    info = {}
    white_action = None
    black_action = None
    
    # Print initial board
    print_board_and_info(env, reward, info, args.mode, 
                        args.color if args.mode == "human_vs_ai" else None,
                        white_action, black_action)
    
    while not done:
        # Determine current player (White=1, Black=0)
        current_player = 1 if env.board.turn else 0
        
        # Determine who makes the move based on the game mode
        if args.mode == "human_vs_human":
            # Both players are human
            action = human_move(env)
        elif args.mode == "ai_vs_ai":
            # Both players are AI
            obs_dict = {
                'board': torch.FloatTensor(obs['board']).unsqueeze(0).to(args.device),
                'action_mask': torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(args.device)
            }
            
            if args.mcts_sims > 0:
                # Use MCTS
                state = env.board.copy()
                action, _ = model.mcts.search(state)
            else:
                # Use model directly
                with torch.no_grad():
                    result = model(obs_dict, deterministic=True)
                    action = result['actions'][0].item()
            
            # Add a delay to see AI moves
            sleep(args.delay)
        else:  # human_vs_ai
            human_plays_white = args.color == "white"
            
            if (current_player == 1 and human_plays_white) or (current_player == 0 and not human_plays_white):
                # Human's turn
                action = human_move(env)
            else:
                # AI's turn
                obs_dict = {
                    'board': torch.FloatTensor(obs['board']).unsqueeze(0).to(args.device),
                    'action_mask': torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(args.device)
                }
                
                if args.mcts_sims > 0:
                    # Use MCTS
                    state = env.board.copy()
                    action, _ = model.mcts.search(state)
                else:
                    # Use model directly
                    with torch.no_grad():
                        result = model(obs_dict, deterministic=True)
                        action = result['actions'][0].item()
                
                # Add a delay to see AI moves
                sleep(args.delay)
        
        # Take the action
        obs, reward, done, info = env.step(action)
        
        # Record the last action
        if current_player == 1:  # White
            white_action = action
        else:  # Black
            black_action = action
        
        # Print the updated board
        print_board_and_info(env, reward, info, args.mode, 
                            args.color if args.mode == "human_vs_ai" else None,
                            white_action, black_action)
        
        # If game is over, display result
        if done:
            if info.get('white_won', False):
                print("\n### White won! ###")
            elif info.get('black_won', False):
                print("\n### Black won! ###")
            else:
                print("\n### Draw! ###")
            
            # Ask if the player wants to play again
            if args.mode != "ai_vs_ai":
                play_again = input("\nPlay again? (y/n): ").strip().lower()
                if play_again == 'y':
                    obs = env.reset()
                    done = False
                    reward = 0
                    info = {}
                    white_action = None
                    black_action = None
                    print_board_and_info(env, reward, info, args.mode, 
                                         args.color if args.mode == "human_vs_ai" else None,
                                         white_action, black_action)

if __name__ == "__main__":
    main()