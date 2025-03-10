import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyspiel
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

piece_mapping = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
}

piece_map_human = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',  # White pawn as P
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': 'p'   # Black pawn as p
}

def render_board(state):
    fen = str(state)
    board_fen = fen.split()[0]
    rows = board_fen.split('/')
    
    # Debug: Print raw FEN
    print(f"FEN: {fen}", flush=True)
    board = []
    for row in rows:
        line = []
        for char in row:
            if char.isdigit():
                line.extend(['_'] * int(char))
            else:
                line.append(piece_map_human.get(char, char))
        # Ensure exactly 8 squares
        if len(line) > 8:
            print(f"Row too long: {row} -> {line}", flush=True)
            line = line[:8]  # Truncate to 8
        elif len(line) < 8:
            print(f"Row too short: {row} -> {line}", flush=True)
            line.extend(['·'] * (8 - len(line)))  # Pad to 8
        board.append(line)
    
    output = ""
    for i, row in enumerate(board):
        row_num = 8 - i
        output += f"{row_num}  " + "  ".join(row) + "\n"
    output += "   " + "  ".join(list("abcdefgh")) + "\n"
    return output

def encode_board(state):
    fen = str(state)
    board_fen = fen.split()[0]
    rows = board_fen.split('/')
    board = np.zeros((8, 8), dtype=np.int8)
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                board[i, col] = piece_mapping.get(char, 0)
                col += 1
    return board

def canonical_encode_board(state):
    board = encode_board(state)
    # In our environment white is represented as player 1.
    # We want the board always to be rendered with white at the bottom.
    # Thus, if it's black's turn (current_player() returns 0), flip the board.
    if state.current_player() == 0:
        board = np.rot90(board, 2)
        board = -board
    return board

def canonical_encode_board_for_cnn(state):
    """
    Encode a chess state into a 13-channel format for CNN:
    - 12 channels for piece types (6 per player)
    - 1 channel for current player (1s for white, 0s for black)
    
    Args:
        state: A pyspiel chess state object
        
    Returns:
        numpy.ndarray: A 13x8x8 array representing the board state
    """
    # Get the canonical board representation
    board = canonical_encode_board(state)
    
    # Initialize 13-channel board representation
    # 0-5: current player's pieces (P,N,B,R,Q,K)
    # 6-11: opponent's pieces (p,n,b,r,q,k)
    # 12: current player channel (1s for white, 0s for black)
    board_array = np.zeros((13, 8, 8), dtype=np.float32)
    
    # Fill piece channels
    for row in range(8):
        for col in range(8):
            piece_value = board[row, col]
            if piece_value > 0:  # Current player pieces (positive values)
                channel = piece_value - 1  # Map 1-6 to channels 0-5
                board_array[channel, row, col] = 1.0
            elif piece_value < 0:  # Opponent pieces (negative values)
                channel = abs(piece_value) + 5  # Map -1 to -6 to channels 6-11
                board_array[channel, row, col] = 1.0
    
    # Add current player channel (1s for white, 0s for black)
    current_player = state.current_player()
    if current_player == 1:  # White's turn (assuming white is player 1)
        board_array[12, :, :] = 1.0
    else:  # Black's turn
        board_array[12, :, :] = 0.0
    
    return board_array

def get_game_result(state):
    result = {
        'terminal': state.is_terminal(),
        'white_won': False,
        'black_won': False,
        'draw': False,
        'draw_reason': None,
        'checkmate': False,
        'stalemate': False
    }
    
    rewards = state.rewards()
    # Since white is player 1 and black is player 0, swap the indices:
    if rewards[1] > 0:
        result['white_won'] = True
        result['checkmate'] = True
    elif rewards[0] > 0:
        result['black_won'] = True
        result['checkmate'] = True
    else:
        result['draw'] = True
        fen = str(state)
        if '50' in fen:
            result['draw_reason'] = 'fifty_move_rule'
        elif state.returns()[0] == 0 and not any(piece.isalpha() for piece in fen.split()[0] if piece not in 'Kk'):
            result['draw_reason'] = 'insufficient_material'
        elif 'repetition' in fen.lower():
            result['draw_reason'] = 'threefold_repetition'
        else:
            result['draw_reason'] = 'stalemate'
            result['stalemate'] = True
    return result

class ChessEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.game = pyspiel.load_game("chess")
        self.state = self.game.new_initial_state()
        self.action_space = spaces.Discrete(self.game.num_distinct_actions())
        
        self.piece_channels = 13  # 12 piece channels + player channel
        self.board_size = 8  # 8x8 board
        self.num_actions = 4672
        
        obs_size = self.piece_channels * self.board_size * self.board_size
        
        self.observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.max_moves = 150
        self.move_count = 0
        self.last_move = None
        self.position_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.move_count = 0
        self.state = self.game.new_initial_state()
        self.last_move = None
        self.position_history = [str(self.state)]
        return self._get_obs(), {'starting_player': self.state.current_player()}

    def step(self, action):
        current_player = self.state.current_player()
        legal_actions = self.state.legal_actions()
        
        if action not in legal_actions:
            raise ValueError(f"Action {action} is illegal. Legal actions: {legal_actions}")
        
        previous_fen = str(self.state)
        self.state.apply_action(action)
        self.move_count += 1
        self.last_move = action
        self.position_history.append(str(self.state))
        
        truncated = self.move_count >= self.max_moves
        done = self.state.is_terminal() or truncated
        
        # Get base rewards from environment
        rewards = self.state.rewards()
        reward = rewards[current_player]

        # Calculate intermediate rewards to provide signal during training
        intermediate_reward = 0.0
        
        # Get board encoding to calculate material balance
        board = encode_board(self.state)
        material_balance = 0
        
        # Calculate material balance (positive for white advantage, negative for black)
        # Piece values: pawn=1, knight/bishop=3, rook=5, queen=9
        piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0,  # White pieces (king has no material value)
                      -1: -1, -2: -3, -3: -3, -4: -5, -5: -9, -6: 0}  # Black pieces
        
        for value in board.flatten():
            if value != 0:  # Skip empty squares
                material_balance += piece_values.get(value, 0)
        
        # Scale material balance to a small reward (-0.01 to 0.01 per move)
        material_reward = material_balance * 0.001
        
        # Legal move count reward (encourage positions with more options)
        next_legal_count = len(self.state.legal_actions())
        move_count_reward = min(0.0001 * next_legal_count, 0.001)  # Cap at 0.001
        
        # Center control reward
        center_squares = board[3:5, 3:5]  # e4, d4, e5, d5
        center_control = 0
        for r in range(2):
            for c in range(2):
                if center_squares[r, c] > 0:  # White piece
                    center_control += 0.0005
                elif center_squares[r, c] < 0:  # Black piece
                    center_control -= 0.0005
        
        # Combine intermediate rewards
        if current_player == 1:  # White
            intermediate_reward = material_reward + move_count_reward + center_control
        else:  # Black
            intermediate_reward = -material_reward + move_count_reward - center_control
        
        # If terminal state, use the game result reward (much larger)
        if done:
            # Terminal rewards are much more significant
            result = get_game_result(self.state)
            if result['white_won'] or result['black_won']:
                reward = reward * 20.0  # Amplify win/loss rewards even more
            elif result['draw']:
                # Slightly penalize draws to encourage decisive play
                reward = -0.1
        else:
            # During the game, use small intermediate rewards
            reward = intermediate_reward
        
        result = get_game_result(self.state)
        info = {
            'move_count': self.move_count,
            'current_player': current_player,
            'last_move': self.last_move,
            'previous_position': previous_fen,
            'current_position': str(self.state),
            'legal_moves_count': len(legal_actions),
            'is_terminal': result['terminal'],
            'is_truncated': truncated,
            'white_won': result['white_won'],
            'black_won': result['black_won'],
            'is_draw': result['draw'],
            'draw_reason': result['draw_reason'],
            'is_checkmate': result['checkmate'],
            'is_stalemate': result['stalemate'],
            'position_repetition_count': self.position_history.count(str(self.state)),
            'action_mask': self._get_action_mask(),
            'material_balance': material_balance,
            'intermediate_reward': intermediate_reward
        }
        
        # Set outcome based on game result: white win -> outcome=1, black win -> outcome=-1, draw -> outcome=0.
        if result['white_won']:
            info['outcome'] = 1  
        elif result['black_won']:
            info['outcome'] = -1  
        else:
            info['outcome'] = 0
            
        if done:
            opponent = 1 - current_player
            info['opponent_reward'] = float(self.state.rewards()[opponent])
    
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        board_array = canonical_encode_board_for_cnn(self.state)
        
        return board_array.flatten()
    
    def _get_action_mask(self):
        legal_actions = self.state.legal_actions()
        mask = np.zeros(4672, dtype=np.float32)
        mask[legal_actions] = 1.0
        return mask
    
    def get_legal_actions(self):
        return self.state.legal_actions()
    
    def render(self, mode="human"):
        print(render_board(self.state))
    
    def close(self):
        pass
    
    def get_current_player(self):
        """Get the current player (0 for black, 1 for white)"""
        return self.state.current_player()

    def get_state(self):
        """Get the current game state"""
        return self.state
        
# Special wrapper to handle action masking with stable-baselines3
class ActionMaskWrapper(gym.Wrapper):
    """
    Wrapper that exposes the action mask as part of the observation for easy use with
    standard RL algorithms that don't natively support action masking.
    """
    def __init__(self, env):
        super(ActionMaskWrapper, self).__init__(env)
        board_size = self.env.piece_channels * self.env.board_size * self.env.board_size
        mask_size = self.env.action_space.n
        
        self.observation_space = spaces.Dict({
            'board': self.env.observation_space,
            'action_mask': spaces.Box(
                low=0, high=1, 
                shape=(mask_size,), 
                dtype=np.float32
            )
        })
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        action_mask = self.env._get_action_mask()
        return {'board': obs, 'action_mask': action_mask}, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action_mask = self.env._get_action_mask()
        return {'board': obs, 'action_mask': action_mask}, reward, terminated, truncated, info
