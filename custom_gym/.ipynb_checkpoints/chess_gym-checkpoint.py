import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyspiel
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv

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
        board_size = 64
        num_actions = 4672
        obs_size = board_size + num_actions
        
        self.observation_space = spaces.Box(
            low=-6,
            high=6,
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
        
        rewards = self.state.rewards()
        reward = rewards[current_player]

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

        if current_player == 0:
            reward += 0.005  # Small constant bonus for black
    
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        board_obs = canonical_encode_board(self.state).flatten()
        legal_actions = self.state.legal_actions()
        mask = np.zeros(self.game.num_distinct_actions(), dtype=np.float32)
        mask[legal_actions] = 1.0
        return np.concatenate([board_obs, mask])
    
    def render(self, mode="human"):
        print(render_board(self.state))
    
    def close(self):
        pass
