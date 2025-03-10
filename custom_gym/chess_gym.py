import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyspiel
import chess
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
    """
    Renders a chess board state in a text-based format.
    """
    board_str = str(state.board)
    rendered = ""
    for row in board_str.split('\n'):
        rendered += row + '\n'
    return rendered

def encode_board(state):
    """
    Encodes the board state as a 2D array for RL.
    """
    board = state.board
    encoded = np.zeros((8, 8), dtype=np.int8)
    
    for i in range(8):
        for j in range(8):
            square = chess.square(j, 7-i)
            piece = board.piece_at(square)
            if piece:
                value = piece_mapping.get(piece.symbol(), 0)
                encoded[i, j] = value
    
    return encoded

def canonical_encode_board(state):
    """
    Encode the board from perspective of current player.
    White pieces are always positive, black pieces always negative.
    """
    board = state.board
    player = 1 if board.turn else -1
    encoded = encode_board(state)
    
    if player == -1:  # If black's turn, flip the board and values
        encoded = np.flip(encoded, axis=0)
        encoded = -encoded  # Negate values to maintain white positive, black negative
    
    return encoded

def canonical_encode_board_for_cnn(state):
    """
    Encode the board from the perspective of the current player for CNN input.
    Uses a 13-channel representation (6 for player's pieces, 6 for opponent's, 1 for empty squares).
    """
    board = state.board
    encoded = np.zeros((13, 8, 8), dtype=np.float32)
    
    # Current player is always white in our encoding
    player_is_white = board.turn
    
    for i in range(8):
        for j in range(8):
            square = chess.square(j, 7-i)
            piece = board.piece_at(square)
            
            if piece:
                piece_type = piece.piece_type  # 1=pawn, 2=knight, ..., 6=king
                is_white = piece.color
                
                if (player_is_white and is_white) or (not player_is_white and not is_white):
                    # Current player's pieces (always in channels 0-5)
                    encoded[piece_type - 1, i, j] = 1
                else:
                    # Opponent's pieces (always in channels 6-11)
                    encoded[piece_type + 5, i, j] = 1
            else:
                # Empty square (channel 12)
                encoded[12, i, j] = 1
    
    # If current player is black, flip the board
    if not player_is_white:
        encoded = np.flip(encoded, axis=1)
    
    return encoded

def get_game_result(state):
    """
    Returns the game result from the current player's perspective:
    1.0 for win, 0.0 for draw, -1.0 for loss
    """
    board = state.board
    
    if board.is_game_over():
        if board.is_checkmate():
            # If it's checkmate, the side whose turn it was lost
            return -1.0
        else:
            # Draw (stalemate, insufficient material, 50-move rule, etc.)
            return 0.0
    return None  # Game not yet over

class ChessState:
    """
    A wrapper around chess.Board to provide consistent interface.
    """
    def __init__(self, board=None):
        self.board = board if board is not None else chess.Board()
        
    def legal_actions(self):
        """Get legal actions in the current state."""
        return [self.move_to_action(move) for move in self.board.legal_moves]
    
    def move_to_action(self, move):
        """Convert a chess move to an action index."""
        from_square = move.from_square
        to_square = move.to_square
        
        # Promotion pieces
        promotion_pieces = {
            None: 0,
            chess.QUEEN: 1,
            chess.ROOK: 2,
            chess.BISHOP: 3,
            chess.KNIGHT: 4
        }
        promotion = promotion_pieces.get(move.promotion, 0)
        
        # Action is from_square * 64 + to_square + promotion_offset
        # This gives a unique index for each possible move
        action = from_square * 64 + to_square
        if promotion > 0:
            # Add promotion offset (there are 4 promotion piece types)
            action = 64 * 64 + (from_square * 64 + to_square) * 4 + (promotion - 1)
        
        return action
    
    def action_to_move(self, action):
        """Convert an action index to a chess move."""
        if action < 64 * 64:
            # Regular move
            from_square = action // 64
            to_square = action % 64
            return chess.Move(from_square, to_square)
        else:
            # Promotion move
            promotion_index = (action - 64 * 64) % 4
            action_without_promotion = (action - 64 * 64) // 4
            from_square = action_without_promotion // 64
            to_square = action_without_promotion % 64
            
            promotion_pieces = [
                chess.QUEEN,
                chess.ROOK,
                chess.BISHOP,
                chess.KNIGHT
            ]
            return chess.Move(from_square, to_square, promotion=promotion_pieces[promotion_index])
    
    def is_terminal(self):
        """Return True if the game is over."""
        return self.board.is_game_over()
    
    def current_player(self):
        """Return the current player (1 for white, 0 for black)."""
        return 1 if self.board.turn else 0
    
    def copy(self):
        """Return a deep copy of the state."""
        return ChessState(self.board.copy())

def create_simple_endgame(random_side=True):
    """
    Create a simple endgame position for testing.
    """
    board = chess.Board()
    board.clear()
    
    # Place kings
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    
    # Place a queen for white
    if random_side:
        white_side = random.choice([chess.WHITE, chess.BLACK])
        board.set_piece_at(chess.D1 if white_side == chess.WHITE else chess.D8, 
                          chess.Piece(chess.QUEEN, white_side))
        board.turn = not white_side  # Make it the other side's turn
    else:
        # Always place white queen
        board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
        board.turn = chess.BLACK
        
    return board

class ChessEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, simple_test=False, white_advantage=None):
        super(ChessEnv, self).__init__()
        
        # Initialize the chess game
        self.board = chess.Board()
        self.state = ChessState(self.board)
        
        # Create a simple endgame test position if requested
        if simple_test:
            self.board = create_simple_endgame(random_side=True)
            self.state = ChessState(self.board)
        
        # Give white an advantage if requested (for testing value function)
        if white_advantage is not None:
            # Add extra pawns for white to create advantage
            if white_advantage > 0:
                for i in range(min(white_advantage, 8)):
                    square = chess.square(i, 1)  # 2nd rank
                    self.board.set_piece_at(square, chess.Piece(chess.PAWN, chess.WHITE))
            # Add extra pawns for black to create advantage
            elif white_advantage < 0:
                for i in range(min(-white_advantage, 8)):
                    square = chess.square(i, 6)  # 7th rank
                    self.board.set_piece_at(square, chess.Piece(chess.PAWN, chess.BLACK))
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4672)  # Maximum number of possible moves in chess
        
        # Observation space includes board state and action mask
        self.board_channels = 13  # 6 piece types for each color + empty squares
        self.board_size = 8  # 8x8 board
        board_shape = (self.board_channels, self.board_size, self.board_size)
        action_mask_shape = (self.action_space.n,)
        
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=1, shape=board_shape, dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=action_mask_shape, dtype=np.float32)
        })
        
        self.done = False
        self.reward = 0
        self.move_count = 0
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new game."""
        super().reset(seed=seed)
        self.board = chess.Board()
        self.state = ChessState(self.board)
        self.done = False
        self.reward = 0
        self.move_count = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: The action to take (an integer index)
            
        Returns:
            observation: The new state
            reward: Reward for this step
            done: Whether the game is done
            info: Additional information
        """
        if self.done:
            return self._get_obs(), 0, True, {"message": "Game already over"}
        
        # Convert action index to move
        try:
            move = self.state.action_to_move(action)
            
            # Check if the move is legal
            if move not in self.board.legal_moves:
                # Illegal move, return negative reward and don't change state
                return self._get_obs(), -0.1, False, {"message": "Illegal move"}
            
            # Make the move
            self.board.push(move)
            self.move_count += 1
            
            # Check if the game is over
            info = {}
            reward = 0
            
            if self.board.is_checkmate():
                self.done = True
                # Who won?
                if self.board.turn:  # Black's turn now, so White won
                    reward = 1.0 if self.board.turn == chess.BLACK else -1.0
                    info["white_won"] = True
                else:  # White's turn now, so Black won
                    reward = 1.0 if self.board.turn == chess.WHITE else -1.0
                    info["black_won"] = True
                    
            elif self.board.is_stalemate() or self.board.is_insufficient_material() or \
                 self.board.is_fifty_moves() or self.board.is_repetition():
                self.done = True
                reward = 0.0
                info["draw"] = True
                
            # Small negative reward per move to encourage efficiency
            if not self.done:
                reward -= 0.001
                
            # Small reward for checks to encourage aggressive play
            if self.board.is_check():
                reward += 0.01
                
            # Update state
            self.state = ChessState(self.board)
            self.reward = reward
                
            # Add move count to info
            info["move_count"] = self.move_count
            info["current_player"] = 1 if self.board.turn else 0
            
            return self._get_obs(), reward, self.done, info
            
        except Exception as e:
            # If there's an error, return the current state and a negative reward
            print(f"Error making move: {e}")
            return self._get_obs(), -0.1, False, {"message": f"Error: {e}"}
    
    def _get_obs(self):
        """Get the current observation."""
        return {
            'board': canonical_encode_board_for_cnn(self.state),
            'action_mask': self._get_action_mask()
        }
    
    def _get_action_mask(self):
        """Create a mask for valid actions."""
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        for action in self.state.legal_actions():
            mask[action] = 1.0
        return mask
    
    def legal_actions(self):
        """Get legal actions for the current state."""
        return self.state.legal_actions()
    
    def render(self, mode="human"):
        """Render the chess board."""
        return render_board(self.state)
    
    def close(self):
        """Close the environment."""
        pass
    
    def get_current_player(self):
        """Get the current player (1 for white, 0 for black)."""
        return 1 if self.board.turn else 0
        
    def move_to_action(self, move):
        """Convert a chess move to an action index."""
        return self.state.move_to_action(move)
        
    def action_to_move(self, action):
        """Convert an action index to a chess move."""
        return self.state.action_to_move(action)

class ActionMaskWrapper(gym.Wrapper):
    """
    A wrapper that adds an action mask to the observation,
    which can be used by the agent to mask out illegal moves.
    """
    def __init__(self, env):
        super(ActionMaskWrapper, self).__init__(env)
        
        # Update observation space to include action mask
        self.observation_space = spaces.Dict({
            'board': env.observation_space['board'],
            'action_mask': spaces.Box(
                low=0, high=1, 
                shape=(env.action_space.n,), 
                dtype=np.float32
            )
        })
    
    def reset(self, **kwargs):
        """Reset the environment and add action mask to observation."""
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        """Take a step and add action mask to the new observation."""
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

def make_env(rank, seed=0, simple_test=False, white_advantage=None):
    """
    Create a function that returns a chess environment with proper wrapper.
    Used for compatibility with distributed training frameworks.
    """
    def _init():
        env = ChessEnv(simple_test=simple_test, white_advantage=white_advantage)
        env = ActionMaskWrapper(env)
        return env
    return _init
