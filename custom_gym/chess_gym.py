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
    
    Returns:
        numpy.ndarray: A 13×8×8 representation of the board state
    """
    board = state.board if hasattr(state, 'board') else state
    encoded = np.zeros((13, 8, 8), dtype=np.float32)
    
    # Current player is always white in our encoding
    player_is_white = board.turn if hasattr(board, 'turn') else True
    
    for i in range(8):
        for j in range(8):
            square = chess.square(j, 7-i)
            piece = board.piece_at(square)
            
            if piece:
                piece_type = piece.piece_type  # 1=pawn, 2=knight, ..., 6=king
                is_white = piece.color
                
                if (player_is_white and is_white) or (not player_is_white and not is_white):
                    # Current player's pieces (always in channels 0-5)
                    channel = piece_type - 1  # Convert from 1-based to 0-based indexing
                    if 0 <= channel < 6:  # Ensure valid index
                        encoded[channel, i, j] = 1
                else:
                    # Opponent's pieces (always in channels 6-11)
                    channel = piece_type + 5  # Offset by 6, then -1 for 0-based indexing
                    if 6 <= channel < 12:  # Ensure valid index
                        encoded[channel, i, j] = 1
            else:
                # Empty square (channel 12)
                encoded[12, i, j] = 1
    
    # If current player is black, flip the board
    if not player_is_white:
        encoded = np.flip(encoded, axis=1)
    
    # Final safety check to ensure correct shape
    if encoded.shape != (13, 8, 8):
        print(f"Warning: canonical_encode_board_for_cnn produced incorrect shape: {encoded.shape}, fixing to (13, 8, 8)")
        # Create a proper shape board with zeros
        proper_board = np.zeros((13, 8, 8), dtype=np.float32)
        # Set empty squares to 1
        proper_board[12, :, :] = 1
        # Try to copy data if possible
        if len(encoded.shape) == 3 and encoded.shape[1:] == (8, 8):
            # Copy the available channels
            for c in range(min(encoded.shape[0], 13)):
                proper_board[c] = encoded[c]
        encoded = proper_board
    
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
        # Update action space to accommodate the full range of possible move indices
        # Regular moves: 64*64 = 4096
        # Promotion moves: 64*64 + (64*64*4) = 20480
        self.action_space = spaces.Discrete(20480)  # Increased from 4672 to handle all possible action indices
        
        # Observation space includes board state and action mask
        self.board_channels = 13  # 6 piece types for each color + empty squares
        self.board_size = 8  # 8x8 board
        board_shape = (self.board_channels, self.board_size, self.board_size)
        action_mask_shape = (self.action_space.n,)
        
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=1, shape=board_shape, dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=action_mask_shape, dtype=np.float32),
            'white_to_move': spaces.Discrete(2)  # Boolean flag: 0=False (Black's turn), 1=True (White's turn)
        })
        
        self.done = False
        self.reward = 0
        self.move_count = 0
        self.max_moves = 150  # Limit games to 150 moves (150 plies) to prevent infinite games
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new game."""
        super().reset(seed=seed)
        self.board = chess.Board()
        self.state = ChessState(self.board)
        self.done = False
        self.reward = 0
        self.move_count = 0
        
        # Return observation and info dict (new Gymnasium API)
        return self._get_obs(), {"current_player": 1 if self.board.turn else 0}
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: The action to take (an integer index)
            
        Returns:
            observation: The new state
            reward: Reward for this step
            terminated: Whether the game is done due to termination
            truncated: Whether the game is done due to truncation
            info: Additional information
        """
        if self.done:
            return self._get_obs(), 0, True, False, {"message": "Game already over"}
        
        # Convert action index to move
        try:
            move = self.state.action_to_move(action)
            
            # Check if the move is legal
            if move not in self.board.legal_moves:
                # Illegal move, return negative reward and don't change state
                return self._get_obs(), -0.1, False, False, {"message": "Illegal move"}
            
            # Make the move
            self.board.push(move)
            self.move_count += 1
            
            # Check if the game is over
            info = {}
            reward = 0
            
            # Check game state conditions
            checkmate = self.board.is_checkmate()
            stalemate = self.board.is_stalemate()
            insufficient = self.board.is_insufficient_material()
            fifty_move = self.board.is_fifty_moves()
            repetition = self.board.is_repetition()
            
            if checkmate:
                self.done = True
                # Who won?
                if self.board.turn:  # Black's turn now, so White won
                    reward = 1.0  # White win from White's perspective is +1
                    info["white_won"] = True
                    info["game_outcome"] = "white_win"
                    info["outcome"] = "white_win"
                else:  # White's turn now, so Black won
                    reward = -1.0  # Black win from White's perspective is -1
                    info["black_won"] = True
                    info["game_outcome"] = "black_win"
                    info["outcome"] = "black_win"
                    
            elif stalemate or insufficient or fifty_move or repetition:
                self.done = True
                reward = 0.0  # Draw is 0 from any perspective
                info["draw"] = True
                info["game_outcome"] = "draw"
                info["outcome"] = "draw"
                if stalemate:
                    info["termination_reason"] = "stalemate"
                elif insufficient:
                    info["termination_reason"] = "insufficient_material"
                elif fifty_move:
                    info["termination_reason"] = "fifty_move_rule"
                elif repetition:
                    info["termination_reason"] = "threefold_repetition"
                
            # Small negative reward per move to encourage efficiency
            if not self.done:
                reward -= 0.001
                
            # Small reward for checks to encourage aggressive play
            if self.board.is_check():
                reward += 0.01
                
            # Check for move limit exceeded (draw)
            if self.move_count >= self.max_moves:
                self.done = True
                reward = 0.0
                info["draw"] = True
                info["game_outcome"] = "draw"
                info["outcome"] = "draw"
                info["termination_reason"] = "move_limit_exceeded"
                
            # Update state
            self.state = ChessState(self.board)
            self.reward = reward
                
            # Add move count to info
            info["move_count"] = self.move_count
            info["current_player"] = 1 if self.board.turn else 0
            
            # Always include game state info
            if self.done:
                info["game_completed"] = True
                if not info.get("termination_reason"):
                    if info.get("white_won"):
                        info["termination_reason"] = "white_checkmate"
                    elif info.get("black_won"):
                        info["termination_reason"] = "black_checkmate"
                    elif info.get("draw"):
                        info["termination_reason"] = "draw"
            else:
                info["game_completed"] = False
            
            # Follow new Gymnasium API: (observation, reward, terminated, truncated, info)
            return self._get_obs(), reward, self.done, False, info
            
        except Exception as e:
            # If there's an error, return the current state and a negative reward
            print(f"Error making move: {e}")
            return self._get_obs(), -0.1, False, False, {"message": f"Error: {e}"}
    
    def _get_obs(self):
        """Returns the current observation."""
        # Use the canonical_encode_board_for_cnn function which produces (13, 8, 8) shape
        board_state = canonical_encode_board_for_cnn(self.state)
        action_mask = self._get_action_mask()
        
        return {
            'board': board_state,
            'action_mask': action_mask,
            'white_to_move': int(not self.board.turn)  # chess.BLACK is True, chess.WHITE is False
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
    """Wrapper to add action mask to observation"""
    
    def __init__(self, env):
        super().__init__(env)
        # Define observation space with action mask added
        self.observation_space = spaces.Dict({
            'board': env.observation_space['board'],
            'action_mask': spaces.Box(
                low=0, 
                high=1, 
                shape=(env.action_space.n,), 
                dtype=np.float32
            ),
            'white_to_move': spaces.Discrete(2)  # Boolean flag: 0=False (Black's turn), 1=True (White's turn)
        })
        
    def reset(self, **kwargs):
        """Reset environment and add action mask to observation"""
        obs, info = self.env.reset(**kwargs)
        action_mask = self._get_action_mask()
        
        # Ensure white_to_move is present in all observations
        if 'white_to_move' not in obs:
            # Add white_to_move based on board state
            white_to_move = not self.env.board.turn  # chess.BLACK is True, chess.WHITE is False
            obs['white_to_move'] = white_to_move
        
        # Create new observation with action mask
        dict_obs = {
            'board': obs['board'],
            'action_mask': action_mask,
            'white_to_move': int(obs['white_to_move'])  # Convert to int for Discrete space
        }
        
        # Ensure info dictionary is initialized with basic information
        if info is None:
            info = {}
            
        # Add game state info - fresh game
        info["move_count"] = 0
        info["game_ongoing"] = True
        
        return dict_obs, info
        
    def step(self, action):
        """Take a step and add action mask to new observation"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        action_mask = self._get_action_mask()
        
        # Ensure white_to_move is present in all observations
        if 'white_to_move' not in obs:
            # Add white_to_move based on board state
            white_to_move = not self.env.board.turn  # chess.BLACK is True, chess.WHITE is False
            obs['white_to_move'] = white_to_move
        
        # Create new observation with action mask
        dict_obs = {
            'board': obs['board'],
            'action_mask': action_mask,
            'white_to_move': int(obs['white_to_move'])  # Convert to int for Discrete space
        }
        
        # Ensure info dictionary is complete, especially for terminal states
        if terminated or truncated:
            # If info is None, create an empty dict
            if info is None:
                info = {}
                
            # Ensure outcome information is present for terminal states
            if "outcome" not in info:
                # Try to get game state from environment
                if hasattr(self.env, 'board'):
                    board = self.env.board
                    # Check game state conditions
                    if board.is_checkmate():
                        # Who won?
                        if board.turn:  # Black's turn now, so White won
                            info["outcome"] = "white_win"
                            info["game_outcome"] = "white_win"
                            info["white_won"] = True
                        else:  # White's turn now, so Black won
                            info["outcome"] = "black_win"
                            info["game_outcome"] = "black_win"
                            info["black_won"] = True
                    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
                        info["outcome"] = "draw"
                        info["game_outcome"] = "draw"
                        info["draw"] = True
                    elif hasattr(self.env, 'move_count') and hasattr(self.env, 'max_moves') and self.env.move_count >= self.env.max_moves:
                        info["outcome"] = "draw"
                        info["game_outcome"] = "draw"
                        info["draw"] = True
                        info["termination_reason"] = "move_limit_exceeded"
                    else:
                        info["outcome"] = "unknown"
                        info["game_outcome"] = "unknown"
                else:
                    # Default to unknown if we can't determine the outcome
                    info["outcome"] = "unknown"
                    info["game_outcome"] = "unknown"
            
            # Make sure game_outcome is consistent with outcome
            if "game_outcome" not in info and "outcome" in info:
                info["game_outcome"] = info["outcome"]
            
            # Make sure outcome is consistent with game_outcome
            if "outcome" not in info and "game_outcome" in info:
                info["outcome"] = info["game_outcome"]
        
        return dict_obs, reward, terminated, truncated, info
    
    def _get_action_mask(self):
        """Create a mask of legal actions."""
        legal_actions = self.env.legal_actions()
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        
        if legal_actions:
            # Set 1 for each legal action
            mask[legal_actions] = 1.0
            
        return mask

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
