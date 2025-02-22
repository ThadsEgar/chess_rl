import gym
from gym import spaces
import numpy as np
import pyspiel

def encode_board(state):
    return np.zeros(64, dtype=np.int8)

class ChessEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self):
        super(ChessEnv, self).__init__()
        # Load the chess game from OpenSpiel
        self.game = pyspiel.load_game("chess")
        self.state = self.game.new_initial_state()
        
        # OpenSpiel represents moves as integers.
        # Use the total number of distinct moves as the action space.
        self.action_space = spaces.Discrete(self.game.num_distinct_actions())
        
        # For a minimal observation, we'll use a flat array of length 64.
        # In practice, you would design a more expressive board encoding.
        self.observation_space = spaces.Box(low=0, high=12, shape=(64,), dtype=np.int8)
    
    def reset(self):
        self.state = self.game.new_initial_state()
        return self._get_obs()
    
    def step(self, action):
        # Check if the chosen action is legal
        legal_actions = self.state.legal_actions()
        if action not in legal_actions:
            raise ValueError(f"Action {action} is illegal. Legal actions: {legal_actions}")
        
        # Apply the action
        self.state.apply_action(action)
         
        # Get observation, reward, done, and any additional info
        obs = self._get_obs()
        done = self.state.is_terminal()
        # Reward from the perspective of the current player
        reward = self.state.reward() if done else 0.0
        info = {}
        return obs, reward, done, info
    
    def _calculate_reward(self):
        return 0
    
    def _get_obs(self):
        # Here you should convert the current state into an observation.
        # For chess, a common approach is to convert the board into a matrix encoding piece types.
        # For simplicity, we use a dummy encoder.
        return encode_board(self.state)
    
    def render(self, mode="human"):
        print(self.state)   
    
    def close(self):
        pass

# Example usage:
if __name__ == "__main__":
    env = ChessEnv()
    obs = env.reset()
    env.render()
    legal = env.state.legal_actions()
    print("Legal actions:", legal)
    # For demonstration, pick the first legal move:
    action = legal[0]
    obs, reward, done, info = env.step(action)
    env.render()
