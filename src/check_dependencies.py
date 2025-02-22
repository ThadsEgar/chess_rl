import numpy as np
import torch
import gym
import pyspiel

print("NumPy version:", np.__version__)
print("Torch version:", torch.__version__)
print("Gym version:", gym.__version__)

# Load a simple OpenSpiel game (e.g., Tic Tac Toe)
game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print("Initial state of tic_tac_toe:", state)