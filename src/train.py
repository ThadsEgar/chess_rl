"""
Simplified training utilities for chess RL.
This module provides helper functions and forwards to the new Ray implementation.
"""

import os
import numpy as np
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper
import torch

def make_env(rank, seed=0):
    """
    Create a function that returns a chess environment with proper wrapper.
    Used for compatibility with test_learning.py and other existing code.
    """
    def _init():
        env = ChessEnv()
        env = ActionMaskWrapper(env)
        return env
    return _init

def train(args=None):
    """
    Wrapper function to call the Ray A3C training.
    """
    from src.run_ray_a3c import main as ray_main
    if args is None:
        import sys
        sys.argv = [sys.argv[0], '--mode', 'train']
    ray_main()

if __name__ == '__main__':
    train() 