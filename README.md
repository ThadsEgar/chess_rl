# Chess Reinforcement Learning with PyTorch and Ray

A reinforcement learning system for learning to play chess using a CNN architecture, Monte Carlo Tree Search (MCTS), and distributed training with Ray.

## Overview

This project implements a chess reinforcement learning agent using:

- **PyTorch-only** neural network architecture (no Stable Baselines 3 dependencies)
- **CNN-based** policy for board evaluation
- **Monte Carlo Tree Search (MCTS)** for improved policy decisions
- **Ray** for distributed training using the A3C algorithm
- Python-chess for game mechanics

## Key Features

- Clean PyTorch implementation with no RL framework dependencies
- Distributed training with Ray for efficient parallelization
- MCTS integration for stronger play and exploration
- Automatic resource detection and optimization
- Comprehensive logging and metrics
- Support for self-play data generation

## Project Structure

- `src/chess_model.py` - Pure PyTorch CNN model and MCTS implementation
- `src/ray_a3c.py` - Ray-based A3C implementation
- `src/run_ray_a3c.py` - Entry point script with command-line interface
- `custom_gym/` - Chess environment implementation

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd chess_rl

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basic training with default parameters
python -m src.run_ray_a3c --mode train

# Training with specific parameters
python -m src.run_ray_a3c --mode train --num_workers 8 --mcts_sims 50 --mcts_workers 2 --device cuda
```

### Evaluation

```bash
# Evaluate a trained model
python -m src.run_ray_a3c --mode evaluate --checkpoint checkpoints/model_final.pt --num_games 20 --render

# Evaluate with MCTS
python -m src.run_ray_a3c --mode evaluate --checkpoint checkpoints/model_final.pt --mcts_sims 100
```

### Self-Play Data Generation

```bash
# Generate self-play data for supervised learning
python -m src.run_ray_a3c --mode self_play --checkpoint checkpoints/model_final.pt --num_games 50
```

## Command-Line Options

### General Options
- `--mode`: Operation mode (train, evaluate, self_play)
- `--device`: Device to run on (cpu, cuda, auto)
- `--checkpoint`: Path to model checkpoint for loading

### Training Options
- `--num_workers`: Number of workers for parallel training
- `--mcts_workers`: Number of workers that use MCTS
- `--mcts_sims`: Number of MCTS simulations per move
- `--steps_per_update`: Number of steps per gradient update
- `--max_iterations`: Maximum number of training iterations
- `--checkpoint_dir`: Directory to save checkpoints
- `--checkpoint_interval`: Iterations between checkpoints
- `--log_interval`: Iterations between logging

### Evaluation Options
- `--num_games`: Number of games to play during evaluation
- `--render`: Enable game rendering
- `--render_interval`: Render every N steps

### Self-Play Options
- `--data_dir`: Directory to save self-play data

## Architecture Details

### CNN Model

The model uses a CNN-based architecture with the following components:
- Input: 13-channel 8x8 board representation
- Feature extraction layers with residual connections
- Separate policy and value heads
- Action masking for legal moves

### MCTS Implementation

The MCTS implementation includes:
- UCT-based tree search with exploration parameter
- Integration with policy network for move selection
- Batch search capability for multiple states
- Value estimation for tree positions

### Ray A3C

The Ray implementation includes:
- Parameter server for weight synchronization
- Multiple rollout workers for experience collection
- Gradient-based updates
- Metrics tracking and checkpointing
