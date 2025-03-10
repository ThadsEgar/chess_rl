#!/usr/bin/env python
"""
Run Ray-based A3C training for Chess RL with the new PyTorch-only model.
This script serves as the entry point for training, evaluation, and self-play.
"""

import os
import argparse
import logging
import torch
import ray
import platform
import psutil

from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ray_a3c.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ray_a3c")

def detect_resources():
    """
    Detect available system resources to inform Ray configuration.
    """
    # Detect CPU resources
    cpu_count = os.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    
    # Detect GPU resources
    gpu_info = []
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_info.append(f"GPU {i}: {gpu_name}")
    else:
        gpu_count = 0
    
    # Detect if we're on a Lambda Labs machine with A10 GPU
    on_lambda_a10 = False
    if platform.node().startswith('lambda-') and gpu_count > 0:
        for i in range(gpu_count):
            if 'A10' in torch.cuda.get_device_name(i):
                on_lambda_a10 = True
                break
    
    resources = {
        'cpu_count': cpu_count,
        'cpu_percent': cpu_percent,
        'memory_gb': memory_gb,
        'gpu_count': gpu_count,
        'gpu_info': gpu_info,
        'on_lambda_a10': on_lambda_a10
    }
    
    logger.info(f"Detected resources: {resources}")
    return resources

def configure_ray(resources):
    """
    Configure Ray based on detected resources.
    """
    # Initialize Ray with appropriate resources
    if not ray.is_initialized():
        # Default config
        ray_config = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }
        
        # Adjust config based on detected resources
        if resources['gpu_count'] > 0:
            ray_config["num_gpus"] = resources['gpu_count']
        
        # Special configuration for Lambda Labs A10 machine
        if resources['on_lambda_a10']:
            # A10 has 24GB VRAM, configure for better performance
            ray_config["_memory"] = 10 * 1024 * 1024 * 1024  # 10GB
            ray_config["object_store_memory"] = 10 * 1024 * 1024 * 1024  # 10GB
        
        # Initialize Ray
        ray.init(**ray_config)
        logger.info(f"Ray initialized with config: {ray_config}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Ray A3C Chess Training')
    
    # General arguments
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'evaluate', 'self_play'],
                        help='Mode: train, evaluate, or self_play')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (cpu, cuda, auto)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for loading')
    
    # Training arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for parallel training')
    parser.add_argument('--mcts_workers', type=int, default=1,
                        help='Number of workers that use MCTS (0 to disable)')
    parser.add_argument('--mcts_sims', type=int, default=20,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--steps_per_update', type=int, default=200,
                        help='Number of steps per gradient update')
    parser.add_argument('--max_iterations', type=int, default=5000,
                        help='Maximum number of training iterations')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Iterations between checkpoints')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Iterations between logging')
    
    # Evaluation arguments
    parser.add_argument('--num_games', type=int, default=10,
                        help='Number of games to play during evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Render games during evaluation')
    parser.add_argument('--render_interval', type=int, default=1,
                        help='Render every N steps')
    
    # Self-play arguments
    parser.add_argument('--data_dir', type=str, default='self_play_data',
                        help='Directory to save self-play data')
    
    args = parser.parse_args()
    
    # Detect system resources
    resources = detect_resources()
    
    # Configure Ray
    configure_ray(resources)
    
    # Initialize appropriate device
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Log some information about the run
    logger.info(f"Starting Ray A3C in {args.mode} mode on {device}")
    logger.info(f"Command-line arguments: {args}")
    
    try:
        # Import training script (import here to avoid circular imports)
        from src.ray_a3c import train, evaluate, self_play
        
        # Run in the selected mode
        if args.mode == 'train':
            train(args)
        elif args.mode == 'evaluate':
            evaluate(args)
        elif args.mode == 'self_play':
            self_play(args)
            
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
    finally:
        # Clean up Ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")

if __name__ == '__main__':
    main() 