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
import socket

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
    
    # Detect GPU resources with better validation
    gpu_info = []
    gpu_count = 0
    cuda_available = False
    
    # First check if CUDA is actually usable (not just installed)
    try:
        if torch.cuda.is_available():
            # Try to actually initialize CUDA to confirm it works
            test_tensor = torch.zeros(1).cuda()
            del test_tensor  # Clean up
            
            # If we get here, CUDA is actually working
            cuda_available = True
            gpu_count = torch.cuda.device_count()
            
            for i in range(gpu_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_info.append(f"GPU {i}: {gpu_name}")
                except Exception as e:
                    gpu_info.append(f"GPU {i}: Error getting name ({str(e)})")
    except Exception as e:
        logger.warning(f"CUDA reported as available but failed initialization: {str(e)}")
        logger.warning("Falling back to CPU mode")
        cuda_available = False
        gpu_count = 0
    
    # Detect if we're on a Lambda Labs machine with A10 GPU
    on_lambda_a10 = False
    if cuda_available and platform.node().startswith('lambda-'):
        for i in range(gpu_count):
            if 'A10' in torch.cuda.get_device_name(i):
                on_lambda_a10 = True
                break
    
    # Detect if we're on a Mac
    is_mac = platform.system() == 'Darwin'
    has_mps = False
    
    # Check for Metal Performance Shaders (Apple Silicon)
    if is_mac:
        try:
            has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if has_mps:
                # Verify MPS actually works
                test_tensor = torch.zeros(1).to('mps')
                del test_tensor
        except Exception as e:
            logger.warning(f"MPS reported as available but failed: {str(e)}")
            has_mps = False
    
    resources = {
        'hostname': socket.gethostname(),
        'cpu_count': cpu_count,
        'cpu_percent': cpu_percent,
        'memory_gb': memory_gb,
        'gpu_count': gpu_count,
        'gpu_info': gpu_info,
        'cuda_available': cuda_available,
        'on_lambda_a10': on_lambda_a10,
        'is_mac': is_mac,
        'has_mps': has_mps
    }
    
    logger.info(f"Detected resources: {resources}")
    return resources

def configure_ray(resources, args):
    """
    Configure Ray based on detected resources.
    """
    # Initialize Ray with appropriate resources
    if not ray.is_initialized():
        if args.distributed:
            # Connect to existing Ray cluster
            if args.head_address:
                logger.info(f"Connecting to Ray cluster at {args.head_address}")
                ray_config = {
                    "address": args.head_address,
                    "ignore_reinit_error": True,
                }
                if args.redis_password:
                    ray_config["_redis_password"] = args.redis_password
                
                ray.init(**ray_config)
                logger.info(f"Connected to Ray cluster at {args.head_address}")
                return
        
        # Default config for local mode
        ray_config = {
            "ignore_reinit_error": True,
            "include_dashboard": args.dashboard,
        }
        
        # Start Ray head node
        if args.head:
            ray_config["dashboard_host"] = "0.0.0.0"
            ray_config["num_cpus"] = resources['cpu_count']
            
            # Only set GPU resources if CUDA is actually available
            if resources['cuda_available'] and resources['gpu_count'] > 0:
                ray_config["num_gpus"] = resources['gpu_count']
            
            # If the password is provided, use it
            if args.redis_password:
                ray_config["_redis_password"] = args.redis_password
                
            logger.info(f"Starting Ray head node with config: {ray_config}")
        
        # Special configuration for Lambda Labs A10 machine
        if resources['on_lambda_a10']:
            # A10 has 24GB VRAM, configure for better performance
            ray_config["_memory"] = 10 * 1024 * 1024 * 1024  # 10GB
            ray_config["object_store_memory"] = 10 * 1024 * 1024 * 1024  # 10GB
        
        # Special configuration for Mac
        if resources['is_mac']:
            # Macs often need less aggressive resource allocation
            ray_config["num_cpus"] = max(1, resources['cpu_count'] - 2)  # Leave CPU cores for system
            ray_config["object_store_memory"] = 2 * 1024 * 1024 * 1024  # 2GB
        
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
                        help='Device to run on (cpu, cuda, mps, auto)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for loading')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU use even if GPU is available')
    
    # Distributed mode arguments
    parser.add_argument('--distributed', action='store_true',
                        help='Run in distributed mode across multiple machines')
    parser.add_argument('--head', action='store_true',
                        help='Start Ray as a head node')
    parser.add_argument('--head_address', type=str, default=None,
                        help='Address of the Ray head node to connect to (e.g., "ip:port")')
    parser.add_argument('--redis_password', type=str, default=None,
                        help='Password for connecting to the Redis server in the Ray cluster')
    parser.add_argument('--dashboard', action='store_true',
                        help='Include Ray dashboard')
    
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
    configure_ray(resources, args)
    
    # Initialize appropriate device
    device = args.device
    if args.force_cpu:
        device = 'cpu'
        logger.info("Forcing CPU use as requested")
    elif device == 'auto':
        if resources['cuda_available']:
            device = 'cuda'
        elif resources['has_mps']:
            device = 'mps'
        else:
            device = 'cpu'
    elif device == 'cuda' and not resources['cuda_available']:
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    elif device == 'mps' and not resources['has_mps']:
        logger.warning("MPS requested but not available, falling back to CPU")
        device = 'cpu'
    
    # Log some information about the run
    logger.info(f"Starting Ray A3C in {args.mode} mode on {device}")
    if args.distributed:
        if args.head:
            logger.info(f"Running as head node")
        else:
            logger.info(f"Running as worker node connected to {args.head_address}")
    
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
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
    finally:
        # Clean up Ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")

if __name__ == '__main__':
    main() 