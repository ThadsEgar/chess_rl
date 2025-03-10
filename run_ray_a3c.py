#!/usr/bin/env python3
import os
import sys
import multiprocessing as mp
import argparse
import torch
import socket

# Get number of available CPU cores
cpu_count = mp.cpu_count()
optimal_actors = max(1, cpu_count - 1)

# Get hostname
hostname = socket.gethostname()

# Get GPU count and memory
def get_gpu_info():
    gpu_count = 0
    gpu_memory = 0
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_memory += torch.cuda.get_device_properties(i).total_memory
            # Convert to GB
            gpu_memory = gpu_memory / (1024**3)
    except:
        pass
    return gpu_count, gpu_memory

# Detect if running on Lambda Labs
def is_lambda_labs():
    import platform
    import socket
    hostname = socket.gethostname()
    return 'lambda' in hostname.lower() or 'gpu-' in hostname.lower()

def main():
    parser = argparse.ArgumentParser(description='Run Ray-based A3C training for chess')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['local', 'head', 'worker'], default='local',
                      help='Running mode: local (single machine), head (cluster head), worker (cluster worker)')
    
    # Cluster settings (only for distributed mode)
    parser.add_argument('--head_address', type=str, default=None,
                      help='Address of the Ray head node (required for worker mode)')
    
    # Resource allocation
    parser.add_argument('--num_actors', type=int, default=None,
                      help='Number of actor processes (default: auto-detected based on hardware)')
    parser.add_argument('--n_envs_per_actor', type=int, default=4,
                      help='Number of environments per actor (default: 4)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu, default: auto-detect)')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                      help='Learning rate (default: 2e-4)')
    parser.add_argument('--ent_coef', type=float, default=0.1,
                      help='Entropy coefficient (default: 0.1)')
    parser.add_argument('--max_steps', type=int, default=10000000,
                      help='Total steps to train for (default: 10M)')
    parser.add_argument('--mcts_sims', type=int, default=None,
                      help='Number of MCTS simulations (default: auto-based on hardware)')
    parser.add_argument('--mcts_freq', type=float, default=0.2,
                      help='Frequency of using MCTS (default: 0.2)')
    parser.add_argument('--auto_adjust_actors', action='store_true', default=True,
                      help='Automatically adjust resources based on hardware')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Auto-detect hardware and set defaults
    gpu_count, gpu_memory = get_gpu_info()
    
    # Set optimal actors and MCTS sims based on hardware
    if args.num_actors is None:
        if gpu_count > 0:
            # For A10 with 24GB, we want to use around 16 actors
            if gpu_memory > 20:  # A10 or better
                args.num_actors = min(16, optimal_actors)
            else:  # Smaller GPU
                args.num_actors = min(8, optimal_actors)
        else:
            args.num_actors = optimal_actors
    
    if args.mcts_sims is None:
        # Set MCTS sims based on GPU memory
        if gpu_count > 0:
            if gpu_memory > 20:  # A10 or better
                args.mcts_sims = 50  # Moderate MCTS for A10
            elif gpu_memory > 10:  # Mid-range GPU
                args.mcts_sims = 30
            else:  # Small GPU
                args.mcts_sims = 20
        else:
            args.mcts_sims = 10  # CPU-only mode
    
    # Import Ray and our A3C implementation here to avoid import errors
    # if Ray is not installed
    try:
        import ray
        from src.ray_a3c import A3CTrainer
    except ImportError:
        print("Error: Ray is not installed. Please install it with 'pip install ray[default]'")
        sys.exit(1)
    
    # Prepare Ray address based on mode
    ray_address = None  # Default for local mode
    
    if args.mode == 'head':
        # Get IP address for the head node
        ip_address = socket.gethostbyname(hostname)
        ray_address = f"{ip_address}:6379"
        print(f"Starting Ray head node at {ray_address}")
        
        # Use all available resources on head node
        ray.init(address=None, include_dashboard=True)
        
        # Print instructions for workers
        print("\n" + "="*50)
        print(f"To connect workers to this head node, run:")
        print(f"python run_ray_a3c.py --mode worker --head_address {ray_address}")
        print("="*50 + "\n")
        
    elif args.mode == 'worker':
        if not args.head_address:
            print("Error: --head_address is required in worker mode")
            sys.exit(1)
        
        ray_address = args.head_address
        print(f"Connecting to Ray head node at {ray_address}")
        
        # Connect to the existing Ray cluster
        ray.init(address=f"ray://{ray_address}")
        
        # In worker mode, we don't need to run the training logic
        print("Successfully connected to the Ray cluster.")
        print("This worker will now be available for distributed tasks.")
        
        # Keep the process running
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("Worker shutting down.")
        
        sys.exit(0)
    
    elif args.mode == 'local':
        # Use default local Ray init
        print("Starting Ray in local mode")
        ray.init(ignore_reinit_error=True)
    
    # Print hardware information
    print(f"\nHardware detected: {cpu_count} CPU cores, {gpu_count} GPUs with {gpu_memory:.1f}GB memory")
    print(f"Using {args.num_actors} actors with {args.mcts_sims} MCTS simulations per decision")
    
    # Create configuration from arguments
    config = {
        "ray_address": ray_address,
        "num_actors": args.num_actors,
        "n_envs_per_actor": args.n_envs_per_actor,
        "device": args.device,
        "learning_rate": args.learning_rate,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": args.ent_coef,
        "vf_coef": 0.5,
        "clip_range": 0.2,
        "max_grad_norm": 0.5,
        "trajectory_length": 128,
        "max_steps": args.max_steps,
        "save_interval": 50,
        "log_interval": 5,
        "save_dir": "data/models",
        "mcts_sims": args.mcts_sims,
        "mcts_freq": args.mcts_freq,
        "simple_test": False,
        "auto_adjust_actors": args.auto_adjust_actors
    }
    
    # Print configuration
    print("\nTraining configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create trainer and run training
    if args.mode != 'worker':  # Only start training in local or head mode
        trainer = A3CTrainer(config)
        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            trainer.shutdown()
    
if __name__ == "__main__":
    main() 