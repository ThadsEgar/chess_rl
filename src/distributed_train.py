import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import socket
import numpy as np
from src.a3c import CNNMCTSActorCriticPolicy, make_env, A3CWorker

def run_master(args):
    """
    Run as the master node that coordinates training.
    Master initializes the model, sets up the distributed process group,
    and aggregates updates from workers.
    """
    print(f"Running as master on {socket.gethostname()}")
    
    # Set up environment to get spaces for model creation
    env = make_env(0)()
    
    # Initialize model
    model = CNNMCTSActorCriticPolicy(
        env.observation_space,
        env.action_space,
        lambda _: args.learning_rate
    ).to(args.device)
    
    # Enable shared memory
    model.share_memory()
    
    # Initialize process group
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{args.master_addr}:{args.master_port}',
        world_size=args.world_size,
        rank=0
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Set up worker processes on master node
    processes = []
    for local_rank in range(args.local_workers):
        p = mp.Process(target=run_worker, args=(0, local_rank, model, optimizer, args))
        p.daemon = True
        processes.append(p)
        p.start()
    
    # Save model periodically
    os.makedirs(args.save_dir, exist_ok=True)
    start_time = time.time()
    step_counter = mp.Value('i', 0)
    
    # Monitor and save progress
    try:
        while True:
            current_steps = step_counter.value
            if current_steps >= args.max_steps:
                print(f"Reached max steps: {current_steps}")
                break
            
            elapsed = time.time() - start_time
            steps_per_sec = current_steps / elapsed if elapsed > 0 else 0
            
            print(f"Steps: {current_steps}/{args.max_steps} ({current_steps/args.max_steps*100:.1f}%), "
                  f"Steps/sec: {steps_per_sec:.1f}, Elapsed: {elapsed:.1f}s")
            
            # Save checkpoint periodically
            if current_steps > 0 and (current_steps % args.save_freq == 0 or current_steps >= args.max_steps):
                model_path = os.path.join(args.save_dir, f"a3c_chess_model_{current_steps}_steps.pt")
                torch.save(model.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")
            
            time.sleep(60)  # Check progress every minute
    
    except KeyboardInterrupt:
        print("Training interrupted. Saving final model...")
    
    # Save final model
    final_path = os.path.join(args.save_dir, "a3c_chess_model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")
    
    # Clean up processes
    for p in processes:
        p.terminate()

def run_worker_node(args):
    """
    Run as a worker node that connects to the master.
    Each worker node spawns multiple local workers.
    """
    print(f"Running as worker node on {socket.gethostname()}")
    
    # Initialize process group
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{args.master_addr}:{args.master_port}',
        world_size=args.world_size,
        rank=args.node_rank
    )
    
    # Create environment to get spaces
    env = make_env(0)()
    
    # Initialize model
    model = CNNMCTSActorCriticPolicy(
        env.observation_space,
        env.action_space,
        lambda _: args.learning_rate
    ).to(args.device)
    
    # Enable shared memory
    model.share_memory()
    
    # Receive model from master node
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    # Create optimizer (local to this node)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Launch local workers
    processes = []
    for local_rank in range(args.local_workers):
        p = mp.Process(target=run_worker, args=(args.node_rank, local_rank, model, optimizer, args))
        p.daemon = True
        processes.append(p)
        p.start()
    
    # Wait for processes to finish
    for p in processes:
        p.join()

def run_worker(node_rank, local_rank, shared_model, shared_optimizer, args):
    """
    Individual worker process that runs on either master or worker nodes.
    Collects trajectories, computes updates, and shares gradients.
    """
    # Calculate global rank
    global_rank = node_rank * args.local_workers + local_rank
    print(f"Worker process starting: node {node_rank}, local_rank {local_rank}, global_rank {global_rank}")
    
    # Create worker
    worker = A3CWorker(
        global_model=shared_model,
        optimizer=shared_optimizer,
        rank=global_rank,
        args=args,
        device=args.device
    )
    
    # Run worker
    worker.run()

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed A3C for Chess')
    
    # Distributed training settings
    parser.add_argument('--role', type=str, choices=['master', 'worker'], required=True,
                      help='Role of this machine (master or worker)')
    parser.add_argument('--master_addr', type=str, default='localhost',
                      help='IP address of the master node')
    parser.add_argument('--master_port', type=str, default='29500',
                      help='Port for master node communication')
    parser.add_argument('--node_rank', type=int, default=0,
                      help='Rank of this node in the distributed setup (0=master)')
    parser.add_argument('--world_size', type=int, default=1,
                      help='Total number of nodes (master + workers)')
    parser.add_argument('--local_workers', type=int, default=4,
                      help='Number of worker processes to run on this machine')
    
    # Training parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                      help='GAE lambda parameter')
    parser.add_argument('--ent_coef', type=float, default=0.1,
                      help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                      help='Value function coefficient')
    parser.add_argument('--clip_range', type=float, default=0.2,
                      help='PPO clipping parameter')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                      help='Maximum norm for gradients')
    parser.add_argument('--trajectory_length', type=int, default=128,
                      help='Length of trajectory to collect before update')
    parser.add_argument('--n_envs_per_worker', type=int, default=4,
                      help='Number of environments per worker')
    parser.add_argument('--max_steps', type=int, default=10000000,
                      help='Total number of steps to train for')
    parser.add_argument('--save_freq', type=int, default=50000,
                      help='Frequency (in timesteps) to save model checkpoints')
    parser.add_argument('--save_dir', type=str, default='data/models',
                      help='Directory to save models')
    
    # MCTS parameters
    parser.add_argument('--mcts_sims', type=int, default=100,
                      help='Number of MCTS simulations (0 to disable)')
    parser.add_argument('--mcts_freq', type=float, default=0.2,
                      help='Frequency of using MCTS (0-1)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set up multiprocessing 
    mp.set_start_method('spawn')
    
    # Run based on role
    if args.role == 'master':
        run_master(args)
    else:
        run_worker_node(args) 