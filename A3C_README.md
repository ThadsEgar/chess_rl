# Distributed A3C Chess Training

This document explains how to set up and use the Asynchronous Advantage Actor-Critic (A3C) implementation for training chess models across multiple machines.

## What is A3C?

A3C stands for Asynchronous Advantage Actor-Critic, which is a distributed reinforcement learning algorithm that allows for parallel training across multiple processors and machines. Unlike A2C, which synchronizes updates, A3C allows workers to update a global model asynchronously, which can significantly speed up training.

Key benefits of A3C:
- **Improved sample efficiency**: Multiple agents collect experiences in parallel
- **Better stability**: Asynchronous updates can lead to more stable training
- **Scalability**: Can run efficiently across multiple machines

## Files Overview

- `src/a3c.py`: Core implementation of the A3C algorithm
- `src/distributed_train.py`: Script for distributed training across multiple machines
- `src/cnn.py`: The CNN architecture used for the policy and value networks

## Setup for Distributed Training

### Requirements

- PyTorch with distributed support
- Multiple machines (e.g., your laptop and Lambda Labs instance)
- Network connectivity between machines
- Same codebase on all machines

### Setting Up Environments

On each machine, make sure you have:

1. The same Python environment with all dependencies installed
2. The chess_rl codebase with identical files
3. Network access to communicate between machines

## Running Distributed Training

### 1. Master Node Setup (e.g., Lambda Labs)

The master node coordinates the training and aggregates updates from all workers.

```bash
python -m src.distributed_train \
    --role master \
    --master_addr <MASTER_IP_ADDRESS> \
    --master_port 29500 \
    --world_size 2 \
    --local_workers 8 \
    --device cuda \
    --learning_rate 2e-4 \
    --max_steps 50000000 \
    --save_freq 100000 \
    --mcts_sims 100 \
    --mcts_freq 0.2
```

- `master_addr`: Public IP address of the master node
- `world_size`: Total number of machines (nodes) in the distributed setup (including master)
- `local_workers`: Number of worker processes to run on this machine

### 2. Worker Node Setup (e.g., Your Laptop)

Worker nodes connect to the master and run parallel workers to collect experiences.

```bash
python -m src.distributed_train \
    --role worker \
    --master_addr <MASTER_IP_ADDRESS> \
    --master_port 29500 \
    --node_rank 1 \
    --world_size 2 \
    --local_workers 4 \
    --device cuda \
    --learning_rate 2e-4
```

- `node_rank`: Unique ID for this worker node (must be between 1 and world_size-1)
- `local_workers`: Adjust based on your machine's capabilities

## Training Considerations

### Hardware Optimization

- For Lambda Labs (high-end GPU): Use more local workers (8-16) and MCTS simulations
- For laptop: Use fewer local workers (2-4) and possibly disable MCTS to avoid overheating

### Communication

Ensure that:
- The master node is accessible from worker nodes (check firewall settings)
- The port specified in `master_port` is open for TCP communication
- All nodes can resolve the `master_addr` IP address

### Checkpointing and Recovery

Models are automatically saved to the `--save_dir` directory on the master node at intervals specified by `--save_freq`. Training can be resumed from a checkpoint with the `--checkpoint` argument.

## Tuning Parameters

- `--learning_rate`: Start with 2e-4, adjust based on training stability
- `--entropy_coef`: Higher values (0.1-0.2) encourage exploration
- `--trajectory_length`: 128-256 works well for most setups
- `--mcts_sims`: More simulations improve quality but slow down training (50-200)
- `--mcts_freq`: How often to use MCTS (0.2-0.5 is a good range)

## Monitoring Training

Training progress is logged on the master node, showing:
- Steps completed
- Training speed (steps/second)
- Checkpoint saving information

To monitor performance metrics in more detail, you may want to integrate TensorBoard by modifying the code to log relevant statistics.

## Troubleshooting

### Connection Issues
- Ensure the IP address in `master_addr` is correct and accessible
- Check that firewalls allow traffic on the specified port
- Test connectivity between nodes with a simple ping test

### Out of Memory
- Reduce `--local_workers` or `--n_envs_per_worker`
- Decrease `--mcts_sims` to use less memory

### Training Instability
- Lower the learning rate
- Adjust the entropy coefficient
- Reduce gradient clipping threshold (`--max_grad_norm`)

## Performance Comparison to PPO

A3C should provide better performance than PPO in these ways:
- Faster exploration of the state space due to parallel workers
- More updates per second due to asynchronous nature
- Better utilization of heterogeneous hardware (laptop + cloud GPU)

However, A3C may require more careful tuning and may be more sensitive to hyperparameters. 