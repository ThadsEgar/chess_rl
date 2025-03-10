# Ray-based A3C Chess Training

This document explains how to use the Ray-based Asynchronous Advantage Actor-Critic (A3C) implementation for training chess models. This implementation makes connections optional and runs the process by default, addressing the issues with the previous implementation.

## Advantages of Ray-based A3C

- **Fully Distributed**: Uses Ray to handle distributed computing across multiple machines
- **Fault Tolerant**: Can continue training even if some workers disconnect
- **Auto-scaling**: Workers can join or leave the cluster at any time without disrupting training
- **Resource Efficient**: Automatically utilizes available CPU cores and GPUs
- **Simple to Use**: Single command to start training locally or in distributed mode

## Installation Requirements

Before running the Ray-based A3C implementation, install the required dependencies:

```bash
pip install "ray[default]>=2.0.0" torch numpy gym
```

## Running the Training

### 1. Single Machine (Local) Mode

The simplest way to run the training is in local mode, which uses all available cores on your machine:

```bash
python run_ray_a3c.py --mode local
```

This will automatically:
- Detect the optimal number of CPU cores to use
- Initialize Ray locally
- Start training with the detected resources

### 2. Distributed Mode (Lambda Labs + Your Laptop)

#### Step 1: Start the head node on Lambda Labs

On your Lambda Labs instance, run:

```bash
python run_ray_a3c.py --mode head --device cuda --num_actors 16 --mcts_sims 100
```

This will:
- Start a Ray head node on Lambda Labs
- Print the address for workers to connect to
- Begin training using the GPU resources available

#### Step 2: Connect your laptop as a worker

On your laptop, run the command displayed by the head node:

```bash
python run_ray_a3c.py --mode worker --head_address <LAMBDA_IP>:6379
```

Your laptop will now:
- Connect to the Lambda Labs instance
- Contribute its computing resources to the cluster
- Help with collecting experiences and computing gradients

You can disconnect your laptop at any time without stopping the training on Lambda Labs.

## Configuration Options

The script provides several options for customizing the training:

```
--mode {local,head,worker}  Running mode: local, head, or worker
--head_address HEAD_ADDRESS Address of Ray head node (for worker mode)
--num_actors NUM_ACTORS     Number of actor processes
--n_envs_per_actor N_ENVS   Number of environments per actor
--device {cuda,cpu}         Device to use (cuda or cpu)
--learning_rate LR          Learning rate
--ent_coef ENT_COEF         Entropy coefficient
--max_steps MAX_STEPS       Total steps to train for
--mcts_sims MCTS_SIMS       Number of MCTS simulations (0 to disable)
--mcts_freq MCTS_FREQ       Frequency of using MCTS (0-1)
```

### Resource Optimization

- **Lambda Labs (GPU)**: Use higher `--num_actors` (8-16) and `--mcts_sims` (50-200)
- **Laptop (CPU)**: Use lower `--num_actors` (equal to CPU cores - 1) and consider setting `--mcts_sims 0` to disable MCTS

## Monitoring Training

The training script provides real-time logging of:
- Training updates and step counts
- Steps per second (training speed)
- Mean reward over time
- Checkpoint saving information

## Resuming Training

To resume training from a checkpoint, simply restart the head node with the same parameters. Ray will automatically load the latest available checkpoint.

## Implementation Details

The key components of the Ray-based A3C implementation:

1. **A3CTrainer**: Coordinates the entire training process
2. **A3CLearner**: Maintains the global model and applies gradients
3. **A3CActor**: Collects experiences and computes gradients remotely

The Ray implementation uses a mix of A3C (asynchronous updates) and PPO (proximal policy optimization) techniques to ensure stable and efficient learning.

## Troubleshooting

### Connection Issues
- Ensure the firewall on the head node allows connections on port 6379
- Check that the IP address is correctly specified and accessible 
- Try running `ping <LAMBDA_IP>` to verify connectivity

### Resource Issues
- If your Lambda instance becomes unresponsive, reduce `--num_actors` or `--mcts_sims`
- For laptops with limited memory, reduce `--n_envs_per_actor` to 2

### Performance Issues
- If training is slow, try reducing `--mcts_sims` or setting it to 0
- Adjust `--ent_coef` if training becomes stuck (higher values encourage exploration)

## Advantages Over Previous Implementation

This Ray-based implementation addresses several limitations of the previous approach:

1. **No Dependency Wait**: Lambda Labs will train even without your laptop connected
2. **Dynamic Scaling**: Workers can join and leave at any time
3. **Simpler Setup**: No manual port configuration or process management
4. **Better Fault Tolerance**: Training continues even if workers crash
5. **Resource Efficiency**: Automatically balances workload across available resources 