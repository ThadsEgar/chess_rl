# Chess RL on Vast.ai

This guide explains how to set up and run your Chess Reinforcement Learning project on Vast.ai.

## Setting Up Your Vast.ai Instance

1. **Select an Instance**:
   - **Container Image**: Choose a recent PyTorch + CUDA image (e.g., `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`)
   - **GPUs**: 2x GPUs (RTX 3090 or better recommended)
   - **CPUs**: As many as available (ideally 30+ cores)
   - **RAM**: 32GB+ recommended
   - **Disk Space**: 20GB+ recommended
   - **On-demand vs. Interruptible**: Interruptible is cheaper but may stop unexpectedly

2. **SSH Access**:
   - Generate and add your SSH key for secure access
   - For persistent access, enable bidirectional ssh

3. **Set Up Persistent Storage**:
   - Enable persistent storage to keep your trained models between sessions

## Initial Setup

Once your instance is running, SSH into it and perform these steps:

```bash
# 1. Clone your repository
git clone https://github.com/YOUR_USERNAME/chess_rl.git
cd chess_rl

# 2. Make setup script executable
chmod +x setup_vast.sh

# 3. Run the setup script
./setup_vast.sh
```

## Running Training

After setup, start your training:

```bash
# 1. Activate environment (if starting a new session)
source setup_env.sh

# 2. Start training with optimal settings
python src/run_rllib.py train --device cuda --num_workers $(expr $(nproc) - 3)
```

## Monitoring Training

Monitor your training using:

1. **Ray Dashboard**:
   ```bash
   ray start --head --dashboard-host=0.0.0.0
   ```
   Access the dashboard at `http://YOUR_INSTANCE_IP:8265`

2. **TensorBoard**:
   ```bash
   tensorboard --logdir=./rllib_checkpoints --port=6006 --host=0.0.0.0
   ```
   Access TensorBoard at `http://YOUR_INSTANCE_IP:6006`

## Optimizing Performance

For best performance:

1. **Adjust Worker Count**: 
   ```bash
   # Use all CPUs minus a few for overhead
   --num_workers $(expr $(nproc) - 3)
   ```

2. **Multi-GPU Settings**:
   - Ensure `num_gpus` is set to 2 in your configuration
   - Batch size of 32768 recommended for 2 GPUs

3. **Checkpointing**:
   - Use `--checkpoint_interval` to save progress periodically
   - Example: `--checkpoint_interval 10`

## Downloading Trained Models

To download your trained models:

```bash
# On your local machine
scp -r vastai_user@YOUR_INSTANCE_IP:/path/to/chess_rl/rllib_checkpoints ./local_path
```

## Troubleshooting

- **Out of Memory Errors**: Reduce batch sizes in `src/run_rllib.py`
- **Slow Training**: Ensure you're using GPU (`--device cuda`)
- **Connection Issues**: Check Vast.ai console for instance status

## Advanced: Setting Up a Ray Cluster

For multi-node training across multiple Vast.ai instances:

1. **On Head Node**:
   ```bash
   ray start --head --port=6379 --dashboard-host=0.0.0.0
   ```

2. **On Worker Nodes**:
   ```bash
   ray start --address=HEAD_NODE_IP:6379
   ```

3. **Update Training Command**:
   ```bash
   python src/run_rllib.py train --device cuda --distributed --head_address=HEAD_NODE_IP:6379
   ``` 