# Chess RL on Vast.ai

This guide explains how to set up and run your Chess Reinforcement Learning project on Vast.ai.

## Key Dependencies

Your Chess RL implementation relies on several critical dependencies:

- **PyTorch**: For neural network operations and GPU acceleration
- **Ray/RLlib**: For distributed reinforcement learning
- **OpenSpiel**: Framework providing the chess environment
- **Python-Chess**: Library for chess game representation and moves

The setup script automatically installs all these dependencies.

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

## Multi-GPU Scaling

Your setup with 4 GPUs and 128 CPU cores is powerful and can be optimized as follows:

### 4-GPU Configuration
For optimal performance with 4 GPUs:

```python
"num_gpus": 4,
"train_batch_size": 65536,  # 16K samples per GPU
"sgd_minibatch_size": 4096, # Larger minibatches for efficient processing
"num_sgd_iter": 5,          # Keep iteration count moderate
```

### 128 CPU Core Allocation
- Use 122-124 workers (`--num_workers 124`)
- Reserve 4-6 cores for system and driver processes
- With `num_envs_per_worker=2`, this provides 244-248 parallel environments

### Memory Management
- Ensure your script uses `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"`
- Consider enabling GPU memory sharing with `"_memory_shared_prefix": "chess_rl_"` if running low on GPU memory

### Batch Distribution Across 4 GPUs
With a total batch size of 65,536:
- Each GPU processes ~16,384 samples per iteration
- Each GPU handles 4 minibatches of 4,096 samples each
- All gradients are synchronized across GPUs automatically

This configuration maximizes both your CPU cores for environment simulation and your GPUs for neural network training.

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