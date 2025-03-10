# Setting Up Distributed Ray Training for Chess RL

This guide explains how to set up distributed training for the chess RL project between a Lambda Labs instance and a Mac.

## Lambda Labs Setup (Head Node)

### 1. First-time Setup

On your Lambda Labs instance, run these commands:

```bash
# Install conda if not already installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate

# Create and activate environment
conda create -n chess_rl python=3.10 -y
conda activate chess_rl

# Clone the repository
mkdir -p ~/code/thads_projects
cd ~/code/thads_projects
git clone <your-repo-url> chess_rl
cd chess_rl

# Install dependencies
pip install -r requirements.txt
```

### 2. Fix CUDA Issues (Optional)

If you want to use the GPU, ensure the correct CUDA version is installed:

```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch version (example for CUDA 11.8)
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. Start the Head Node

```bash
# Navigate to the project
cd ~/code/thads_projects/chess_rl
conda activate chess_rl

# Start Ray as head node
# For CPU-only mode (recommended for distributed setup)
python -m src.run_ray_a3c --mode train --device cpu --force_cpu --head --distributed \
    --dashboard --redis_password your_password --num_workers 4 --mcts_workers 0

# Alternative: For GPU mode (if CUDA is working)
python -m src.run_ray_a3c --mode train --device cuda --head --distributed \
    --dashboard --redis_password your_password --num_workers 4 --mcts_workers 2 --mcts_sims 50
```

Note: When starting the head node, look for the line:
```
INFO worker.py:1832 -- Started a local Ray instance. View the dashboard at X.X.X.X:8265
```
This IP address is what you'll need to connect from your Mac.

## Mac Setup (Worker Node)

### 1. First-time Setup

On your Mac, run these commands:

```bash
# Install conda if not already installed
# For Intel Macs
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# For Apple Silicon Macs
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Create and activate environment
conda create -n chess_rl python=3.10 -y
conda activate chess_rl

# Clone the repository
mkdir -p ~/Code/thads_projects
cd ~/Code/thads_projects
git clone <your-repo-url> chess_rl
cd chess_rl

# Install dependencies
pip install -r requirements.txt
```

### 2. Connect to the Lambda Labs Instance

There are two ways to connect to the Lambda Labs instance:

#### Option 1: Direct Connection (if Lambda allows incoming connections)

```bash
# Navigate to the project
cd ~/Code/thads_projects/chess_rl
conda activate chess_rl

# Connect to the Lambda Labs head node
python -m src.run_ray_a3c --mode train --device auto --distributed \
    --head_address "lambda-ip:6379" --redis_password your_password \
    --num_workers 2
```

Replace `lambda-ip` with the actual IP address of your Lambda Labs instance.

#### Option 2: SSH Tunneling (if Lambda has firewall restrictions)

```bash
# Open an SSH tunnel in a terminal window
ssh -L 6379:localhost:6379 -L 8265:localhost:8265 username@lambda-ip

# In a different terminal, start your worker
cd ~/Code/thads_projects/chess_rl
conda activate chess_rl

python -m src.run_ray_a3c --mode train --device auto --distributed \
    --head_address "localhost:6379" --redis_password your_password \
    --num_workers 2
```

## Monitoring Training

### Ray Dashboard

Access the Ray dashboard at:
```
http://lambda-ip:8265
```

Or if using SSH tunneling:
```
http://localhost:8265
```

### Log Files

Check logs in the log files on both machines:
```bash
tail -f ray_a3c.log
```

## Common Issues and Solutions

### Network Issues

If you have connectivity problems:

1. **Check Firewall Rules**: Make sure port 6379 is open on Lambda Labs.
2. **Use SSH Tunneling**: If direct connection doesn't work, use SSH tunneling.
3. **Check Password**: Make sure the redis password matches on both machines.

### Memory Issues

If you run into memory errors:

1. **Reduce Workers**: Use `--num_workers 2` on Mac and `--num_workers 4` on Lambda.
2. **Disable MCTS**: Use `--mcts_workers 0` to disable MCTS.
3. **Force CPU Mode**: Use `--force_cpu` to avoid CUDA issues.

### Synchronization Issues

If code changes aren't reflecting:

1. **Pull Latest Code**: Make sure both machines have the latest code with `git pull`.
2. **Check Directory Paths**: Make sure you're in the correct directory on both machines.
3. **Reload Ray**: If necessary, restart the Ray clusters on both machines.

## Tips for Efficient Training

1. **Balance Workers**: Assign more workers to Lambda and fewer to Mac for better performance.
2. **Device Selection**: Use CPU on both machines for consistency, or GPU on Lambda and CPU on Mac.
3. **Checkpoint Sharing**: Periodically copy checkpoints from Lambda to Mac for evaluation.
4. **Resource Monitoring**: Use `htop` on Linux and Activity Monitor on Mac to watch resource usage.
5. **Regular Saving**: Use a small checkpoint interval (e.g., `--checkpoint_interval 5`) to avoid losing progress. 