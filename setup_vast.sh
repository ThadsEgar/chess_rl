#!/bin/bash
# setup_vast.sh - Setup script for Chess RL on Vast.ai instances
# Designed for instances with CUDA-enabled GPUs

set -e  # Exit on error

echo "==============================================="
echo "Setting up Chess RL environment on Vast.ai"
echo "==============================================="

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi
    
    # Count available GPUs
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Found $GPU_COUNT GPUs"
else
    echo "WARNING: No GPU detected. Training will be very slow!"
    GPU_COUNT=0
fi

# Count CPU cores
CPU_COUNT=$(nproc)
echo "Found $CPU_COUNT CPU cores"

# Update system packages
echo "Updating system packages..."
apt-get update -y
apt-get install -y build-essential cmake wget curl git htop tmux

# Install Miniconda if not already installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    
    # Add conda to path
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# Create and activate conda environment
echo "Setting up conda environment..."
conda create -y -n chess_rl python=3.10
source activate chess_rl || conda activate chess_rl

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Ray and RLlib
echo "Installing Ray and RLlib..."
pip install ray[rllib]==2.9.0
pip install "ray[default]"

# Install Gymnasium and related packages
echo "Installing Gymnasium and dependencies..."
pip install gymnasium python-chess matplotlib pandas tensorboard

# Install PySpiel for chess environment
echo "Checking for PySpiel..."
if ! python -c "import pyspiel" &> /dev/null; then
    echo "Installing PySpiel..."
    pip install open_spiel
fi

# Setup CUDA paths
echo "Setting up CUDA paths..."
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib/" >> ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Clone the repository if it doesn't exist
if [ ! -d "chess_rl" ]; then
    echo "Cloning Chess RL repository..."
    git clone https://github.com/ThadsEgar/chess_rl.git
fi

cd chess_rl

# Create directories for checkpoints and logs
echo "Creating directories for checkpoints and logs..."
mkdir -p rllib_checkpoints
mkdir -p logs

# Optimize config based on detected hardware
echo "Generating optimized configuration for $CPU_COUNT CPUs and $GPU_COUNT GPUs..."

# Calculate optimal worker count (leave some cores for system and driver)
WORKER_COUNT=$(($CPU_COUNT - 3))
echo "Recommended worker count: $WORKER_COUNT"

# Set batch size based on GPU count
if [ $GPU_COUNT -ge 2 ]; then
    echo "Multi-GPU setup detected. Optimizing for $GPU_COUNT GPUs."
    echo "Recommended train_batch_size: 32768"
    echo "Recommended sgd_minibatch_size: 2048"
else
    echo "Single GPU setup detected."
    echo "Recommended train_batch_size: 16384"
    echo "Recommended sgd_minibatch_size: 1024"
fi

# Optimize system for training
echo "Optimizing system for RL training..."
echo "vm.overcommit_memory = 1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Configure PyTorch for better performance
echo 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"' >> ~/.bashrc

# Create a tmux session for persistent training
echo "Setting up tmux for persistent training sessions..."
cat > start_training.sh << EOL
#!/bin/bash
source $HOME/miniconda/bin/activate
conda activate chess_rl
cd \$(dirname \$0)
python src/run_rllib.py train --device cuda --num_workers $WORKER_COUNT --dashboard
EOL
chmod +x start_training.sh

# Print GPU information for verification
echo "==============================================="
echo "Environment setup complete!"
echo "GPU configuration:"
nvidia-smi
echo "==============================================="

# Test imports to verify installation
echo "Testing imports..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"
python -c "import ray; print('Ray version:', ray.__version__)"
python -c "import pyspiel; print('PySpiel installed successfully')"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import chess; print('Python-chess version:', chess.__version__)"

# Create additional directories for data storage
echo "Creating additional data directories..."
mkdir -p data/models
mkdir -p self_play_data
mkdir -p checkpoints

# Set up Ray dashboard for monitoring (optional)
echo "To start Ray dashboard for monitoring:"
echo "ray start --head --dashboard-host=0.0.0.0"

echo "==============================================="
echo "Setup complete! You can now run training with:"
echo "./start_training.sh"
echo "or"
echo "python src/run_rllib.py train --device cuda --num_workers $WORKER_COUNT"
echo "==============================================="

# Save environment variables for future sessions
cat > setup_env.sh << 'EOL'
#!/bin/bash
source $HOME/miniconda/bin/activate
conda activate chess_rl
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
EOL

chmod +x setup_env.sh
echo "Created setup_env.sh - run 'source setup_env.sh' in new sessions"

# Create a helper script to start tmux session with training
cat > tmux_train.sh << 'EOL'
#!/bin/bash
SESSION="chess_training"
tmux new-session -d -s $SESSION
tmux send-keys -t $SESSION "source setup_env.sh" C-m
tmux send-keys -t $SESSION "./start_training.sh" C-m
echo "Started training in tmux session '$SESSION'"
echo "Connect to session with: tmux attach -t $SESSION"
echo "Detach from session with: Ctrl+b then d"
EOL

chmod +x tmux_train.sh
echo "Created tmux_train.sh - run to start training in a persistent tmux session" 