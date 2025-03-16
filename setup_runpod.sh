#!/bin/bash
# setup_runpod.sh - Setup script for Chess RL on RunPod instances
# Adapted from Vast.ai setup script

set -e  # Exit on error

echo "==============================================="
echo "Setting up Chess RL environment on RunPod"
echo "==============================================="

# RunPod workspace directory
WORKSPACE="/workspace"
DATA_DIR="$WORKSPACE/data"

# Check if CUDA is available (using RunPod environment variables first)
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "RunPod GPU environment detected!"
    GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    echo "Found $GPU_COUNT GPUs through CUDA_VISIBLE_DEVICES"
    nvidia-smi
elif command -v nvidia-smi &> /dev/null; then
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

# Check if we're using a standard RunPod template with preinstalled components
PYTORCH_INSTALLED=false
CONDA_INSTALLED=false

# Check if PyTorch is already installed
if python3 -c "import torch" &> /dev/null; then
    echo "PyTorch is already installed."
    python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
    PYTORCH_INSTALLED=true
fi

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda is already installed."
    CONDA_INSTALLED=true
fi

# Update system packages
echo "Updating system packages..."
apt-get update -y
apt-get install -y build-essential cmake wget curl git htop tmux

# Install system dependencies for Python packages (especially matplotlib)
echo "Installing system dependencies for Python packages..."
apt-get install -y pkg-config libfreetype6-dev libpng-dev libffi-dev \
                   python3-dev python3-tk python3-setuptools \
                   libssl-dev zlib1g-dev libbz2-dev liblzma-dev \
                   libncurses5-dev libreadline-dev libsqlite3-dev

# Install Miniconda if not already installed
if [ "$CONDA_INSTALLED" = false ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $WORKSPACE/miniconda
    rm miniconda.sh
    
    # Add conda to path
    echo 'export PATH="$WORKSPACE/miniconda/bin:$PATH"' >> ~/.bashrc
    export PATH="$WORKSPACE/miniconda/bin:$PATH"
    
    # Initialize conda for bash
    conda init bash
    source ~/.bashrc
fi

# Create and activate conda environment
echo "Setting up conda environment..."
if ! conda info --envs | grep -q "chess_rl"; then
    conda create -y -n chess_rl python=3.10
fi
source activate chess_rl || conda activate chess_rl

# Install PyTorch with CUDA support if not already installed
if [ "$PYTORCH_INSTALLED" = false ]; then
    echo "Installing PyTorch with CUDA support..."
    
    # Try installing with CUDA 11.8 first
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || {
        echo "Failed to install PyTorch with CUDA 11.8, trying CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
            echo "Failed with CUDA packages, falling back to default PyTorch (will use available CUDA if detected)..."
            pip install torch torchvision torchaudio
        }
    }
    
    # Verify installation
    python -c "import torch; print('PyTorch installed successfully. Version:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
fi

# Install core scientific packages with conda (better dependency handling)
echo "Installing core scientific packages with conda..."
conda install -y matplotlib pandas numpy scipy

# Install Ray and RLlib
echo "Installing Ray and RLlib (latest version)..."
pip install "ray[rllib]" "ray[default]" "ray[tune]"

# Check Ray installation
python -c "import ray; print('Ray installed successfully. Version:', ray.__version__)"

# Create compatibility layer for different Ray versions
mkdir -p $REPO_DIR/src/compat
cat > $REPO_DIR/src/compat/__init__.py << 'EOF'
# Compatibility layer for different Ray versions
import ray

# Get Ray version (first two components like 2.9)
RAY_VERSION = '.'.join(ray.__version__.split('.')[:2])
RAY_VERSION_FLOAT = float(RAY_VERSION)

print(f"Detected Ray version: {ray.__version__}")

# If using Ray 2.9.0 or older code with newer Ray versions
if RAY_VERSION_FLOAT > 2.9:
    try:
        # For newer Ray versions, provide backwards compatibility
        print("Setting up compatibility layer for newer Ray version")
        
        # Compatibility imports that might be needed
        try:
            from ray.rllib.models.torch.attention_net import AttentionWrapper
        except ImportError:
            # Define compatibility classes if needed
            pass
    except Exception as e:
        print(f"Warning: Error setting up Ray compatibility layer: {e}")
EOF

# Install Gymnasium and related packages
echo "Installing Gymnasium and dependencies..."
pip install gymnasium python-chess tensorboard

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

# Create persistent directories in the RunPod volume
mkdir -p $DATA_DIR
mkdir -p $DATA_DIR/rllib_checkpoints
mkdir -p $DATA_DIR/logs
mkdir -p $DATA_DIR/models
mkdir -p $DATA_DIR/self_play_data

# Repository setup - check if we're in a git repo or need to clone
REPO_DIR="$WORKSPACE/chess_rl"
if [ ! -d "$REPO_DIR" ]; then
    echo "Creating chess_rl directory..."
    mkdir -p $REPO_DIR
fi

cd $REPO_DIR

# Check if we're already in a git repo or need to clone
if [ ! -d ".git" ]; then
    echo "Cloning Chess RL repository..."
    # If repository doesn't exist, clone it
    # Remove this comment and uncomment the next line with your actual repo URL
    # git clone https://github.com/YOUR_USERNAME/chess_rl.git .
    # Or copy existing code if already in the workspace
    if [ -d "$WORKSPACE/src" ]; then
        echo "Found existing code in workspace, copying..."
        cp -r $WORKSPACE/src .
        cp -r $WORKSPACE/*.py ./ 2>/dev/null || true
    fi
else
    echo "Git repository already initialized."
    # Optionally, pull latest changes
    # git pull
fi

# Create symbolic links for persistent storage in RunPod
if [ ! -L "rllib_checkpoints" ]; then
    ln -sf $DATA_DIR/rllib_checkpoints rllib_checkpoints
fi

if [ ! -L "logs" ]; then
    ln -sf $DATA_DIR/logs logs
fi

if [ ! -L "models" ]; then
    ln -sf $DATA_DIR/models models
fi

if [ ! -L "self_play_data" ]; then
    ln -sf $DATA_DIR/self_play_data self_play_data
fi

# Optimize config based on detected hardware
echo "Generating optimized configuration for $CPU_COUNT CPUs and $GPU_COUNT GPUs..."

# Calculate optimal worker count (leave some cores for system and driver)
if [ $CPU_COUNT -gt 16 ]; then
    WORKER_COUNT=$(($CPU_COUNT / 4))  # Use 25% of CPUs for workers on large instances
else
    WORKER_COUNT=$(($CPU_COUNT - 4))  # Leave 4 cores for system on smaller instances
fi
if [ $WORKER_COUNT -lt 1 ]; then
    WORKER_COUNT=1  # Ensure at least one worker
fi
echo "Recommended worker count: $WORKER_COUNT"

# Set batch size based on GPU count and VRAM
if [ $GPU_COUNT -ge 4 ]; then
    echo "Multi-GPU setup (4+ GPUs) detected. Optimizing for $GPU_COUNT GPUs."
    echo "Recommended train_batch_size: 65536"
    echo "Recommended sgd_minibatch_size: 4096"
    BATCH_SIZE="65536"
    MINIBATCH_SIZE="4096"
elif [ $GPU_COUNT -ge 2 ]; then
    echo "Multi-GPU setup (2 GPUs) detected. Optimizing for $GPU_COUNT GPUs."
    echo "Recommended train_batch_size: 32768"
    echo "Recommended sgd_minibatch_size: 2048"
    BATCH_SIZE="32768"
    MINIBATCH_SIZE="2048"
else
    echo "Single GPU setup detected."
    echo "Recommended train_batch_size: 16384"
    echo "Recommended sgd_minibatch_size: 1024"
    BATCH_SIZE="16384"
    MINIBATCH_SIZE="1024"
fi

# Optimize system for training
echo "Optimizing system for RL training..."
# Only run if we have sudo access
if command -v sudo &> /dev/null; then
    echo "vm.overcommit_memory = 1" | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
else
    echo "No sudo access. Skipping system optimization."
fi

# Configure PyTorch for better performance
echo 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"' >> ~/.bashrc
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

# Create a training script with optimal settings for this hardware
cat > start_training.sh << EOL
#!/bin/bash
source activate chess_rl || conda activate chess_rl
cd \$(dirname \$0)

# Use RunPod-optimized settings
python src/run_rllib.py train \\
    --device cuda \\
    --num_workers $WORKER_COUNT \\
    --dashboard \\
    --max_iterations 5000 \\
    --entropy_coeff 0.05 \\
    --checkpoint_dir $DATA_DIR/rllib_checkpoints

# Add --checkpoint path/to/checkpoint parameter to resume from a checkpoint
EOL

chmod +x start_training.sh

# Create evaluation script
cat > evaluate_model.sh << EOL
#!/bin/bash
source activate chess_rl || conda activate chess_rl
cd \$(dirname \$0)

# Usage: ./evaluate_model.sh path/to/checkpoint
if [ -z "\$1" ]; then
    echo "Error: No checkpoint provided."
    echo "Usage: ./evaluate_model.sh path/to/checkpoint"
    exit 1
fi

python src/run_rllib.py eval \\
    --device cuda \\
    --checkpoint "\$1" \\
    --render
EOL

chmod +x evaluate_model.sh

# Print GPU information for verification
echo "==============================================="
echo "Environment setup complete!"
echo "GPU configuration:"
nvidia-smi
echo "==============================================="

# Test imports to verify installation
echo "==============================================="
echo "Testing imports..."

test_import() {
    package=$1
    command=$2
    echo -n "Testing $package: "
    if python -c "$command" 2>/dev/null; then
        echo "[SUCCESS]"
        return 0
    else
        echo "[FAILED]"
        echo "  - Attempting to reinstall $package..."
        if conda install -y $package 2>/dev/null || pip install $package; then
            echo "  - Reinstalled $package, retesting..."
            if python -c "$command" 2>/dev/null; then
                echo "  - [SUCCESS] after reinstall"
                return 0
            else
                echo "  - [FAILED] after reinstall"
                return 1
            fi
        else
            echo "  - Reinstallation failed"
            return 1
        fi
    fi
}

test_import "torch" "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"
test_import "ray" "import ray; print('Ray version:', ray.__version__)"
test_import "gymnasium" "import gymnasium; print('Gymnasium version:', gymnasium.__version__)"
test_import "numpy" "import numpy; print('NumPy version:', numpy.__version__)"
test_import "matplotlib" "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"
test_import "chess" "import chess; print('Python-chess version:', chess.__version__)"
test_import "pandas" "import pandas; print('Pandas version:', pandas.__version__)"

echo "==============================================="

# Set up instructions for Ray dashboard
echo "==============================================="
echo "To start Ray dashboard for monitoring on RunPod:"
echo "ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265"
echo "Then access through RunPod UI using port 8265"
echo "==============================================="

# Save environment variables for future sessions
cat > setup_env.sh << 'EOL'
#!/bin/bash
export PATH="$WORKSPACE/miniconda/bin:$PATH"
source activate chess_rl || conda activate chess_rl
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

# Add Ray compatibility import to run_rllib.py if using newer Ray version
if python -c "import ray; from pkg_resources import parse_version; exit(0 if parse_version(ray.__version__) > parse_version('2.9.0') else 1)" 2>/dev/null; then
    echo "Adding compatibility import to run_rllib.py..."
    
    # Check if the file exists
    if [ -f "$REPO_DIR/src/run_rllib.py" ]; then
        # Back up the original file
        cp "$REPO_DIR/src/run_rllib.py" "$REPO_DIR/src/run_rllib.py.bak"
        
        # Add the import after the existing imports
        sed -i '0,/^import/s/^import/# Import compatibility layer for Ray version differences\ntry:\n    from compat import *\nexcept ImportError:\n    print("Compatibility layer not found, proceeding without it")\n\nimport/' "$REPO_DIR/src/run_rllib.py"
        
        echo "Added compatibility import to run_rllib.py"
    else
        echo "Warning: run_rllib.py not found, skipping modification"
    fi
fi

echo "==============================================="
echo "RunPod Setup Complete!"
echo "To start training, run:"
echo "./tmux_train.sh"
echo ""
echo "Or manually with:"
echo "./start_training.sh"
echo "===============================================" 