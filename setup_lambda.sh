#!/bin/bash
# Setup script for Lambda Labs A10 machine with chess_rl

set -e

echo "Setting up Lambda Labs environment for chess_rl..."

# Clone repository if it doesn't exist
REPO_DIR="code/thads_projects/chess_rl"
if [ ! -d "$HOME/$REPO_DIR" ]; then
    echo "Cloning repository..."
    mkdir -p $(dirname "$HOME/$REPO_DIR")
    cd $(dirname "$HOME/$REPO_DIR")
    git clone https://github.com/yourusername/chess_rl.git
    cd chess_rl
else
    echo "Repository already exists, updating..."
    cd "$HOME/$REPO_DIR"
    git pull
fi

# Ensure we're in the repo directory
cd "$HOME/$REPO_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "$HOME/miniforge3" ]; then
    echo "Installing Miniforge3..."
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p "$HOME/miniforge3"
    rm Miniforge3-$(uname)-$(uname -m).sh
fi

# Initialize conda
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# Create or update the environment
if conda env list | grep -q "chess_rl"; then
    echo "Updating chess_rl environment..."
    conda env update -f environment.yml
else
    echo "Creating chess_rl environment..."
    conda env create -f environment.yml
fi

# Activate the environment
conda activate chess_rl

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install PySpiel for chess environment
if ! python -c "import pyspiel" &> /dev/null; then
    echo "Installing PySpiel..."
    pip install open_spiel
fi

# Setup CUDA paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Test imports
echo "Testing imports..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"
python -c "import ray; print('Ray version:', ray.__version__)"
python -c "import pyspiel; print('PySpiel installed successfully')"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import chess; print('Python-chess version:', chess.__version__)"

# Create data directory structure if it doesn't exist
mkdir -p data/models
mkdir -p checkpoints
mkdir -p self_play_data

echo ""
echo "Setup complete! Your Lambda Labs environment is ready for chess_rl."
echo ""
echo "To start using it:"
echo "  1. Run 'conda activate chess_rl'"
echo "  2. Navigate to $REPO_DIR"
echo "  3. Run 'python -m src.run_ray_a3c --mode train' to start training"
echo ""
echo "GPU Information:"
nvidia-smi

echo ""
echo "Creating helpful aliases..."
echo "alias chess_train='cd $HOME/$REPO_DIR && conda activate chess_rl && python -m src.run_ray_a3c --mode train'" >> ~/.bashrc
echo "alias chess_eval='cd $HOME/$REPO_DIR && conda activate chess_rl && python -m src.run_ray_a3c --mode evaluate'" >> ~/.bashrc
echo "alias chess_test='cd $HOME/$REPO_DIR && conda activate chess_rl && python -m src.test_model'" >> ~/.bashrc

echo "Done! Restart your shell or run 'source ~/.bashrc' to enable the aliases." 