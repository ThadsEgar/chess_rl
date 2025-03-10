#!/bin/bash

# Exit on any error
set -e

# Update package list and install essentials + OpenSpiel build dependencies
echo "Updating package list and installing essentials..."
sudo apt-get update
sudo apt-get install -y git python3 python3-pip python3-dev cmake g++ \
    libopenmpi-dev zlib1g-dev curl unzip

# Create project directory structure
echo "Creating project directory structure..."
mkdir -p ~/code/thads_projects
cd ~/code/thads_projects

# Clone your Git repo (using HTTPS to avoid SSH key setup)
echo "Cloning chess_rl repository..."
if [ ! -d "chess_rl" ]; then
    # Replace with your actual GitHub repository URL
    git clone https://github.com/thadsegar/chess_rl.git
else
    echo "Repository already exists, pulling latest changes..."
    cd chess_rl
    git pull
    cd ..
fi

cd chess_rl

# Install Miniconda if not already installed
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
fi

# Add Miniconda to PATH
export PATH="$HOME/miniconda3/bin:$PATH"

# Initialize conda for bash
echo "Initializing conda..."
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# Create and activate Conda environment
echo "Setting up Conda environment 'chess_rl'..."
$HOME/miniconda3/bin/conda create -n chess_rl python=3.10 -y
$HOME/miniconda3/bin/conda activate chess_rl || source $HOME/miniconda3/bin/activate chess_rl

# Install Python dependencies
echo "Installing Python dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3 gymnasium numpy matplotlib chess

# Install OpenSpiel using pip
echo "Installing OpenSpiel using pip..."
python3 -m pip install open_spiel

# Create necessary directories
mkdir -p data/logs
mkdir -p data/models

# Verify setup
echo "Verifying setup..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import stable_baselines3; print('Stable-Baselines3 version:', stable_baselines3.__version__)"
python -c "try: import pyspiel; print('OpenSpiel (pyspiel) installed successfully'); except ImportError: print('OpenSpiel installation failed')"

echo "Setup complete! To start working:"
echo "1. Run 'source ~/.bashrc'"
echo "2. Run 'conda activate chess_rl'"
echo "3. Run 'python -m src.train'" 