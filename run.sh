#!/bin/bash
#SBATCH --job-name=train-shiplog-unet      # Job name
#SBATCH --output=slurm_logs/%x_%j.out  # Output file with job name and ID
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --mem=80G                   # Memory required per node
#SBATCH --time=20:00:00             # Time limit (d-hh:mm:ss)
#SBATCH --gres=gpu:1             # Request 2 GPUs
#SBATCH --partition=gpu               # Use the general GPU partition
#SBATCH --verbose                   # Enable verbose output for debugging


# Load Conda
source /home/dxq257/miniforge3/etc/profile.d/conda.sh 

# Activate the 'mgr' environment
echo "Activating conda environment: mgr"
conda activate mgr

# Confirm environment activation
echo "Using Python from $(which python)"
python --version

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Run your Python script
echo "Starting training script"
python -u scripts/train_w_dice_ugly.py
