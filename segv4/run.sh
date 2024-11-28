#!/bin/bash
#SBATCH --job-name=train-shiplog-unet      # Job name
#SBATCH --output=slurm_logs/%x_%j.out  # Output file with job name and ID
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --mem=24G                   # Memory required per node
#SBATCH --time=02:00:00             # Time limit (d-hh:mm:ss)
#SBATCH --gres=gpu:2             # Request 2 GPUs
#SBATCH --partition=gpu               # Use the general GPU partition

# Load Conda
source /home/rmw874/miniforge3/etc/profile.d/conda.sh 

# Activate the 'piratbog' environment
echo "Activating conda environment: piratbog"
conda activate piratbog

# Confirm environment activation
echo "Using Python from $(which python)"
python --version

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Run your Python script
echo "Starting training script"
python -u /home/rmw874/piratbog/segv4/scripts/train.py