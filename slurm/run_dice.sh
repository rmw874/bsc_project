#!/bin/bash
#SBATCH --job-name=train-shiplog-unet_w_dice # Job name
#SBATCH --output=slurm_logs/%x_%j.out        # Output file with job name and ID
#SBATCH --cpus-per-task=4                    # Number of CPU cores per task
#SBATCH --mem=32G                            # Memory required per node
#SBATCH --gres=gpu:1                         # Request 3 GPUs
#SBATCH --exclude=hendrixgpu12fl,hendrixgpu11fl

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
python -u /home/rmw874/piratbog/segv4/scripts/train_w_dice.py