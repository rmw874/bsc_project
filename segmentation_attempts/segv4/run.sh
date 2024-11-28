#!/bin/bash
#SBATCH --job-name=train-model     # Job name
#SBATCH --output=slurm_logs/%x_%j.out # Output file where x is the job name and j is the job ID
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem=8G                   # Memory required per node
#SBATCH --time=02:00:00            # Time limit dys:hrs:min:sec
#SBATCH --gres=gpu:2              # Request 1 GPU
#SBATCH --verbose                  # Enable verbose output for debugging

# Load Conda
source /home/dxq257/miniforge3/etc/profile.d/conda.sh 

# Activate the 'mgr' environment
echo "Activating conda environment: mgr"
conda activate mgr

# Confirm environment activation
echo "Using Python from $(which python)"
python --version

# Run your Python script
echo "Starting training script"
python -u /home/rmw874/piratbog/segmentationV4/scripts2/train.py > /home/rmw874/piratbog/segmentationV4/slurm_logs/test.out
# python -u /home/dxq257/piratbog/segmentationV4/scripts2/train.py > /home/dxq257/piratbog/segmentationV4/slurm_logs/test.out
